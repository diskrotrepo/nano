"""Audio codec wrappers (the model/data seam).

Built as a thin facade so we can swap codecs without touching the model or
training code. Every codec guarantees the same interface:
  - SAMPLE_RATE, N_CODEBOOKS, VOCAB_SIZE, FRAME_RATE_HZ  (read as ``codec.X``)
  - encode(source) -> LongTensor[N_CODEBOOKS, T]          (T = ceil(samples / hop))
  - encode_batch(list) -> list[LongTensor[N_CODEBOOKS, T]]
  - decode(tokens)  -> FloatTensor[C, samples] | [samples] at SAMPLE_RATE

Two implementations:
  - ``DACodec``           — Descript DAC 44.1kHz, MONO, 9 cb, 1024 vocab, 86 Hz.
                            decode() returns mono [samples] (or [B, samples]).
  - ``SpectroStreamCodec`` — Magenta RealTime SpectroStream 48kHz, JOINT STEREO,
                            25 Hz, up to 64 RVQ cb (truncatable by a clean slice),
                            1024 vocab. decode() returns stereo [2, samples].
                            Used for v9. Requires the magenta_rt stack (Linux+CUDA),
                            so it is lazy-imported inside the class — importing this
                            module never pulls magenta_rt.

Pick one with ``get_codec()`` (env ``NANO_CODEC``: "dac" default, "spectrostream"
for v9).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import librosa
import torch


class DACodec:
    SAMPLE_RATE = 44100
    N_CODEBOOKS = 9
    VOCAB_SIZE = 1024
    FRAME_RATE_HZ = 44100 // 512  # 86 (DAC 44.1kHz uses hop=512)
    N_CHANNELS = 1  # mono

    def __init__(self, device: str = "cpu"):
        # Lazy import: constructing DACodec is what needs `dac`, so importing this
        # module (e.g. to use SpectroStreamCodec) never requires the DAC package.
        import dac

        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def _load_audio(self, path: str | Path) -> torch.Tensor:
        # librosa shells out to ffmpeg via audioread for mp3/m4a/etc. — works
        # with any ffmpeg version installed on PATH (no shared-lib ABI issues).
        y, _ = librosa.load(str(path), sr=self.SAMPLE_RATE, mono=True)
        return torch.from_numpy(y).unsqueeze(0)  # [1, samples]

    @torch.no_grad()
    def encode(self, source: str | Path | torch.Tensor) -> torch.Tensor:
        """Encode audio to DAC token codes.

        source: a filesystem path OR a mono [1, samples] tensor at SAMPLE_RATE.
        returns: LongTensor[N_CODEBOOKS, T_frames]
        """
        if isinstance(source, (str, Path)):
            wav = self._load_audio(source)
        else:
            wav = source
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            assert wav.shape[0] == 1, "expected mono input"
        x = wav.unsqueeze(0).to(self.device)  # [1, 1, samples]
        x = self.model.preprocess(x, self.SAMPLE_RATE)
        _, codes, _, _, _ = self.model.encode(x)
        return codes[0].detach().cpu().long()  # [K, T]

    @torch.no_grad()
    def encode_batch(self, audios: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode multiple audio tensors in a single DAC forward pass.

        audios: list of mono [1, samples] tensors at SAMPLE_RATE. Lengths may differ.
        returns: list of LongTensor[N_CODEBOOKS, T_i], with T_i = ceil(samples_i / hop).

        Amortizes GPU kernel launch and Python overhead across the batch. Pads to
        the longest audio with zeros, then crops each output to the frame count it
        would have had in a standalone encode."""
        if not audios:
            return []
        # DAC 44.1kHz uses hop_length=512; frame count is ceil(samples / hop).
        hop = self.SAMPLE_RATE // self.FRAME_RATE_HZ  # 512

        # Validate + collect lengths
        sample_lengths: list[int] = []
        for a in audios:
            if a.dim() == 1:
                a_ = a.unsqueeze(0)
            else:
                a_ = a
            assert a_.shape[0] == 1, "expected mono input"
            sample_lengths.append(int(a_.shape[-1]))
        max_samples = max(sample_lengths)

        # Pad and stack to [B, 1, max_samples]
        batch = torch.zeros(len(audios), 1, max_samples, dtype=audios[0].dtype)
        for i, a in enumerate(audios):
            a_ = a if a.dim() == 2 else a.unsqueeze(0)
            batch[i, 0, : a_.shape[-1]] = a_.squeeze(0)
        x = batch.to(self.device)
        x = self.model.preprocess(x, self.SAMPLE_RATE)
        _, codes, _, _, _ = self.model.encode(x)
        codes = codes.detach().cpu().long()  # [B, K, T_max]

        # Crop each item to the frame count its own length would produce.
        out: list[torch.Tensor] = []
        for i, samples in enumerate(sample_lengths):
            t_i = (samples + hop - 1) // hop
            out.append(codes[i, :, :t_i].contiguous())
        return out

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode DAC token codes to audio waveform.

        tokens: LongTensor[N_CODEBOOKS, T] or [B, N_CODEBOOKS, T]
        returns: FloatTensor[samples] (mono) if input was 2D, else [B, samples]
        """
        squeeze_batch = tokens.dim() == 2
        if squeeze_batch:
            tokens = tokens.unsqueeze(0)
        tokens = tokens.to(self.device)
        z, _, _ = self.model.quantizer.from_codes(tokens)
        audio = self.model.decode(z)  # [B, 1, samples]
        audio = audio.squeeze(1).detach().cpu()
        return audio[0] if squeeze_batch else audio


# Default stored RVQ depth for SpectroStream re-tokenization. The model's
# n_codebooks (the by-ear K, e.g. 16/24) is a SLICE of this — truncation is a
# byte-identical prefix (verified in the codec spike), so storing headroom lets
# K be retuned at train time without re-encoding the corpus.
SS_DEFAULT_DEPTH = 32


def bucket_target_samples(
    n_samples: int, hop: int, bucket_frames: int, max_samples: int
) -> int:
    """Padded sample count that rounds a clip's frame count UP to a multiple of
    ``bucket_frames``, so the SpectroStream STFT front-end sees only a HANDFUL of
    distinct input lengths (its cuFFT plans are keyed by length — hundreds of unique
    song lengths overflow the 512-plan cache and thrash "constantly creating new
    plans"; bucketing collapses that to a few reused plans).

    Clamped so the padded length never exceeds ``max_samples`` (the int32
    launch-config ceiling). Returns ``n_samples`` UNCHANGED when bucketing is off
    (``bucket_frames <= 0``), when the clip already lands on the grid, or when
    padding would breach the cap. The caller pads with zeros to this length,
    encodes, then crops the codes back to ``ceil(n_samples / hop)`` — crop-equivalent
    to an unpadded encode (guarded by ``modal_spectrostream_spike.bucket_check``)."""
    if bucket_frames <= 0:
        return n_samples
    true_frames = (n_samples + hop - 1) // hop
    tgt_frames = ((true_frames + bucket_frames - 1) // bucket_frames) * bucket_frames
    tgt_samples = tgt_frames * hop
    if tgt_samples <= n_samples or tgt_samples > max_samples:
        return n_samples
    return tgt_samples


def frame_safe_target_samples(tgt: int, hop: int, sample_rate: int, frame_rate: int) -> int:
    """Nudge a padded sample count off SpectroStream's float-ceil off-by-one boundary.

    ``SpectroStream.encode`` asserts ``embeddings.shape[1] == int(np.ceil(
    (T/sample_rate) * frame_rate))`` computed in float64 — but for ~4.8% of
    frame-aligned ``T`` that expression rounds ONE ABOVE the conv encoder's true
    frame count ``ceil(T/hop)`` (e.g. ``(13440/48000)*25 == 7.000000000000001 ->
    ceil 8``) and aborts the container with ``Expected (1,F+1,256) but got
    (1,F,256)``. When ``tgt`` sits on such a boundary, add half-frames until SS's
    float ``expected`` matches ``ceil(tgt/hop)`` — one half-frame always suffices.
    Returns ``tgt`` UNCHANGED for the other ~95% (guard inert → byte-identical
    encode). The caller crops codes back to ``true_frames``, so the extra frame is
    dropped. Pure integer/float arithmetic (no model) — unit-tested; the crash + fix
    are GPU-validated by ``modal_spectrostream_spike.py::verify_offbyone_fix``."""
    def _ss_expected(t: int) -> int:  # SS's exact float computation
        return math.ceil((t / float(sample_rate)) * float(frame_rate))
    guard = 0
    while _ss_expected(tgt) != (tgt + hop - 1) // hop and guard < 8:
        tgt += hop // 2
        guard += 1
    return tgt


class SpectroStreamCodec:
    """Magenta RealTime SpectroStream codec — 48kHz, joint stereo, 25 Hz, 1024 vocab.

    Same facade as DACodec, but:
      - input/output are STEREO ([2, samples]); a mono source is duplicated to L=R.
      - encode() returns LongTensor[depth, T] (T = ceil(samples / 1920)); ``depth``
        defaults to ``NANO_SS_DEPTH`` (SS_DEFAULT_DEPTH). Codes are plain per-codebook
        indices in [0, 1024) — no offset to undo.
      - decode() returns FloatTensor[2, samples] (or [B, 2, samples]); it accepts any
        depth <= the stored depth (RVQ residual prefix).

    magenta_rt is lazy-imported in __init__ so this module imports cleanly without it.
    """

    SAMPLE_RATE = 48000
    VOCAB_SIZE = 1024
    FRAME_RATE_HZ = 25  # 48000 / 1920; SpectroStream config frame_rate=25.0
    N_CHANNELS = 2  # joint stereo
    MAX_DEPTH = 64
    # Class-level default so codec_constants() can read N_CODEBOOKS without
    # constructing the (heavy) codec; __init__ overrides it per-instance.
    N_CODEBOOKS = SS_DEFAULT_DEPTH

    def __init__(self, device: str = "cuda", depth: int | None = None):
        # device is accepted for interface parity; SpectroStream places itself on
        # the available accelerator via its TF/JAX SavedModels.
        self.device = device
        self.N_CODEBOOKS = int(
            depth if depth is not None else os.environ.get("NANO_SS_DEPTH", SS_DEFAULT_DEPTH)
        )
        if not (0 < self.N_CODEBOOKS <= self.MAX_DEPTH):
            raise ValueError(f"depth must be in (0, {self.MAX_DEPTH}], got {self.N_CODEBOOKS}")
        # Frame-alignment / length-bucketing grid (in frames). `encode` ALWAYS pads a
        # clip's tail up to a whole frame (at least 1-frame alignment) — this is a
        # CORRECTNESS guard, not just a speed knob: SpectroStream floors a partial
        # final frame while a downstream stage expects ceil, so a non-frame-aligned
        # length crashes with an internal "(1,T,256) vs (1,T-1,256)" mismatch for a
        # subset of files. NANO_SS_BUCKET_FRAMES>1 aligns to a COARSER grid so the
        # encoder also reuses cuFFT plans across songs (the tokenize speedup); 0 or 1
        # both mean minimal frame-alignment. Crop-back keeps the output frame count
        # exact; equivalence across grids is checked by
        # modal_spectrostream_spike.bucket_check.
        self._bucket_frames = int(os.environ.get("NANO_SS_BUCKET_FRAMES", "1"))
        self._max_encode_samples = int(
            float(os.environ.get("NANO_MAX_ENCODE_SECONDS_SS", "360.0")) * self.SAMPLE_RATE
        )
        from magenta_rt import audio

        self._audio = audio
        # Construct at the stored depth: encode then yields exactly [S, depth].
        try:
            # v1 stack (TF/JAX, Linux+CUDA — the Modal images that tokenize the corpus).
            from magenta_rt import spectrostream

            self.model = spectrostream.SpectroStream(max_rvq_depth=self.N_CODEBOOKS)
        except ImportError:
            # magenta-rt 2.x dropped the top-level module but ships an MLX port of
            # the SAME codec (token-parity verified) — the Apple-Silicon path.
            from model.spectrostream_mlx import MLXSpectroStream

            self.model = MLXSpectroStream(max_rvq_depth=self.N_CODEBOOKS)

    def _to_waveform(self, source: str | Path | torch.Tensor):
        """Resolve a path or [C, samples]/[samples] tensor to a stereo Waveform."""
        import numpy as np

        if isinstance(source, (str, Path)):
            y, _ = librosa.load(str(source), sr=self.SAMPLE_RATE, mono=False)
            if y.ndim == 1:
                y = np.stack([y, y], axis=0)  # mono -> L=R
            samples = np.ascontiguousarray(y.T, dtype=np.float32)  # [N, 2]
        else:
            x = source.detach().cpu().numpy() if isinstance(source, torch.Tensor) else source
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = np.stack([x, x], axis=-1)  # [N, 2]
            elif x.shape[0] in (1, 2) and x.shape[0] < x.shape[-1]:
                x = x.T  # [C, N] -> [N, C]
            samples = np.ascontiguousarray(x, dtype=np.float32)
        wav = self._audio.Waveform(samples, self.SAMPLE_RATE)
        return wav.resample(self.SAMPLE_RATE).as_stereo()

    def encode(self, source: str | Path | torch.Tensor) -> torch.Tensor:
        """Encode stereo audio to SpectroStream codes -> LongTensor[depth, T].

        The waveform is zero-padded so its length is a whole number of codec frames
        before the TF forward, then the codes are cropped back to the true frame count
        (``ceil(samples / hop)``). This is REQUIRED for correctness, not just speed:
        SpectroStream floors a partial final frame while a downstream stage expects
        ceil, so a non-frame-aligned input length crashes with an internal
        ``(1, T, 256) vs (1, T-1, 256)`` shape mismatch for a subset of files. Aligning
        the tail to a full frame (optionally to a coarser ``NANO_SS_BUCKET_FRAMES`` grid,
        which also lets the encoder reuse cuFFT plans across songs) makes the framing
        unambiguous. Shared by train + inference, so both encode a given file
        identically."""
        import numpy as np

        wav = self._to_waveform(source)
        hop = self.SAMPLE_RATE // self.FRAME_RATE_HZ  # 1920
        n = int(wav.samples.shape[0])
        true_frames = (n + hop - 1) // hop
        grid = max(1, self._bucket_frames)  # >=1 → always at least frame-aligned
        tgt = bucket_target_samples(n, hop, grid, self._max_encode_samples)
        # Dodge SpectroStream's float-ceil off-by-one (see frame_safe_target_samples):
        # a frame-aligned `tgt` lands on the unstable boundary for ~4.8% of lengths,
        # aborting the encode; a half-frame nudge fixes it and the crop below drops
        # the extra frame. Guard inert (byte-identical) for the other ~95%.
        tgt = frame_safe_target_samples(tgt, hop, self.SAMPLE_RATE, self.FRAME_RATE_HZ)
        if tgt > n:
            pad = np.zeros((tgt - n, wav.samples.shape[1]), dtype=np.float32)
            wav = self._audio.Waveform(
                np.ascontiguousarray(np.concatenate([wav.samples, pad], axis=0), dtype=np.float32),
                self.SAMPLE_RATE,
            )
        codes = self.model.encode(wav)  # [>=true_frames, depth] int32, frame-major
        codes = np.ascontiguousarray(codes[:true_frames])  # crop to true frame count
        return torch.from_numpy(codes).to(torch.long).T.contiguous()  # [depth, T]

    def encode_batch(self, audios: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode multiple audios. Loops encode() per item (one TF forward each);
        the tokenize fan-out provides the parallelism, so this stays simple and
        avoids ragged-length batching assumptions in the codec."""
        return [self.encode(a) for a in audios]

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode codes to stereo audio.

        tokens: LongTensor[K, T] or [B, K, T] with K <= stored depth (RVQ prefix).
        returns: FloatTensor[2, samples] (if 2D input) or [B, 2, samples].
        """
        import numpy as np

        squeeze_batch = tokens.dim() == 2
        arr = tokens.detach().cpu().to(torch.int32).numpy()
        if squeeze_batch:
            toks = np.ascontiguousarray(arr.T)  # [K, T] -> [T, K]
            recon = self.model.decode(toks)  # stereo Waveform
            return torch.from_numpy(np.ascontiguousarray(recon.samples.T)).float()  # [2, N]
        batch = np.ascontiguousarray(arr.transpose(0, 2, 1))  # [B, K, T] -> [B, T, K]
        recons = self.model.decode(batch)  # list[Waveform]
        outs = [torch.from_numpy(np.ascontiguousarray(w.samples.T)).float() for w in recons]
        return torch.stack(outs, dim=0)  # [B, 2, N]


def _codec_class(codec: str | None = None):
    name = (codec or os.environ.get("NANO_CODEC", "dac")).lower()
    if name in ("spectrostream", "ss"):
        return SpectroStreamCodec
    if name == "dac":
        return DACodec
    raise ValueError(f"unknown NANO_CODEC={name!r} (expected 'dac' or 'spectrostream')")


def get_codec(device: str = "cuda", codec: str | None = None):
    """Construct the configured codec. ``codec`` (or env NANO_CODEC):
    "dac" (default, backward-compatible) or "spectrostream"/"ss" (v9 stereo)."""
    return _codec_class(codec)(device=device)


def codec_constants(codec: str | None = None) -> dict:
    """The configured codec's shape constants, read from CLASS attributes WITHOUT
    constructing the (heavy) codec model — for frame-rate/vocab-derived call sites
    (melody chroma alignment, seconds<->frames math, GPTConfig defaults).

    Returns {sample_rate, n_codebooks, vocab_size, frame_rate_hz, hop}. For
    SpectroStream n_codebooks is the stored depth (env NANO_SS_DEPTH default);
    the model's n_codebooks may be a smaller slice of it.
    """
    cls = _codec_class(codec)
    sr, fr = cls.SAMPLE_RATE, cls.FRAME_RATE_HZ
    return {
        "sample_rate": sr,
        "n_codebooks": cls.N_CODEBOOKS,
        "vocab_size": cls.VOCAB_SIZE,
        "frame_rate_hz": fr,
        "hop": sr // fr,
    }
