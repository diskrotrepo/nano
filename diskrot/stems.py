"""Stem-token extraction for generative stem conditioning — the train==inference contract.

The "/addstem" path conditions on a song's separated stems (drums/bass/vocals/other)
and generates a NEW isolated stem that fits the song. This module is the SINGLE
source of truth for turning audio into the per-stem codec-token streams the model
consumes — shared by the Modal stem job + packer (train) and ``server/inference.py``
(inference). A train/inference mismatch here is the same footgun class as g2p drift
in the lyric path, so both sides call THIS code; ``tests/test_stem.py`` guards it.

Pipeline (per song):
  mp3 -> ffmpeg stereo @44.1kHz -> Demucs (htdemucs, 4 stems) -> resample each stem
  to the codec sample rate -> codec.encode -> per-stem ``[depth, T]`` codes.

Key contract:
- Stems are keyed by ``STEM_TYPES`` NAME (drums/bass/vocals/other), NOT Demucs'
  source order (drums/bass/OTHER/VOCALS) — vocals/other are swapped, so we map via
  ``demucs_model.sources`` names, never positional index.
- Demucs is trained at 44.1 kHz; SpectroStream wants 48 kHz STEREO. We Demucs at
  44.1 kHz then resample each stem to the codec rate before encoding (the codec's
  ``encode`` assumes its own sample rate, so the resample MUST happen here).
- Each stem's token frame count is forced to the song's own token frame count
  (``n_frames``, the ``.pt`` length) so it indexes byte-identically into the
  parallel packed stem mmap — mirrors ``melody.extract_chroma``'s force-align.
- Stems are stored at the codec's full stored depth (``N_CODEBOOKS``); the dataset
  slices to the model's ``n_codebooks`` exactly like the main token stream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.stem_encoder import STEM_TYPES
# Reuse the transcribe stage's Demucs loader + stereo ffmpeg decode + overlap so the
# separation is byte-identical to the corpus's existing vocal-isolation pass.
from diskrot.transcribe_lyrics import _load_demucs, _ffmpeg_load_stereo, _DEMUCS_OVERLAP

DEMUCS_SAMPLE_RATE = 44100  # htdemucs native rate


def load_demucs(device: str = "cuda"):
    """Construct the Demucs model + apply fn (``(model, apply_fn)``). Thin re-export
    of the transcribe loader so callers import one stem module."""
    return _load_demucs(device)


def _demucs_all_stems(
    demucs_model, apply_fn, audio_stereo: np.ndarray, device: str
) -> dict[str, np.ndarray]:
    """Run Demucs on a ``[2, T]`` float32 mix @44.1kHz, return every stem keyed by
    name -> ``[2, T]`` float32 @44.1kHz. Keys come from ``demucs_model.sources``
    (drums/bass/other/vocals) so the mapping is by name, never positional."""
    wav = torch.from_numpy(np.ascontiguousarray(audio_stereo, dtype=np.float32))
    wav = wav.unsqueeze(0).to(device)  # [1, 2, T]
    with torch.no_grad():
        sources = apply_fn(demucs_model, wav, device=device, overlap=_DEMUCS_OVERLAP)
    src = sources[0].cpu().numpy()  # [n_sources, 2, T]
    return {name: src[i] for i, name in enumerate(demucs_model.sources)}


def _stem_to_codec_input(stem_stereo_44k: np.ndarray, codec) -> torch.Tensor:
    """Resample a ``[2, T]`` @44.1kHz stem to the codec's rate + channel layout.

    Returns a ``[C, samples]`` float32 tensor ready for ``codec.encode`` (stereo for
    SpectroStream, mono-downmixed for DAC). The codec treats its input as already at
    its own sample rate, so the resample MUST happen here, not inside encode."""
    import librosa

    sr = codec.SAMPLE_RATE
    y = np.ascontiguousarray(stem_stereo_44k, dtype=np.float32)  # [2, T] @44.1k
    if sr != DEMUCS_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=DEMUCS_SAMPLE_RATE, target_sr=sr, axis=-1)
    n_ch = getattr(codec, "N_CHANNELS", 1)
    if n_ch == 1:
        y = y.mean(axis=0, keepdims=True)  # [1, T] mono downmix for DAC
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


def _force_frames(codes: torch.Tensor, n_frames: int, pad_id: int) -> torch.Tensor:
    """Force ``[K, T]`` codes to exactly ``n_frames`` along time (truncate / pad_id-pad),
    mirroring melody's frame-count force-align."""
    K, T = codes.shape
    if T == n_frames:
        return codes
    if T > n_frames:
        return codes[:, :n_frames].contiguous()
    pad = torch.full((K, n_frames - T), pad_id, dtype=codes.dtype)
    return torch.cat([codes, pad], dim=1)


def extract_stem_tokens(
    source: str | Path | np.ndarray,
    codec,
    demucs,
    n_frames: int | None = None,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Separate + tokenize every stem of one song.

    source: an audio path, OR a pre-decoded ``[2, T]`` float32 stereo array @44.1kHz.
    codec: an active codec instance (``encode``/``SAMPLE_RATE``/``N_CODEBOOKS``).
    demucs: the ``(model, apply_fn)`` pair from ``load_demucs``.
    n_frames: when given, force each stem's token length to it (the song's ``.pt``
        frame count); when None, leave each stem at its natural encoded length
        (inference, where the gap length governs alignment).

    returns: ``{stem_name: LongTensor[depth, T]}`` for all four ``STEM_TYPES``.
    """
    model, apply_fn = demucs
    if isinstance(source, np.ndarray):
        audio = np.ascontiguousarray(source, dtype=np.float32)
    else:
        audio = _ffmpeg_load_stereo(source, DEMUCS_SAMPLE_RATE)  # [2, T] @44.1k
    by_name = _demucs_all_stems(model, apply_fn, audio, device)
    pad_id = codec.VOCAB_SIZE  # model.pad_id when n_control == 1 (no FIM)
    out: dict[str, torch.Tensor] = {}
    for name in STEM_TYPES:
        codes = codec.encode(_stem_to_codec_input(by_name[name], codec))  # [depth, T]
        if n_frames is not None:
            codes = _force_frames(codes, n_frames, pad_id)
        out[name] = codes.long()
    return out


def extract_stem_array(
    source: str | Path | np.ndarray,
    codec,
    demucs,
    n_frames: int,
    device: str = "cuda",
) -> np.ndarray:
    """The packer-facing form: a single ``[len(STEM_TYPES), depth, n_frames]`` int16
    array, stems stacked in ``STEM_TYPES`` order and force-aligned to the song's
    token frame count. Saved per song as ``<name>.stems.npy`` on the nano-stems
    volume; the packer folds it into ``packed_NNN.stem.bin`` at the song's offset."""
    tokens = extract_stem_tokens(source, codec, demucs, n_frames=n_frames, device=device)
    depth = codec.N_CODEBOOKS
    out = np.empty((len(STEM_TYPES), depth, n_frames), dtype=np.int16)
    for ti, name in enumerate(STEM_TYPES):
        out[ti] = tokens[name][:depth].cpu().numpy().astype(np.int16)
    return out
