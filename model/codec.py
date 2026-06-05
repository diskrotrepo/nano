"""DAC codec wrapper.

Built as a thin facade so we can swap in EnCodec or another codec later
without touching the model or training code. The interface guarantees:
  - SAMPLE_RATE, N_CODEBOOKS, VOCAB_SIZE, FRAME_RATE_HZ are constants
  - encode(source) -> LongTensor[N_CODEBOOKS, T]
  - decode(tokens) -> FloatTensor[samples]  (mono, at SAMPLE_RATE)
"""
from __future__ import annotations

from pathlib import Path

import librosa
import torch

import dac


class DACodec:
    SAMPLE_RATE = 44100
    N_CODEBOOKS = 9
    VOCAB_SIZE = 1024
    FRAME_RATE_HZ = 44100 // 512  # 86 (DAC 44.1kHz uses hop=512)

    def __init__(self, device: str = "cpu"):
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
