"""Vendored LP-MusicCaps captioning model (inference only).

Minimal extraction from https://github.com/seungheondoh/lp-music-caps (MIT).
Original paper: "LP-MusicCaps: LLM-Based Pseudo Music Captioning" (ISMIR 2023).

Generates natural-language captions from 10-second audio clips.  Audio is
resampled to 16 kHz mono, converted to a 128-bin mel spectrogram, passed through
stride convolutions, and decoded by facebook/bart-base.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

# ── audio constants ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
DURATION = 10
N_SAMPLES = DURATION * SAMPLE_RATE  # 160 000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = int(0.01 * SAMPLE_RATE)  # 160

# ── helpers ──────────────────────────────────────────────────────────────────

def _sinusoids(length: int, channels: int, max_timescale: int = 10_000) -> Tensor:
    log_inc = np.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-log_inc * torch.arange(channels // 2))
    scaled = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


# ── mel encoder ──────────────────────────────────────────────────────────────

class _MelEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft=N_FFT, win_length=N_FFT, hop_length=HOP_LENGTH, power=None,
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            N_MELS, SAMPLE_RATE, 0.0, 8000.0, N_FFT // 2 + 1,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav: Tensor) -> Tensor:
        spec = self.spec_fn(wav)
        power = spec.real.abs().pow(2)
        return self.amplitude_to_db(self.mel_scale(power))


# ── audio encoder (mel → stride convs → positional embedding) ───────────────

class _AudioEncoder(nn.Module):
    def __init__(self, n_ctx: int, audio_dim: int, text_dim: int,
                 num_stride_convs: int) -> None:
        super().__init__()
        self.mel_encoder = _MelEncoder()
        self.conv1 = nn.Conv1d(N_MELS, audio_dim, kernel_size=3, padding=1)
        self.conv_stack = nn.ModuleList(
            [nn.Conv1d(audio_dim, audio_dim, kernel_size=3, stride=2, padding=1)
             for _ in range(num_stride_convs)])
        self.register_buffer("positional_embedding", _sinusoids(n_ctx, text_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.mel_encoder(x)
        x = F.gelu(self.conv1(x))
        for conv in self.conv_stack:
            x = F.gelu(conv(x))
        x = x.permute(0, 2, 1)
        return (x + self.positional_embedding).to(x.dtype)


# ── caption model (audio encoder + BART decoder) ────────────────────────────

class BartCaptionModel(nn.Module):
    """LP-MusicCaps: audio encoder → frozen BART decoder → caption text."""

    def __init__(self, *, num_of_conv: int = 6, audio_dim: int = 768,
                 max_length: int = 128, bart_type: str = "facebook/bart-base",
                 ) -> None:
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained(bart_type)
        self.bart = BartForConditionalGeneration(BartConfig.from_pretrained(bart_type))

        n_frames = N_SAMPLES // HOP_LENGTH  # 1000
        num_stride_convs = num_of_conv - 1  # 5
        n_ctx = n_frames // (2 ** num_stride_convs) + 1  # 32

        self.audio_encoder = _AudioEncoder(
            n_ctx=n_ctx, audio_dim=audio_dim,
            text_dim=self.bart.config.hidden_size,
            num_stride_convs=num_stride_convs,
        )
        self.max_length = max_length

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(self, samples: Tensor, *, num_beams: int = 5,
                 max_length: int = 128, min_length: int = 2,
                 repetition_penalty: float = 1.0) -> list[str]:
        """Generate captions for a batch of 10-second waveforms.

        Args:
            samples: ``[B, N_SAMPLES]`` float tensor (16 kHz mono).

        Returns:
            List of *B* caption strings.
        """
        audio_embs = self.audio_encoder(samples)
        enc_out = self.bart.model.encoder(
            input_ids=None, inputs_embeds=audio_embs, return_dict=True,
        )
        B = enc_out["last_hidden_state"].size(0)
        dec_ids = torch.full((B, 1), self.bart.config.decoder_start_token_id,
                             dtype=torch.long, device=self.device)
        dec_mask = torch.ones_like(dec_ids)

        out_ids = self.bart.generate(
            input_ids=None,
            decoder_input_ids=dec_ids,
            decoder_attention_mask=dec_mask,
            encoder_outputs=enc_out,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)


# ── public loader ────────────────────────────────────────────────────────────

def load_captioner(device: str = "cpu", ckpt_path: str | None = None) -> BartCaptionModel:
    """Load the LP-MusicCaps transfer model.

    Downloads ``transfer.pth`` from HuggingFace on first call (cached
    automatically by ``huggingface_hub``).
    """
    if ckpt_path is None:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id="seungheondoh/lp-music-caps", filename="transfer.pth")

    model = BartCaptionModel(max_length=128)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()
