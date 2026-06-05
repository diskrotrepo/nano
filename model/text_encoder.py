"""Thin CLAP wrapper for text and audio conditioning.

Encodes text descriptions or reference audio into embeddings that the
transformer consumes via cross-attention. CLAP maps both modalities into
the same 1024-dim space, so audio conditioning requires no model changes.
The CLAP model is frozen — never trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CLAPTextEncoder(nn.Module):
    """Frozen CLAP text encoder (Microsoft msclap).

    Produces a single embedding vector per input string, projected to d_out
    and returned as [B, 1, d_out] so it can feed directly into cross-attention.
    """

    CLAP_DIM = 1024  # msclap 2023 embedding size

    def __init__(self, d_out: int, device: str = "cpu"):
        super().__init__()
        self.d_out = d_out
        self._device = device
        self.proj = nn.Linear(self.CLAP_DIM, d_out, bias=False)
        self._clap = None  # lazy-loaded

    def _ensure_clap(self) -> None:
        if self._clap is not None:
            return
        from msclap import CLAP

        use_cuda = self._device == "cuda"
        self._clap = CLAP(version="2023", use_cuda=use_cuda)
        self._patch_clap_truncation()
        self._patch_torchaudio_load()

    @staticmethod
    def _patch_torchaudio_load() -> None:
        """Make ``torchaudio.load`` resilient to a missing/broken torchcodec.

        Recent torchaudio routes ``load`` through ``load_with_torchcodec``;
        when torchcodec is absent (Modal serve image) or too old for the
        system ffmpeg (local ffmpeg 8 — see the torchcodec_ffmpeg8 memory),
        the native call raises and CLAP's ``read_audio`` dies on every
        style-audio upload. msclap resamples after loading, so it only needs
        ``(waveform[channels, frames], sample_rate)`` — which librosa
        (already a dependency) can supply for wav and mp3 alike. Installed
        once, as a fallback that defers to the native loader when it works."""
        import torchaudio

        if getattr(torchaudio, "_nano_load_patched", False):
            return
        _orig_load = torchaudio.load

        def _load(path, *args, **kwargs):
            try:
                return _orig_load(path, *args, **kwargs)
            except Exception:
                import librosa
                import numpy as np

                y, sr = librosa.load(path, sr=None, mono=False)  # [frames] or [ch, frames]
                if y.ndim == 1:
                    y = y[None, :]
                return torch.from_numpy(np.ascontiguousarray(y)), sr

        torchaudio.load = _load
        torchaudio._nano_load_patched = True

    def _patch_clap_truncation(self) -> None:
        """Force ``truncation=True`` on CLAP's tokenizer.

        msclap's ``preprocess_text`` tokenizes with ``padding='max_length',
        max_length=text_len`` (77) but omits ``truncation=True`` (see
        site-packages/msclap/CLAPWrapper.py). A caption longer than text_len —
        the LP-MusicCaps prose descriptions routinely tokenize to 80-90+ — is
        therefore left at its full length instead of being cut, so a batch that
        mixes a short tag (padded to 77) and a long one (e.g. 91) blows up in
        the ``torch.stack`` inside ``default_collate``. Defaulting
        ``encode_plus`` to ``truncation=True`` restores CLAP's intended
        fixed-length behaviour and covers every call path that goes through
        ``_ensure_clap`` (training precompute, inference, lyrics)."""
        tokenizer = getattr(self._clap, "tokenizer", None)
        if tokenizer is None or getattr(tokenizer, "_nano_trunc_patched", False):
            return
        _orig_encode_plus = tokenizer.encode_plus

        def _encode_plus(*args, **kwargs):
            kwargs.setdefault("truncation", True)
            return _orig_encode_plus(*args, **kwargs)

        tokenizer.encode_plus = _encode_plus
        tokenizer._nano_trunc_patched = True

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of text strings -> [B, 1, d_out].

        msclap's ``get_text_embeddings`` accepts a list and handles padding
        internally — verified by
        tests/test_pipeline_audit.py::test_clap_batched_matches_single_tag,
        which compares batched vs per-tag output to <1e-4 atol. An older
        comment here claimed otherwise; it was wrong / outdated."""
        self._ensure_clap()
        emb = self._clap.get_text_embeddings(texts)            # [B, 1024]
        emb = emb.to(self.proj.weight.device)
        projected = self.proj(emb)                             # [B, d_out]
        return projected.unsqueeze(1)                          # [B, 1, d_out]

    @torch.no_grad()
    def encode_audio(self, file_paths: list[str]) -> torch.Tensor:
        """Encode audio files -> [B, 1, d_out]. Same embedding space as text."""
        self._ensure_clap()
        emb = self._clap.get_audio_embeddings(file_paths)     # [B, 1024]
        emb = emb.to(self.proj.weight.device)
        projected = self.proj(emb)                             # [B, d_out]
        return projected.unsqueeze(1)                          # [B, 1, d_out]

    @torch.no_grad()
    def audio_text_similarity(self, audio_path: str, text: str) -> float:
        """Raw CLAP-1024 cosine similarity between an audio file and a text prompt.

        Uses the frozen CLAP embedding space directly (not the learned ``proj``
        to d_out) — this is the standard CLAP score for "does this audio match
        this description", i.e. prompt adherence. Higher is closer."""
        self._ensure_clap()
        a = self._clap.get_audio_embeddings([audio_path])  # [1, 1024]
        t = self._clap.get_text_embeddings([text])         # [1, 1024]
        a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-9)
        return float((a @ t.T).squeeze().item())

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts)
