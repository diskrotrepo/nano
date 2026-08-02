"""Thin CLAP wrapper for text and audio conditioning.

Encodes text descriptions or reference audio into embeddings that the
transformer consumes via cross-attention. CLAP maps both modalities into
the same 1024-dim space, so audio conditioning requires no model changes.
The CLAP model is frozen — never trained.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Default chunking knobs (shared train==inference contract). CLAP's GPT2 text
# branch caps at 77 tokens; CHUNK_TOKENS<77 leaves headroom for BPE boundary
# drift on the decode→re-encode round-trip. MAX_CHUNKS bounds a pathological
# description (~3000 dense/multilingual chars ≈ ~950 tokens ≈ ~13 chunks).
CHUNK_TOKENS = 72
MAX_CHUNKS = 16


def chunk_text_ids(tokenizer, text: str, max_chunks: int = MAX_CHUNKS,
                   chunk_tokens: int = CHUNK_TOKENS) -> list[str]:
    """Split *text* into ``<=chunk_tokens``-GPT2-token windows using *tokenizer*.

    The single source of truth for chunking, so the train precompute (which has
    only a bare GPT2 ``AutoTokenizer``) and inference (``CLAPTextEncoder``) build
    byte-identical chunk streams. A text that fits one window is returned as
    ``[text]`` unchanged (the terse-tag backward-compat invariant). Empty ->
    ``[]``. Over ``max_chunks`` windows logs and drops the tail (no silent
    truncation)."""
    text = (text or "").strip()
    if not text:
        return []
    # Explicit truncation=False bypasses any tokenizer truncation patch so we
    # get the FULL token list (add_special_tokens=False -> pure content).
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) <= chunk_tokens:
        return [text]
    windows = [ids[i:i + chunk_tokens] for i in range(0, len(ids), chunk_tokens)]
    if len(windows) > max_chunks:
        dropped = len(windows) - max_chunks
        print(f"[clap-chunk] description is {len(ids)} tokens -> {len(windows)} "
              f"chunks; capping at {max_chunks} (dropping {dropped} tail chunk(s))")
        windows = windows[:max_chunks]
    return [tokenizer.decode(w) for w in windows]


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
        # House-style init (std=0.02) instead of PyTorch's Linear default
        # (~2.2x larger at this fan-in) — the projection feeds the decoder's
        # cross-attention cond and shouldn't start louder than everything else.
        nn.init.normal_(self.proj.weight, std=0.02)
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

    def chunk_text(
        self, text: str, max_chunks: int = MAX_CHUNKS, chunk_tokens: int = CHUNK_TOKENS,
    ) -> list[str]:
        """Split *text* into ``<=chunk_tokens``-GPT2-token windows for chunked CLAP.

        CLAP's text branch is capped at ``text_len`` (77) tokens and pools the
        whole input to ONE vector, so a long description must be split into
        windows that each fit, then encoded into a *sequence* of pooled vectors
        (see ``encode_chunked``). Delegates to the shared ``chunk_text_ids`` with
        CLAP's own GPT2 tokenizer, so train precompute and inference chunk
        identically. A text that fits one window is returned as ``[text]``
        unchanged (the terse-tag backward-compat invariant)."""
        self._ensure_clap()
        return chunk_text_ids(self._clap.tokenizer, text, max_chunks, chunk_tokens)

    @staticmethod
    def additive_kv_mask(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """bool keep-mask [B, N] -> additive cross-attn mask [B, 1, 1, N].

        0 at kept chunks, -inf at padding — the same convention as
        ``NanoAudioGPT.encode_lyrics``. Callers must keep >=1 chunk per row (a
        fully -inf row NaNs the cross-attn softmax)."""
        return torch.zeros(
            keep.shape[0], 1, 1, keep.shape[1], dtype=dtype, device=keep.device,
        ).masked_fill(~keep[:, None, None, :], float("-inf"))

    @torch.no_grad()
    def encode_chunked(
        self, texts: list[str], max_chunks: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (possibly long) strings -> (emb [B, N_max, d_out], additive
        kv_mask [B, 1, 1, N_max]).

        Each string is split into ``<=77``-token chunks (``chunk_text``), every
        chunk encoded by frozen CLAP to a pooled [1024] vector, and the padded
        sequence projected to ``d_out`` — so a 3000-char description conditions
        the decoder as ``N`` cross-attention positions instead of one averaged
        vector. The mask is 0 at real chunks, -inf at padding; every row keeps
        >=1 un-masked chunk (an empty string becomes a single un-masked *zero*
        chunk, which a bias-free cross-attn maps to zero output == "no tags"), so
        no row is ever fully masked. A string that fits one chunk reproduces
        ``encode`` exactly."""
        self._ensure_clap()
        per_text = [self.chunk_text(t, max_chunks=max_chunks) for t in texts]
        n_per = [len(c) for c in per_text]
        n_max = max(max(n_per, default=1), 1)
        flat = [c for chunks in per_text for c in chunks]
        device = self.proj.weight.device
        if flat:
            flat_emb = self._clap.get_text_embeddings(flat).to(device)  # [sum_n, 1024]
            dtype = flat_emb.dtype
        else:
            dtype = self.proj.weight.dtype
        B = len(texts)
        raw = torch.zeros(B, n_max, self.CLAP_DIM, device=device, dtype=dtype)
        keep = torch.zeros(B, n_max, dtype=torch.bool, device=device)
        idx = 0
        for i, n in enumerate(n_per):
            if n == 0:
                keep[i, 0] = True  # un-masked zero chunk == "no tags"
                continue
            raw[i, :n] = flat_emb[idx:idx + n]
            keep[i, :n] = True
            idx += n
        emb = self.proj(raw)  # [B, n_max, d_out]
        return emb, self.additive_kv_mask(keep, emb.dtype)

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
