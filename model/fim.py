"""Fill-in-the-middle (infill) reorder for nano.

Canonical FIM (Bavarian et al., "Efficient Training of Language Models to Fill
in the Middle"): rearrange a contiguous span into

    prefix  <SUF>  suffix  <MID>  middle

and train the *existing* causal next-token objective on the reordered sequence.
The decoder learns that, having been shown the prefix and the suffix (separated
by the sentinels), it should generate the middle that bridges into that suffix.

This is a pure frame-domain transform applied BEFORE the delay pattern, so the
attention mask and ``delay_pattern`` are untouched. The two sentinels (<SUF>,
<MID>) are whole frames — every codebook of a sentinel frame holds the sentinel
id — so they survive ``apply_delay``/``revert_delay`` like any other frame.

Train and inference build the same ``prefix <SUF> suffix <MID>`` prefix layout
(``fim_reorder_batch`` for training, ``build_fim_prompt`` for inference); the
``tests/test_fim.py`` equivalence guard keeps them byte-identical.

Melody (chroma) is co-reordered with the IDENTICAL permutation; the two sentinel
frames get a zero chroma row, which the ``MelodyEncoder`` already sees as
in-distribution (the packer zero-fills missing chroma), so no encoder change is
needed. Lyrics cannot be co-reordered (their near-monotonic sung alignment would
scramble), so FIM batches drop the lyric stream — handled by the caller.
"""
from __future__ import annotations

import random

import torch
from torch import Tensor

# A FIM example is only meaningful if each region carries some signal. Keep the
# prefix and suffix non-trivial and the middle (the gap to bridge) substantial.
MIN_PREFIX_FRAMES = 8
MIN_SUFFIX_FRAMES = 8
MIN_MIDDLE_FRAMES = 8


def _sample_splits(content_len: int, rng: random.Random) -> tuple[int, int]:
    """Pick 0 < a <= b < content_len giving prefix [0,a), middle [a,b),
    suffix [b,content_len) with each region >= its minimum. Falls back to a
    centered split if the crop is too short to honor the minimums."""
    lo_a = MIN_PREFIX_FRAMES
    hi_a = content_len - MIN_MIDDLE_FRAMES - MIN_SUFFIX_FRAMES
    if hi_a < lo_a:
        # Crop too small for the configured minimums — split into thirds.
        a = content_len // 3
        b = 2 * content_len // 3
        return max(1, a), max(a + 1, b)
    a = rng.randint(lo_a, hi_a)
    b = rng.randint(a + MIN_MIDDLE_FRAMES, content_len - MIN_SUFFIX_FRAMES)
    return a, b


def fim_reorder_batch(
    codes: Tensor,
    melody: Tensor | None,
    suf_id: int,
    mid_id: int,
    rng: random.Random,
) -> tuple[Tensor, Tensor | None]:
    """Reorder a whole batch into FIM layout, preserving the [B,K,T] shape.

    codes:  [B, K, T] int token ids.
    melody: [B, T, n_bins] chroma or None (co-reordered with zero sentinel rows).
    returns (codes', melody') with the SAME shapes — two sentinel frames are
    inserted and the trailing two middle frames are dropped to keep length T.

    Per example: content = the full T frames, split into prefix/middle/suffix;
    emit ``prefix <SUF> suffix <MID> middle`` then truncate to T (so at most the
    last two middle frames are sacrificed to make room for the sentinels).
    """
    B, K, T = codes.shape
    device = codes.device
    out_codes = torch.empty_like(codes)
    out_melody = torch.empty_like(melody) if melody is not None else None
    n_bins = melody.shape[2] if melody is not None else 0

    for i in range(B):
        a, b = _sample_splits(T, rng)
        P, M, S = codes[i, :, :a], codes[i, :, a:b], codes[i, :, b:]
        suf_col = torch.full((K, 1), suf_id, dtype=codes.dtype, device=device)
        mid_col = torch.full((K, 1), mid_id, dtype=codes.dtype, device=device)
        row = torch.cat([P, suf_col, S, mid_col, M], dim=1)[:, :T]
        out_codes[i] = row
        if melody is not None:
            mP, mM, mS = melody[i, :a], melody[i, a:b], melody[i, b:]
            zero = torch.zeros((1, n_bins), dtype=melody.dtype, device=device)
            mrow = torch.cat([mP, zero, mS, zero, mM], dim=0)[:T]
            out_melody[i] = mrow
    return out_codes, out_melody


def build_fim_prompt(
    prefix_codes: Tensor,
    suffix_codes: Tensor,
    suf_id: int,
    mid_id: int,
) -> Tensor:
    """Inference-side FIM prompt: ``prefix <SUF> suffix <MID>``.

    prefix_codes / suffix_codes: [K, Tp] / [K, Ts] DAC token ids. Returns
    [K, Tp + 1 + Ts + 1]; ``generate`` then produces the middle as the new
    frames after this prompt (mirrors the training layout up to the <MID>
    sentinel, which is the last column here). Must stay byte-identical to the
    prefix that ``fim_reorder_batch`` builds — guarded by tests/test_fim.py.
    """
    assert prefix_codes.dim() == 2 and suffix_codes.dim() == 2
    K = prefix_codes.shape[0]
    assert suffix_codes.shape[0] == K
    device, dtype = prefix_codes.device, prefix_codes.dtype
    suf_col = torch.full((K, 1), suf_id, dtype=dtype, device=device)
    mid_col = torch.full((K, 1), mid_id, dtype=dtype, device=device)
    return torch.cat([prefix_codes, suf_col, suffix_codes, mid_col], dim=1)
