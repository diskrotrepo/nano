"""MusicGen-style delay pattern for multi-codebook autoregressive prediction.

With K codebooks and T frames, codebook k is shifted right by k positions, so the
delayed tensor has T + K - 1 timesteps. At position p, codebook k's value is
codes[k, p - k] when 0 <= p - k < T, else PAD.

This lets a single transformer predict one timestep at a time while still letting
each codebook's prediction condition on earlier codebooks at the same original
frame (because codebook k at original frame t is at delayed position t+k, after
codebooks 0..k-1 at the same original frame which sit at positions t+0..t+k-1).
"""
from __future__ import annotations

import torch
from torch import Tensor


def apply_delay(codes: Tensor, pad_token: int) -> Tensor:
    """codes: [..., K, T] -> [..., K, T + K - 1] with codebook k shifted by k."""
    *prefix, K, T = codes.shape
    out = torch.full((*prefix, K, T + K - 1), pad_token, dtype=codes.dtype, device=codes.device)
    for k in range(K):
        out[..., k, k:k + T] = codes[..., k, :]
    return out


def revert_delay(delayed: Tensor, original_T: int) -> Tensor:
    """delayed: [..., K, T + K - 1] -> [..., K, T]. Inverse of apply_delay."""
    *prefix, K, _ = delayed.shape
    out = torch.empty((*prefix, K, original_T), dtype=delayed.dtype, device=delayed.device)
    for k in range(K):
        out[..., k, :] = delayed[..., k, k:k + original_T]
    return out


def build_train_inputs(codes: Tensor, pad_token: int) -> tuple[Tensor, Tensor]:
    """For teacher-forced training. Returns (inputs, targets) both [..., K, T + K - 2]."""
    delayed = apply_delay(codes, pad_token)
    inputs = delayed[..., :, :-1]
    targets = delayed[..., :, 1:]
    return inputs, targets


if __name__ == "__main__":
    # Quick sanity check.
    codes = torch.arange(2 * 9 * 5).reshape(2, 9, 5)
    pad = 9999
    delayed = apply_delay(codes, pad)
    assert delayed.shape == (2, 9, 13)
    recovered = revert_delay(delayed, 5)
    assert torch.equal(recovered, codes)
    print("delay_pattern OK")
