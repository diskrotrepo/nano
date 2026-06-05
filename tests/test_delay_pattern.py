"""Tests for model/delay_pattern.py.

The delay pattern is silent-corruption territory: an off-by-one in the k-shift
keeps the loss looking reasonable but trains the model on misaligned codebooks.
"""
from __future__ import annotations

import pytest
import torch

from model.delay_pattern import apply_delay, build_train_inputs, revert_delay


PAD = 9999


def test_apply_delay_shape_no_batch():
    codes = torch.arange(9 * 5).reshape(9, 5)
    out = apply_delay(codes, PAD)
    assert out.shape == (9, 5 + 9 - 1)


def test_apply_delay_shape_batched():
    codes = torch.arange(2 * 9 * 5).reshape(2, 9, 5)
    out = apply_delay(codes, PAD)
    assert out.shape == (2, 9, 13)


def test_apply_delay_shape_double_batched():
    codes = torch.arange(3 * 2 * 9 * 5).reshape(3, 2, 9, 5)
    out = apply_delay(codes, PAD)
    assert out.shape == (3, 2, 9, 13)


def test_apply_delay_values_at_correct_positions():
    """For codebook k, delayed[..., k, k:k+T] must equal codes[..., k, :]."""
    K, T = 9, 5
    codes = torch.arange(K * T).reshape(K, T)
    out = apply_delay(codes, PAD)
    for k in range(K):
        assert torch.equal(out[k, k:k + T], codes[k]), f"codebook {k} misaligned"


def test_apply_delay_pad_fills_complement():
    """Positions outside [k, k+T) must be pad."""
    K, T = 9, 5
    codes = torch.arange(K * T).reshape(K, T)
    out = apply_delay(codes, PAD)
    for k in range(K):
        # before the window
        if k > 0:
            assert (out[k, :k] == PAD).all(), f"codebook {k} left-pad missing"
        # after the window
        tail_start = k + T
        if tail_start < out.shape[-1]:
            assert (out[k, tail_start:] == PAD).all(), f"codebook {k} right-pad missing"


def test_apply_delay_first_codebook_unshifted():
    """k=0 is shifted by 0 so the leading T positions are the original values
    and the rest is pad."""
    K, T = 9, 5
    codes = torch.arange(K * T).reshape(K, T)
    out = apply_delay(codes, PAD)
    assert torch.equal(out[0, :T], codes[0])
    assert (out[0, T:] == PAD).all()


def test_apply_delay_last_codebook_fully_shifted():
    """k=K-1 starts at position K-1 and runs to the end."""
    K, T = 9, 5
    codes = torch.arange(K * T).reshape(K, T)
    out = apply_delay(codes, PAD)
    assert (out[K - 1, :K - 1] == PAD).all()
    assert torch.equal(out[K - 1, K - 1:], codes[K - 1])


@pytest.mark.parametrize("shape", [(9, 5), (2, 9, 5), (3, 2, 9, 5), (1, 1, 9, 7)])
def test_revert_is_inverse_of_apply(shape):
    codes = torch.arange(int(torch.tensor(shape).prod())).reshape(*shape)
    T = shape[-1]
    recovered = revert_delay(apply_delay(codes, PAD), T)
    assert torch.equal(recovered, codes)


def test_build_train_inputs_shape():
    K, T = 9, 5
    codes = torch.arange(K * T).reshape(K, T)
    inputs, targets = build_train_inputs(codes, PAD)
    # delayed length is T + K - 1, then we drop one for input/target shift.
    assert inputs.shape == (K, T + K - 2)
    assert targets.shape == (K, T + K - 2)


def test_build_train_inputs_shift_relationship():
    """``targets`` is ``inputs`` shifted left by one timestep along the
    delayed-sequence axis."""
    codes = torch.arange(2 * 9 * 5).reshape(2, 9, 5)
    inputs, targets = build_train_inputs(codes, PAD)
    # inputs[..., 1:] aligns with targets[..., :-1]
    assert torch.equal(inputs[..., 1:], targets[..., :-1])


def test_build_train_inputs_dtype_preserved():
    codes = torch.zeros(9, 5, dtype=torch.long)
    inputs, targets = build_train_inputs(codes, PAD)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
