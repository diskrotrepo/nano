"""Tests for diskrot/train.py — _loss_fn and _cosine_lr.

These two are pure tensor / float math but easy to break silently. The loss
function in particular was just rewritten from a Python loop into a single
masked reduction, and the warmup boundary semantics matter more now that
``warmup_steps`` jumped from 750 → 2500.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from diskrot.train import TrainConfig, _cosine_lr, _distill_loss, _loss_fn


def _loss_fn_loopy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int):
    """Reference: the pre-refactor implementation. One ``F.cross_entropy`` for
    the global total, then K more for the per-codebook breakdown. Slow but
    obviously correct — this is the oracle the new fused version must match."""
    B, K, T, V = logits.shape
    total = F.cross_entropy(
        logits.reshape(B * K * T, V),
        targets.reshape(B * K * T),
        ignore_index=pad_id,
    )
    per_cb = torch.stack([
        F.cross_entropy(
            logits[:, k].reshape(B * T, V),
            targets[:, k].reshape(B * T),
            ignore_index=pad_id,
        )
        for k in range(K)
    ])
    return total, per_cb


# ----- _loss_fn -----

def test_loss_fn_matches_loop_no_pad():
    torch.manual_seed(0)
    B, K, T, V = 2, 9, 13, 1025
    pad = V - 1
    logits = torch.randn(B, K, T, V)
    targets = torch.randint(0, pad, (B, K, T))  # no pad tokens at all
    total, per_cb = _loss_fn(logits, targets, pad)
    ref_total, ref_per_cb = _loss_fn_loopy(logits, targets, pad)
    assert torch.allclose(total, ref_total, atol=1e-6)
    assert torch.allclose(per_cb, ref_per_cb, atol=1e-6)


def test_loss_fn_matches_loop_with_pad():
    """Mixed pad / non-pad targets — the path that gets exercised in real
    training (every batch has the delay-pattern's pad positions)."""
    torch.manual_seed(1)
    B, K, T, V = 3, 9, 17, 1025
    pad = V - 1
    logits = torch.randn(B, K, T, V)
    targets = torch.randint(0, pad, (B, K, T))
    # sprinkle pad into a structured pattern: first k positions of codebook k.
    for k in range(K):
        targets[:, k, :k] = pad
    total, per_cb = _loss_fn(logits, targets, pad)
    ref_total, ref_per_cb = _loss_fn_loopy(logits, targets, pad)
    assert torch.allclose(total, ref_total, atol=1e-6)
    assert torch.allclose(per_cb, ref_per_cb, atol=1e-6)


def test_loss_fn_pad_positions_ignored():
    """If we replace some non-pad targets with pad, only the non-pad ones
    contribute. Build two target tensors that differ only in pad positions and
    verify the loss is unchanged."""
    torch.manual_seed(2)
    B, K, T, V = 2, 9, 11, 1025
    pad = V - 1
    logits = torch.randn(B, K, T, V)
    base = torch.randint(0, pad, (B, K, T))

    # version A: every position contributes
    masked = base.clone()
    masked[:, :, :3] = pad  # first 3 positions get masked
    # version B: same masked positions, but with different (still-pad) target ids
    # — since they're ignored, the loss should not change vs masked.
    # (We can't put a different pad id, but we can sanity-check that the masked
    # positions' *logits* don't affect the result.)
    logits_perturbed = logits.clone()
    logits_perturbed[:, :, :3] = torch.randn_like(logits_perturbed[:, :, :3]) * 100
    loss_a, per_cb_a = _loss_fn(logits, masked, pad)
    loss_b, per_cb_b = _loss_fn(logits_perturbed, masked, pad)
    # Pad positions are ignored so perturbing only their logits is a no-op.
    assert torch.allclose(loss_a, loss_b, atol=1e-6)
    assert torch.allclose(per_cb_a, per_cb_b, atol=1e-6)


def test_loss_fn_all_pad_is_finite():
    """``mask.sum().clamp(min=1.0)`` guards against div-by-zero when an entire
    batch is pad — confirm we get a finite zero rather than NaN."""
    B, K, T, V = 1, 9, 5, 1025
    pad = V - 1
    logits = torch.randn(B, K, T, V)
    targets = torch.full((B, K, T), pad, dtype=torch.long)
    total, per_cb = _loss_fn(logits, targets, pad)
    assert torch.isfinite(total).all(), f"total loss is non-finite: {total}"
    assert torch.isfinite(per_cb).all(), f"per-cb loss is non-finite: {per_cb}"
    # All-pad → numerator is 0, denominator clamped to 1 → loss is 0.
    assert total.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_fn_shapes():
    B, K, T, V = 4, 9, 21, 1025
    pad = V - 1
    logits = torch.randn(B, K, T, V)
    targets = torch.randint(0, pad, (B, K, T))
    total, per_cb = _loss_fn(logits, targets, pad)
    assert total.shape == ()
    assert per_cb.shape == (K,)


# ----- _distill_loss (knowledge distillation) -----

def test_distill_loss_zero_kd_when_teacher_equals_student():
    """Identical teacher/student logits → KL is exactly 0, so KD=0 and the total
    collapses to alpha*CE. ce returned must equal the plain _loss_fn total."""
    torch.manual_seed(0)
    B, K, T, V = 2, 9, 13, 1025
    pad = V - 1
    logits = torch.randn(B, K, T, V)
    targets = torch.randint(0, pad, (B, K, T))
    alpha, tau = 0.5, 2.0
    total, ce, kd, per_cb = _distill_loss(logits, logits.clone(), targets, pad, tau, alpha)

    ref_total, ref_per_cb = _loss_fn(logits, targets, pad)
    assert kd.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(ce, ref_total, atol=1e-6)
    assert torch.allclose(per_cb, ref_per_cb, atol=1e-6)
    assert total.item() == pytest.approx(alpha * ce.item(), abs=1e-6)


def test_distill_loss_decomposition():
    """total == alpha*CE + (1-alpha)*KD for distinct teacher/student, and KD>0."""
    torch.manual_seed(1)
    B, K, T, V = 3, 9, 17, 1025
    pad = V - 1
    student = torch.randn(B, K, T, V)
    teacher = torch.randn(B, K, T, V)
    targets = torch.randint(0, pad, (B, K, T))
    alpha, tau = 0.3, 1.5
    total, ce, kd, _ = _distill_loss(student, teacher, targets, pad, tau, alpha)
    assert kd.item() > 0.0
    assert total.item() == pytest.approx(alpha * ce.item() + (1 - alpha) * kd.item(), abs=1e-5)


def test_distill_loss_pad_positions_ignored():
    """KD (like CE) is masked at pad targets: perturbing BOTH teacher and student
    logits only at pad positions leaves total/ce/kd unchanged."""
    torch.manual_seed(2)
    B, K, T, V = 2, 9, 15, 1025
    pad = V - 1
    student = torch.randn(B, K, T, V)
    teacher = torch.randn(B, K, T, V)
    targets = torch.randint(0, pad, (B, K, T))
    targets[:, :, :3] = pad  # mask the first 3 positions
    alpha, tau = 0.5, 2.0

    total_a, ce_a, kd_a, _ = _distill_loss(student, teacher, targets, pad, tau, alpha)
    s2, t2 = student.clone(), teacher.clone()
    s2[:, :, :3] = torch.randn_like(s2[:, :, :3]) * 100
    t2[:, :, :3] = torch.randn_like(t2[:, :, :3]) * 100
    total_b, ce_b, kd_b, _ = _distill_loss(s2, t2, targets, pad, tau, alpha)

    assert torch.allclose(total_a, total_b, atol=1e-5)
    assert torch.allclose(ce_a, ce_b, atol=1e-6)
    assert torch.allclose(kd_a, kd_b, atol=1e-5)


def test_distill_loss_all_pad_is_finite():
    """Entire batch pad → CE=0 and KD=0 (mask.sum() clamped), total finite zero."""
    B, K, T, V = 1, 9, 5, 1025
    pad = V - 1
    student = torch.randn(B, K, T, V)
    teacher = torch.randn(B, K, T, V)
    targets = torch.full((B, K, T), pad, dtype=torch.long)
    total, ce, kd, per_cb = _distill_loss(student, teacher, targets, pad, 2.0, 0.5)
    for t in (total, ce, kd, per_cb):
        assert torch.isfinite(t).all()
    assert total.item() == pytest.approx(0.0, abs=1e-6)
    assert kd.item() == pytest.approx(0.0, abs=1e-6)


# ----- _cosine_lr -----

def _cfg(lr=1e-3, warmup=100, steps=1000):
    # TrainConfig requires cache_dir/ckpt_dir as strings; values are arbitrary
    # for LR-schedule purposes — we never touch the filesystem here.
    return TrainConfig(
        cache_dir="/tmp/x", ckpt_dir="/tmp/y", device="cpu",
        lr=lr, warmup_steps=warmup, steps=steps,
    )


def test_cosine_lr_warmup_starts_above_zero():
    """Linear warmup uses (step + 1) / warmup, so step=0 → lr/warmup, not 0.
    This is intentional (avoid a true-zero LR on the first step)."""
    cfg = _cfg(lr=1.0, warmup=100)
    assert _cosine_lr(0, cfg) == pytest.approx(1.0 / 100)


def test_cosine_lr_warmup_last_step_reaches_peak():
    """At step=warmup-1, linear ramp is (warmup) / warmup = 1.0 → lr."""
    cfg = _cfg(lr=1.0, warmup=100)
    assert _cosine_lr(99, cfg) == pytest.approx(1.0)


def test_cosine_lr_at_warmup_boundary():
    """At step==warmup, cosine progress=0 → factor=1 → still lr."""
    cfg = _cfg(lr=1.0, warmup=100, steps=1000)
    assert _cosine_lr(100, cfg) == pytest.approx(1.0)


def test_cosine_lr_terminal_is_zero():
    """At step==cfg.steps, cosine progress is 1 → factor=0 → lr=0."""
    cfg = _cfg(lr=1.0, warmup=100, steps=1000)
    assert _cosine_lr(1000, cfg) == pytest.approx(0.0, abs=1e-9)


def test_cosine_lr_past_end_clamps_to_zero():
    """Schedule clamps progress at 1.0 so post-terminal steps stay at zero
    rather than going negative."""
    cfg = _cfg(lr=1.0, warmup=100, steps=1000)
    assert _cosine_lr(2000, cfg) == pytest.approx(0.0, abs=1e-9)


def test_cosine_lr_monotonic_decay_after_warmup():
    cfg = _cfg(lr=1.0, warmup=100, steps=1000)
    prev = _cosine_lr(100, cfg)
    for step in range(101, 1001, 25):
        cur = _cosine_lr(step, cfg)
        assert cur <= prev + 1e-9, f"lr increased between {step-25} → {step}: {prev} → {cur}"
        prev = cur


def test_cosine_lr_monotonic_ramp_during_warmup():
    cfg = _cfg(lr=1.0, warmup=100, steps=1000)
    prev = _cosine_lr(0, cfg)
    for step in range(1, 100):
        cur = _cosine_lr(step, cfg)
        assert cur >= prev - 1e-9, f"lr decreased during warmup at step {step}"
        prev = cur


def test_cosine_lr_midpoint_value():
    """Halfway through the decay window the cosine factor is 0.5, so the lr
    should be lr/2."""
    cfg = _cfg(lr=1.0, warmup=100, steps=1100)  # decay window = 1000 steps
    midpoint = 100 + 500  # halfway through decay
    expected = 0.5 * (1 + math.cos(math.pi * 0.5))  # = 0.5
    assert _cosine_lr(midpoint, cfg) == pytest.approx(expected, abs=1e-9)
