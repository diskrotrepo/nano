"""Tests for the SpectroStream length-bucketing math (model.codec.bucket_target_samples).

Pure frame arithmetic — no codec/GPU needed. The byte-identity of the pad-encode-crop
against an un-bucketed encode is a GPU property verified separately by
diskrot/modal_spectrostream_spike.py::bucket_check.
"""
from __future__ import annotations

from model.codec import bucket_target_samples

HOP = 1920  # SpectroStream 48000/25
MAXS = 360 * 48000  # int32-overflow guard ceiling (NANO_MAX_ENCODE_SECONDS_SS)


def test_off_returns_unchanged():
    assert bucket_target_samples(123_456, HOP, 0, MAXS) == 123_456


def test_rounds_up_to_grid():
    n = 500 * HOP  # exactly 500 frames
    assert bucket_target_samples(n, HOP, 256, MAXS) == 512 * HOP  # → next mult of 256


def test_partial_frame_rounds_up():
    n = 500 * HOP + 5  # ceil → 501 frames → 512
    assert bucket_target_samples(n, HOP, 256, MAXS) == 512 * HOP


def test_already_on_grid_no_pad():
    n = 512 * HOP
    assert bucket_target_samples(n, HOP, 256, MAXS) == n


def test_mid_length_pads():
    n = 100 * 48000  # 100 s → 2500 frames → 2560
    out = bucket_target_samples(n, HOP, 256, MAXS)
    assert out == 2560 * HOP and out > n


def test_clamp_to_cap_returns_unchanged():
    # Just under the cap: rounding up would breach max_samples → no pad.
    n = (MAXS // HOP) * HOP - 10
    out = bucket_target_samples(n, HOP, 256, MAXS)
    assert out == n and out <= MAXS


def test_output_never_exceeds_cap():
    for secs in range(20, 361, 7):
        out = bucket_target_samples(secs * 48000, HOP, 256, MAXS)
        assert out <= MAXS
