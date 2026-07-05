"""Tests for the SpectroStream length-bucketing math (model.codec.bucket_target_samples).

Pure frame arithmetic — no codec/GPU needed. The byte-identity of the pad-encode-crop
against an un-bucketed encode is a GPU property verified separately by
diskrot/modal_spectrostream_spike.py::bucket_check.
"""
from __future__ import annotations

import math

from model.codec import bucket_target_samples, frame_safe_target_samples

HOP = 1920  # SpectroStream 48000/25
SR, FR = 48000, 25
MAXS = 360 * 48000  # int32-overflow guard ceiling (NANO_MAX_ENCODE_SECONDS_SS)


def _ss_expected(t: int) -> int:
    """SpectroStream's exact float-ceil frame count (the assertion in its encode)."""
    return math.ceil((t / float(SR)) * float(FR))


def _would_crash(t: int) -> bool:
    """SS aborts when its float `expected` != the conv output ceil(t/hop)."""
    return _ss_expected(t) != (t + HOP - 1) // HOP


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


# --- frame_safe_target_samples: the SpectroStream float-ceil off-by-one guard ---

# Known boundary frame counts where (F*hop/sr)*fr floats to F+1 (see the analysis
# in modal_spectrostream_spike.py::verify_offbyone_fix, which reproduces the crash
# on these exact lengths on GPU).
_BOUNDARY_F = [7, 14, 28, 55, 56, 109, 110, 111, 112]


def test_boundary_lengths_are_detected_as_crashing():
    # Sanity: these frame-aligned lengths really do hit SS's float off-by-one.
    for F in _BOUNDARY_F:
        assert _would_crash(F * HOP), F


def test_guard_fixes_every_boundary_length():
    for F in _BOUNDARY_F:
        out = frame_safe_target_samples(F * HOP, HOP, SR, FR)
        assert out != F * HOP                    # it nudged
        assert not _would_crash(out)             # now SS-safe
        assert (out + HOP - 1) // HOP >= F        # conv frames >= true_frames → crop valid


def test_guard_inert_on_non_boundary_lengths():
    # The ~95% that already encode must be returned UNCHANGED (byte-identical encode →
    # existing tokens are not invalidated). This is the "does it break my tokens" guard.
    for F in range(1, 4000):
        t = F * HOP
        if _would_crash(t):
            continue
        assert frame_safe_target_samples(t, HOP, SR, FR) == t, F


def test_guard_fires_iff_would_crash_over_full_range():
    # The core safety invariant: the guard changes the length EXACTLY when (and only
    # when) SS would otherwise crash. So any length that previously succeeded (has a
    # cached .pt) is untouched, and every previously-crashing length is now fixed.
    changed = fixed = 0
    for F in range(1, 12000):
        t = F * HOP
        out = frame_safe_target_samples(t, HOP, SR, FR)
        if _would_crash(t):
            assert out != t and not _would_crash(out), F
            fixed += 1
        else:
            assert out == t, F
        changed += int(out != t)
    assert changed == fixed > 0               # only-and-all boundary lengths changed
    assert fixed / 12000 < 0.06               # ~4.8% — a small, bounded fraction


def test_guard_also_covers_non_frame_aligned_targets():
    # bucket_target_samples can hand a non-frame-aligned tgt (grid>1 or the n-unchanged
    # path); the guard must still only nudge genuinely-crashing lengths.
    for n in range(1, 3_000_000, 4801):
        out = frame_safe_target_samples(n, HOP, SR, FR)
        assert not _would_crash(out)          # always resolves to an SS-safe length
        assert out >= n                       # never shrinks below the requested length
        assert (out - n) <= 8 * (HOP // 2)    # within the guard's nudge cap
