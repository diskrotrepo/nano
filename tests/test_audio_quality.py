"""Tests for diskrot/audio_quality.py — the prepare content-quality gate.

Pure signal-level metrics, so the synthetic-signal tests below fully exercise the
verdict logic without any audio files or Modal."""
from __future__ import annotations

import numpy as np
import pytest

from diskrot.audio_quality import (
    QualityThresholds,
    assess_quality,
)

SR = 22_050


def _sine(seconds: float, amp: float, freq: float = 220.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_clean_signal_passes():
    # A clean -12 dBFS tone (amp ~0.25): no clipping, not silent, healthy RMS.
    x = _sine(5.0, amp=0.25)
    r = assess_quality(x, SR, bit_rate=320_000)
    assert r.status == "ok", r.reasons
    assert r.reasons == []
    assert r.metrics["clip_ratio"] == 0.0
    assert r.metrics["silence_ratio"] == 0.0


def test_hard_clipping_flagged():
    # A full-scale tone hard-clipped to a square-ish wave: a large fraction of
    # samples sit at +/-1.0 → clipped.
    x = np.clip(_sine(5.0, amp=4.0), -1.0, 1.0).astype(np.float32)
    r = assess_quality(x, SR, bit_rate=320_000)
    assert r.status == "low_quality"
    assert any("clipped" in reason for reason in r.reasons)


def test_mostly_silent_flagged():
    # 1 s of tone followed by 9 s of digital silence → >60% silent frames.
    x = np.concatenate([_sine(1.0, amp=0.3), np.zeros(int(9 * SR), dtype=np.float32)])
    r = assess_quality(x, SR, bit_rate=320_000)
    assert r.status == "low_quality"
    assert any("silent" in reason for reason in r.reasons)


def test_dead_quiet_flagged():
    # An extremely quiet signal (~-60 dBFS) trips the dead/near-silent floor.
    x = _sine(5.0, amp=0.001)
    r = assess_quality(x, SR, bit_rate=320_000)
    assert r.status == "low_quality"
    assert any("dead" in reason for reason in r.reasons)


def test_low_bitrate_flagged():
    x = _sine(5.0, amp=0.25)
    r = assess_quality(x, SR, bit_rate=64_000)
    assert r.status == "low_quality"
    assert any("lowbitrate" in reason for reason in r.reasons)


def test_lossless_unknown_bitrate_not_flagged_on_bitrate():
    # bit_rate None (lossless / unknown) must never trip the bitrate axis.
    x = _sine(5.0, amp=0.25)
    r = assess_quality(x, SR, bit_rate=None)
    assert r.status == "ok", r.reasons


def test_empty_array_is_neutral():
    r = assess_quality(np.empty(0, dtype=np.float32), SR, bit_rate=320_000)
    assert r.status == "ok"
    assert r.metrics["n_samples"] == 0


def test_thresholds_are_tunable():
    # Tightening the clip threshold flips a borderline-clean signal.
    x = np.clip(_sine(5.0, amp=1.05), -1.0, 1.0).astype(np.float32)
    lenient = assess_quality(x, SR, thresholds=QualityThresholds(max_clip_ratio=0.5))
    strict = assess_quality(x, SR, thresholds=QualityThresholds(max_clip_ratio=0.001))
    assert lenient.status == "ok"
    assert strict.status == "low_quality"
