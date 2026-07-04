"""Stage-2 loudnorm speedup: measuring integrated LUFS on a decimated copy must
yield a gain within a fraction of a dB of the full-rate measurement (LUFS is
low-frequency-weighted and robust), so the resulting tokens shift only marginally.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("pyloudnorm")

import diskrot.tokenize as tok


def _signal(sr: int = 48000, secs: float = 3.0) -> np.ndarray:
    t = np.linspace(0, secs, int(sr * secs), endpoint=False).astype(np.float32)
    y = (
        0.10 * np.sin(2 * np.pi * 440 * t)
        + 0.06 * np.sin(2 * np.pi * 110 * t)
        + 0.03 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)
    return y[None, :]  # [1, N]


def test_decimated_gain_close_to_full(monkeypatch):
    sr = 48000
    y = _signal(sr)
    monkeypatch.setattr(tok, "_LOUDNORM_DECIMATE", 1)
    a = tok._normalize_loudness(y.copy(), sr)
    monkeypatch.setattr(tok, "_LOUDNORM_DECIMATE", 4)
    b = tok._normalize_loudness(y.copy(), sr)
    ga, gb = float(np.abs(a).max()), float(np.abs(b).max())
    assert ga > 0 and gb > 0
    # Neither should have hit the anti-clip divide (keeps this a gain comparison).
    assert ga < 1.0 and gb < 1.0
    ddb = 20 * math.log10(ga / gb)
    assert abs(ddb) < 0.5, f"decimated gain differs {ddb:.3f} dB from full-rate"
