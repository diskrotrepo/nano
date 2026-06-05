"""Shared librosa feature extraction + scoring (single source of truth).

CLAP scoring can't run locally (torchcodec vs ffmpeg8 crash), so we rank on
librosa features only. From eval/analyze_samples.py's findings: spectral flatness
is useless (DAC decode is always tonal); the informative axes are BEAT STRENGTH
and SILENCE-COLLAPSE %. Score gates on collapse, then rewards rhythm + healthy
loudness.
"""
import numpy as np
import librosa


def features(path: str, sr: int = 22050) -> dict:
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if y.size < sr:  # < 1s -> treat as collapsed
        return {"rms": 0.0, "beat": 0.0, "sil": 1.0, "centroid": 0.0}
    rms = float(np.sqrt(np.mean(y**2)))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(onset - onset.mean())
    ac = ac / (ac[0] + 1e-9)
    beat = float(np.max(ac[4:200])) if ac.size > 200 else 0.0
    sil = float(np.mean(np.abs(y) < 0.01))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return {"rms": rms, "beat": beat, "sil": sil, "centroid": centroid}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def score(feat: dict) -> float:
    """0..1. Collapse is a multiplicative gate (the dominant failure mode);
    within non-collapsed clips, 0.6*beat + 0.4*rms_band."""
    collapse_gate = _clip(1 - feat["sil"] / 0.30, 0.0, 1.0)   # sil >= 30% -> 0
    beat_term = min(feat["beat"], 1.0)
    rms = feat["rms"]
    if rms < 0.02 or rms > 0.35:                              # inaudible or blown out
        rms_term = 0.0
    else:
        rms_term = _clip(1 - abs(rms - 0.10) / 0.10, 0.0, 1.0)  # peaks at ~0.10
    return collapse_gate * (0.60 * beat_term + 0.40 * rms_term)


def aggregate(scores: list[float]) -> dict:
    """Rank settings on mean - 0.5*std so collapse-prone (high-variance) settings
    are penalized as risky defaults."""
    a = np.array(scores, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "rank_score": float(a.mean() - 0.5 * a.std()),
    }
