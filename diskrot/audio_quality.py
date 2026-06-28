"""Audio-quality assessment for the prepare gate (raise the training-data floor).

``modal_prepare`` already drops undecodable / too-short / too-long / byte-duplicate
files. This adds a *content* quality gate: cheap signal-level metrics that catch the
recordings that cap the model's output fidelity no matter how much of them you have —
hard-clipped/distorted rips, near-silent or mostly-silent files, and very-low-bitrate
lossy junk. Garbage-in is a hard ceiling, so flagging it is the highest-ROI data lever
that doesn't touch the architecture (see the "additional steps" analysis).

Two pure pieces, both unit-testable without Modal:
  - ``decode_mono_pcm`` — decode an audio file to a mono float32 array via the
    ``ffmpeg`` binary already in the prepare image (no librosa/soundfile — keeps the
    debian_slim image light). Downsamples + optionally truncates so the gate stays
    cheap relative to ffprobe+sha.
  - ``assess_quality`` — turn a decoded array (+ the ffprobe bit_rate) into metrics
    and an ``ok`` / ``low_quality`` verdict with human-readable reasons.

The gate is OPT-IN (``modal_prepare --quality-gate``): decoding every file is real CPU
work on top of the ffprobe+sha pass, and the thresholds are deliberately conservative
(drop only clear garbage), so it's a flag rather than the default.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    """Conservative defaults — trip only on clear garbage, not on artistic choices
    (a deliberately quiet ambient intro or a loud-but-clean master must pass)."""

    sr: int = 22_050            # decode rate for the metrics (downsampled — cheap)
    max_seconds: float = 120.0  # only analyze the first N s (a song is homogeneous enough)
    # Hard clipping: fraction of samples pinned at/near full-scale. A clean master
    # peaks at full-scale only momentarily; >2% of samples there is sustained
    # clipping / a brick-walled distorted rip.
    clip_level: float = 0.992
    max_clip_ratio: float = 0.02
    # Mostly-silent: fraction of short frames below the silence floor. A song with
    # >60% silence is a fragment, a long fade of dead air, or a mis-rip.
    silence_dbfs: float = -50.0
    max_silence_ratio: float = 0.60
    # Dead / near-silent overall: an entire file quieter than this is unusable.
    min_rms_dbfs: float = -40.0
    # Lossy junk: MP3/AAC encoded below this bit rate is audibly degraded. None
    # bit_rate (lossless / unknown) is never flagged on this axis.
    min_bit_rate: int = 96_000


@dataclass
class QualityResult:
    status: str                       # "ok" | "low_quality"
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def decode_mono_pcm(
    path: str | Path, sr: int = 22_050, max_seconds: float | None = None,
) -> np.ndarray:
    """Decode ``path`` to a mono float32 array in [-1, 1] via the ffmpeg binary.

    Uses ffmpeg (already in the prepare image) → raw ``f32le`` on stdout → numpy, so
    no librosa/soundfile/audioread dependency. Returns an empty array if ffmpeg
    produces no audio (caller treats that as undecodable, not low-quality)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds:.3f}"]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _dbfs(rms: float) -> float:
    return 20.0 * float(np.log10(max(rms, 1e-10)))


def assess_quality(
    samples: np.ndarray,
    sr: int,
    bit_rate: int | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityResult:
    """Score a decoded mono array → (status, reasons, metrics).

    ``status`` is ``"low_quality"`` if ANY conservative threshold trips, else
    ``"ok"``. ``bit_rate`` is the ffprobe value (bits/sec) or None for
    lossless/unknown. Pure + deterministic — the unit tests drive it with synthetic
    clipped / silent / quiet signals."""
    t = thresholds or QualityThresholds()
    x = np.asarray(samples, dtype=np.float32).ravel()
    if x.size == 0:
        # No audio decoded — let the caller's undecodable path own this; we report
        # an empty verdict rather than inventing a low-quality reason.
        return QualityResult(status="ok", reasons=[], metrics={"n_samples": 0})

    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))

    # Clipping: share of samples pinned at/near full-scale.
    clip_ratio = float(np.mean(np.abs(x) >= t.clip_level))

    # Silence: frame-level RMS over ~50 ms windows; share of frames below the floor.
    win = max(1, int(0.05 * sr))
    n_full = (x.size // win) * win
    if n_full >= win:
        frames = x[:n_full].reshape(-1, win)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, 1e-10))
        silence_ratio = float(np.mean(frame_dbfs < t.silence_dbfs))
    else:
        silence_ratio = 1.0 if _dbfs(rms) < t.silence_dbfs else 0.0

    rms_dbfs = _dbfs(rms)
    metrics = {
        "n_samples": int(x.size),
        "duration_s": round(x.size / sr, 2),
        "peak_dbfs": round(_dbfs(peak), 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "clip_ratio": round(clip_ratio, 5),
        "silence_ratio": round(silence_ratio, 4),
        "bit_rate": bit_rate,
    }

    reasons: list[str] = []
    if clip_ratio > t.max_clip_ratio:
        reasons.append(f"clipped({clip_ratio:.1%}>{t.max_clip_ratio:.0%})")
    if silence_ratio > t.max_silence_ratio:
        reasons.append(f"silent({silence_ratio:.0%}>{t.max_silence_ratio:.0%})")
    if rms_dbfs < t.min_rms_dbfs:
        reasons.append(f"dead({rms_dbfs:.0f}dBFS<{t.min_rms_dbfs:.0f})")
    if bit_rate is not None and 0 < bit_rate < t.min_bit_rate:
        reasons.append(f"lowbitrate({bit_rate // 1000}k<{t.min_bit_rate // 1000}k)")

    return QualityResult(
        status="low_quality" if reasons else "ok",
        reasons=reasons,
        metrics=metrics,
    )


def assess_file(
    path: str | Path,
    bit_rate: int | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityResult:
    """Decode ``path`` and assess it (the one call the prepare worker makes)."""
    t = thresholds or QualityThresholds()
    samples = decode_mono_pcm(path, sr=t.sr, max_seconds=t.max_seconds)
    return assess_quality(samples, t.sr, bit_rate=bit_rate, thresholds=t)
