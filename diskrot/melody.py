"""Chromagram extraction for melody conditioning — the train==inference contract.

A hummed/uploaded melody is conditioned on as a time-aligned **chromagram**
(MusicGen-Melody style): one octave-invariant 12-bin pitch-class vector per audio
frame, at the SAME ~86 Hz frame rate as the DAC tokens (``hop_length=512`` at
44.1 kHz, matching ``DACodec.FRAME_RATE_HZ``). The decoder adds a projection of
this sequence to its per-frame input, so the model regenerates the melodic contour
in whatever timbre the text prompt asks for.

``extract_chroma`` is the SINGLE source of truth for the chroma id stream, shared
by the dataset (train-time, via the Modal melody job + packer) and inference
(``server/inference.py`` cover path). A train/inference mismatch here is the same
class of footgun as a g2p drift in the lyric path, so both sides call THIS
function — see ``tests/test_melody.py`` for the byte-identical guard.

Key contract:
- Octave-invariant ``chroma_cqt`` (CQT log-frequency bins track a sung/hummed
  pitch far better than STFT chroma; octave-invariance lets the cover sit in its
  own register, e.g. a violin an octave off the hum).
- Each frame L2-normalized (so loudness/silence doesn't leak in); a near-silent
  frame becomes an all-zero 12-vector.
- Frame count forced to exactly ``ceil(samples / 512)`` — the DAC frame-count rule
  (``codec.encode_batch`` line) — so a song's chroma indexes byte-identically to
  its token tensor in the parallel packed mmap. When ``n_frames`` is passed
  explicitly (the packer passes the token tensor's frame count), it is forced to
  that instead, guarding against any off-by-one between librosa and DAC.
"""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

N_CHROMA = 12
SAMPLE_RATE = 44100
HOP_LENGTH = 512  # -> 86.13 Hz; DACodec uses the same hop, frame count = ceil(n/hop)
_EPS = 1e-8


def _dac_frame_count(n_samples: int) -> int:
    """Frames DAC would emit for ``n_samples`` mono samples: ceil(n / hop).

    Mirrors ``DACodec.encode_batch`` (``t_i = (samples + hop - 1) // hop``) so the
    chroma frame count matches the token frame count for the same audio.
    """
    return (n_samples + HOP_LENGTH - 1) // HOP_LENGTH


def _resample_time(chroma: np.ndarray, target: int) -> np.ndarray:
    """Resample ``chroma`` [12, T] along time to exactly ``target`` frames.

    Linear interpolation per pitch-class row (deterministic — the train==inference
    guard depends on it). A no-op when already the right length. ``target`` frames
    are sampled at evenly spaced positions across the original [0, T-1] grid.
    """
    t = chroma.shape[1]
    if t == target:
        return chroma
    if t == 0:
        return np.zeros((N_CHROMA, target), dtype=chroma.dtype)
    src_x = np.arange(t)
    # Map each target index onto the source grid; endpoints inclusive so the
    # first/last frames anchor (avoids a half-frame shift that would break the
    # crop alignment guard).
    dst_x = np.linspace(0, t - 1, target)
    out = np.empty((N_CHROMA, target), dtype=chroma.dtype)
    for c in range(N_CHROMA):
        out[c] = np.interp(dst_x, src_x, chroma[c])
    return out


def _l2_normalize_frames(chroma: np.ndarray) -> np.ndarray:
    """L2-normalize each time frame (column) of [12, T]; zero frames stay zero."""
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    return chroma / np.maximum(norms, _EPS)


def extract_chroma(
    source: str | Path | np.ndarray,
    n_frames: int | None = None,
) -> np.ndarray:
    """Octave-invariant chromagram for melody conditioning.

    source: a filesystem path (any ffmpeg-decodable audio) OR a mono float waveform
        ``np.ndarray[samples]`` already at ``SAMPLE_RATE``.
    n_frames: force the output to exactly this many frames (the packer passes the
        song's DAC token frame count). When None, uses ``ceil(samples / 512)`` —
        the DAC convention — so train (n_frames given) and inference (None) land on
        the identical force-align target for the same audio.

    returns: ``np.ndarray[12, n_frames]`` float32, each frame L2-normalized.
    """
    if isinstance(source, (str, Path)):
        y, _ = librosa.load(str(source), sr=SAMPLE_RATE, mono=True)
    else:
        y = np.asarray(source, dtype=np.float32)
        if y.ndim != 1:
            y = y.reshape(-1)

    target = n_frames if n_frames is not None else _dac_frame_count(len(y))
    target = max(1, int(target))

    if len(y) == 0:
        return np.zeros((N_CHROMA, target), dtype=np.float32)

    chroma = librosa.feature.chroma_cqt(
        y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, n_chroma=N_CHROMA,
    ).astype(np.float32)  # [12, T_raw]
    chroma = _l2_normalize_frames(chroma)
    chroma = _resample_time(chroma, target)
    # Interpolation can perturb unit norm slightly; renormalize so every stored
    # frame is exactly unit (or zero), keeping the model's input distribution tight.
    chroma = _l2_normalize_frames(chroma)
    return np.ascontiguousarray(chroma, dtype=np.float32)
