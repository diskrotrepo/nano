"""Shared audio decode helper — the single ffmpeg → raw PCM contract.

Every corpus and inference audio-load path should route through ``decode_pcm`` so
decode is fast, quiet, and — critically — IDENTICAL on both sides of the
train==inference seam: the codec tokenizes the corpus at train time and re-encodes
prompt/seed audio at inference, and both must use the same decoder or the model
sees off-distribution seed tokens.

Why ffmpeg and not ``librosa.load``: librosa opens via libsndfile (soundfile),
which can't read most of the corpus's MP3s, so it silently falls back to the
deprecated audioread + libmpg123 path — slow, a "PySoundFile failed" / junk-header
warning storm across hundreds of thousands of songs, and it GIVES UP on some
marginally broken files ffmpeg recovers. ffmpeg decodes MP3 natively, resamples in
the same C process (swr), tolerates junk/ID3 headers, and is quiet. This mirrors
the already-migrated melody / transcribe / stems / audio_quality loaders and lets
them collapse onto one contract.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def decode_pcm(path: str | Path, sr: int, n_channels: int = 1) -> np.ndarray:
    """Decode any ffmpeg-readable audio to ``[n_channels, samples]`` float32 at
    ``sr`` via one ffmpeg subprocess (decode + resample fused in C).

    - ``n_channels=1`` downmixes to mono; ``n_channels=2`` upmixes a mono source to
      L=R (ffmpeg's ``-ac 2``, matching the old ``np.stack([y, y])`` intent).
    - Always returns a 2-D ``[C, N]`` array (``C == n_channels``), including for an
      empty/failed decode (``[C, 0]`` — callers' quality gate treats that as silent).
    - Falls back to ``librosa.load`` only if the ffmpeg binary isn't on PATH, so
      environments without ffmpeg still work (just noisily). A file ffmpeg runs on
      but can't decode raises ``CalledProcessError`` — the caller owns that as a
      per-file failure, exactly as the old librosa exception did.
    """
    if n_channels not in (1, 2):
        raise ValueError(f"n_channels must be 1 or 2, got {n_channels}")
    cmd = [
        "ffmpeg", "-nostdin", "-v", "quiet", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", str(n_channels), "-ar", str(sr), "-",
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True
        )
    except FileNotFoundError:
        # ffmpeg binary missing — fall back to librosa (the noisy audioread path).
        import librosa

        y, _ = librosa.load(str(path), sr=sr, mono=(n_channels == 1))
        if y.ndim == 1:
            y = np.stack([y, y], axis=0) if n_channels == 2 else y[None, :]
        return np.ascontiguousarray(y, dtype=np.float32)
    buf = np.frombuffer(proc.stdout, dtype=np.float32)
    if buf.size == 0:
        return np.zeros((n_channels, 0), dtype=np.float32)
    # ffmpeg writes interleaved f32le ([L0,R0,L1,R1,...]) → [N, C] → [C, N].
    return np.ascontiguousarray(buf.reshape(-1, n_channels).T)
