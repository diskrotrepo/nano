"""Tests for diskrot/audio_io.py — the shared ffmpeg PCM decoder.

The ffmpeg-dependent cases skip when ffmpeg isn't on PATH; the contract cases
(validation, empty decode, librosa fallback) run everywhere.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from diskrot.audio_io import decode_pcm

_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_skip_no_ffmpeg = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")


def _write_stereo_wav(path, sr=48000, seconds=0.5, freqs=(220.0, 277.18)):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    stereo = np.stack(
        [0.2 * np.sin(2 * np.pi * freqs[0] * t), 0.2 * np.sin(2 * np.pi * freqs[1] * t)],
        axis=1,
    )  # [N, 2] for soundfile
    sf.write(str(path), stereo, sr)
    return stereo.T  # [2, N]


def _write_mono_wav(path, sr=48000, seconds=0.5, freq=220.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    mono = 0.2 * np.sin(2 * np.pi * freq * t)
    sf.write(str(path), mono, sr)
    return mono


@_skip_no_ffmpeg
def test_decode_stereo_shape_dtype_content(tmp_path):
    p = tmp_path / "s.wav"
    ref = _write_stereo_wav(p, sr=48000)
    y = decode_pcm(p, 48000, 2)
    assert y.ndim == 2 and y.shape[0] == 2 and y.dtype == np.float32
    assert abs(y.shape[1] - ref.shape[1]) <= 4  # same-rate: no length change
    n = min(y.shape[1], ref.shape[1])
    assert np.allclose(y[:, :n], ref[:, :n], atol=1e-3)  # WAV passthrough ~lossless


@_skip_no_ffmpeg
def test_decode_mono_shape(tmp_path):
    p = tmp_path / "m.wav"
    ref = _write_mono_wav(p, sr=48000)
    y = decode_pcm(p, 48000, 1)
    assert y.shape[0] == 1 and y.ndim == 2 and y.dtype == np.float32
    n = min(y.shape[1], ref.shape[0])
    assert np.allclose(y[0, :n], ref[:n], atol=1e-3)


@_skip_no_ffmpeg
def test_mono_source_upmixed_to_LR(tmp_path):
    p = tmp_path / "m.wav"
    _write_mono_wav(p, sr=48000)
    y = decode_pcm(p, 48000, 2)  # request stereo from a mono source
    assert y.shape[0] == 2
    assert np.allclose(y[0], y[1], atol=1e-6)  # L == R


@_skip_no_ffmpeg
def test_decode_resamples_in_process(tmp_path):
    p = tmp_path / "s.wav"
    ref = _write_stereo_wav(p, sr=48000, seconds=1.0)
    y = decode_pcm(p, 24000, 2)  # half rate → ~half the samples (ffmpeg swr)
    assert abs(y.shape[1] - ref.shape[1] // 2) <= 4


def test_invalid_n_channels_raises(tmp_path):
    with pytest.raises(ValueError):
        decode_pcm(tmp_path / "x.wav", 48000, 3)


def test_empty_decode_returns_empty_2d(monkeypatch, tmp_path):
    # ffmpeg produced no audio → [C, 0] (caller's quality gate treats it as silent),
    # not a crash.
    class _P:
        stdout = b""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _P())
    y = decode_pcm(tmp_path / "x.wav", 48000, 2)
    assert y.shape == (2, 0) and y.dtype == np.float32


def test_missing_ffmpeg_falls_back_to_librosa(monkeypatch, tmp_path):
    p = tmp_path / "s.wav"
    _write_stereo_wav(p, sr=48000)

    def _raise(*a, **k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "run", _raise)
    y2 = decode_pcm(p, 48000, 2)
    assert y2.ndim == 2 and y2.shape[0] == 2 and y2.dtype == np.float32
    y1 = decode_pcm(p, 48000, 1)
    assert y1.shape[0] == 1 and y1.dtype == np.float32
