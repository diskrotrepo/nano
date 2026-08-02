"""Tests for diskrot/tempo_detect.py — the cheap dense tempo (BPM) source.

Three surfaces: (1) ``estimate_tempo`` must recover a clean periodic tempo and
refuse degenerate input (silent/short songs must not get a guessed bpm), (2) the
octave fold must guarantee every output lands in [60,180) so it maps to a real
``bpm_to_id`` bucket, and (3) ``detect_tempo`` must sweep a corpus, emit the
dataset's tempo.json schema, and resume by skip.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from diskrot.tempo_detect import (
    _TEMPO_FOLD_HI,
    _TEMPO_FOLD_LO,
    _octave_fold,
    estimate_tempo,
)
from model.lyric_encoder import UNKNOWN_TEMPO_ID, bpm_to_id

SR = 44100


def _click_track(bpm: float, seconds: float = 16.0, sr: int = SR) -> np.ndarray:
    """Deterministic percussive click train at ``bpm`` — a sharp onset every beat."""
    period = int(round(sr * 60.0 / bpm))
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(160) / sr
    click = (np.sin(2 * np.pi * 3000 * t) * np.exp(-t * 180.0)).astype(np.float32)
    for start in range(0, n - click.size, period):
        y[start:start + click.size] += click
    return y


# --- octave fold (deterministic, no librosa) ---------------------------------
def test_octave_fold_lands_in_band():
    # The fold only pulls OUT-of-band estimates into [60,180); an in-band value
    # (e.g. 150) is left alone — the start_bpm prior, not the fold, resolves an
    # in-band half/double ambiguity.
    cases = {40: 80, 50: 100, 59: 118, 75: 75, 120: 120, 150: 150, 181: 90.5,
             200: 100, 240: 120, 30: 60}
    for raw, want in cases.items():
        folded = _octave_fold(float(raw))
        assert folded == pytest.approx(want)
        assert _TEMPO_FOLD_LO <= folded < _TEMPO_FOLD_HI


# --- estimate_tempo ----------------------------------------------------------
def test_estimate_tempo_recovers_clean_120():
    librosa = pytest.importorskip("librosa")  # noqa: F841
    est = estimate_tempo(_click_track(120.0), SR)
    assert est is not None
    assert abs(est - 120.0) <= 6.0, f"got {est}"
    assert bpm_to_id(est) != UNKNOWN_TEMPO_ID


def test_estimate_tempo_always_in_band():
    pytest.importorskip("librosa")
    # Even a fast track (200 BPM) must fold into [60,180) — never returns a bpm
    # that bpm_to_id would push into the top open bucket via a double-tempo error.
    for bpm in (75.0, 90.0, 140.0, 200.0):
        est = estimate_tempo(_click_track(bpm), SR)
        assert est is not None
        assert _TEMPO_FOLD_LO <= est < _TEMPO_FOLD_HI, f"bpm={bpm} -> {est}"


def test_estimate_tempo_rejects_degenerate():
    pytest.importorskip("librosa")
    assert estimate_tempo(np.zeros(SR * 4, dtype=np.float32), SR) is None      # silence
    assert estimate_tempo(np.full(SR * 4, 1e-9, np.float32), SR) is None       # ~silent
    assert estimate_tempo(np.zeros(100, dtype=np.float32), SR) is None         # too short
    assert estimate_tempo(np.full(SR * 4, np.nan, np.float32), SR) is None     # non-finite


# --- detect_tempo sweep + resume ---------------------------------------------
def _audio_io_ok(tmp_path) -> bool:
    """True iff we can write an audio file and decode it back (soundfile + ffmpeg
    or libsndfile). The sweep test needs real decodable files."""
    try:
        import soundfile as sf

        from diskrot.melody import _load_audio_file
        p = tmp_path / "_probe.wav"
        sf.write(p, _click_track(120.0, seconds=2.0), SR)
        return _load_audio_file(str(p)).size > 0
    except Exception:
        return False


def test_detect_tempo_sweeps_and_resumes(tmp_path):
    pytest.importorskip("librosa")
    sf = pytest.importorskip("soundfile")
    if not _audio_io_ok(tmp_path):
        pytest.skip("no decodable-audio backend (soundfile/ffmpeg) available")
    from diskrot.tempo_detect import detect_tempo

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # ffmpeg/libsndfile probe by CONTENT, so wav-content named .mp3 decodes fine.
    sf.write(corpus / "song0.mp3", _click_track(120.0), SR, format="WAV")
    sf.write(corpus / "song1.mp3", np.zeros(SR * 4, np.float32), SR, format="WAV")  # silent -> omitted

    out = detect_tempo(corpus, verbose=False)
    payload = json.loads(out.read_text())
    assert "song1" not in payload                      # silent song omitted, never guessed
    assert set(payload) == {"song0"}
    bpm0 = payload["song0"]["bpm"]
    assert _TEMPO_FOLD_LO <= bpm0 < _TEMPO_FOLD_HI
    assert abs(bpm0 - 120.0) <= 6.0

    # Resume-by-skip: a second run leaves the existing estimate untouched.
    out.write_text(json.dumps({"song0": {"bpm": 99.0}}))  # poison to prove skip
    detect_tempo(corpus, verbose=False)
    assert json.loads(out.read_text())["song0"]["bpm"] == 99.0

    # The dataset loader consumes the schema as the <tempo_*> marker source.
    out.write_text(json.dumps(payload))
    from diskrot.dataset import _load_tempo
    assert _load_tempo(out, verbose=False) == {"song0": pytest.approx(bpm0)}
