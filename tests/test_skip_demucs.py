"""Guards for the skip-Demucs transcribe lever (transcribe_lyrics).

Covers the parts that don't need a GPU or the real Demucs/Whisper weights:
the Demucs-lite gender-clip CLAMPING (the easy-to-get-wrong bit — an unclamped
mid-song slice goes empty on short songs), the _prepare_transcribe_audio routing
(stem vs raw-mix + clip), and that _transcribe estimates gender from the SEPARATE
clean signal in skip-mode rather than the raw mix.
"""
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("librosa")

from diskrot import transcribe_lyrics as tl

SR = 44100


def _fake_apply(model, wav, device, overlap):
    """Stand in for demucs.apply_model: wav is [1, 2, T]; return [1, 4, 2, T]
    with the LAST source (vocals) equal to the input, so _demucs_vocals returns
    the mono mix of whatever clip it was handed (lets us check lengths/slicing)."""
    return wav.unsqueeze(1).repeat(1, 4, 1, 1)


def _stereo(n_samples):
    """Deterministic [2, n] stereo where the two channels differ (so a mono
    mean is distinguishable from either channel)."""
    t = np.arange(n_samples, dtype=np.float32)
    return np.stack([np.sin(t / 100.0), np.cos(t / 100.0)]).astype(np.float32)


@pytest.mark.parametrize("dur_sec, expect_sec", [
    (200, 30),   # normal: 30s clip from the mid-song window [60s, 90s]
    (45, 30),    # window [60,90] overruns 45s -> slid back to [15,45], still 30s
    (20, 20),    # song shorter than the 30s clip -> whole song
])
def test_gender_clip_clamps_to_song_length(monkeypatch, dur_sec, expect_sec):
    monkeypatch.setattr(tl, "_GENDER_CLIP_START_SEC", 60.0)
    monkeypatch.setattr(tl, "_GENDER_CLIP_SEC", 30.0)
    audio = _stereo(SR * dur_sec)
    vocals = tl._gender_clip_vocals(None, _fake_apply, audio, "cpu")
    assert vocals.ndim == 1
    assert vocals.shape[0] == SR * expect_sec


def test_prepare_skip_mode_returns_mix_and_clip(monkeypatch):
    monkeypatch.setattr(tl, "_GENDER_CLIP_START_SEC", 60.0)
    monkeypatch.setattr(tl, "_GENDER_CLIP_SEC", 30.0)
    audio = _stereo(SR * 200)
    monkeypatch.setattr(tl, "_ffmpeg_load_stereo", lambda p, sr=SR: audio)

    asr, gender = tl._prepare_transcribe_audio(
        None, _fake_apply, "x.mp3", "cpu", skip_demucs=True)
    # ASR signal is the full-length raw mix (mono mean of the two channels)...
    assert asr.shape[0] == SR * 200
    np.testing.assert_allclose(asr, audio.mean(axis=0), rtol=0, atol=1e-6)
    # ...and gender comes from a SEPARATE 30s clip, not the whole track.
    assert gender is not None and gender.shape[0] == SR * 30


def test_prepare_normal_mode_returns_stem_and_none(monkeypatch):
    audio = _stereo(SR * 90)
    monkeypatch.setattr(tl, "_ffmpeg_load_stereo", lambda p, sr=SR: audio)

    asr, gender = tl._prepare_transcribe_audio(
        None, _fake_apply, "x.mp3", "cpu", skip_demucs=False)
    assert gender is None  # signals "reuse the ASR stem for gender"
    assert asr.shape[0] == SR * 90  # full-track stem (mono)


class _FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class _FakeSeg:
    def __init__(self):
        self.text = "la la la"
        self.avg_logprob = -0.3
        self.words = [_FakeWord("la", 0.0, 0.5)]


class _FakeWhisper:
    def transcribe(self, audio16k, word_timestamps, vad_filter):
        info = types.SimpleNamespace(language="en", language_probability=0.9)
        return [_FakeSeg()], info


def test_transcribe_gender_uses_separate_clean_signal(monkeypatch):
    """In skip-mode the ASR input is the raw mix but the F0 gender estimate must
    run on the SEPARATE clip signal — verify estimate_vocal_gender receives the
    clip-derived array (its 16k length), not the mix's."""
    captured = {}

    def _fake_gender(arr, sr):
        captured["len"] = arr.shape[0]
        return "male"

    monkeypatch.setattr(tl, "estimate_vocal_gender", _fake_gender)

    asr_mix = np.zeros(SR * 10, dtype=np.float32)        # 10s raw mix -> 16k = 160000
    gender_clip = np.zeros(SR * 30, dtype=np.float32)    # 30s clip    -> 16k = 480000

    res = tl._transcribe(_FakeWhisper(), asr_mix, gender_vocals=gender_clip)
    assert res is not None and res["gender"] == "male"
    assert captured["len"] == 16000 * 30  # the clip's 16k length, not the mix's

    # And when gender_vocals is None (normal path) it reuses the ASR 16k signal.
    captured.clear()
    tl._transcribe(_FakeWhisper(), asr_mix, gender_vocals=None)
    assert captured["len"] == 16000 * 10
