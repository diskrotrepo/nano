"""Audio-LLM captioner — the CPU-safe surface.

The Qwen2-Audio model (~7B) can't run in CI, so this covers everything around it:
- load_song_windows: the train==inference "what audio the captioner sees" contract
  (window count / length / padding / offsets), exercised with a fake librosa so no
  audio decode or librosa install is needed.
- _clean / _build_conversation: backend-agnostic helpers shared by both engines
  (so the vLLM prompt is byte-identical to the HF one).
- load_captioner: NANO_CAPTIONER_BACKEND selects the engine; construction is lazy
  (no torch/vllm import), so we can assert the class without the heavy deps.
"""
from __future__ import annotations

import sys
import types

import numpy as np

from model.audio_llm_captioner import (
    MAX_CHARS,
    MAX_WINDOWS,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    AudioLLMCaptioner,
    VLLMAudioCaptioner,
    _build_conversation,
    _CAPTION_INSTRUCTION,
    _clean,
    load_captioner,
    load_song_windows,
)


def _patch_librosa(monkeypatch, audio: np.ndarray) -> None:
    """Inject a fake ``librosa`` whose ``load`` returns *audio* — load_song_windows
    imports librosa lazily, so this tests the pure windowing math with no decode."""
    fake = types.ModuleType("librosa")

    def _load(path, sr=SAMPLE_RATE, mono=True):
        return np.asarray(audio, dtype=np.float32), sr

    fake.load = _load
    monkeypatch.setitem(sys.modules, "librosa", fake)


# ----------------------------- load_song_windows -----------------------------

def test_short_song_single_padded_window(monkeypatch):
    total = SAMPLE_RATE * 10  # 10 s < 30 s window
    audio = np.linspace(-1.0, 1.0, total, dtype=np.float32)
    _patch_librosa(monkeypatch, audio)

    wins = load_song_windows("x.wav")

    assert len(wins) == 1
    assert wins[0].shape[0] == SAMPLE_RATE * WINDOW_SECONDS
    # original samples kept at the front, zero-padded after
    assert np.allclose(wins[0][:total], audio)
    assert np.all(wins[0][total:] == 0.0)


def test_long_song_uses_max_windows(monkeypatch):
    total = SAMPLE_RATE * 130  # 130 // 30 = 4 windows fit
    _patch_librosa(monkeypatch, np.zeros(total, dtype=np.float32))

    wins = load_song_windows("x.wav")

    assert len(wins) == MAX_WINDOWS == 4
    assert all(w.shape[0] == SAMPLE_RATE * WINDOW_SECONDS for w in wins)


def test_max_windows_param_caps_count(monkeypatch):
    total = SAMPLE_RATE * 130
    _patch_librosa(monkeypatch, np.zeros(total, dtype=np.float32))

    assert len(load_song_windows("x.wav", max_windows=2)) == 2


def test_medium_song_single_window(monkeypatch):
    total = SAMPLE_RATE * 45  # 45 // 30 = 1 -> single 25%-offset window
    _patch_librosa(monkeypatch, np.zeros(total, dtype=np.float32))

    wins = load_song_windows("x.wav")

    assert len(wins) == 1
    assert wins[0].shape[0] == SAMPLE_RATE * WINDOW_SECONDS


# --------------------------------- _clean ------------------------------------

def test_clean_collapses_whitespace():
    assert _clean("  a   b\n\tc  ") == "a b c"


def test_clean_strips_wrapping_quotes():
    assert _clean('"hello world"') == "hello world"
    assert _clean("'hello'") == "hello"


def test_clean_trims_to_maxchars_at_word_boundary():
    out = _clean("word " * 2000)  # ~10k chars
    assert len(out) <= MAX_CHARS
    assert set(out.split()) == {"word"}  # no partial token at the cut


def test_clean_handles_empty_and_none():
    assert _clean("") == ""
    assert _clean(None) == ""


# ----------------------------- _build_conversation ---------------------------

def test_build_conversation_audio_count_and_instruction():
    conv = _build_conversation([np.zeros(4), np.zeros(4), np.zeros(4)])

    assert len(conv) == 1 and conv[0]["role"] == "user"
    content = conv[0]["content"]
    audio_items = [c for c in content if c["type"] == "audio"]
    text_items = [c for c in content if c["type"] == "text"]
    assert len(audio_items) == 3
    assert len(text_items) == 1
    assert text_items[0]["text"] == _CAPTION_INSTRUCTION
    # audio entries precede the instruction (order matters for the template)
    assert content[-1]["type"] == "text"


# ------------------------------- load_captioner ------------------------------

def test_load_captioner_default_is_vllm(monkeypatch):
    monkeypatch.delenv("NANO_CAPTIONER_BACKEND", raising=False)
    cap = load_captioner("cuda")
    assert isinstance(cap, VLLMAudioCaptioner)
    assert hasattr(cap, "caption_many")  # the batched API only the vLLM path has


def test_load_captioner_hf_backend(monkeypatch):
    monkeypatch.setenv("NANO_CAPTIONER_BACKEND", "hf")
    cap = load_captioner("cpu")
    assert isinstance(cap, AudioLLMCaptioner)
    assert not hasattr(cap, "caption_many")


def test_load_captioner_backend_case_insensitive(monkeypatch):
    monkeypatch.setenv("NANO_CAPTIONER_BACKEND", "HF")
    assert isinstance(load_captioner(), AudioLLMCaptioner)


def test_both_backends_expose_single_song_contract():
    # The Modal worker / local CLI call .caption(windows) regardless of backend.
    for cls in (AudioLLMCaptioner, VLLMAudioCaptioner):
        inst = cls()
        assert hasattr(inst, "caption") and hasattr(inst, "caption_path")
