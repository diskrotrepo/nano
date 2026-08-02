"""Audio-LLM captioner — the CPU-safe surface.

The Qwen2-Audio model (~7B) can't run in CI, so this covers everything around it:
- load_song_windows: the train==inference "what audio the captioner sees" contract
  (window count / length / padding / offsets), exercised with a fake decode_pcm so
  no ffmpeg/audio decode is needed.
- _clean / _build_conversation: backend-agnostic helpers shared by both engines
  (so the vLLM prompt is byte-identical to the HF one).
- load_captioner: NANO_CAPTIONER_BACKEND selects the engine; construction is lazy
  (no torch/vllm import), so we can assert the class without the heavy deps.
"""
from __future__ import annotations

import numpy as np

from model.audio_llm_captioner import (
    CAPTIONER_MARKER,
    MAX_CHARS,
    MAX_WINDOWS,
    SAMPLE_RATE,
    STEM_CAPTION_KEYS,
    WINDOW_SECONDS,
    AudioLLMCaptioner,
    VLLMAudioCaptioner,
    _build_conversation,
    _CAPTION_INSTRUCTION,
    _clean,
    load_captioner,
    load_song_windows,
    parse_gender,
    parse_stems,
)


def _patch_decode(monkeypatch, audio: np.ndarray) -> None:
    """Patch ``diskrot.audio_io.decode_pcm`` to return *audio* as ``[1, N]`` —
    load_song_windows imports it lazily, so this tests the pure windowing math with
    no ffmpeg/decode. (Mirrors decode_pcm's ``[n_channels, samples]`` contract.)"""
    import diskrot.audio_io as audio_io

    def _decode(path, sr=SAMPLE_RATE, n_channels=1):
        return np.asarray(audio, dtype=np.float32)[None, :]

    monkeypatch.setattr(audio_io, "decode_pcm", _decode)


# ----------------------------- load_song_windows -----------------------------

def test_short_song_single_padded_window(monkeypatch):
    total = SAMPLE_RATE * 10  # 10 s < 30 s window
    audio = np.linspace(-1.0, 1.0, total, dtype=np.float32)
    _patch_decode(monkeypatch, audio)

    wins = load_song_windows("x.wav")

    assert len(wins) == 1
    assert wins[0].shape[0] == SAMPLE_RATE * WINDOW_SECONDS
    # original samples kept at the front, zero-padded after
    assert np.allclose(wins[0][:total], audio)
    assert np.all(wins[0][total:] == 0.0)


def test_long_song_uses_max_windows(monkeypatch):
    total = SAMPLE_RATE * 130  # 130 // 30 = 4 windows fit
    _patch_decode(monkeypatch, np.zeros(total, dtype=np.float32))

    wins = load_song_windows("x.wav")

    assert len(wins) == MAX_WINDOWS == 4
    assert all(w.shape[0] == SAMPLE_RATE * WINDOW_SECONDS for w in wins)


def test_max_windows_param_caps_count(monkeypatch):
    total = SAMPLE_RATE * 130
    _patch_decode(monkeypatch, np.zeros(total, dtype=np.float32))

    assert len(load_song_windows("x.wav", max_windows=2)) == 2


def test_medium_song_single_window(monkeypatch):
    total = SAMPLE_RATE * 45  # 45 // 30 = 1 -> single 25%-offset window
    _patch_decode(monkeypatch, np.zeros(total, dtype=np.float32))

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


# --------------------------------- parse_gender ------------------------------
# The captioner emits a trailing "GENDER: male|female|instrumental" tag; parse_gender
# splits it off into a canonical label (the vocal-gender source that replaced the
# F0-on-Demucs estimate) and returns the description with the tag stripped.

def test_parse_gender_canonical_and_strips_tag():
    desc, g = parse_gender("Soaring synthpop with bright vocals.\nGENDER: female")
    assert g == "female"
    assert "GENDER" not in desc.upper() and desc.endswith("vocals.")


def test_parse_gender_male_and_dash_and_case_insensitive():
    assert parse_gender("Gritty blues.\n\nGENDER: male")[1] == "male"
    assert parse_gender("Lo-fi beat.\ngender - m")[1] == "male"
    assert parse_gender("Choir.\nGENDER:   Female  ")[1] == "female"


def test_parse_gender_instrumental_and_missing_are_none():
    assert parse_gender("Ambient drone, no singing.\nGENDER: instrumental")[1] is None
    assert parse_gender("No tag at all.")[1] is None
    assert parse_gender("")[1] is None


def test_parse_gender_does_not_false_match_prose():
    # "gender-bending" must not be read as a gender tag (no colon/dash + label).
    desc, g = parse_gender("A gender-bending art-pop number with no marker")
    assert g is None and desc == "A gender-bending art-pop number with no marker"


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


def test_both_backends_expose_gender_and_stems_contract():
    # auto_tag / the Modal worker call the *_with_gender_and_stems APIs.
    for cls in (AudioLLMCaptioner, VLLMAudioCaptioner):
        inst = cls()
        assert hasattr(inst, "caption_with_gender_and_stems")
    assert hasattr(VLLMAudioCaptioner(), "caption_many_with_gender_and_stems")


# --------------------------------- parse_stems -------------------------------
# v5: after GENDER the captioner emits DRUMS:/BASS:/VOCALS:/OTHER: lines, parsed
# into the per-stem caption dict (the /addstem target-stem tag source).

_FULL = (
    "A bright synthpop number with shimmering pads and a driving pulse.\n"
    "GENDER: female\n"
    "DRUMS: punchy four-on-the-floor kit with crisp closed hats.\n"
    "BASS: a round analog synth bass walking in syncopated octaves.\n"
    "VOCALS: an airy female lead, breathy and double-tracked.\n"
    "OTHER: glassy electric piano and a soaring string pad."
)


def test_parse_stems_extracts_all_four_in_order():
    rest, stems = parse_stems(_FULL)
    assert set(stems) == set(STEM_CAPTION_KEYS)
    assert "analog synth bass" in stems["bass"]
    assert "four-on-the-floor" in stems["drums"]
    # The stem lines are stripped; the description + GENDER line survive for
    # parse_gender (run next in the real pipeline).
    assert "DRUMS:" not in rest and "BASS:" not in rest
    desc, gender = parse_gender(rest)
    assert gender == "female"
    assert "shimmering pads" in desc and "GENDER" not in desc


def test_parse_stems_drops_none_sentinel():
    text = ("Ambient instrumental drone.\nGENDER: instrumental\n"
            "DRUMS: none\nBASS: a deep sustained sine swell.\n"
            "VOCALS: none\nOTHER: bowed metallic textures.")
    _rest, stems = parse_stems(text)
    assert "drums" not in stems and "vocals" not in stems  # sentinels dropped
    assert "bass" in stems and "other" in stems


def test_parse_stems_tolerates_missing_and_markdown_and_reorder():
    text = ("Lo-fi beat.\nGENDER: male\n"
            "- BASS: woody upright bass.\n"
            "* DRUMS - brushed jazz kit.")  # reordered, bullets, dash, 2 of 4
    _rest, stems = parse_stems(text)
    assert set(stems) == {"bass", "drums"}
    assert "upright bass" in stems["bass"]


def test_parse_stems_empty_and_absent():
    assert parse_stems("")[1] == {}
    assert parse_stems("No stem lines here at all.")[1] == {}


def test_stem_caption_keys_match_stem_types():
    # The tags.json stems dict is keyed by STEM_CAPTION_KEYS; the dataset reads it
    # in STEM_TYPES order — a drift would silently misroute per-stem captions.
    from model.stem_encoder import STEM_TYPES
    assert tuple(STEM_CAPTION_KEYS) == tuple(STEM_TYPES)


def test_captioner_marker_literal_matches_modal_auto_tag():
    # modal_auto_tag.py duplicates the marker literal (slim image can't import this
    # module); they MUST agree or --redo skips/redoes the wrong entries.
    import re
    src = (importlib_path := __import__("pathlib").Path(
        "diskrot/modal_auto_tag.py")).read_text()
    m = re.search(r'CAPTIONER_MARKER\s*=\s*"([^"]+)"', src)
    assert m and m.group(1) == CAPTIONER_MARKER
