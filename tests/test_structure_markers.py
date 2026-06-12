"""Tests for song-structure marker conditioning.

Covers the load-bearing contract: the dataset's train-time injection
(``TokenDataset._get_segment_lyric_ids``) and the inference parser
(``text_with_markers_to_phoneme_ids``) MUST build byte-identical phoneme+marker
streams. The equivalence test is the guard against the two drifting apart.
"""
from __future__ import annotations

import json

import pytest

from diskrot.dataset import TokenDataset
from diskrot.pack_cache import pack
from model.lyric_encoder import (
    BOS_PHONEME_ID,
    GENDER_TOKEN_TO_ID,
    KEY_TOKEN_TO_ID,
    NO_SECTION_ID,
    STRUCTURE_TOKEN_TO_ID,
    UNKNOWN_GENDER_ID,
    UNKNOWN_KEY_ID,
    UNKNOWN_TEMPO_ID,
    UNKNOWN_VOCALS_ID,
    VOCAL_TOKEN_TO_ID,
    bpm_to_id,
    text_with_markers_to_phoneme_ids,
)

VOCALS_ID = VOCAL_TOKEN_TO_ID["vocals"]
INSTRUMENTAL_ID = VOCAL_TOKEN_TO_ID["instrumental"]


def _g2p_available() -> bool:
    try:
        from model.lyric_encoder import text_to_word_phoneme_groups
        text_to_word_phoneme_groups(["test"])
        return True
    except Exception:
        return False


g2p_required = pytest.mark.skipif(
    not _g2p_available(), reason="g2p_en / nltk data not installed"
)


def _make_ds(synth_tokens_dir, tmp_path, words, segments, bpm=None, gender=None, key=None):
    """Build a TokenDataset over a tiny synth corpus with the given lyrics +
    structure (and optional bpm/gender/key) for song_000. ``words=[]`` exercises
    the transcribed-but-wordless (instrumental) path; ``words=None`` writes a
    null entry — the on-disk instrumental convention."""
    tokens_dir = synth_tokens_dir(n_files=2, T=1000)
    pack(tokens_dir, verbose=False)
    lyrics_path = tmp_path / "lyrics.json"
    lyric_entry = {"words": words} if words is not None else None
    if gender is not None and lyric_entry is not None:
        lyric_entry["gender"] = gender
    lyrics_path.write_text(json.dumps({"song_000": lyric_entry}))
    structure_path = tmp_path / "structure.json"
    struct_entry = {"segments": segments}
    if bpm is not None:
        struct_entry["bpm"] = bpm
    structure_path.write_text(json.dumps({"song_000": struct_entry}))
    keys_path = None
    if key is not None:
        keys_path = tmp_path / "keys.json"
        keys_path.write_text(json.dumps({"song_000": {"key": key}}))
    return TokenDataset(
        tokens_dir, segment_frames=500, lyrics_path=lyrics_path,
        structure_path=structure_path, keys_path=keys_path,
        val_ratio=0.5, max_lyric_len=256,
    )


# --- structure loading -------------------------------------------------------
def test_structure_load_drops_sentinels_and_sorts(synth_tokens_dir, tmp_path):
    segments = [
        {"start": 10.0, "end": 20.0, "label": "chorus"},
        {"start": 0.0, "end": 30.0, "label": "start"},   # sentinel -> dropped
        {"start": 0.0, "end": 10.0, "label": "verse"},
    ]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hi", "start": 0.0, "end": 1.0}], segments)
    segs = ds._structure["song_000"]
    assert [s["label"] for s in segs] == ["verse", "chorus"]  # sentinel gone, sorted


def test_structure_load_carries_bpm(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hi", "start": 0.0, "end": 1.0}], segments, bpm=128.0)
    assert ds._bpm["song_000"] == 128.0
    # A song with no bpm field simply isn't in the map (-> <unknown_tempo>).
    ds2 = _make_ds(synth_tokens_dir, tmp_path,
                   [{"word": "hi", "start": 0.0, "end": 1.0}], segments)
    assert "song_000" not in ds2._bpm


# --- injection ---------------------------------------------------------------
@g2p_required
def test_prefix_marker_for_active_section(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hello", "start": 0.5, "end": 1.5}], segments)
    ids = ds._get_segment_lyric_ids("song_000", 0.0, 5.0)
    assert ids[:6] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID,
                       UNKNOWN_KEY_ID, VOCALS_ID, STRUCTURE_TOKEN_TO_ID["verse"]]
    # A crop starting inside the chorus gets the chorus prefix.
    ids2 = ds._get_segment_lyric_ids("song_000", 12.0, 18.0)
    assert ids2[:6] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID,
                        UNKNOWN_KEY_ID, VOCALS_ID, STRUCTURE_TOKEN_TO_ID["chorus"]]


@g2p_required
def test_prefix_no_section_in_gap(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 5.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hi", "start": 0.0, "end": 1.0}], segments)
    ids = ds._get_segment_lyric_ids("song_000", 6.0, 9.0)  # gap [5,10)
    assert ids[:6] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID,
                       UNKNOWN_KEY_ID, VOCALS_ID, NO_SECTION_ID]


@g2p_required
def test_inline_boundary_injected(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    words = [{"word": "hello", "start": 0.5, "end": 1.5},
             {"word": "world", "start": 2.0, "end": 3.0},
             {"word": "yeah", "start": 11.0, "end": 12.0}]
    ds = _make_ds(synth_tokens_dir, tmp_path, words, segments)
    ids = ds._get_segment_lyric_ids("song_000", 0.0, 30.0)
    assert ids[1] == UNKNOWN_GENDER_ID                    # dense gender prefix
    assert ids[2] == UNKNOWN_TEMPO_ID                     # dense tempo prefix
    assert ids[3] == UNKNOWN_KEY_ID                       # dense key prefix
    assert ids[4] == VOCALS_ID                            # dense vocal prefix (has words)
    assert ids[5] == STRUCTURE_TOKEN_TO_ID["verse"]       # section prefix
    chorus = STRUCTURE_TOKEN_TO_ID["chorus"]
    assert chorus in ids                                  # inline boundary present
    # chorus marker comes after the verse words, before the chorus word's phonemes
    assert ids.index(chorus) > 6


# --- train == inference equivalence (the critical guard) ---------------------
@g2p_required
def test_train_inference_stream_equivalence(synth_tokens_dir, tmp_path):
    """A crop's train-time stream must equal the inference parser's output for the
    equivalent bracketed string."""
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    words = [{"word": "hello", "start": 0.5, "end": 1.5},
             {"word": "world", "start": 2.0, "end": 3.0},
             {"word": "yeah", "start": 11.0, "end": 12.0}]
    ds = _make_ds(synth_tokens_dir, tmp_path, words, segments)

    train_ids = ds._get_segment_lyric_ids("song_000", 0.0, 30.0)
    infer_ids = text_with_markers_to_phoneme_ids("[verse] hello world [chorus] yeah")
    assert train_ids == infer_ids


@g2p_required
def test_tempo_prefix_bucket(synth_tokens_dir, tmp_path):
    """A song's bpm lands in the tempo header slot, bucketed via bpm_to_id."""
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hello", "start": 0.5, "end": 1.5}], segments, bpm=120.0)
    ids = ds._get_segment_lyric_ids("song_000", 0.0, 5.0)
    assert ids[:6] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, bpm_to_id(120.0),
                       UNKNOWN_KEY_ID, VOCALS_ID, STRUCTURE_TOKEN_TO_ID["verse"]]
    assert bpm_to_id(120.0) != UNKNOWN_TEMPO_ID  # a real bucket, not the fallback


@g2p_required
def test_train_inference_equivalence_with_gender_and_tempo(synth_tokens_dir, tmp_path):
    """Full dense header (gender + tempo + section) must match the bracket parser."""
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    words = [{"word": "hello", "start": 0.5, "end": 1.5},
             {"word": "world", "start": 2.0, "end": 3.0},
             {"word": "yeah", "start": 11.0, "end": 12.0}]
    ds = _make_ds(synth_tokens_dir, tmp_path, words, segments,
                  bpm=128.0, gender="female")

    train_ids = ds._get_segment_lyric_ids("song_000", 0.0, 30.0)
    # Prefix markers in any order; tempo accepts the numeric bpm form.
    infer_ids = text_with_markers_to_phoneme_ids(
        "[female] [128bpm] [verse] hello world [chorus] yeah")
    assert train_ids == infer_ids
    assert train_ids[1] == GENDER_TOKEN_TO_ID["female"]
    assert train_ids[2] == bpm_to_id(128.0)


@g2p_required
def test_key_prefix_from_keys_json(synth_tokens_dir, tmp_path):
    """A song's key_detect estimate lands in the key header slot, and the
    train-time stream matches the bracket parser's equivalent."""
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hello", "start": 0.5, "end": 1.5}], segments,
                  key="a_minor")
    train_ids = ds._get_segment_lyric_ids("song_000", 0.0, 5.0)
    assert train_ids[3] == KEY_TOKEN_TO_ID["a_minor"]
    # User forms [a minor] / [key:Am] produce the same stream.
    assert train_ids == text_with_markers_to_phoneme_ids("[a minor] [verse] hello")
    assert train_ids == text_with_markers_to_phoneme_ids("[key:Am] [verse] hello")
    # No keys.json -> <unknown_key> slot.
    ds2 = _make_ds(synth_tokens_dir, tmp_path,
                   [{"word": "hello", "start": 0.5, "end": 1.5}], segments)
    assert ds2._get_segment_lyric_ids("song_000", 0.0, 5.0)[3] == UNKNOWN_KEY_ID


@g2p_required
def test_instrumental_marker_and_equivalence(synth_tokens_dir, tmp_path):
    """A transcribed-but-wordless song carries <instrumental>; a never-transcribed
    one carries <unknown_vocals>; both match their bracketed equivalents."""
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"}]
    ds = _make_ds(synth_tokens_dir, tmp_path, [], segments)  # words=[] -> instrumental
    assert "song_000" in ds._instrumental and "song_000" not in ds._lyrics
    train_ids = ds._get_segment_lyric_ids("song_000", 0.0, 5.0)
    assert train_ids == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID,
                         UNKNOWN_KEY_ID, INSTRUMENTAL_ID,
                         STRUCTURE_TOKEN_TO_ID["verse"]]
    assert train_ids == text_with_markers_to_phoneme_ids("[instrumental] [verse]")
    # song_001 was never transcribed (no lyrics.json entry) -> <unknown_vocals>.
    other = ds._get_segment_lyric_ids("song_001", 0.0, 5.0)
    assert other[4] == UNKNOWN_VOCALS_ID


@g2p_required
def test_null_lyric_entry_is_instrumental(synth_tokens_dir, tmp_path):
    """A NULL lyric entry is the on-disk instrumental convention (written by
    transcribe and the hallucination filter) and must feed the instrumental set
    exactly like an empty-words dict. Regression: 2026-06-12 — nulls were
    silently dropped, so all 173k instrumentals trained as <unknown_vocals>."""
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"}]
    ds = _make_ds(synth_tokens_dir, tmp_path, None, segments)  # null entry
    assert "song_000" in ds._instrumental and "song_000" not in ds._lyrics
    assert ds._get_segment_lyric_ids("song_000", 0.0, 5.0)[4] == INSTRUMENTAL_ID


@g2p_required
def test_vocals_marker_on_crop_without_words(synth_tokens_dir, tmp_path):
    """A vocal song's crop with no words in the window still carries <vocals>
    (per-song attribute), matching an explicit [vocals] bracket."""
    segments = [{"start": 0.0, "end": 30.0, "label": "verse"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hello", "start": 0.5, "end": 1.5}], segments)
    train_ids = ds._get_segment_lyric_ids("song_000", 20.0, 25.0)  # no words here
    assert train_ids[4] == VOCALS_ID
    assert train_ids == text_with_markers_to_phoneme_ids("[vocals] [verse]")
