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
    NO_SECTION_ID,
    STRUCTURE_TOKEN_TO_ID,
    UNKNOWN_GENDER_ID,
    UNKNOWN_TEMPO_ID,
    bpm_to_id,
    text_with_markers_to_phoneme_ids,
)


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


def _make_ds(synth_tokens_dir, tmp_path, words, segments, bpm=None, gender=None):
    """Build a TokenDataset over a tiny synth corpus with the given lyrics +
    structure (and optional bpm/gender) for song_000."""
    tokens_dir = synth_tokens_dir(n_files=2, T=1000)
    pack(tokens_dir, verbose=False)
    lyrics_path = tmp_path / "lyrics.json"
    lyric_entry = {"words": words}
    if gender is not None:
        lyric_entry["gender"] = gender
    lyrics_path.write_text(json.dumps({"song_000": lyric_entry}))
    structure_path = tmp_path / "structure.json"
    struct_entry = {"segments": segments}
    if bpm is not None:
        struct_entry["bpm"] = bpm
    structure_path.write_text(json.dumps({"song_000": struct_entry}))
    return TokenDataset(
        tokens_dir, segment_frames=500, lyrics_path=lyrics_path,
        structure_path=structure_path, val_ratio=0.5, max_lyric_len=256,
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
    assert ids[:4] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID,
                       STRUCTURE_TOKEN_TO_ID["verse"]]
    # A crop starting inside the chorus gets the chorus prefix.
    ids2 = ds._get_segment_lyric_ids("song_000", 12.0, 18.0)
    assert ids2[:4] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID,
                        STRUCTURE_TOKEN_TO_ID["chorus"]]


@g2p_required
def test_prefix_no_section_in_gap(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 5.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hi", "start": 0.0, "end": 1.0}], segments)
    ids = ds._get_segment_lyric_ids("song_000", 6.0, 9.0)  # gap [5,10)
    assert ids[:4] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID, NO_SECTION_ID]


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
    assert ids[3] == STRUCTURE_TOKEN_TO_ID["verse"]       # section prefix
    chorus = STRUCTURE_TOKEN_TO_ID["chorus"]
    assert chorus in ids                                  # inline boundary present
    # chorus marker comes after the verse words, before the chorus word's phonemes
    assert ids.index(chorus) > 4


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
    assert ids[:4] == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, bpm_to_id(120.0),
                       STRUCTURE_TOKEN_TO_ID["verse"]]
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
