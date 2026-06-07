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
    NO_SECTION_ID,
    STRUCTURE_TOKEN_TO_ID,
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


def _make_ds(synth_tokens_dir, tmp_path, words, segments):
    """Build a TokenDataset over a tiny synth corpus with the given lyrics +
    structure for song_000."""
    tokens_dir = synth_tokens_dir(n_files=2, T=1000)
    pack(tokens_dir, verbose=False)
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps({"song_000": {"words": words}}))
    structure_path = tmp_path / "structure.json"
    structure_path.write_text(json.dumps({"song_000": {"segments": segments}}))
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


# --- injection ---------------------------------------------------------------
@g2p_required
def test_prefix_marker_for_active_section(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hello", "start": 0.5, "end": 1.5}], segments)
    ids = ds._get_segment_lyric_ids("song_000", 0.0, 5.0)
    assert ids[:2] == [BOS_PHONEME_ID, STRUCTURE_TOKEN_TO_ID["verse"]]
    # A crop starting inside the chorus gets the chorus prefix.
    ids2 = ds._get_segment_lyric_ids("song_000", 12.0, 18.0)
    assert ids2[:2] == [BOS_PHONEME_ID, STRUCTURE_TOKEN_TO_ID["chorus"]]


@g2p_required
def test_prefix_no_section_in_gap(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 5.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    ds = _make_ds(synth_tokens_dir, tmp_path,
                  [{"word": "hi", "start": 0.0, "end": 1.0}], segments)
    ids = ds._get_segment_lyric_ids("song_000", 6.0, 9.0)  # gap [5,10)
    assert ids[:2] == [BOS_PHONEME_ID, NO_SECTION_ID]


@g2p_required
def test_inline_boundary_injected(synth_tokens_dir, tmp_path):
    segments = [{"start": 0.0, "end": 10.0, "label": "verse"},
                {"start": 10.0, "end": 20.0, "label": "chorus"}]
    words = [{"word": "hello", "start": 0.5, "end": 1.5},
             {"word": "world", "start": 2.0, "end": 3.0},
             {"word": "yeah", "start": 11.0, "end": 12.0}]
    ds = _make_ds(synth_tokens_dir, tmp_path, words, segments)
    ids = ds._get_segment_lyric_ids("song_000", 0.0, 30.0)
    assert ids[1] == STRUCTURE_TOKEN_TO_ID["verse"]       # prefix
    chorus = STRUCTURE_TOKEN_TO_ID["chorus"]
    assert chorus in ids                                  # inline boundary present
    # chorus marker comes after the verse words, before the chorus word's phonemes
    assert ids.index(chorus) > 2


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
