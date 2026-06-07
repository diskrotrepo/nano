"""Tests for diskrot/dataset.py — TokenDataset.

Uses the ``synth_tokens_dir`` factory fixture from conftest to materialize
small synthetic .pt corpora in tmp_path. All tests pack the corpus into the
sharded mmap layout first (the only supported path).
"""
from __future__ import annotations

import json

import pytest
import torch

from diskrot.dataset import TokenDataset
from diskrot.pack_cache import pack
from model.codec import DACodec


def _g2p_available() -> bool:
    try:
        from model.lyric_encoder import text_to_word_phoneme_groups
        text_to_word_phoneme_groups(["test"])
        return True
    except Exception:
        return False


def _packed_dir(tokens_dir):
    """Pack loose .pt files into the sharded layout that TokenDataset requires."""
    pack(tokens_dir, verbose=False)
    return tokens_dir


def test_loads_files_and_reports_length(synth_tokens_dir):
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=5, T=1000))
    ds = TokenDataset(tokens_dir, segment_frames=500)
    assert len(ds) > 0


def test_train_val_split_deterministic(synth_tokens_dir):
    """Same seed → same partition of files into train vs val, across separate
    TokenDataset constructions. This is what makes val curves comparable
    across runs."""
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=10, T=1000))
    train_a = TokenDataset(tokens_dir, segment_frames=500, split="train", seed=42)
    train_b = TokenDataset(tokens_dir, segment_frames=500, split="train", seed=42)
    assert train_a.names == train_b.names


def test_train_val_split_disjoint(synth_tokens_dir):
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=10, T=1000))
    train = TokenDataset(tokens_dir, segment_frames=500, split="train", seed=42)
    val = TokenDataset(tokens_dir, segment_frames=500, split="val", seed=42)
    assert set(train.names).isdisjoint(set(val.names))
    # Together they cover the full corpus.
    assert set(train.names) | set(val.names) == {f"song_{i:03d}" for i in range(10)}


def test_val_ratio_respected(synth_tokens_dir):
    """val_ratio=0.2 over 10 files → 2 val files."""
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=10, T=1000))
    val = TokenDataset(tokens_dir, segment_frames=500, split="val", seed=42, val_ratio=0.2)
    assert len(val) == 2


def test_skips_too_short_files(synth_tokens_dir):
    """Files with T < segment_frames + 2 are filtered out. Make 4 short
    (T=100) and 4 long (T=1000), train split should contain only the long ones."""
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=8, T=[100] * 4 + [1000] * 4))
    ds = TokenDataset(tokens_dir, segment_frames=500, split="train", seed=42, val_ratio=0.125)
    # All retained entries must be long enough. Verify via __getitem__ shape.
    for i in range(len(ds)):
        tokens, _, _ = ds[i]
        assert tokens.shape[1] == 500


def test_getitem_shape_and_dtype(synth_tokens_dir):
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=4, T=1000))
    ds = TokenDataset(tokens_dir, segment_frames=300)
    tokens, tags, lyric_ids = ds[0]
    assert tokens.shape == (9, 300)
    # int16 stays int16 on host (saves ~24 GB shared RAM at production scale).
    # Training loop casts to int64 via .long() after .to(device) — see
    # tests/test_pipeline_audit.py::test_int16_storage_round_trips_through_embedding
    # for the correctness contract.
    assert tokens.dtype == torch.int16
    assert tags == ""
    # No lyrics/structure path → BOS + the dense <unknown_gender> + <no_section>
    # prefixes (never an empty/fully-padded sequence, which would NaN the lyric
    # cross-attention).
    from model.lyric_encoder import BOS_PHONEME_ID, NO_SECTION_ID, UNKNOWN_GENDER_ID
    assert lyric_ids.tolist() == [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, NO_SECTION_ID]


def test_getitem_uses_random_crop_within_bounds(synth_tokens_dir):
    """The slice must stay inside [0, T_full]. Sample many crops and confirm
    each one has the requested shape (which would fail if start went too far)."""
    import random
    random.seed(0)
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=2, T=600))
    ds = TokenDataset(tokens_dir, segment_frames=500)
    for _ in range(50):
        tokens, _, _ = ds[0]
        assert tokens.shape == (9, 500)


def test_tags_loaded(synth_tokens_dir, tmp_path):
    """tags.json with {name: {description: ...}} maps each song name to its
    description string at __getitem__ time."""
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=3, T=1000))
    tags_path = tmp_path / "tags.json"
    tags_path.write_text(json.dumps({
        "song_000": {"description": "ambient drone"},
        "song_001": {"description": "fast techno"},
        # song_002 intentionally missing — should resolve to empty string.
    }))
    ds = TokenDataset(tokens_dir, segment_frames=500, tags_path=tags_path, val_ratio=0.34)
    # The split is determined by the seeded shuffle; walk every sample and
    # confirm the tag matches whatever song landed at that index.
    for i in range(len(ds)):
        _, tag, _ = ds[i]
        if ds.names[i] == "song_000":
            assert tag == "ambient drone"
        elif ds.names[i] == "song_001":
            assert tag == "fast techno"
        else:
            assert tag == ""


def test_lyrics_window_filtering(synth_tokens_dir, tmp_path):
    """_get_segment_lyrics keeps words whose [start,end] overlaps the crop
    window. We can't control the random crop position from outside, so call
    the method directly instead of going through __getitem__."""
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=2, T=1000))
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps({
        "song_000": {
            "words": [
                {"word": "alpha", "start": 0.0, "end": 1.0},
                {"word": "beta",  "start": 2.0, "end": 3.0},
                {"word": "gamma", "start": 5.0, "end": 6.0},
            ],
        },
    }))
    ds = TokenDataset(tokens_dir, segment_frames=500,
                      lyrics_path=lyrics_path, val_ratio=0.5)
    # Window [1.5s, 4.0s] → only "beta" overlaps.
    out = ds._get_segment_lyrics("song_000", start_sec=1.5, end_sec=4.0)
    assert out == "beta"
    # Window covering everything.
    out = ds._get_segment_lyrics("song_000", start_sec=0.0, end_sec=10.0)
    assert out == "alpha beta gamma"
    # Window with no overlap.
    out = ds._get_segment_lyrics("song_000", start_sec=10.0, end_sec=11.0)
    assert out == ""
    # Missing entry.
    out = ds._get_segment_lyrics("nonexistent", start_sec=0.0, end_sec=10.0)
    assert out == ""


def test_malformed_word_entries_filtered_at_load(synth_tokens_dir, tmp_path):
    """Malformed word entries (missing/non-numeric start/end, missing word) are
    dropped at load so the crop builders never hit a KeyError/TypeError. A song
    left with no valid words is treated as instrumental (no entry)."""
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=2, T=1000))
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps({
        # song_000: mix of good and malformed words — only the good ones survive.
        "song_000": {"gender": "female", "words": [
            {"word": "good", "start": 0.0, "end": 1.0},
            {"word": "no_times"},                                  # missing start/end
            {"word": "bad_start", "start": "x", "end": 3.0},       # non-numeric start
            {"start": 4.0, "end": 5.0},                            # missing word
            {"word": "alsogood", "start": 6.0, "end": 7.0},
        ]},
        # song_001: every word malformed → dropped entirely (instrumental).
        "song_001": {"words": [{"word": "x"}, {"start": 1.0}]},
    }))
    ds = TokenDataset(tokens_dir, segment_frames=500,
                      lyrics_path=lyrics_path, val_ratio=0.5)
    # song_000 kept only the two well-formed words; gender field preserved.
    assert "song_000" in ds._lyrics
    assert [w["word"] for w in ds._lyrics["song_000"]["words"]] == ["good", "alsogood"]
    assert ds._lyrics["song_000"]["gender"] == "female"
    # song_001 has no usable words → not loaded (treated as instrumental).
    assert "song_001" not in ds._lyrics
    # The hot path must not raise on the cleaned entry.
    assert ds._get_segment_lyrics("song_000", 0.0, 10.0) == "good alsogood"


@pytest.mark.skipif(not _g2p_available(), reason="g2p_en / nltk data not installed")
def test_word_phones_cache_eviction_is_lossless(synth_tokens_dir, tmp_path):
    """The bounded LRU on _word_phones must change only WHEN g2p runs, never the
    ids: a cache miss re-runs g2p deterministically and yields the same groups."""
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=6, T=1000))
    lyrics_path = tmp_path / "lyrics.json"
    # Give every song its own lyric so any two names in the loaded split have one.
    lyrics_path.write_text(json.dumps({
        f"song_{i:03d}": {"words": [{"word": w, "start": 0.0, "end": 1.0}]}
        for i, w in enumerate(["hello", "world", "singing", "blues", "tonight", "yeah"])
    }))
    ds = TokenDataset(tokens_dir, segment_frames=500,
                      lyrics_path=lyrics_path, val_ratio=0.34, max_lyric_len=256)
    a, b = ds.names[0], ds.names[1]  # two songs guaranteed in this split
    ds._word_phones_cap = 1  # force eviction between the two songs

    first = ds._get_segment_lyric_ids(a, 0.0, 10.0)
    assert a in ds._word_phones
    # Touch the other song — with cap=1 this evicts a's cached groups.
    ds._get_segment_lyric_ids(b, 0.0, 10.0)
    assert a not in ds._word_phones  # evicted
    # Recomputed from scratch → must be byte-identical to the cached result.
    again = ds._get_segment_lyric_ids(a, 0.0, 10.0)
    assert again == first
    assert len(ds._word_phones) <= 1  # cap honored


@pytest.mark.skipif(not _g2p_available(), reason="g2p_en / nltk data not installed")
def test_segment_lyric_ids_window_and_bos(synth_tokens_dir, tmp_path):
    """_get_segment_lyric_ids returns BOS + <gender> + section prefix + phonemes for
    overlapping words, and BOS + dense prefixes when nothing overlaps or the song is
    instrumental (no structure entry / no gender field here → <unknown_gender> +
    <no_section>)."""
    from model.lyric_encoder import BOS_PHONEME_ID, NO_SECTION_ID, UNKNOWN_GENDER_ID

    prefix = [BOS_PHONEME_ID, UNKNOWN_GENDER_ID, NO_SECTION_ID]
    tokens_dir = _packed_dir(synth_tokens_dir(n_files=2, T=1000))
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps({
        "song_000": {"words": [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": 2.0, "end": 3.0},
        ]},
    }))
    ds = TokenDataset(tokens_dir, segment_frames=500,
                      lyrics_path=lyrics_path, val_ratio=0.5, max_lyric_len=256)

    full = ds._get_segment_lyric_ids("song_000", 0.0, 10.0)
    assert full[:3] == prefix  # BOS + <unknown_gender> + <no_section> prefix
    assert len(full) > 3  # phonemes present
    # No overlap → dense prefix only (never a lone BOS).
    assert ds._get_segment_lyric_ids("song_000", 10.0, 11.0) == prefix
    # Instrumental / missing → dense prefix only.
    assert ds._get_segment_lyric_ids("nonexistent", 0.0, 10.0) == prefix


@pytest.mark.skipif(not _g2p_available(), reason="g2p_en / nltk data not installed")
def test_segment_lyric_ids_gender_prefix(synth_tokens_dir, tmp_path):
    """A song's F0-labeled ``gender`` field becomes the dense gender prefix, and
    the train-time stream matches the inference parser's bracketed equivalent."""
    from model.lyric_encoder import (
        BOS_PHONEME_ID, GENDER_TOKEN_TO_ID, NO_SECTION_ID,
        text_with_markers_to_phoneme_ids,
    )

    tokens_dir = _packed_dir(synth_tokens_dir(n_files=2, T=1000))
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps({
        "song_000": {"gender": "female", "words": [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": 2.0, "end": 3.0},
        ]},
    }))
    ds = TokenDataset(tokens_dir, segment_frames=500,
                      lyrics_path=lyrics_path, val_ratio=0.5, max_lyric_len=256)

    train_ids = ds._get_segment_lyric_ids("song_000", 0.0, 10.0)
    assert train_ids[:3] == [BOS_PHONEME_ID, GENDER_TOKEN_TO_ID["female"], NO_SECTION_ID]
    # train-time stream == inference parser for the equivalent bracketed string
    infer_ids = text_with_markers_to_phoneme_ids("[female] hello world")
    assert train_ids == infer_ids


def test_collate_lyrics_pads_and_masks(synth_tokens_dir):
    """collate_lyrics pads ragged phoneme sequences and builds the bool mask."""
    from diskrot.dataset import collate_lyrics
    from model.lyric_encoder import BOS_PHONEME_ID, PAD_PHONEME_ID

    batch = [
        (torch.zeros(9, 300, dtype=torch.int16), "tagA",
         torch.tensor([BOS_PHONEME_ID, 10, 11], dtype=torch.long)),
        (torch.zeros(9, 300, dtype=torch.int16), "tagB",
         torch.tensor([BOS_PHONEME_ID], dtype=torch.long)),
    ]
    tokens, tags, ids, mask = collate_lyrics(batch)
    assert tokens.shape == (2, 9, 300)
    assert tags == ["tagA", "tagB"]
    assert ids.shape == (2, 3)
    assert ids[1].tolist() == [BOS_PHONEME_ID, PAD_PHONEME_ID, PAD_PHONEME_ID]
    assert mask[0].tolist() == [True, True, True]
    assert mask[1].tolist() == [True, False, False]


def test_frame_rate_constant_unchanged():
    """TokenDataset.__getitem__ divides by DACodec.FRAME_RATE_HZ to convert
    frame offsets to seconds for lyric lookup. Pin the constant so changes are
    intentional."""
    assert DACodec.FRAME_RATE_HZ == 86
