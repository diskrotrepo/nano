"""Tests for diskrot/phonemize.py — the pre-phonemized lyric cache.

The load-bearing contract: the stored per-word groups must be byte-identical to
what live g2p produces (the dataset serves whichever is available, so any drift
would silently change the training stream), and a stale entry (word count
changed by a re-transcribe) must fall back to live g2p, never serve wrong ids.
"""
from __future__ import annotations

import json

import pytest

from diskrot.dataset import TokenDataset, _load_phonemes
from diskrot.pack_cache import pack
from diskrot.phonemize import load_phoneme_shards, phonemize_corpus


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

WORDS = [
    {"word": "hello", "start": 0.5, "end": 1.5},
    {"word": "world", "start": 2.0, "end": 3.0},
    {"word": "oooh", "start": 4.0, "end": 5.0},  # OOV — exercises the seq2seq path
]


@g2p_required
def test_phonemize_corpus_matches_live_g2p_and_resumes(tmp_path):
    from model.lyric_encoder import text_to_word_phoneme_groups

    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps({
        "song_000": {"words": WORDS},
        "song_001": {"words": []},  # instrumental — no phoneme entry
    }))
    out = phonemize_corpus(lyrics_path, tmp_path / "phonemes", n_workers=1, verbose=False)

    stored = load_phoneme_shards(out)
    assert set(stored) == {"song_000"}
    live = text_to_word_phoneme_groups([w["word"] for w in WORDS])
    assert stored["song_000"] == live  # byte-identical to the live contract

    # Resume-by-skip: an up-to-date entry isn't recomputed (poison to prove it).
    bucket_file = next(out.glob("phonemes_*.json"))
    poisoned = {"song_000": [[1]] * len(WORDS)}  # same group count -> "fresh"
    bucket_file.write_text(json.dumps(poisoned))
    phonemize_corpus(lyrics_path, out, n_workers=1, verbose=False)
    assert load_phoneme_shards(out)["song_000"] == poisoned["song_000"]


@g2p_required
def test_dataset_serves_precomputed_groups_identically(synth_tokens_dir, tmp_path):
    """The same crop must produce the same lyric stream with and without the
    pre-phonemized store, and a stale entry must fall back to live g2p."""
    tokens_dir = synth_tokens_dir(n_files=2, T=1000)
    pack(tokens_dir, verbose=False)
    lyrics_path = tmp_path / "lyrics.json"
    lyrics_path.write_text(json.dumps({"song_000": {"words": WORDS}}))
    phonemes_dir = phonemize_corpus(
        lyrics_path, tmp_path / "phonemes", n_workers=1, verbose=False)

    def make_ds(phonemes_path):
        return TokenDataset(
            tokens_dir, segment_frames=500, lyrics_path=lyrics_path,
            phonemes_path=phonemes_path, val_ratio=0.5, max_lyric_len=256,
        )

    live_ds = make_ds(None)
    pre_ds = make_ds(phonemes_dir)
    assert pre_ds._phonemes  # store actually loaded
    expect = live_ds._get_segment_lyric_ids("song_000", 0.0, 10.0)
    got = pre_ds._get_segment_lyric_ids("song_000", 0.0, 10.0)
    assert got == expect
    assert all(isinstance(i, int) for i in got)  # int16 store -> python ints

    # Stale store (word count mismatch) -> live g2p fallback, same stream.
    stale = make_ds(phonemes_dir)
    flat, offs = stale._phonemes["song_000"]
    stale._phonemes["song_000"] = (flat, offs[:-1])  # one group short
    assert stale._get_segment_lyric_ids("song_000", 0.0, 10.0) == expect


def test_load_phonemes_compact_form(tmp_path):
    """_load_phonemes converts groups to (flat int16, int32 offsets) correctly."""
    import numpy as np

    p = tmp_path / "phonemes"
    p.mkdir()
    (p / "phonemes_000.json").write_text(json.dumps({
        "a": [[1, 2, 3], [], [4]],
        "bad": "not-a-list",  # ignored
    }))
    store = _load_phonemes(p, verbose=False)
    assert set(store) == {"a"}
    flat, offs = store["a"]
    assert flat.dtype == np.int16 and offs.dtype == np.int32
    assert offs.tolist() == [0, 3, 3, 4]
    assert flat.tolist() == [1, 2, 3, 4]
