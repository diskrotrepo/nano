"""Smoke tests for the v2 mmap-backed TokenDataset path.

Validates that load_mmap_bundle + TokenDataset.from_mmap produce samples
byte-equivalent to the existing in-memory path, that train/val splits stay
deterministic by seed, and that the DataLoader pickle round-trip (which
clears mmap handles via __getstate__) doesn't break iteration.
"""
from __future__ import annotations

import pickle

import torch

from diskrot import pack_cache
from diskrot.dataset import TokenDataset, load_mmap_bundle


def _crop_for_idx(ds: TokenDataset, idx: int, seed: int) -> torch.Tensor:
    """Sample idx with a deterministic random.seed for reproducibility."""
    import random as _r
    _r.seed(seed)
    tokens, *_ = ds[idx]
    return tokens


def test_from_mmap_yields_int16_tensor_with_correct_shape(synth_tokens_dir):
    cache = synth_tokens_dir(n_files=6, T=1000, n_codebooks=9, vocab=1024, seed=0)
    out_dir = pack_cache.pack(cache, shard_target_songs=4, verbose=False)
    bundle = load_mmap_bundle(out_dir, segment_frames=400, val_ratio=0.2, seed=42)
    train = TokenDataset.from_mmap(bundle, split="train", segment_frames=400)
    val = TokenDataset.from_mmap(bundle, split="val", segment_frames=400)
    # Stable per-song hash split: total is preserved and val is a non-empty
    # proper subset. (Exact counts depend on the name hashes now, not on the old
    # max(1, int(N*ratio)) arithmetic.)
    assert len(train) + len(val) == 6
    assert 1 <= len(val) < 6
    assert len(train) >= 1
    tokens, tag, lyric_ids, _melody, *_ = train[0]
    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.int16
    assert tokens.shape == (9, 400)
    assert tag == ""  # no tags fixture passed
    # no lyrics/structure/keys fixture → BOS + the dense all-unknown prefix
    # header (see lyric_encoder)
    from model.lyric_encoder import (
        BOS_PHONEME_ID, NO_SECTION_ID, UNKNOWN_GENDER_ID, UNKNOWN_KEY_ID,
        UNKNOWN_LANG_ID, UNKNOWN_TEMPO_ID, UNKNOWN_VOCALS_ID,
    )
    # v9 6-marker header: BOS <gender> <tempo> <key> <vocals> <lang> <section>.
    assert lyric_ids.tolist() == [
        BOS_PHONEME_ID, UNKNOWN_GENDER_ID, UNKNOWN_TEMPO_ID,
        UNKNOWN_KEY_ID, UNKNOWN_VOCALS_ID, UNKNOWN_LANG_ID, NO_SECTION_ID,
    ]


def test_from_mmap_matches_in_memory_dataset(synth_tokens_dir):
    """Same .pt corpus -> in-memory dataset and mmap dataset must serve byte-
    equivalent samples at the same random seed."""
    cache = synth_tokens_dir(n_files=4, T=600, n_codebooks=9, vocab=1024, seed=7)
    out_dir = pack_cache.pack(cache, shard_target_songs=10, verbose=False)

    in_mem = TokenDataset(cache, segment_frames=300, split="train", seed=42, val_ratio=0.25)
    bundle = load_mmap_bundle(out_dir, segment_frames=300, val_ratio=0.25, seed=42)
    mm = TokenDataset.from_mmap(bundle, split="train", segment_frames=300)

    # Same split discipline -> same name list (order may differ; compare sets).
    assert set(in_mem.names) == set(mm.names)

    # Sample each name with the same seed and verify byte-equal crops.
    common_name = sorted(set(in_mem.names))[0]
    in_idx = in_mem.names.index(common_name)
    mm_idx = mm.names.index(common_name)
    t_in = _crop_for_idx(in_mem, in_idx, seed=123)
    t_mm = _crop_for_idx(mm, mm_idx, seed=123)
    assert torch.equal(t_in.to(torch.int16), t_mm), (
        f"in-memory and mmap crops for {common_name} differ"
    )


def test_split_is_deterministic_by_seed(synth_tokens_dir):
    cache = synth_tokens_dir(n_files=10, T=500, seed=0)
    out_dir = pack_cache.pack(cache, shard_target_songs=4, verbose=False)
    b1 = load_mmap_bundle(out_dir, segment_frames=200, val_ratio=0.3, seed=99)
    b2 = load_mmap_bundle(out_dir, segment_frames=200, val_ratio=0.3, seed=99)
    train1 = TokenDataset.from_mmap(b1, "train", 200).names
    train2 = TokenDataset.from_mmap(b2, "train", 200).names
    val1 = TokenDataset.from_mmap(b1, "val", 200).names
    val2 = TokenDataset.from_mmap(b2, "val", 200).names
    assert train1 == train2
    assert val1 == val2
    # train and val must be disjoint.
    assert set(train1).isdisjoint(set(val1))


def test_pickle_roundtrip_clears_mmap_handles(synth_tokens_dir):
    """DataLoader workers pickle the dataset; __getstate__ must drop the
    mmap handle dict so the worker re-opens lazily without inheriting a
    stale handle from the parent process."""
    cache = synth_tokens_dir(n_files=3, T=500, seed=0)
    out_dir = pack_cache.pack(cache, shard_target_songs=10, verbose=False)
    bundle = load_mmap_bundle(out_dir, segment_frames=200, val_ratio=0.4, seed=42)
    ds = TokenDataset.from_mmap(bundle, split="train", segment_frames=200)
    # Force-open at least one shard handle.
    _ = ds[0]
    assert len(ds._mmap_handles) > 0
    blob = pickle.dumps(ds)
    revived = pickle.loads(blob)
    assert revived._mmap_handles == {}  # cleared by __getstate__
    # And the revived dataset can still serve samples (re-opens lazily).
    tokens, *_ = revived[0]
    assert tokens.shape == (9, 200)


def test_short_songs_are_filtered(synth_tokens_dir):
    """Songs shorter than segment_frames + 2 must be filtered from the index."""
    cache = synth_tokens_dir(
        n_files=4, T=[100, 100, 500, 500], n_codebooks=9, vocab=1024, seed=0,
    )
    out_dir = pack_cache.pack(cache, shard_target_songs=10, verbose=False)
    bundle = load_mmap_bundle(out_dir, segment_frames=300, val_ratio=0.5, seed=42)
    train = TokenDataset.from_mmap(bundle, "train", 300).names
    val = TokenDataset.from_mmap(bundle, "val", 300).names
    # Only the two 500-frame songs survive; one to each split.
    assert len(train) + len(val) == 2
    assert set(train + val) == {"song_002", "song_003"}
