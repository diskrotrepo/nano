"""Smoke tests for the v2 sharded streaming packer in diskrot/pack_cache.py.

The v6 dataset path mmaps these shards directly, so byte-exact round-trip from
per-song .pt files through the packer back out via the mmap is load-bearing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from diskrot import pack_cache


def test_pack_writes_sharded_layout(synth_tokens_dir):
    cache = synth_tokens_dir(n_files=7, T=500, n_codebooks=9, vocab=1024, seed=0)
    out_dir = pack_cache.pack(cache, shard_target_songs=3, verbose=False)
    out = Path(out_dir)

    # 7 songs / 3 per shard -> 3 shards (3 + 3 + 1).
    index = pack_cache.load_shard_index(out)
    assert index["format_version"] == pack_cache.FORMAT_VERSION
    assert index["n_codebooks"] == 9
    assert index["n_shards"] == 3
    assert index["n_songs_total"] == 7
    assert index["dtype"] == "int16"

    # Sidecars exist for every shard.
    for s in range(3):
        assert (out / f"packed_{s:03d}.bin").exists()
        assert (out / f"packed_{s:03d}.json").exists()


def test_pack_roundtrip_byte_exact(synth_tokens_dir):
    """For every song, mmap-read slice must equal the original .pt content."""
    cache = synth_tokens_dir(n_files=5, T=400, n_codebooks=9, vocab=1024, seed=42)
    out_dir = pack_cache.pack(cache, shard_target_songs=2, verbose=False)

    # Walk every (shard, local_idx, name, n_frames) and verify byte equality.
    for shard_id, local_idx, name, n_frames in pack_cache.iter_all_names(out_dir):
        mm, meta = pack_cache.open_shard_mmap(out_dir, shard_id)
        try:
            off_start = meta["offsets"][local_idx]
            off_end = meta["offsets"][local_idx + 1]
            assert off_end - off_start == n_frames
            mmap_slice = np.array(mm[:, off_start:off_end])

            original = torch.load(Path(cache) / f"{name}.pt", weights_only=True)
            assert original.dtype == torch.int16
            assert original.shape == (9, n_frames)
            assert np.array_equal(mmap_slice, original.numpy()), (
                f"shard {shard_id} local {local_idx} ({name}) mismatch"
            )
        finally:
            del mm


def test_pack_uneven_song_lengths(synth_tokens_dir):
    """Songs of different lengths must pack with correct per-song offsets."""
    cache = synth_tokens_dir(
        n_files=4, T=[100, 250, 800, 50], n_codebooks=9, vocab=1024, seed=1,
    )
    out_dir = pack_cache.pack(cache, shard_target_songs=10, verbose=False)
    # All 4 in one shard.
    index = pack_cache.load_shard_index(out_dir)
    assert index["n_shards"] == 1
    meta = pack_cache.load_shard_meta(out_dir, 0)
    assert meta["total_T"] == 100 + 250 + 800 + 50
    # Offsets are sorted by file stem (song_000, song_001, ...). The fixture
    # writes them in that order, so offsets should be cumulative of the lengths
    # in the order [100, 250, 800, 50].
    assert meta["offsets"] == [0, 100, 350, 1150, 1200]


