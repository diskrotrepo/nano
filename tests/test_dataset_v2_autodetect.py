"""TokenDataset.__init__ should detect the sharded packed dir and use the mmap
path. Single-GPU runs that construct TokenDataset(cache_dir=...) land here
without any caller changes.
"""
from __future__ import annotations

import torch

from diskrot import pack_cache
from diskrot.dataset import TokenDataset


def test_init_uses_mmap_when_packed_dir_present(synth_tokens_dir):
    cache = synth_tokens_dir(n_files=5, T=800, seed=0)
    pack_cache.pack(cache, shard_target_songs=3, verbose=False)
    ds = TokenDataset(cache, segment_frames=400, split="train", seed=42)

    # Mmap path: _mmap_entries populated.
    assert ds._mmap_entries is not None
    assert len(ds) > 0


def test_init_mmap_serves_correct_crops(synth_tokens_dir):
    import random
    cache = synth_tokens_dir(n_files=4, T=600, seed=11)
    pack_cache.pack(cache, shard_target_songs=10, verbose=False)
    ds = TokenDataset(cache, segment_frames=300, split="train", seed=42, val_ratio=0.25)

    random.seed(99)
    tokens, *_ = ds[0]
    assert tokens.shape == (9, 300)
    assert tokens.dtype == torch.int16
