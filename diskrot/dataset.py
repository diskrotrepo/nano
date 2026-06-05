"""Mmap-backed sharded dataset of cached DAC token tensors.

Uses the v2 sharded ``packed/`` layout produced by ``diskrot.pack_cache.pack``
and reads it via ``np.memmap`` so tokens page in from disk on demand. The bundle
returned by ``load_mmap_bundle`` is tiny — just a global index of (shard_id,
local_idx, name, n_frames) plus paths and tag/lyric dicts. Workers open their
own mmap handles lazily on first ``__getitem__``.

Serves random fixed-length crops via ``__getitem__``.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from model.codec import DACodec


def _load_tags(tags_path: str | Path | None, verbose: bool = True) -> dict[str, str]:
    if tags_path is None:
        return {}
    tp = Path(tags_path)
    if not tp.exists():
        return {}
    raw = json.loads(tp.read_text())
    tags = {key: val["description"] for key, val in raw.items()
            if isinstance(val, dict) and "description" in val}
    if verbose:
        print(f"[tags] loaded {len(tags)} entries from {tp}", flush=True)
    return tags


def _load_lyrics(lyrics_path: str | Path | None, verbose: bool = True) -> dict[str, dict]:
    if lyrics_path is None:
        return {}
    lp = Path(lyrics_path)
    if not lp.exists():
        return {}
    if lp.is_dir():
        from diskrot.transcribe_lyrics import load_lyrics_shards
        raw = load_lyrics_shards(lp)
    else:
        raw = json.loads(lp.read_text())
    lyrics = {key: val for key, val in raw.items()
              if isinstance(val, dict) and val.get("words")}
    if verbose:
        print(f"[lyrics] loaded {len(lyrics)} entries from {lp}", flush=True)
    return lyrics



# ---- mmap-backed sharded loader -------------------------------------------

def _build_mmap_split_index(
    packed_dir: Path,
    segment_frames: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[int, int, str, int]], list[tuple[int, int, str, int]], dict]:
    """Build (train_entries, val_entries, shard_metas) over a packed/ dir.

    ``entries`` is a list of (shard_id, local_idx, name, n_frames). Songs too
    short for ``segment_frames`` are filtered out. Train/val split uses the
    same ``random.Random(seed).shuffle()`` discipline as ``_split_files`` and
    ``_split_packed`` (sort by name first for shard-order independence).

    ``shard_metas`` is a dict ``{shard_id: meta}`` so the dataset can read
    offsets without re-opening each JSON sidecar per access.
    """
    from diskrot.pack_cache import iter_all_names, load_shard_meta, load_shard_index

    index = load_shard_index(packed_dir)
    shard_metas: dict[int, dict] = {}
    all_entries: list[tuple[int, int, str, int]] = []
    for shard_id, local_idx, name, n_frames in iter_all_names(packed_dir):
        if shard_id not in shard_metas:
            shard_metas[shard_id] = load_shard_meta(packed_dir, shard_id)
        # +2 mirrors _load_one: leave headroom for the random crop offset.
        if n_frames < segment_frames + 2:
            continue
        all_entries.append((shard_id, local_idx, name, n_frames))

    # Same deterministic shuffle as _split_packed: sort by name then
    # random.Random(seed).shuffle(). Same seed -> same split as the v1 path
    # so re-training on a repacked corpus with the same seed gives the same
    # train/val partition.
    all_entries.sort(key=lambda e: e[2])
    rng = random.Random(seed)
    rng.shuffle(all_entries)
    n_val = max(1, int(len(all_entries) * val_ratio))
    return all_entries[n_val:], all_entries[:n_val], shard_metas


def load_mmap_bundle(
    packed_dir: str | Path,
    segment_frames: int,
    val_ratio: float = 0.1,
    seed: int = 42,
    tags_path: str | Path | None = None,
    lyrics_path: str | Path | None = None,
) -> dict:
    """Parent-process bundle for the v2 sharded mmap path.

    Reads only the JSON sidecars + tags/lyrics dicts (no tokens). Returns a
    dict that ``TokenDataset.from_mmap`` can claim views into. Designed for
    DDP: build once in the parent before ``mp.spawn``; workers open their
    own ``np.memmap`` handles lazily on first ``__getitem__``.
    """
    packed_dir = Path(packed_dir)
    t0 = time.time()
    train_entries, val_entries, shard_metas = _build_mmap_split_index(
        packed_dir, segment_frames, val_ratio, seed,
    )
    print(f"[mmap-bundle] {len(train_entries)} train + {len(val_entries)} val "
          f"entries across {len(shard_metas)} shards ({time.time()-t0:.1f}s)",
          flush=True)
    return {
        "packed_dir": str(packed_dir),
        "train_entries": train_entries,
        "val_entries": val_entries,
        "shard_metas": shard_metas,
        "tags": _load_tags(tags_path),
        "lyrics": _load_lyrics(lyrics_path),
    }


class TokenDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        segment_frames: int = 860,  # 10s at 86Hz (overridden by TrainConfig.segment_seconds)
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        tags_path: str | Path | None = None,
        lyrics_path: str | Path | None = None,
    ):
        from diskrot.pack_cache import PACKED_DIR, SHARD_INDEX_NAME

        self.segment_frames = segment_frames
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        packed_dir = Path(cache_dir) / PACKED_DIR
        if not (packed_dir / SHARD_INDEX_NAME).exists():
            raise FileNotFoundError(
                f"No sharded packed layout at {packed_dir} — run diskrot.pack_cache first"
            )
        bundle = load_mmap_bundle(
            packed_dir=packed_dir,
            segment_frames=segment_frames,
            val_ratio=val_ratio,
            seed=seed,
            tags_path=tags_path,
            lyrics_path=lyrics_path,
        )
        # Delegate to from_mmap and steal its state into self.
        ds = TokenDataset.from_mmap(bundle, split, segment_frames)
        self.__dict__.update(ds.__dict__)

    @classmethod
    def from_mmap(
        cls,
        bundle: dict,
        split: str,
        segment_frames: int,
    ) -> "TokenDataset":
        """Wire up a TokenDataset over the sharded mmap bundle.

        No disk I/O at construction; mmap handles are opened lazily inside
        ``_get`` so each DataLoader worker owns its own kernel-page-cache
        view.
        """
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        ds = cls.__new__(cls)
        ds.segment_frames = segment_frames
        ds._packed_dir = bundle["packed_dir"]
        ds._mmap_entries = bundle[f"{split}_entries"]
        ds._shard_metas = bundle["shard_metas"]
        ds._mmap_handles = {}
        ds.names = [e[2] for e in ds._mmap_entries]
        name_set = set(ds.names)
        all_tags = bundle.get("tags", {})
        all_lyrics = bundle.get("lyrics", {})
        ds._tags = {n: all_tags[n] for n in name_set if n in all_tags}
        ds._lyrics = {n: all_lyrics[n] for n in name_set if n in all_lyrics}
        ds.has_tags = len(ds._tags) > 0 or len(ds._lyrics) > 0
        return ds

    def __getstate__(self) -> dict:
        # mmap handles don't survive DataLoader's spawn pickling cleanly; drop
        # them and let each worker reopen lazily on first __getitem__.
        state = self.__dict__.copy()
        state["_mmap_handles"] = {}
        return state

    def __len__(self) -> int:
        return len(self.names)

    def _get_shard_mmap(self, shard_id: int) -> np.memmap:
        mm = self._mmap_handles.get(shard_id)
        if mm is not None:
            return mm
        from diskrot.pack_cache import open_shard_mmap

        assert self._packed_dir is not None
        mm, _ = open_shard_mmap(self._packed_dir, shard_id)
        self._mmap_handles[shard_id] = mm
        return mm

    def _get(self, idx: int):
        shard_id, local_idx, _, _ = self._mmap_entries[idx]
        mm = self._get_shard_mmap(shard_id)
        offsets = self._shard_metas[shard_id]["offsets"]
        return mm[:, offsets[local_idx]:offsets[local_idx + 1]]  # numpy view

    def _get_segment_lyrics(self, name: str, start_sec: float, end_sec: float) -> str:
        entry = self._lyrics.get(name)
        if not entry or not entry.get("words"):
            return ""
        words = [w["word"] for w in entry["words"]
                 if w["end"] > start_sec and w["start"] < end_sec]
        return " ".join(words)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, str]:
        t = self._get(idx)  # [K, T_full] int16 — torch.Tensor or np.memmap view
        T = t.shape[1]
        start = random.randint(0, T - self.segment_frames)
        crop = t[:, start:start + self.segment_frames]
        if not isinstance(crop, torch.Tensor):
            # mmap path: materialize the ~46 KB int16 crop (30s case) so the
            # collate doesn't carry a memmap view across the worker boundary.
            crop = torch.from_numpy(np.ascontiguousarray(crop))
        tokens = crop
        tags = self._tags.get(self.names[idx], "")
        # time-aligned lyrics for this segment
        start_sec = start / DACodec.FRAME_RATE_HZ
        end_sec = (start + self.segment_frames) / DACodec.FRAME_RATE_HZ
        lyrics = self._get_segment_lyrics(self.names[idx], start_sec, end_sec)
        return tokens, tags, lyrics
