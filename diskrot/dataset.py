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


def _load_structure(structure_path: str | Path | None, verbose: bool = True) -> dict[str, list[dict]]:
    """Load song-structure segments keyed by song name.

    Each value is a list of ``{"start","end","label"}`` segments sorted by start,
    with allin1's non-section ``start``/``end`` sentinel labels filtered out. Songs
    with no structure entry simply get a ``<no_section>`` prefix at crop time, so a
    partial structure pass is fine. Mirrors ``_load_lyrics`` (sharded dir or a
    single JSON).
    """
    if structure_path is None:
        return {}
    sp = Path(structure_path)
    if not sp.exists():
        return {}
    if sp.is_dir():
        from diskrot.structure import load_structure_shards
        raw = load_structure_shards(sp)
    else:
        raw = json.loads(sp.read_text())
    structure: dict[str, list[dict]] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        segs = [s for s in val.get("segments", [])
                if isinstance(s, dict) and s.get("label") not in (None, "start", "end")]
        if segs:
            structure[key] = sorted(segs, key=lambda s: s["start"])
    if verbose:
        print(f"[structure] loaded {len(structure)} entries from {sp}", flush=True)
    return structure


def _active_label_at(segs: list[dict] | None, t: float) -> str:
    """Label of the segment containing time ``t``, or ``no_section`` if in a gap."""
    from model.lyric_encoder import NO_SECTION_LABEL
    if not segs:
        return NO_SECTION_LABEL
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s["label"]
    return NO_SECTION_LABEL


def collate_lyrics(batch):
    """DataLoader collate for TokenDataset's (tokens, tags, lyric_ids) items.

    Pads the variable-length phoneme-id sequences to the batch max and builds a
    bool mask (True = real phoneme). Required because default_collate can't stack
    ragged lyric tensors. Returns (tokens [B,K,T], tags list[str],
    lyric_ids [B,Lmax], lyric_mask [B,Lmax]).
    """
    from model.lyric_encoder import PAD_PHONEME_ID

    tokens = torch.stack([b[0] for b in batch])  # [B, K, T] int16
    tags = [b[1] for b in batch]
    lyrics = [b[2] for b in batch]
    lmax = max(int(t.shape[0]) for t in lyrics)
    ids = torch.full((len(batch), lmax), PAD_PHONEME_ID, dtype=torch.long)
    for i, t in enumerate(lyrics):
        ids[i, : t.shape[0]] = t
    mask = ids != PAD_PHONEME_ID
    return tokens, tags, ids, mask



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
    structure_path: str | Path | None = None,
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
        "structure": _load_structure(structure_path),
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
        structure_path: str | Path | None = None,
        max_lyric_len: int = 256,
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
            structure_path=structure_path,
        )
        # Delegate to from_mmap and steal its state into self.
        ds = TokenDataset.from_mmap(bundle, split, segment_frames, max_lyric_len)
        self.__dict__.update(ds.__dict__)

    @classmethod
    def from_mmap(
        cls,
        bundle: dict,
        split: str,
        segment_frames: int,
        max_lyric_len: int = 256,
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
        ds.max_lyric_len = max_lyric_len
        ds._packed_dir = bundle["packed_dir"]
        ds._mmap_entries = bundle[f"{split}_entries"]
        ds._shard_metas = bundle["shard_metas"]
        ds._mmap_handles = {}
        ds.names = [e[2] for e in ds._mmap_entries]
        name_set = set(ds.names)
        all_tags = bundle.get("tags", {})
        all_lyrics = bundle.get("lyrics", {})
        all_structure = bundle.get("structure", {})
        ds._tags = {n: all_tags[n] for n in name_set if n in all_tags}
        ds._lyrics = {n: all_lyrics[n] for n in name_set if n in all_lyrics}
        # Per-song structure segments (sorted by start), used to inject section
        # markers into the time-aligned lyric stream. Songs without an entry get a
        # <no_section> prefix, so a partial structure pass is fine.
        ds._structure = {n: all_structure[n] for n in name_set if n in all_structure}
        # Per-song phoneme groups (one list[int] per word), built lazily on first
        # access and reused across crops — g2p runs once per song, not per segment.
        ds._word_phones = {}
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

    def _get_segment_lyric_ids(self, name: str, start_sec: float, end_sec: float) -> list[int]:
        """Phoneme + structure-marker ids for the crop window [start_sec, end_sec].

        Stream format (must stay byte-identical to the inference parser
        ``text_with_markers_to_phoneme_ids``):

            BOS  <gender>  <active-section>  w w  <inline-section>  w ...

        - Always starts with BOS, then exactly one gender marker (the song's
          F0-labeled vocal gender, ``<unknown_gender>`` if instrumental /
          unlabeled) and exactly one section marker for the section active at
          ``start_sec`` (``<no_section>`` in a gap / for songs without a structure
          entry). This dense prefix means even a boundary-free or instrumental crop
          carries both slots, never a fully-padded row.
        - Any section boundary that falls inside the crop is injected inline before
          the first word at/after the boundary. (Gender is per-song, prefix-only.)
        Markers and words are appended via the shared ``append_unit`` separator
        rule; truncated to max_lyric_len.
        """
        from model.lyric_encoder import (
            BOS_PHONEME_ID, append_unit, gender_label_to_id,
            structure_label_to_id, text_to_word_phoneme_groups,
        )

        segs = self._structure.get(name)
        entry = self._lyrics.get(name)
        gender = entry.get("gender") if isinstance(entry, dict) else None
        # Compact 2-marker header (no internal word-boundary): BOS <gender> <section>.
        ids = [BOS_PHONEME_ID]
        append_unit(ids, [
            gender_label_to_id(gender),
            structure_label_to_id(_active_label_at(segs, start_sec)),
        ])

        if not entry or not entry.get("words"):
            return ids
        groups = self._word_phones.get(name)
        if groups is None:
            groups = text_to_word_phoneme_groups([w["word"] for w in entry["words"]])
            self._word_phones[name] = groups

        # Boundaries strictly inside the crop, in order, injected as we reach the
        # first word at/after each one.
        pending = [s for s in segs if start_sec < s["start"] < end_sec] if segs else []
        bi = 0
        for w, g in zip(entry["words"], groups):
            if w["end"] > start_sec and w["start"] < end_sec:
                while bi < len(pending) and pending[bi]["start"] <= w["start"]:
                    append_unit(ids, [structure_label_to_id(pending[bi]["label"])])
                    bi += 1
                append_unit(ids, g)
                if len(ids) >= self.max_lyric_len:
                    ids = ids[: self.max_lyric_len]
                    break
        return ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor]:
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
        # time-aligned lyric phoneme ids for this segment
        start_sec = start / DACodec.FRAME_RATE_HZ
        end_sec = (start + self.segment_frames) / DACodec.FRAME_RATE_HZ
        lyric_ids = self._get_segment_lyric_ids(self.names[idx], start_sec, end_sec)
        return tokens, tags, torch.tensor(lyric_ids, dtype=torch.long)
