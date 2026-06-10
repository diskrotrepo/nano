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
from collections import OrderedDict
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


def _load_phonemes(
    phonemes_path: str | Path | None, verbose: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the pre-phonemized per-word groups (diskrot.phonemize) compactly.

    Returns ``{name: (flat int16 phoneme ids, int32 group offsets)}`` — word k's
    group is ``flat[offsets[k]:offsets[k+1]]``. The numpy form (instead of 322k
    Python list-of-lists) keeps the parent bundle small and fork-COW-stable for
    the DataLoader workers. Sparse: a song without an entry (or whose group
    count no longer matches its words — a re-transcribe made it stale) falls
    back to live g2p at crop time, so a partial/absent pass is always safe."""
    if phonemes_path is None:
        return {}
    pp = Path(phonemes_path)
    if not pp.exists():
        return {}
    if pp.is_dir():
        from diskrot.phonemize import load_phoneme_shards
        raw = load_phoneme_shards(pp)
    else:
        raw = json.loads(pp.read_text())
    phonemes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, groups in raw.items():
        if not isinstance(groups, list):
            continue
        offsets = np.zeros(len(groups) + 1, dtype=np.int32)
        for i, g in enumerate(groups):
            offsets[i + 1] = offsets[i] + len(g)
        flat = np.fromiter(
            (pid for g in groups for pid in g), dtype=np.int16, count=int(offsets[-1]),
        )
        phonemes[name] = (flat, offsets)
    if verbose:
        print(f"[phonemes] loaded {len(phonemes)} pre-phonemized entries from {pp}",
              flush=True)
    return phonemes


def _load_keys(keys_path: str | Path | None, verbose: bool = True) -> dict[str, str]:
    """Load per-song key labels (diskrot.key_detect's keys.json) for the
    ``<key_*>`` header marker. Sparse: a song without an entry gets
    ``<unknown_key>`` at crop time, so a partial/absent pass is fine."""
    if keys_path is None:
        return {}
    kp = Path(keys_path)
    if not kp.exists():
        return {}
    raw = json.loads(kp.read_text())
    keys = {name: val["key"] for name, val in raw.items()
            if isinstance(val, dict) and isinstance(val.get("key"), str)}
    if verbose:
        print(f"[keys] loaded {len(keys)} entries from {kp}", flush=True)
    return keys


# Word-entry validity lives in transcribe_lyrics.is_valid_word (the schema
# producer) so the dataset loader and diskrot.phonemize filter with the
# IDENTICAL predicate — the pre-phonemized store's group-count==word-count
# contract depends on it. Imported inside _load_lyrics (its only user here).


def _load_lyrics(
    lyrics_path: str | Path | None, verbose: bool = True, with_instrumental: bool = False,
):
    """Load usable lyric entries, keyed by song name.

    With ``with_instrumental=True`` returns ``(lyrics, instrumental)`` where
    ``instrumental`` is the set of names the transcription pass *processed but
    found no usable words in* — the train-time signal for the ``<instrumental>``
    vocal-presence marker (distinct from "never transcribed", which stays
    ``<unknown_vocals>``). Default return shape is just the dict, for callers
    that predate the marker (e.g. scripts/eval_lyric_wer.py)."""
    from diskrot.transcribe_lyrics import is_valid_word, load_lyrics_shards

    if lyrics_path is None:
        return ({}, set()) if with_instrumental else {}
    lp = Path(lyrics_path)
    if not lp.exists():
        return ({}, set()) if with_instrumental else {}
    raw = load_lyrics_shards(lp) if lp.is_dir() else json.loads(lp.read_text())
    lyrics: dict[str, dict] = {}
    instrumental: set[str] = set()
    n_dropped_words = 0
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        words = val.get("words")
        clean = [w for w in words if is_valid_word(w)] if isinstance(words, list) else []
        n_dropped_words += len(words) - len(clean) if isinstance(words, list) else 0
        if not clean:
            # Transcribed but no usable words — instrumental (or hallucination-free
            # silence). Feeds the <instrumental> vocal-presence marker.
            instrumental.add(key)
            continue
        lyrics[key] = {**val, "words": clean} if len(clean) != len(words) else val
    if verbose:
        msg = (f"[lyrics] loaded {len(lyrics)} entries "
               f"({len(instrumental)} instrumental) from {lp}")
        if n_dropped_words:
            msg += f" (dropped {n_dropped_words} malformed word entries)"
        print(msg, flush=True)
    return (lyrics, instrumental) if with_instrumental else lyrics


def _load_structure(
    structure_path: str | Path | None, verbose: bool = True,
) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """Load song-structure segments + per-song bpm, keyed by song name.

    Returns ``(structure, bpm)``. ``structure[name]`` is a list of
    ``{"start","end","label"}`` segments sorted by start, with allin1's non-section
    ``start``/``end`` sentinel labels filtered out. ``bpm[name]`` is the allin1
    tempo (a float) when present in the shard — used for the tempo marker. Both are
    sparse: songs with no structure entry get a ``<no_section>`` prefix and songs
    with no bpm get ``<unknown_tempo>`` at crop time, so a partial structure pass is
    fine. Mirrors ``_load_lyrics`` (sharded dir or a single JSON).
    """
    if structure_path is None:
        return {}, {}
    sp = Path(structure_path)
    if not sp.exists():
        return {}, {}
    if sp.is_dir():
        from diskrot.structure import load_structure_shards
        raw = load_structure_shards(sp)
    else:
        raw = json.loads(sp.read_text())
    structure: dict[str, list[dict]] = {}
    bpm: dict[str, float] = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            continue
        segs = [s for s in val.get("segments", [])
                if isinstance(s, dict) and s.get("label") not in (None, "start", "end")]
        if segs:
            structure[key] = sorted(segs, key=lambda s: s["start"])
        b = val.get("bpm")
        if isinstance(b, (int, float)):
            bpm[key] = float(b)
    if verbose:
        print(f"[structure] loaded {len(structure)} entries "
              f"({len(bpm)} with bpm) from {sp}", flush=True)
    return structure, bpm


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
    """DataLoader collate for TokenDataset's (tokens, tags, lyric_ids, melody) items.

    Pads the variable-length phoneme-id sequences to the batch max and builds a
    bool mask (True = real phoneme). Required because default_collate can't stack
    ragged lyric tensors. The melody (chroma) crop is fixed-length (== segment
    frames) so it stacks directly; it's ``None`` for non-melody packs. Returns
    (tokens [B,K,T], tags list[str], lyric_ids [B,Lmax], lyric_mask [B,Lmax],
    melody [B,T,12] | None).
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
    melodies = [b[3] for b in batch]
    melody = torch.stack(melodies) if melodies and melodies[0] is not None else None
    return tokens, tags, ids, mask, melody



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
    keys_path: str | Path | None = None,
    phonemes_path: str | Path | None = None,
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
    # Melody is available iff the pack wrote the parallel chroma sidecars (index
    # flag). The dataset then co-crops chroma with the SAME crop window.
    from diskrot.pack_cache import load_shard_index
    has_melody = bool(load_shard_index(packed_dir).get("has_melody", False))
    print(f"[mmap-bundle] {len(train_entries)} train + {len(val_entries)} val "
          f"entries across {len(shard_metas)} shards "
          f"(melody={'on' if has_melody else 'off'}) ({time.time()-t0:.1f}s)",
          flush=True)
    structure, bpm = _load_structure(structure_path)
    lyrics, instrumental = _load_lyrics(lyrics_path, with_instrumental=True)
    return {
        "packed_dir": str(packed_dir),
        "train_entries": train_entries,
        "val_entries": val_entries,
        "shard_metas": shard_metas,
        "tags": _load_tags(tags_path),
        "lyrics": lyrics,
        "instrumental": instrumental,
        "structure": structure,
        "bpm": bpm,
        "keys": _load_keys(keys_path),
        "phonemes": _load_phonemes(phonemes_path),
        "has_melody": has_melody,
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
        keys_path: str | Path | None = None,
        phonemes_path: str | Path | None = None,
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
            keys_path=keys_path,
            phonemes_path=phonemes_path,
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
        ds._has_melody = bool(bundle.get("has_melody", False))
        ds._mel_handles = {}
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
        # Per-song bpm (allin1 tempo), used for the <tempo_*> header marker. Sparse:
        # songs without a bpm aren't in the map and get <unknown_tempo> at crop time.
        all_bpm = bundle.get("bpm", {})
        ds._bpm = {n: all_bpm[n] for n in name_set if n in all_bpm}
        # Per-song key labels (diskrot.key_detect), used for the <key_*> header
        # marker. Sparse: missing -> <unknown_key> at crop time.
        all_keys = bundle.get("keys", {})
        ds._keys = {n: all_keys[n] for n in name_set if n in all_keys}
        # Names the transcription pass processed but found no usable words in —
        # the <instrumental> vocal-presence signal (in _lyrics -> <vocals>,
        # never transcribed -> <unknown_vocals>).
        ds._instrumental = bundle.get("instrumental", set()) & name_set
        # Pre-phonemized per-word groups (diskrot.phonemize) in compact numpy
        # form — consulted before live g2p in _get_segment_lyric_ids.
        all_phonemes = bundle.get("phonemes", {})
        ds._phonemes = {n: all_phonemes[n] for n in name_set if n in all_phonemes}
        # Per-song phoneme groups (one list[int] per word), built lazily on first
        # access and reused across crops — g2p runs once per song while it stays
        # hot. Bounded LRU (OrderedDict): each forked DataLoader worker fills its
        # own copy, so an unbounded dict would grow to ~corpus-size per worker
        # over a long random-sampled run. The cap trades bounded recompute (a
        # cache miss re-runs g2p, deterministically — same ids) for bounded mem.
        ds._word_phones = OrderedDict()
        ds._word_phones_cap = min(len(ds.names), 4096) or 1
        ds.has_tags = len(ds._tags) > 0 or len(ds._lyrics) > 0
        return ds

    def __getstate__(self) -> dict:
        # mmap handles don't survive DataLoader's spawn pickling cleanly; drop
        # them and let each worker reopen lazily on first __getitem__.
        state = self.__dict__.copy()
        state["_mmap_handles"] = {}
        state["_mel_handles"] = {}
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

    def _get_shard_mel_mmap(self, shard_id: int) -> np.memmap:
        mm = self._mel_handles.get(shard_id)
        if mm is not None:
            return mm
        from diskrot.pack_cache import open_shard_mel_mmap

        assert self._packed_dir is not None
        mm, _ = open_shard_mel_mmap(self._packed_dir, shard_id)
        self._mel_handles[shard_id] = mm
        return mm

    def _get_mel(self, idx: int):
        """Per-song chroma view [12, T_full] at the SAME offsets as the tokens."""
        shard_id, local_idx, _, _ = self._mmap_entries[idx]
        mm = self._get_shard_mel_mmap(shard_id)
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

            BOS  <gender>  <tempo>  <key>  <vocals>  <active-section>  w w  <inline-section>  w ...

        - Always starts with BOS, then exactly one gender marker (the song's
          F0-labeled vocal gender, ``<unknown_gender>`` if instrumental /
          unlabeled), one tempo marker (the allin1 bpm bucketed, ``<unknown_tempo>``
          if no bpm), one key marker (the key_detect estimate, ``<unknown_key>``
          if none), one vocal-presence marker (``<vocals>`` if the song has usable
          transcribed words, ``<instrumental>`` if transcription found none,
          ``<unknown_vocals>`` if never transcribed), and exactly one section
          marker for the section active at ``start_sec`` (``<no_section>`` in a
          gap / for songs without a structure entry). This dense prefix means even
          a boundary-free or instrumental crop carries all five slots, never a
          fully-padded row.
        - Any section boundary that falls inside the crop is injected inline before
          the first word at/after the boundary. (Gender/tempo/key/vocals are
          per-song, prefix-only.)
        Markers and words are appended via the shared ``append_unit`` separator
        rule; truncated to max_lyric_len.
        """
        from model.lyric_encoder import (
            BOS_PHONEME_ID, VOCAL_TOKEN_TO_ID, UNKNOWN_VOCALS_ID, append_unit,
            append_unit_capped, bpm_to_id, gender_label_to_id, key_label_to_id,
            structure_label_to_id, text_to_word_phoneme_groups,
        )

        segs = self._structure.get(name)
        entry = self._lyrics.get(name)
        gender = entry.get("gender") if isinstance(entry, dict) else None
        bpm = self._bpm.get(name)
        if entry:
            vocal_id = VOCAL_TOKEN_TO_ID["vocals"]
        elif name in self._instrumental:
            vocal_id = VOCAL_TOKEN_TO_ID["instrumental"]
        else:
            vocal_id = UNKNOWN_VOCALS_ID
        # Compact 5-marker header (no internal word-boundary):
        # BOS <gender> <tempo> <key> <vocals> <section>.
        ids = [BOS_PHONEME_ID]
        append_unit(ids, [
            gender_label_to_id(gender),
            bpm_to_id(bpm),
            key_label_to_id(self._keys.get(name)),
            vocal_id,
            structure_label_to_id(_active_label_at(segs, start_sec)),
        ])

        if not entry or not entry.get("words"):
            return ids
        groups = self._word_phones.get(name)
        if groups is None:
            # Lookup order: pre-phonemized store (diskrot.phonemize) -> live g2p.
            # A stale store entry (group count != word count after a re-transcribe)
            # falls back to g2p — same ids either way, g2p is deterministic.
            pre = self._phonemes.get(name)
            if pre is not None and len(pre[1]) - 1 == len(entry["words"]):
                flat, offs = pre
                groups = [flat[offs[i]:offs[i + 1]].tolist()
                          for i in range(len(offs) - 1)]
            else:
                groups = text_to_word_phoneme_groups([w["word"] for w in entry["words"]])
            self._word_phones[name] = groups
            if len(self._word_phones) > self._word_phones_cap:
                self._word_phones.popitem(last=False)  # evict least-recently-used
        else:
            self._word_phones.move_to_end(name)  # mark as recently used

        # Boundaries strictly inside the crop, in order, injected as we reach the
        # first word at/after each one.
        pending = [s for s in segs if start_sec < s["start"] < end_sec] if segs else []
        bi = 0
        for w, g in zip(entry["words"], groups):
            if w["end"] > start_sec and w["start"] < end_sec:
                stop = False
                while bi < len(pending) and pending[bi]["start"] <= w["start"]:
                    marker = [structure_label_to_id(pending[bi]["label"])]
                    if not append_unit_capped(ids, marker, self.max_lyric_len):
                        stop = True
                        break
                    bi += 1
                if stop or not append_unit_capped(ids, g, self.max_lyric_len):
                    break  # whole-unit truncation — never slice a word/marker mid-unit
        return ids

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor | None]:
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
        # Co-crop the chroma with the IDENTICAL [start, start+segment_frames] window
        # so the melody lines up with the tokens frame-for-frame. -> [seg, 12] float.
        melody = None
        if self._has_melody:
            mel = np.ascontiguousarray(self._get_mel(idx)[:, start:start + self.segment_frames])
            melody = torch.from_numpy(mel).to(torch.float32).transpose(0, 1).contiguous()
        return tokens, tags, torch.tensor(lyric_ids, dtype=torch.long), melody
