"""Mmap-backed sharded dataset of cached DAC token tensors.

Uses the v2 sharded ``packed/`` layout produced by ``diskrot.pack_cache.pack``
and reads it via ``np.memmap`` so tokens page in from disk on demand. The bundle
returned by ``load_mmap_bundle`` is tiny — just a global index of (shard_id,
local_idx, name, n_frames) plus paths and tag/lyric dicts. Workers open their
own mmap handles lazily on first ``__getitem__``.

Serves random fixed-length crops via ``__getitem__``.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from model.codec import DACodec, codec_constants

# Frame rate of the active codec's tokens (NANO_CODEC): DAC=86 Hz, SpectroStream=25
# Hz. Drives the seconds<->frames math in crop biasing + structure markers, so it
# MUST match the codec the corpus was tokenized with.
_FRAME_RATE_HZ = codec_constants()["frame_rate_hz"]


def _load_tags(tags_path: str | Path | None, verbose: bool = True) -> dict[str, str]:
    if tags_path is None:
        return {}
    tp = Path(tags_path)
    if not tp.exists():
        return {}
    # Sharded dir (tags/tags_NNN.json) at scale, or the legacy single tags.json.
    if tp.is_dir():
        from diskrot.sharded_store import load_json_shards
        raw = load_json_shards(tp, "tags")
    else:
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
    # Sharded dir (keys/keys_NNN.json) at scale, or the legacy single keys.json.
    if kp.is_dir():
        from diskrot.sharded_store import load_json_shards
        raw = load_json_shards(kp, "keys")
    else:
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
        if val is None:
            # The on-disk instrumental convention: transcribe (and the
            # hallucination filter) write null for transcribed-but-wordless
            # songs. Distinct from ABSENT = never transcribed = <unknown_vocals>.
            instrumental.add(key)
            continue
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
    pad_short: bool = False,
) -> tuple[list[tuple[int, int, str, int]], list[tuple[int, int, str, int]], dict]:
    """Build (train_entries, val_entries, shard_metas) over a packed/ dir.

    ``entries`` is a list of (shard_id, local_idx, name, n_frames). Songs too
    short for ``segment_frames`` are filtered out UNLESS ``pad_short`` (then they
    are kept and the dataset pads+masks the tail — the v9 full-song path, so every
    song trains on its full length). Train/val split is a stable
    per-song hash of the name (+ seed), so adding songs/waves never moves an
    existing song between splits (see the split block below).

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
        # +2 mirrors _load_one: leave headroom for the random crop offset. With
        # pad_short, keep every song (>=1 frame) — songs shorter than the clip are
        # padded+masked in __getitem__ rather than dropped.
        if not pad_short and n_frames < segment_frames + 2:
            continue
        if pad_short and n_frames < 1:
            continue
        all_entries.append((shard_id, local_idx, name, n_frames))

    # Stable, corpus-growth-invariant split: a song's train/val assignment is a
    # pure function of its name (+ seed), NOT of the corpus size. So appending
    # waves never moves an existing song between splits. (The old
    # sort+random.Random(seed).shuffle()+slice reshuffled the whole corpus every
    # time it grew, silently churning val membership and leaking val<->train
    # across waves.) Bucket on sha1(name) — the same hashing discipline the
    # metadata shards use (transcribe_lyrics._shard_for) — into 10k buckets so
    # val_ratio is honored to 0.0001 and the realized ratio converges as the
    # corpus grows. One-time breaking change vs the shuffle split: val membership
    # differs, so val loss is comparable across all FUTURE runs but not to
    # pre-switch checkpoints.
    all_entries.sort(key=lambda e: e[2])  # stable, shard-order-independent order
    val_cut = int(val_ratio * 10_000)

    def _is_val(name: str) -> bool:
        h = int(hashlib.sha1(f"{seed}:{name}".encode()).hexdigest()[:8], 16)
        return (h % 10_000) < val_cut

    train_entries = [e for e in all_entries if not _is_val(e[2])]
    val_entries = [e for e in all_entries if _is_val(e[2])]
    # Guarantee a non-empty val set on tiny corpora (e.g. tests / pipeline runs)
    # where the hash bucket might miss; mirrors the old max(1, ...) floor.
    if not val_entries and all_entries:
        val_entries = [all_entries[0]]
        train_entries = all_entries[1:]
    return train_entries, val_entries, shard_metas


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
    pad_short: bool = False,
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
        packed_dir, segment_frames, val_ratio, seed, pad_short=pad_short,
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


# When TokenDataset.bias_vocal_crops is set (train split only), the probability
# that a vocal-ready song's crop is steered to overlap a transcribed word rather
# than chosen uniformly. Uniform crops frequently land on a vocal song's
# instrumental intro/solo/outro — a <vocals> header with zero phonemes that
# starves the lyric cross-attention. The remaining 1-p stays uniform so the model
# still sees some <vocals>-over-instrumental crops (and never overfits the header).
_VOCAL_CROP_BIAS_PROB = 0.9


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
        n_codebooks: int | None = None,
        pad_short: bool = False,
        pad_id: int = 1024,
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
            pad_short=pad_short,
        )
        # Delegate to from_mmap and steal its state into self.
        ds = TokenDataset.from_mmap(
            bundle, split, segment_frames, max_lyric_len, n_codebooks,
            pad_short=pad_short, pad_id=pad_id)
        self.__dict__.update(ds.__dict__)

    @classmethod
    def from_mmap(
        cls,
        bundle: dict,
        split: str,
        segment_frames: int,
        max_lyric_len: int = 256,
        n_codebooks: int | None = None,
        pad_short: bool = False,
        pad_id: int = 1024,
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
        # Model's codebook count (RVQ prefix of the stored depth); None = use all
        # stored codebooks. See __getitem__ for the slice.
        ds.n_codebooks = n_codebooks
        # Full-song padding: keep songs shorter than segment_frames and pad their
        # crop to segment_frames with pad_id (tokens, masked in loss via
        # ignore_index) / zeros (melody). Off = legacy drop-short behavior.
        ds.pad_short = pad_short
        ds.pad_id = pad_id
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
        # Steer train crops toward sung regions; set True on the train split in
        # diskrot/train.py. See _choose_crop_start / _VOCAL_CROP_BIAS_PROB.
        ds.bias_vocal_crops = False
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
            lang_label_to_id, structure_label_to_id, text_to_word_phoneme_groups,
        )

        segs = self._structure.get(name)
        entry = self._lyrics.get(name)
        gender = entry.get("gender") if isinstance(entry, dict) else None
        # v9: detected language (transcribe stores it) -> <lang_*> header marker +
        # the language the lyrics are phonemized in. None -> <unknown_lang> / en.
        language = entry.get("language") if isinstance(entry, dict) else None
        bpm = self._bpm.get(name)
        if entry:
            vocal_id = VOCAL_TOKEN_TO_ID["vocals"]
        elif name in self._instrumental:
            vocal_id = VOCAL_TOKEN_TO_ID["instrumental"]
        else:
            vocal_id = UNKNOWN_VOCALS_ID
        # Compact 6-marker header (no internal word-boundary):
        # BOS <gender> <tempo> <key> <vocals> <lang> <section>.
        ids = [BOS_PHONEME_ID]
        append_unit(ids, [
            gender_label_to_id(gender),
            bpm_to_id(bpm),
            key_label_to_id(self._keys.get(name)),
            vocal_id,
            lang_label_to_id(language),
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
                groups = text_to_word_phoneme_groups(
                    [w["word"] for w in entry["words"]], language=language)
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

    def _choose_crop_start(self, idx: int, T: int) -> int:
        """Pick the crop's start frame in ``[0, T - segment_frames]``.

        Default is uniform-random (the historical behavior). When
        ``bias_vocal_crops`` is set — train split only — a vocal-ready song's
        window is biased to overlap a transcribed word so the crop actually
        carries phonemes: uniform crops routinely land on a vocal song's
        instrumental intro/solo/outro and return a ``<vocals>`` header with zero
        words, starving the lyric cross-attention. Instrumental / never-
        transcribed / wordless songs (and the ``1 - _VOCAL_CROP_BIAS_PROB`` share
        of vocal ones) stay uniform.
        """
        max_start = T - self.segment_frames
        if max_start <= 0:
            return 0
        if self.bias_vocal_crops and random.random() < _VOCAL_CROP_BIAS_PROB:
            entry = self._lyrics.get(self.names[idx])
            words = entry.get("words") if isinstance(entry, dict) else None
            if words:
                w = random.choice(words)
                seg_sec = self.segment_frames / _FRAME_RATE_HZ
                # start_sec range that keeps word w fully inside the crop window,
                # clamped to the valid range; drawn uniformly within it.
                lo = max(0.0, float(w["end"]) - seg_sec)
                hi = min(max_start / _FRAME_RATE_HZ, float(w["start"]))
                if lo <= hi:
                    start = int(round(random.uniform(lo, hi) * _FRAME_RATE_HZ))
                    return max(0, min(start, max_start))
        return random.randint(0, max_start)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str, torch.Tensor, torch.Tensor | None]:
        t = self._get(idx)  # [K, T_full] int16 — torch.Tensor or np.memmap view
        T = t.shape[1]
        start = self._choose_crop_start(idx, T)
        # Slice stored codebooks -> the model's n_codebooks (RVQ prefix). The corpus
        # may be packed at a deeper STORED depth (e.g. SpectroStream 32) than the
        # model trains on (e.g. 24); slicing the mmap view here drops the unused
        # codebooks before collate. ``n_codebooks=None`` -> all stored (DAC, or a
        # model that uses the full stored depth). t[:None] == t[:] in Python.
        crop = t[:self.n_codebooks, start:start + self.segment_frames]
        if not isinstance(crop, torch.Tensor):
            # mmap path: materialize the ~46 KB int16 crop (30s case) so the
            # collate doesn't carry a memmap view across the worker boundary.
            crop = torch.from_numpy(np.ascontiguousarray(crop))
        tokens = crop
        # Full-song padding: a song shorter than the clip yields a short crop; pad
        # the time axis up to segment_frames with pad_id so it stacks with full-
        # length crops. The padded targets are pad_id -> ignored by the loss
        # (ignore_index=pad_id), so the model is never graded on the filler tail.
        pad_n = self.segment_frames - tokens.shape[1]
        if self.pad_short and pad_n > 0:
            tokens = torch.nn.functional.pad(tokens, (0, pad_n), value=self.pad_id)
        tags = self._tags.get(self.names[idx], "")
        # time-aligned lyric phoneme ids for this segment
        start_sec = start / _FRAME_RATE_HZ
        end_sec = (start + self.segment_frames) / _FRAME_RATE_HZ
        lyric_ids = self._get_segment_lyric_ids(self.names[idx], start_sec, end_sec)
        # Co-crop the chroma with the IDENTICAL [start, start+segment_frames] window
        # so the melody lines up with the tokens frame-for-frame. -> [seg, 12] float.
        melody = None
        if self._has_melody:
            mel = np.ascontiguousarray(self._get_mel(idx)[:, start:start + self.segment_frames])
            melody = torch.from_numpy(mel).to(torch.float32).transpose(0, 1).contiguous()
            # Co-pad the chroma to segment_frames with zero (silent) frames so it
            # lines up with the padded tokens; zero chroma is in-distribution (the
            # packer zero-fills missing melody) and the MelodyEncoder handles it.
            if self.pad_short and melody.shape[0] < self.segment_frames:
                melody = torch.nn.functional.pad(
                    melody, (0, 0, 0, self.segment_frames - melody.shape[0]))
        return tokens, tags, torch.tensor(lyric_ids, dtype=torch.long), melody
