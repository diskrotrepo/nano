"""Per-song key estimation from the packed chroma sidecars -> keys.json.

The melody pass already extracts an octave-invariant 12-bin chromagram per song
and the packer folds it into ``packed/packed_NNN.mel.bin`` (see
diskrot.pack_cache), so a per-song key estimate is near-free: average the chroma
over time and correlate against the 24 Krumhansl-Schmuckler key profiles. The
winning rotation/mode becomes the song's ``<key_*>`` header marker in the lyric
stream (model.lyric_encoder.KEY_LABELS is the canonical label set).

Output is a single ``keys.json`` mapping ``{name: {"key": "a_minor"}}`` —
tags.json-scale, loaded sparse at train time (a song without an entry gets
``<unknown_key>``). Songs whose chroma is all-zero (the packer's zero-fill for a
missing .mel.npy) are skipped, never guessed.

Resumable: the output is rewritten atomically every ``flush_every_shards``
shards (and once at the end), and a re-run skips songs already present, so a
kill mid-sweep only re-derives the few un-flushed shards' mean-chroma.

CLI::

    python -m diskrot.key_detect --cache-dir ~/.cache/nano_tokens
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from diskrot.progress import ProgressReporter
from model.lyric_encoder import KEY_LABELS

KEYS_JSON_NAME = "keys.json"

# Krumhansl-Schmuckler tone profiles (Krumhansl & Kessler 1982): perceived
# stability of each pitch class relative to the tonic at index 0.
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

# KEY_LABELS layout contract: [unknown, 12 majors (C..B in semitone order),
# 12 minors]. The profile matrix below indexes labels as 1 + mode*12 + tonic,
# so guard the assumption here rather than importing lyric_encoder internals.
assert KEY_LABELS[1] == "c_major" and KEY_LABELS[13] == "c_minor" and len(KEY_LABELS) == 25


def _profile_matrix() -> np.ndarray:
    """[24, 12] z-normalized KS profiles: rows 0-11 = C..B major, 12-23 = minor.

    Row ``mode*12 + tonic`` is the profile rotated so its tonic sits at chroma
    bin ``tonic`` (chroma_cqt bin 0 = C). Z-normalizing both sides turns the
    Pearson correlation into a dot product.
    """
    rows = []
    for profile in (_KS_MAJOR, _KS_MINOR):
        z = (profile - profile.mean()) / profile.std()
        for tonic in range(12):
            rows.append(np.roll(z, tonic))
    return np.stack(rows)


_PROFILES = _profile_matrix()


def estimate_key(mean_chroma: np.ndarray) -> str | None:
    """Key label for a song's time-averaged chroma, or None when degenerate.

    ``mean_chroma`` is [12] (any float dtype). Returns a canonical
    ``model.lyric_encoder.KEY_LABELS`` entry (never ``unknown_key`` — degenerate
    input returns None so the caller can omit the song instead)."""
    c = np.asarray(mean_chroma, dtype=np.float64)
    if c.shape != (12,) or not np.all(np.isfinite(c)):
        return None
    std = c.std()
    # All-zero (packer zero-fill) or flat chroma carries no key information.
    if c.sum() <= 1e-6 or std <= 1e-6:
        return None
    z = (c - c.mean()) / std
    best = int(np.argmax(_PROFILES @ z))
    mode, tonic = divmod(best, 12)
    return KEY_LABELS[1 + mode * 12 + tonic]


def _mean_chroma_per_song(mel_mm: np.memmap, offsets: list[int], local_idx: int) -> np.ndarray:
    a, b = offsets[local_idx], offsets[local_idx + 1]
    # float16 sidecar -> float64 accumulation; [12, T_song] slice streams off mmap.
    return mel_mm[:, a:b].astype(np.float64).mean(axis=1) if b > a else np.zeros(12)


def _atomic_write_json(payload: dict, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)


def detect_keys(
    cache_dir: str | Path,
    out_path: str | Path | None = None,
    verbose: bool = True,
    commit_cb=None,
    flush_every_shards: int = 10,
) -> Path:
    """Sweep every melody-packed shard under ``<cache_dir>/packed`` and write
    ``<cache_dir>/keys.json``. Returns the output path.

    Shards packed without melody are skipped with a note (their songs simply get
    no entry -> ``<unknown_key>`` at train time). ``commit_cb`` (e.g. a Modal
    volume commit) runs after each atomic rewrite. The growing keys.json is
    rewritten whole every ``flush_every_shards`` shards (plus once at the end),
    not after every shard — at corpus scale the per-shard rewrite is ~O(corpus)
    write/commit amplification, and a kill between flushes only re-derives a few
    shards' worth of mean-chroma (seconds of CPU) on resume."""
    from diskrot.pack_cache import (
        PACKED_DIR, load_shard_index, load_shard_meta, open_shard_mel_mmap,
    )

    cache_dir = Path(cache_dir)
    packed_dir = cache_dir / PACKED_DIR
    out_path = Path(out_path) if out_path is not None else cache_dir / KEYS_JSON_NAME

    keys: dict[str, dict] = {}
    if out_path.exists():
        keys = json.loads(out_path.read_text())
        if verbose:
            print(f"[key] resuming: {len(keys)} songs already in {out_path}", flush=True)

    def flush() -> None:
        _atomic_write_json(keys, out_path)
        if commit_cb is not None:
            commit_cb()

    index = load_shard_index(packed_dir)
    n_done = n_skipped_flat = 0
    shards_since_flush = 0
    dirty = False
    t0 = time.time()
    rep = ProgressReporter(len(index["shards"]), "key_detect", unit="shards") if verbose else None
    for shard_entry in index["shards"]:
        shard_id = shard_entry["shard_id"]
        if rep is not None:
            rep.update(1, extra=f"{len(keys):,} keys")
        meta = load_shard_meta(packed_dir, shard_id)
        if not meta.get("has_melody"):
            if verbose:
                print(f"[key] shard {shard_id:03d}: no melody sidecar — skipped", flush=True)
            continue
        names = meta["names"]
        if all(n in keys for n in names):
            continue  # resume-by-skip: shard fully done in a prior run
        mel_mm, _ = open_shard_mel_mmap(packed_dir, shard_id)
        offsets = meta["offsets"]
        for local_idx, name in enumerate(names):
            if name in keys:
                continue
            label = estimate_key(_mean_chroma_per_song(mel_mm, offsets, local_idx))
            if label is None:
                n_skipped_flat += 1
                continue
            keys[name] = {"key": label}
            n_done += 1
        del mel_mm
        dirty = True
        shards_since_flush += 1
        if shards_since_flush >= flush_every_shards:
            flush()
            shards_since_flush = 0
            dirty = False
        if verbose:
            print(f"[key] shard {shard_id:03d}: {len(keys)} total "
                  f"(+{n_done} new, {n_skipped_flat} flat/zero skipped, "
                  f"{time.time() - t0:.0f}s)", flush=True)

    if rep is not None:
        rep.done(extra=f"{len(keys):,} keys")
    if dirty or not out_path.exists():
        flush()
    if verbose:
        print(f"[key] done: {len(keys)} songs -> {out_path} "
              f"({n_skipped_flat} skipped, {time.time() - t0:.0f}s)", flush=True)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-dir", type=str, required=True,
                   help="directory containing the packed/ layout (and the keys.json output)")
    p.add_argument("--out", type=str, default=None,
                   help="output path (default: <cache-dir>/keys.json)")
    args = p.parse_args()
    detect_keys(args.cache_dir, out_path=args.out)
