"""Sharded streaming packer: per-song .pt files -> mmap-friendly shards.

The packer writes a **sharded, mmap-friendly** layout so the dataset can do
lazy, page-cached random access without loading the entire corpus into RAM:

::

    <cache_dir>/packed/
        packed_index.json                # global registry
        packed_000.bin                   # raw int16 bytes, [K, total_T_in_shard]
        packed_000.json                  # per-shard metadata (offsets, names)
        packed_001.bin
        packed_001.json
        ...

``packed_NNN.bin`` is *raw* int16 -- no pickle header -- so it can be opened
with ``np.memmap(path, dtype=np.int16, mode='r', shape=(K, total_T))`` and the
kernel pages tokens in on demand. Each ``__getitem__`` becomes a small slice
of the mmap; the resident set stays at the active crop window x num_workers,
not the whole 139 GB.

The packer itself is **two-pass and streaming**:

1.  Scan every ``*.pt``, record per-song frame counts. Group into shards of
    ~``DEFAULT_SHARD_TARGET_SONGS`` songs each.
2.  For each shard, open its ``.bin`` once, iterate the songs, and write each
    song's int16 bytes directly at its computed offset. Memory stays bounded
    to one song's tokens at a time -- the previous single-file packer built
    a ``list[Tensor]`` of every song's tokens, which OOMs at scale.

CLI::

    # Local:
    python -m diskrot.pack_cache --cache-dir ./token_cache

    # Modal (run once after tokenize+auto-tag finishes):
    modal run diskrot/modal_pack_cache.py    # (thin Modal wrapper, TBD)
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

# Match diskrot.dataset._DEFAULT_LOAD_WORKERS — torch.load over Modal FUSE
# bottoms out at ~7 files/sec aggregate with 16 threads. Going past 16 has
# diminishing returns because the bottleneck is the volume, not the host.
_DEFAULT_LOAD_WORKERS = 16


# Sharded mmap-friendly binary + JSON sidecars under packed/.
PACKED_DIR = "packed"
SHARD_PREFIX = "packed_"
SHARD_INDEX_NAME = "packed_index.json"
FORMAT_VERSION = 2

DEFAULT_SHARD_TARGET_SONGS = 5_000

# int16 token tensors -> 2 bytes/element. Used to validate that an on-disk
# .bin matches the dimensions its sidecar claims (truncation guard on resume).
_BYTES_PER_ELEM = 2


def _shard_bin_name(shard_id: int) -> str:
    return f"{SHARD_PREFIX}{shard_id:03d}.bin"


def _shard_meta_name(shard_id: int) -> str:
    return f"{SHARD_PREFIX}{shard_id:03d}.json"


def _shard_is_valid(
    out_dir: Path,
    shard_id: int,
    expected_names: list[str],
    n_codebooks: int | None,
) -> dict | None:
    """Return the shard's loaded meta dict iff it is complete-and-valid on disk,
    else None (caller rebuilds).

    A shard is the resume unit. Its .json sidecar is written last during packing
    (the commit marker), so its presence + a size-consistent .bin + matching
    membership means the shard fully landed. Anything else — missing/half-written
    files, truncated .bin, drifted membership — returns None so the shard is
    rebuilt rather than silently trusted.

    ``n_codebooks`` may be None while the spec is still being resolved; the
    codebook-count cross-check is skipped in that case (the meta defines it).
    """
    meta_path = out_dir / _shard_meta_name(shard_id)
    bin_path = out_dir / _shard_bin_name(shard_id)
    # 1. .json exists (commit marker), 2. .bin exists.
    if not meta_path.exists() or not bin_path.exists():
        return None
    # 3. .json parses.
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    # 4. format version.
    if meta.get("format_version") != FORMAT_VERSION:
        return None
    names = meta.get("names")
    offsets = meta.get("offsets")
    total_T = meta.get("total_T")
    meta_k = meta.get("n_codebooks")
    if not isinstance(names, list) or not isinstance(offsets, list):
        return None
    # 5. membership matches (drift guard).
    if names != expected_names:
        return None
    # 6. codebook count matches the resolved spec (skip if not yet resolved).
    if n_codebooks is not None and meta_k != n_codebooks:
        return None
    # 7. offsets shape/consistency.
    if (len(offsets) != len(names) + 1 or offsets[0] != 0
            or offsets[-1] != total_T):
        return None
    # 8. .bin size matches declared dimensions (truncation guard).
    try:
        expected_bytes = int(meta_k) * int(total_T) * _BYTES_PER_ELEM
    except (TypeError, ValueError):
        return None
    if bin_path.stat().st_size != expected_bytes:
        return None
    return meta


def _resolve_n_codebooks(
    out_dir: Path, n_shards: int, verbose: bool
) -> int | None:
    """Resolve the on-disk codebook count without loading any tensors, so a
    resumed pack can validate/skip shards even when shard 0 is skipped.

    Priority: the global index (most authoritative prior-run spec) -> the first
    parseable shard sidecar -> None (cold start; the first rebuilt shard sets it
    from ``tensors[0]`` exactly as before)."""
    index_path = out_dir / SHARD_INDEX_NAME
    try:
        idx = json.loads(index_path.read_text())
        if idx.get("format_version") == FORMAT_VERSION and idx.get("n_codebooks"):
            if verbose:
                print(f"[pack] resolved n_codebooks={idx['n_codebooks']} "
                      f"from existing {SHARD_INDEX_NAME}", flush=True)
            return int(idx["n_codebooks"])
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
        pass
    for shard_id in range(n_shards):
        meta_path = out_dir / _shard_meta_name(shard_id)
        try:
            meta = json.loads(meta_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        if meta.get("format_version") == FORMAT_VERSION and meta.get("n_codebooks"):
            if verbose:
                print(f"[pack] resolved n_codebooks={meta['n_codebooks']} "
                      f"from existing shard {shard_id:03d} meta", flush=True)
            return int(meta["n_codebooks"])
    return None


def pack(
    cache_dir: str | Path,
    out_dir: str | Path | None = None,
    shard_target_songs: int = DEFAULT_SHARD_TARGET_SONGS,
    n_workers: int = _DEFAULT_LOAD_WORKERS,
    verbose: bool = True,
    commit_cb: Callable[[], None] | None = None,
) -> Path:
    """Single-pass per-shard parallel pack of every ``*.pt`` in ``cache_dir``.

    For each shard, parallel-loads its member .pt files (``n_workers`` threads)
    into RAM, computes offsets, opens an ``np.memmap`` preallocated to
    ``[K, total_T_shard]``, blits each song into its slot, writes the sidecar,
    then frees. Single read per file (vs the previous two-pass), RAM-bounded
    to one shard (~2-3 GB at 5k songs of 5-min average), parallel reads so
    Modal-FUSE-bound corpora pack at ~7 files/sec (matches the dataset
    loader's throughput).

    Resumable: shards already complete-and-valid on disk are skipped (see
    ``_shard_is_valid``), so a re-run — e.g. after a Modal worker preemption —
    continues from the first incomplete shard instead of rebuilding everything.
    Each shard's ``.bin``/``.json`` are written via temp-file + ``os.replace``
    (atomic) and the ``.json`` is written last as the per-shard commit marker,
    so an interrupted shard is always detected and rebuilt rather than trusted.

    ``commit_cb`` (optional) is invoked after each shard fully lands and once
    after the final index, so a caller can persist progress incrementally —
    on Modal, pass ``tokens_vol.commit`` so committed shards survive a retry.
    ``pack_cache.py`` stays modal-free; the commit is injected, mirroring
    ``train_run(cfg, ckpt_callback=ckpts_vol.commit)``.

    Returns the path to the output directory. Idempotent: same input set in
    alphabetical order produces identical shard layout.
    """
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir) if out_dir else cache_dir / PACKED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cache_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no .pt files in {cache_dir}")
    if verbose:
        print(f"[pack] {len(files)} files in {cache_dir} "
              f"-> shards of {shard_target_songs} songs, workers={n_workers}",
              flush=True)
    t_total = time.time()

    def _commit() -> None:
        """Persist progress so far (no-op if no callback). A failed commit just
        means the affected shard is redone on the next run — never corruption —
        so we log and continue rather than aborting the whole pack."""
        if commit_cb is None:
            return
        try:
            commit_cb()
        except Exception as e:  # noqa: BLE001 — defensive; see docstring
            if verbose:
                print(f"[pack] commit_cb raised: {e} "
                      f"(affected shard will be redone on retry)", flush=True)

    # Contiguous shard assignment by sorted index. Sorted-by-name discipline
    # is the same one diskrot.dataset uses, so train/val splits computed from
    # the packed names match the loose-file path.
    n = len(files)
    n_shards = (n + shard_target_songs - 1) // shard_target_songs

    def _load_one(path: Path) -> "torch.Tensor | None":
        # A single corrupt/truncated .pt (e.g. a torn write from a preempted
        # tokenize worker) must NOT kill the whole pack. Return None so the
        # caller can drop that song from its shard and keep going.
        try:
            return torch.load(path, weights_only=True, map_location="cpu")
        except Exception as e:  # noqa: BLE001 — any unpickling/IO failure = corrupt
            print(f"[pack] CORRUPT .pt skipped: {path.name} "
                  f"({type(e).__name__}: {str(e)[:100]})", flush=True)
            return None

    # Resolve the codebook count up front so skipped shards (which never load a
    # tensor) can still validate. Priority: existing index -> any intact shard
    # meta -> None (a cold run; the first rebuilt shard sets it from tensors[0]
    # exactly as before). dtype on disk is always int16.
    n_codebooks: int | None = _resolve_n_codebooks(out_dir, n_shards, verbose)
    dtype: torch.dtype | None = torch.int16 if n_codebooks is not None else None
    index_entries = []
    total_songs_done = 0
    n_skipped = 0
    first_rebuilt: int | None = None
    corrupt_files: list[str] = []  # .pt files dropped because torch.load failed

    for shard_id in range(n_shards):
        t_shard = time.time()
        start_i = shard_id * shard_target_songs
        end_i = min(start_i + shard_target_songs, n)
        member_files = files[start_i:end_i]
        expected_names = [f.stem for f in member_files]

        # ---- Resume: skip a shard already complete-and-valid on disk ----
        existing = _shard_is_valid(out_dir, shard_id, expected_names, n_codebooks)
        if existing is not None:
            if n_codebooks is None:
                n_codebooks, dtype = int(existing["n_codebooks"]), torch.int16
            index_entries.append({
                "shard_id": shard_id,
                "n_songs": len(existing["names"]),
                "total_frames": int(existing["total_T"]),
            })
            total_songs_done += len(existing["names"])
            n_skipped += 1
            if verbose:
                size_gb = (out_dir / _shard_bin_name(shard_id)).stat().st_size / 1e9
                print(f"[pack:shard {shard_id:03d}/{n_shards-1}] skip "
                      f"(already valid, {len(existing['names'])} songs, "
                      f"{size_gb:.2f} GB)", flush=True)
            continue
        if first_rebuilt is None:
            first_rebuilt = shard_id

        # ---- Parallel load this shard's tensors into RAM ----
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            raw = list(ex.map(_load_one, member_files))

        # Drop corrupt files (None) and keep member_files / tensors / names
        # aligned to the survivors. expected_names is recomputed from the
        # survivors so the shard meta records exactly what was packed.
        corrupt = [member_files[i].name for i, t in enumerate(raw) if t is None]
        if corrupt:
            corrupt_files.extend(corrupt)
        member_files = [f for f, t in zip(member_files, raw) if t is not None]
        tensors = [t for t in raw if t is not None]
        if not tensors:
            print(f"[pack:shard {shard_id:03d}/{n_shards-1}] all "
                  f"{len(raw)} files corrupt — skipping shard", flush=True)
            continue
        expected_names = [f.stem for f in member_files]

        # Resolve/validate the spec. First rebuilt shard of a cold run sets it;
        # if it was already resolved from disk, cross-check that the corpus's
        # codebook count hasn't changed under an existing pack.
        if n_codebooks is None:
            n_codebooks, dtype = tensors[0].shape[0], tensors[0].dtype
            assert dtype == torch.int16, (
                f"packer expects int16 token tensors; got {dtype}. The tokenize "
                f"pipeline writes int16 -- something upstream changed."
            )
        elif tensors[0].shape[0] != n_codebooks:
            raise ValueError(
                f"corpus codebook count changed: shard {shard_id:03d} has "
                f"{tensors[0].shape[0]} codebooks but the existing pack uses "
                f"{n_codebooks}. Delete {out_dir} and repack from scratch."
            )
        for j, t in enumerate(tensors):
            if t.shape[0] != n_codebooks:
                raise ValueError(
                    f"codebook mismatch at {member_files[j].name}: "
                    f"{t.shape[0]} vs {n_codebooks}"
                )
            if t.dtype != dtype:
                raise ValueError(
                    f"dtype mismatch at {member_files[j].name}: "
                    f"{t.dtype} vs {dtype}"
                )

        # ---- Compute offsets and preallocate the shard's mmap ----
        offsets = [0]
        for t in tensors:
            offsets.append(offsets[-1] + int(t.shape[1]))
        shard_total_T = offsets[-1]
        shard_names = expected_names

        # Write the .bin to a temp path, then atomically rename. A preemption
        # mid-blit leaves only the .tmp (ignored on resume), never a partial
        # packed_NNN.bin that could look valid.
        bin_path = out_dir / _shard_bin_name(shard_id)
        bin_tmp = out_dir / (_shard_bin_name(shard_id) + ".tmp")
        mm = np.memmap(
            bin_tmp, dtype=np.int16, mode="w+",
            shape=(int(n_codebooks), int(shard_total_T)),
        )
        try:
            for j, t in enumerate(tensors):
                mm[:, offsets[j]:offsets[j + 1]] = t.numpy()
            mm.flush()
        finally:
            # Release the mmap handle so the file is fully synced/closeable.
            del mm
        # Free the per-song tensors before the next shard's load fans out.
        del tensors
        os.replace(bin_tmp, bin_path)

        # Sidecar metadata for this shard, written LAST (after the .bin is in
        # place) via temp + atomic rename — its presence is the commit marker
        # that _shard_is_valid keys on.
        meta = {
            "format_version": FORMAT_VERSION,
            "shard_id": shard_id,
            "n_codebooks": int(n_codebooks),
            "dtype": "int16",
            "total_T": int(shard_total_T),
            "offsets": offsets,
            "names": shard_names,
        }
        meta_path = out_dir / _shard_meta_name(shard_id)
        meta_tmp = out_dir / (_shard_meta_name(shard_id) + ".tmp")
        with open(meta_tmp, "w") as f:
            json.dump(meta, f)
        os.replace(meta_tmp, meta_path)
        index_entries.append({
            "shard_id": shard_id,
            "n_songs": len(member_files),
            "total_frames": int(shard_total_T),
        })
        total_songs_done += len(member_files)
        # Persist this completed shard so a later preemption+retry skips it.
        _commit()
        if verbose:
            size_gb = bin_path.stat().st_size / 1e9
            elapsed = time.time() - t_shard
            rate = len(member_files) / max(elapsed, 1e-6)
            eta = (n - total_songs_done) / max(rate, 1e-6)
            print(f"[pack:shard {shard_id:03d}/{n_shards-1}] "
                  f"{len(member_files)} songs, {size_gb:.2f} GB "
                  f"in {elapsed:.1f}s ({rate:.1f}/s, ETA {eta:.0f}s "
                  f"for remaining {n - total_songs_done} songs)", flush=True)

    if verbose:
        if n_skipped == n_shards:
            print(f"[pack] all {n_shards} shards already valid — "
                  f"rewriting index only", flush=True)
        elif n_skipped:
            print(f"[pack] skipped {n_skipped} already-packed shards, "
                  f"rebuilt {n_shards - n_skipped} (resumed from shard "
                  f"{first_rebuilt:03d})", flush=True)
    if corrupt_files:
        print(f"[pack] WARNING: dropped {len(corrupt_files)} corrupt .pt file(s) "
              f"— NOT included in shards. Delete + re-tokenize these to recover "
              f"them (re-running pack will then include them):", flush=True)
        for c in corrupt_files[:50]:
            print(f"  CORRUPT: {c}", flush=True)
        if len(corrupt_files) > 50:
            print(f"  …and {len(corrupt_files) - 50} more", flush=True)

    # Persist the full corrupt-file list to a durable manifest — the console
    # list above is capped at 50 and `modal app logs` scrolls. A cleanup tool
    # (diskrot/modal_clean_corrupt.py) reads this to delete them. Cleared when a
    # run finds no corruption so a stale manifest never lingers.
    manifest = Path(out_dir) / "corrupt_files.json"
    if corrupt_files:
        manifest.write_text(json.dumps(sorted(corrupt_files), indent=2))
        if verbose:
            print(f"[pack] wrote corrupt-file manifest ({len(corrupt_files)} "
                  f"files) -> {manifest}", flush=True)
    elif manifest.exists():
        manifest.unlink()

    # ---- Global index across all shards, written LAST and atomically. ----
    # The consumer treats the index's presence as "layout complete", so it must
    # land only after every shard it references is on disk, and never torn.
    index_payload = {
        "format_version": FORMAT_VERSION,
        "n_codebooks": int(n_codebooks),
        "dtype": "int16",
        "n_shards": n_shards,
        "n_songs_total": len(files),
        "shards": index_entries,
    }
    index_tmp = out_dir / (SHARD_INDEX_NAME + ".tmp")
    with open(index_tmp, "w") as f:
        json.dump(index_payload, f, indent=2)
    os.replace(index_tmp, out_dir / SHARD_INDEX_NAME)
    # Final commit so the index itself is durable.
    _commit()

    if verbose:
        total_gb = sum(
            (out_dir / _shard_bin_name(s["shard_id"])).stat().st_size
            for s in index_entries
        ) / 1e9
        print(f"[pack] wrote {n_shards} shards "
              f"({total_gb:.2f} GB) to {out_dir} in {time.time()-t_total:.1f}s",
              flush=True)
    return out_dir


def load_shard_meta(out_dir: str | Path, shard_id: int) -> dict:
    """Load a shard's JSON sidecar (offsets, names, total_T)."""
    out_dir = Path(out_dir)
    with open(out_dir / _shard_meta_name(shard_id)) as f:
        return json.load(f)


def load_shard_index(out_dir: str | Path) -> dict:
    """Load the top-level index that lists every shard."""
    out_dir = Path(out_dir)
    with open(out_dir / SHARD_INDEX_NAME) as f:
        payload = json.load(f)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unexpected packed index format version {payload.get('format_version')} "
            f"(expected {FORMAT_VERSION})"
        )
    return payload


def open_shard_mmap(out_dir: str | Path, shard_id: int) -> tuple[np.memmap, dict]:
    """Open shard ``shard_id`` as a read-only ``np.memmap`` of shape
    [K, total_T]. Returns (mmap, meta) where ``meta`` is the JSON sidecar."""
    out_dir = Path(out_dir)
    meta = load_shard_meta(out_dir, shard_id)
    mm = np.memmap(
        out_dir / _shard_bin_name(shard_id),
        dtype=np.int16, mode="r",
        shape=(int(meta["n_codebooks"]), int(meta["total_T"])),
    )
    return mm, meta


def iter_all_names(out_dir: str | Path) -> Iterable[tuple[int, int, str, int]]:
    """Yield ``(shard_id, local_idx, name, n_frames)`` for every song across
    all shards, in deterministic shard order. Lets the dataset build a global
    index without opening every ``.bin``."""
    out_dir = Path(out_dir)
    index = load_shard_index(out_dir)
    for shard_entry in index["shards"]:
        shard_id = shard_entry["shard_id"]
        meta = load_shard_meta(out_dir, shard_id)
        offsets = meta["offsets"]
        for local_idx, name in enumerate(meta["names"]):
            n_frames = offsets[local_idx + 1] - offsets[local_idx]
            yield shard_id, local_idx, name, n_frames


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cache-dir", type=str, required=True,
                   help="directory containing the per-song .pt files")
    p.add_argument("--out-dir", type=str, default=None,
                   help="output dir (default: <cache-dir>/packed)")
    p.add_argument("--shard-target-songs", type=int,
                   default=DEFAULT_SHARD_TARGET_SONGS,
                   help=f"approx songs per shard (default {DEFAULT_SHARD_TARGET_SONGS})")
    args = p.parse_args()
    pack(args.cache_dir, args.out_dir, shard_target_songs=args.shard_target_songs)
