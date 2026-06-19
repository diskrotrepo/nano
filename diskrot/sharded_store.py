"""Hash-sharded JSON key-value store for per-song metadata at scale.

``tags.json`` and ``keys.json`` are single files loaded whole at startup and
rewritten in full on every metadata flush. At millions of songs that single file
is hundreds of MB of JSON — slow to parse on every train launch and O(corpus) to
rewrite per flush. This shards a ``{song_stem: value}`` mapping into 256
hash-keyed files (``<dir>/<prefix>_NNN.json``), the same scheme the lyrics /
structure / phonemes stores already use (``transcribe_lyrics._shard_for``), so a
flush touches only the shards that changed and startup parses a directory of
bounded-size files.

The dataset loaders accept EITHER the legacy single file or a sharded dir, so
this is backward-compatible: convert when you hit the scale that needs it via
``convert_file_to_shards`` (or the ``--shard`` CLI below).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

N_SHARDS = 256


def shard_index(stem: str, n_shards: int = N_SHARDS) -> int:
    """Stable shard in [0, n_shards) for a song stem — same sha1[:8] discipline
    as transcribe_lyrics / structure / phonemes."""
    return int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) % n_shards


def _shard_path(out_dir: Path, prefix: str, shard: int) -> Path:
    return out_dir / f"{prefix}_{shard:03d}.json"


def load_json_shards(store_dir: str | Path, prefix: str) -> dict:
    """Merge all ``<dir>/<prefix>_NNN.json`` shards into one ``{stem: value}``
    dict. Missing shards are simply absent (a partial pass is safe)."""
    store_dir = Path(store_dir)
    merged: dict = {}
    for shard in range(N_SHARDS):
        p = _shard_path(store_dir, prefix, shard)
        if not p.exists():
            continue
        try:
            merged.update(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue  # a torn/half-written shard is skipped, never fatal
    return merged


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)  # atomic; a reader never sees a torn shard


def write_json_shards(
    store_dir: str | Path, prefix: str, mapping: dict,
    only_shards: set[int] | None = None,
) -> None:
    """Write ``{stem: value}`` into 256 hash shards (atomic per shard). Pass
    ``only_shards`` to rewrite just the shards whose membership changed (the
    O(touched) flush); omit it to rewrite every non-empty shard."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, dict] = {}
    for stem, value in mapping.items():
        s = shard_index(stem)
        if only_shards is not None and s not in only_shards:
            continue
        buckets.setdefault(s, {})[stem] = value
    targets = only_shards if only_shards is not None else range(N_SHARDS)
    for shard in targets:
        bucket = buckets.get(shard, {})
        path = _shard_path(store_dir, prefix, shard)
        if not bucket and not path.exists():
            continue  # nothing to write and nothing to clear
        _atomic_write_json(path, bucket)


def convert_file_to_shards(
    json_path: str | Path, out_dir: str | Path, prefix: str,
) -> int:
    """One-time: explode a legacy single ``{stem: value}`` JSON into a sharded
    dir. Returns the number of entries written."""
    raw = json.loads(Path(json_path).read_text())
    write_json_shards(out_dir, prefix, raw)
    return len(raw)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Shard a legacy tags.json / keys.json")
    p.add_argument("--src", required=True, help="path to the single JSON file")
    p.add_argument("--out-dir", required=True, help="output sharded dir")
    p.add_argument("--prefix", required=True, help="shard filename prefix, e.g. 'tags'")
    args = p.parse_args()
    n = convert_file_to_shards(args.src, args.out_dir, args.prefix)
    print(f"wrote {n} entries -> {args.out_dir}/{args.prefix}_NNN.json")
