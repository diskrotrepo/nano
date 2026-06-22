"""Analyze song structure (intro/verse/chorus/...) with the allin1 analyzer.

Pipeline: MP3 → allin1.analyze (runs Demucs + a joint beat/segment model) →
functional segments with timestamps → sharded structure dir
(``structure/structure_NNN.json``, keyed by a stable hash of the song stem).

These section labels are injected — by timestamp — into the phoneme lyric stream
at train time (see ``diskrot.dataset._get_segment_lyric_ids``) so the model learns
arrangement. Mirrors ``diskrot.transcribe_lyrics`` exactly (same sharded layout,
atomic writes, resume-by-skip).

Usage:
    python -m diskrot.structure --corpus /path/to/mp3s --out ./structure
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

# Number of shard files structure data is split across. Keyed by a stable hash of
# the song stem (sha1, not builtin hash() which is process-salted) so the same
# song always lands in the same shard. Matches N_LYRIC_SHARDS so the two sidecars
# shard identically.
N_STRUCT_SHARDS = 256

# allin1 emits these as segment labels. ``start``/``end`` are song-boundary
# sentinels (not real sections) and are dropped at load time; the rest are the
# functional sections that become marker tokens. "prechorus" is NOT in this set —
# allin1 can't produce it, so the model never learns it (a user typing
# ``[prechorus]`` folds to <no_section> at inference).
ALLIN1_SECTION_LABELS = frozenset(
    {"intro", "verse", "chorus", "bridge", "outro", "break", "inst", "solo"}
)


def _struct_bucket(stem: str) -> int:
    """Stable shard index in [0, N_STRUCT_SHARDS) for a song stem."""
    return int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) % N_STRUCT_SHARDS


def _sample_keep(stem: str, sample_pct: int) -> bool:
    """Deterministic membership in the sampled subset, in [0, sample_pct) of 100.

    The allin1 section pass is the most expensive (optional) data-prep step, and
    downstream the signal collapses to 8 section labels that degrade gracefully to
    ``<no_section>`` for any song without an entry. So we can run it on a fraction
    of the corpus and let the rest fall back. ``sample_pct >= 100`` keeps all,
    ``<= 0`` keeps none. The ``"sample:"`` salt makes this grid INDEPENDENT of
    ``_struct_bucket``'s shard grid, so the kept set isn't correlated with shard id.
    Stable across processes (sha1, not the salted builtin ``hash()``), so a re-run
    with the same ``sample_pct`` never re-decides membership — resume stays correct.
    """
    if sample_pct >= 100:
        return True
    if sample_pct <= 0:
        return False
    return int(hashlib.sha1(("sample:" + stem).encode()).hexdigest()[:8], 16) % 100 < sample_pct


def _shard_path(structure_dir: str | Path, bucket: int) -> Path:
    return Path(structure_dir) / f"structure_{bucket:03d}.json"


def _atomic_write_json(path: str | Path, obj) -> None:
    """Write JSON atomically (temp file + os.replace); a kill mid-write leaves
    only the ``.tmp`` we never read, never a truncated target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_structure_shards(structure_dir: str | Path) -> dict[str, dict | None]:
    """Read all ``structure_*.json`` shards into one merged dict keyed by stem."""
    structure_dir = Path(structure_dir)
    merged: dict[str, dict | None] = {}
    if not structure_dir.exists():
        return merged
    for shard in sorted(structure_dir.glob("structure_*.json")):
        merged.update(json.loads(shard.read_text()))
    return merged


def _segments_from_result(result) -> list[dict]:
    """Serialize allin1 segments to ``[{"start","end","label"}]``.

    Keeps only known functional section labels (drops ``start``/``end`` sentinels
    and anything unexpected), rounds times to ms, and sorts by start.
    """
    segs = []
    for s in getattr(result, "segments", []) or []:
        label = getattr(s, "label", None)
        if label not in ALLIN1_SECTION_LABELS:
            continue
        segs.append({
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "label": label,
        })
    return sorted(segs, key=lambda s: s["start"])


def _result_to_entry(result) -> dict | None:
    """Serialize one allin1 result to ``{"segments":[...], "bpm":...}`` or None.

    Returns None when allin1 produces no usable sections (rare; e.g. very short
    or pathological audio) so the caller can record it and not retry forever.
    """
    segments = _segments_from_result(result)
    if not segments:
        return None
    out: dict = {"segments": segments}
    bpm = getattr(result, "bpm", None)
    if bpm is not None:
        out["bpm"] = bpm
    return out


def analyze_file(analyze_fn, mp3_path: str | Path, device: str) -> dict | None:
    """Run allin1 on one file. Returns ``{"segments":[...], "bpm":...}`` or None."""
    result = analyze_fn(str(mp3_path), device=device)
    # allin1.analyze returns a single result for a single path input.
    if isinstance(result, list):
        result = result[0]
    return _result_to_entry(result)


def analyze_batch(analyze_fn, mp3_paths: list[str | Path], device: str) -> list[dict | None]:
    """Run allin1 on many files in ONE call. Returns entries parallel to input.

    Per-file output is identical to ``analyze_file`` — allin1 separates, extracts
    spectrograms, and infers per file regardless of batching — but a single call
    pays the per-call setup (demucs subprocess spawn + htdemucs load, plus the
    8-fold harmonix-all ensemble construction) once for the whole batch instead
    of once per song.
    """
    results = analyze_fn([str(p) for p in mp3_paths], device=device)
    if not isinstance(results, list):  # single-path input yields a bare result
        results = [results]
    return [_result_to_entry(r) for r in results]


def _flush_shards(structure_dir: Path, structure: dict, dirty: set[int]) -> None:
    """Atomically rewrite only the shard files whose contents changed."""
    by_bucket: dict[int, dict] = {b: {} for b in dirty}
    for key, val in structure.items():
        b = _struct_bucket(key)
        if b in by_bucket:
            by_bucket[b][key] = val
    for b, contents in by_bucket.items():
        _atomic_write_json(_shard_path(structure_dir, b), contents)


def analyze_corpus(
    corpus_dir: str | Path,
    out_path: str | Path,
    device: str = "cuda",
    flush_callback=None,
    flush_every: int = 10,
    sample_pct: int = 100,
) -> None:
    """Analyze structure for all mp3s in corpus_dir into a sharded structure dir.

    Resumes by skipping any stem already present in the shards, and flushes only
    the shards touched since the last flush (atomic temp+rename), so a kill
    mid-write corrupts at most one shard, never the whole corpus. ``sample_pct``
    (<100) runs only the deterministic ``_sample_keep`` subset (cost reduction;
    the rest fall back to ``<no_section>``).
    """
    corpus_dir = Path(corpus_dir)
    structure_dir = Path(out_path)

    mp3s = sorted(corpus_dir.glob("*.mp3"))
    if not mp3s:
        raise SystemExit(f"No mp3s found in {corpus_dir}")
    if sample_pct < 100:
        n_all = len(mp3s)
        mp3s = [m for m in mp3s if _sample_keep(m.stem, sample_pct)]
        print(f"sampling {sample_pct}%: {len(mp3s)}/{n_all} mp3s kept")
    print(f"found {len(mp3s)} mp3s | device: {device}")

    print("loading allin1...")
    import allin1

    structure = load_structure_shards(structure_dir)
    if structure:
        print(f"loaded {len(structure)} existing entries from {structure_dir}")

    n_done, n_skipped, n_empty, n_failed = 0, 0, 0, 0
    dirty: set[int] = set()
    pbar = tqdm(mp3s, desc="analyzing", unit="file")
    for mp3 in pbar:
        key = mp3.stem
        if key in structure:
            n_skipped += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, empty=n_empty, fail=n_failed)
            continue
        try:
            result = analyze_file(allin1.analyze, mp3, device)
        except Exception as e:
            tqdm.write(f"FAILED {mp3.name}: {e}")
            n_failed += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, empty=n_empty, fail=n_failed)
            continue

        if result is None:
            structure[key] = None  # no usable sections
            n_empty += 1
        else:
            structure[key] = result
            n_done += 1
        dirty.add(_struct_bucket(key))
        pbar.set_postfix(done=n_done, skip=n_skipped, empty=n_empty, fail=n_failed)

        processed = n_done + n_empty
        if processed > 0 and processed % flush_every == 0:
            _flush_shards(structure_dir, structure, dirty)
            dirty.clear()
            if flush_callback is not None:
                flush_callback()

    if dirty:
        _flush_shards(structure_dir, structure, dirty)
    print(f"\nanalyzed: {n_done}  empty: {n_empty}  skipped: {n_skipped}  failed: {n_failed}")
    print(f"saved → {structure_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--out", type=str, default="./structure",
                   help="output directory for sharded structure_NNN.json files")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sample-pct", type=int, default=100,
                   help="run allin1 on only this %% of songs (deterministic stem hash); "
                        "the rest fall back to <no_section>. 100 = full corpus.")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    analyze_corpus(args.corpus, args.out, device=device, sample_pct=args.sample_pct)
