"""Read-only validation of the nano-stems sidecars produced by ``modal_stems.py``.

The ``/addstem`` stem stage writes one ``<name>.stems.npy`` per (sampled) song —
a ``[len(STEM_TYPES), depth, T]`` int16 array, stems in ``STEM_TYPES`` order
(drums/bass/vocals/other), force-aligned to the song's ``.pt`` token frame count.
This script audits what actually landed on the **nano-stems** volume:

COVERAGE (per wave subdir found on /stems):
  - #.stems.npy vs #.pt tokenized → the realized sample fraction
  - with --sample-pct N: the PRECISE finished check — of the songs that SHOULD
    have stems (the deterministic _sample_keep subset, re-derived here so we don't
    import the Modal module), how many are MISSING a sidecar, and how many stem
    sidecars are ORPHANS (no matching .pt left).

CONTENT (a random sample of sidecars, --sample files):
  - shape == [4, depth, T], dtype int16
  - T == the matching .pt frame count  (the train==inference align contract)
  - value range in [0, VOCAB_SIZE]; pad (==VOCAB_SIZE) fraction overall + per stem
  - per-stem liveness (distinct cb0 codes; a fully-pad/constant stem is flagged)

Read-only — never writes or deletes.

    modal run scripts/validate_stems.py                          # all subdirs, ratio only
    modal run scripts/validate_stems.py --wave-id 0 --sample-pct 50   # precise finished check
    modal run scripts/validate_stems.py --sample 400             # deeper content sample
"""

import hashlib
import os
import random
from collections import Counter
from pathlib import Path

import modal

app = modal.App("nano-validate-stems")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy>=1.26")
    .add_local_python_source("model", "diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens")
stems_vol = modal.Volume.from_name("nano-stems", create_if_missing=True)

_STEM_EXT = ".stems.npy"  # must match diskrot.modal_stems._STEM_EXT


def _sample_keep(stem: str, sample_pct: int) -> bool:
    """Re-implementation of modal_stems._sample_keep (kept in sync deliberately:
    importing modal_stems would build its image objects + call corpus_mount() at
    import, which needs R2 secrets we don't want here). Same sha1 + 'stem-sample:'
    salt → identical membership."""
    if sample_pct >= 100:
        return True
    if sample_pct <= 0:
        return False
    return int(hashlib.sha1(("stem-sample:" + stem).encode()).hexdigest()[:8], 16) % 100 < sample_pct


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol, "/stems": stems_vol},
    timeout=2 * 60 * 60,
)
def validate(wave_id: str = "", sample: int = 200, sample_pct: int = 100, seed: int = 0):
    import numpy as np
    import torch

    try:
        from model.stem_encoder import STEM_TYPES
    except Exception:  # noqa: BLE001 — fall back to the documented order (see diskrot/stems.py)
        STEM_TYPES = ("drums", "bass", "vocals", "other")
    try:
        from model.codec import SpectroStreamCodec
        VOCAB_SIZE = SpectroStreamCodec.VOCAB_SIZE
    except Exception:  # noqa: BLE001
        VOCAB_SIZE = 1024
    PAD = VOCAB_SIZE
    n_stems_expected = len(STEM_TYPES)

    tokens_vol.reload()
    stems_vol.reload()

    sub = f"waves/wave_{wave_id}" if wave_id else ""
    stems_root = Path("/stems") / sub if sub else Path("/stems")
    tokens_root = Path("/tokens") / sub if sub else Path("/tokens")

    # All stem sidecars, grouped by their subdir (relative to /stems) so a flat or
    # multi-wave volume both work without asking the caller for the layout.
    all_stem_files = [p for p in stems_root.rglob(f"*{_STEM_EXT}")]
    if not all_stem_files:
        print(f"No {_STEM_EXT} files under {stems_root} — nothing to validate.")
        return
    by_dir: dict[Path, list[Path]] = {}
    for p in all_stem_files:
        by_dir.setdefault(p.parent, []).append(p)

    print(f"Found {len(all_stem_files)} stem sidecars across {len(by_dir)} dir(s)")
    print(f"STEM_TYPES={list(STEM_TYPES)}  VOCAB_SIZE={VOCAB_SIZE}  pad_id={PAD}")
    if sample_pct < 100:
        print(f"sample_pct={sample_pct} → checking the deterministic _sample_keep subset")
    print("=" * 72)

    # ---- COVERAGE ----
    total_done = total_pt = total_expected = total_missing = total_orphan = 0
    for d in sorted(by_dir):
        rel = d.relative_to("/stems")
        pt_dir = Path("/tokens") / rel
        done = {p.name[: -len(_STEM_EXT)] for p in by_dir[d]}
        pt_names = {p.stem for p in pt_dir.glob("*.pt")} if pt_dir.exists() else set()
        orphans = done - pt_names  # stem sidecar with no matching .pt
        expected = {s for s in pt_names if _sample_keep(s, sample_pct)}
        missing = expected - done  # should-have-stems but don't
        total_done += len(done)
        total_pt += len(pt_names)
        total_expected += len(expected)
        total_missing += len(missing)
        total_orphan += len(orphans)
        ratio = (len(done) / len(pt_names) * 100) if pt_names else 0.0
        print(f"[{rel}]  stems={len(done):>7}  tokenized={len(pt_names):>7}  "
              f"realized={ratio:5.1f}%  expected={len(expected):>7}  "
              f"MISSING={len(missing):>6}  orphan={len(orphans):>5}")
        for nm in sorted(missing)[:5]:
            print(f"    missing: {nm}")
        for nm in sorted(orphans)[:5]:
            print(f"    orphan : {nm}")

    print("-" * 72)
    print(f"TOTAL  stems={total_done}  tokenized={total_pt}  "
          f"expected@{sample_pct}%={total_expected}  "
          f"MISSING={total_missing}  orphan={total_orphan}")
    print("=" * 72)

    # ---- CONTENT (random sample) ----
    rng = random.Random(seed)
    pick = all_stem_files if len(all_stem_files) <= sample else rng.sample(all_stem_files, sample)
    print(f"Deep-checking {len(pick)} sidecars...")

    shape_bad = []          # wrong rank / stem count
    dtype_bad = []          # not int16
    align_bad = []          # T != .pt frame count
    range_bad = []          # value outside [0, VOCAB_SIZE]
    no_pt = 0               # couldn't find the .pt to compare frames
    depth_hist = Counter()
    pad_fracs = []          # overall pad fraction per file
    dead_stems = Counter()  # stem name -> #files where that stem is all-pad/constant
    per_stem_pad = {name: [] for name in STEM_TYPES}

    for p in pick:
        try:
            arr = np.load(p)
        except Exception as e:  # noqa: BLE001
            shape_bad.append((p.name, f"load failed: {type(e).__name__}: {str(e)[:60]}"))
            continue
        if arr.ndim != 3 or arr.shape[0] != n_stems_expected:
            shape_bad.append((p.name, f"shape={arr.shape}"))
            continue
        if arr.dtype != np.int16:
            dtype_bad.append((p.name, str(arr.dtype)))
        depth_hist[arr.shape[1]] += 1
        T = arr.shape[2]

        # alignment vs the matching .pt
        rel = p.relative_to("/stems")
        name = p.name[: -len(_STEM_EXT)]
        pt = Path("/tokens") / rel.parent / f"{name}.pt"
        if pt.exists():
            try:
                toks = torch.load(pt, weights_only=True, map_location="cpu")
                if int(toks.shape[1]) != T:
                    align_bad.append((p.name, f"stem T={T} vs pt T={int(toks.shape[1])}"))
            except Exception:  # noqa: BLE001
                no_pt += 1
        else:
            no_pt += 1

        amin, amax = int(arr.min()), int(arr.max())
        if amin < 0 or amax > VOCAB_SIZE:
            range_bad.append((p.name, f"[{amin},{amax}]"))
        pad_fracs.append(float((arr == PAD).mean()))

        # per-stem pad fraction + liveness (cb0 distinct non-pad codes)
        for si, sname in enumerate(STEM_TYPES):
            s = arr[si]
            per_stem_pad[sname].append(float((s == PAD).mean()))
            cb0 = s[0]
            distinct = np.unique(cb0[cb0 != PAD])
            if distinct.size <= 1:  # all pad, or one constant code = dead/degenerate
                dead_stems[sname] += 1

    n = len(pick)
    print("\n--- shape / dtype / alignment ---")
    print(f"depth (shape[1]) histogram: {dict(depth_hist)}")
    print(f"wrong shape/rank : {len(shape_bad)}")
    print(f"wrong dtype      : {len(dtype_bad)} (want int16)")
    print(f"FRAME MISALIGNED : {len(align_bad)}  (stem T != .pt T — train/infer contract)")
    print(f"value out of range: {len(range_bad)}  (want [0,{VOCAB_SIZE}])")
    print(f".pt not found for : {no_pt}")
    for label, bad in (("shape", shape_bad), ("dtype", dtype_bad),
                       ("align", align_bad), ("range", range_bad)):
        for nm, info in bad[:8]:
            print(f"    {label}: {nm}  {info}")

    if pad_fracs:
        pads = sorted(pad_fracs)
        print("\n--- pad fraction (tail padding from short stems) ---")
        print(f"overall pad: mean={sum(pads)/len(pads):.3f}  "
              f"median={pads[len(pads)//2]:.3f}  max={pads[-1]:.3f}")
        print("per-stem mean pad fraction:")
        for sname in STEM_TYPES:
            v = per_stem_pad[sname]
            mean = sum(v) / len(v) if v else 0.0
            print(f"    {sname:>7}: {mean:.3f}   dead/degenerate in {dead_stems[sname]}/{n} files")

    # ---- VERDICT ----
    print("\n" + "=" * 72)
    hard_fail = (total_missing > 0 and sample_pct <= 100 and total_expected > 0) \
        or shape_bad or dtype_bad or align_bad or range_bad
    if not hard_fail:
        print("PASS — sidecars are well-formed, frame-aligned, and coverage is complete.")
    else:
        print("ISSUES FOUND:")
        if total_missing:
            print(f"  - {total_missing} sampled songs are MISSING a stem sidecar "
                  f"(stage not finished, or some failed) — re-run modal_stems to backfill.")
        if align_bad:
            print(f"  - {len(align_bad)} sidecars are FRAME-MISALIGNED — these would "
                  f"corrupt the co-crop; investigate before pack.")
        if shape_bad or dtype_bad or range_bad:
            print(f"  - {len(shape_bad)} shape, {len(dtype_bad)} dtype, "
                  f"{len(range_bad)} value-range anomalies.")
    print("=" * 72)


@app.local_entrypoint()
def main(wave_id: str = "", sample: int = 200, sample_pct: int = 100, seed: int = 0):
    validate.remote(wave_id=wave_id, sample=sample, sample_pct=sample_pct, seed=seed)
