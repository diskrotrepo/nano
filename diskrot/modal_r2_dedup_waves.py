"""Deduplicate songs across the numeric ``waves/wave_<N>/`` folders in R2.

The v9 waves accumulated ~29k redundant copies (the same basename stored in two
or more waves). This keeps exactly ONE copy of each basename and deletes the rest,
so every basename becomes globally unique -- a prerequisite for merging waves
during rebalance (``modal_r2_rebalance_waves.py``) and avoids training v9 on the
same song twice.

**Keep policy:** keep the copy with the **largest file size** (highest bitrate /
least-truncated), tie-broken by the **lowest wave number**. So byte-identical dups
collapse to the lowest-numbered wave, and same-name/different-bytes clashes keep the
biggest file. A basename's sole surviving copy is never deleted.

Server-side delete only (the kept copy is never touched), **dry-run by default**,
and idempotent/resumable (the keep is always the max-size copy, which is never in
the delete set, so a re-run recomputes the identical decision on whatever remains).
Only touches folders named exactly ``wave_<digits>`` -- ``wave_<N>_<genre>/`` ``.part``
staging dirs are left alone.

    export NANO_AUDIO_BUCKET=nano-audio
    export NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
    modal run --detach diskrot/modal_r2_dedup_waves.py          # dry-run report
    modal run --detach diskrot/modal_r2_dedup_waves.py --apply   # delete redundant copies
"""
import os
import re

import modal

app = modal.App("nano-r2-dedup-waves")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("boto3>=1.34")
    .env({"NANO_AUDIO_BUCKET": os.environ.get("NANO_AUDIO_BUCKET", "nano-audio"),
          "NANO_AUDIO_ENDPOINT": os.environ.get("NANO_AUDIO_ENDPOINT", "")})
)

WAVE_RE = re.compile(r"^wave_(\d+)$")
SRC_EXT = ".mp3"


def _wave_num(folder: str) -> int:
    return int(WAVE_RE.match(folder).group(1))


def _list_meta(s3, bucket: str) -> dict:
    """basename -> list of (wave_folder, key, size)."""
    from collections import defaultdict

    occ: dict = defaultdict(list)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="waves/"):
        for o in page.get("Contents", []):
            parts = o["Key"].split("/")
            if len(parts) != 3 or parts[0] != "waves":
                continue
            folder, base = parts[1], parts[2]
            if WAVE_RE.match(folder) and base.lower().endswith(SRC_EXT):
                occ[base].append((folder, o["Key"], o["Size"]))
    return occ


def _compute_deletes(occ: dict):
    """Return (deletes, per_wave_now, per_wave_del, n_identical, n_clash).

    deletes: list of keys (every copy that is NOT the keeper). Keeper = max size,
    tie-break lowest wave number.
    """
    from collections import Counter

    deletes: list = []
    per_wave_now: Counter = Counter()
    per_wave_del: Counter = Counter()
    n_identical = n_clash = 0
    for base, copies in occ.items():
        for folder, _key, _sz in copies:
            per_wave_now[folder] += 1
        if len(copies) == 1:
            continue
        # keeper: largest size, then lowest wave number, then key for full determinism
        ordered = sorted(copies, key=lambda c: (-c[2], _wave_num(c[0]), c[1]))
        for folder, key, _sz in ordered[1:]:
            deletes.append(key)
            per_wave_del[folder] += 1
        if len({sz for _, _, sz in copies}) == 1:
            n_identical += 1
        else:
            n_clash += 1
    return deletes, per_wave_now, per_wave_del, n_identical, n_clash


@app.function(image=image, secrets=[modal.Secret.from_name("r2-creds")], timeout=60 * 60 * 6)
def dedup(apply: bool = False) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import boto3

    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    s3 = boto3.client("s3", endpoint_url=os.environ.get("NANO_AUDIO_ENDPOINT") or None)

    occ = _list_meta(s3, bucket)
    deletes, now, dele, n_identical, n_clash = _compute_deletes(occ)
    total = sum(len(v) for v in occ.values())

    print("\n=== R2 wave dedup (keep largest copy, tie-break lowest wave) ===", flush=True)
    print(f"  total objects:    {total:,}", flush=True)
    print(f"  unique basenames: {len(occ):,}", flush=True)
    print(f"  to delete:        {len(deletes):,}  "
          f"({n_identical:,} identical groups, {n_clash:,} differing-byte groups)", flush=True)
    print("\n  per wave  (current -> after dedup):", flush=True)
    for f in sorted(now, key=_wave_num):
        print(f"    {f}: {now[f]:,} -> {now[f] - dele.get(f, 0):,}  (-{dele.get(f, 0):,})", flush=True)

    if not apply:
        print("\nDRY RUN -- re-run with --apply to delete the redundant copies "
              "(server-side, keeps one per song).", flush=True)
        return

    def _del(key: str) -> None:
        s3.delete_object(Bucket=bucket, Key=key)  # idempotent: no-op if already gone

    n = 0
    with ThreadPoolExecutor(max_workers=64) as pool:
        for i, _ in enumerate(pool.map(_del, deletes)):
            n += 1
            if (i + 1) % 10_000 == 0:
                print(f"  deleted {i + 1:,}/{len(deletes):,}", flush=True)
    print(f"\ndone: deleted {n:,} redundant copies. Every basename is now unique.", flush=True)
    print("next: modal run --detach diskrot/modal_r2_rebalance_waves.py --apply", flush=True)


@app.local_entrypoint()
def main(apply: bool = False):
    # spawn (not .remote()) for BOTH modes so the report survives a flaky local
    # client; pair with `modal run --detach` and read the report from app logs.
    call = dedup.spawn(apply=apply)
    print(f"dedup {'APPLY' if apply else 'dry-run'} launched (detached); call {call.object_id}.")
    print("watch / read report: modal app logs <ap-...> -f")
