"""Rebalance the v9 R2 audio waves so no wave exceeds TARGET (default 100k) MP3s.

Caps each over-full ``waves/wave_<N>/`` folder at ``TARGET`` and relocates the
excess: it first tops up any EXISTING under-target wave (lowest wave number first),
then sinks whatever is left into a single NEW overflow wave (``--overflow-wave``).
With the current corpus that means: trim wave_18/20/21/23/24 to 100k, fill wave_19
from 91,649 -> 100,000, and drop the remaining ~85,797 into ``wave_22``.

Like ``modal_r2_wavify.py`` this uses **server-side** S3 ``CopyObject`` (no bytes
leave R2) + delete, is **dry-run by default**, and only touches folders whose name
is exactly ``wave_<digits>`` -- the ``wave_<N>_<genre>/`` ``.part`` staging dirs are
left alone.

It is **resumable**: ``--apply`` first writes a plan manifest to
``waves/.rebalance_plan.json`` and then executes it idempotently, so a kill mid-copy
(this op DELETES source objects) re-runs to the *identical* assignment instead of
re-bucketing a half-moved corpus. The manifest is deleted once the move is fully
clean; pass ``--reset`` to discard a stale one and recompute from scratch.

    export NANO_AUDIO_BUCKET=nano-audio
    export NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
    modal run diskrot/modal_r2_rebalance_waves.py                       # dry-run plan
    modal run --detach diskrot/modal_r2_rebalance_waves.py --apply      # copy + delete
"""
import json
import os
import re

import modal

app = modal.App("nano-r2-rebalance-waves")

# Same image/secret/env contract as modal_r2_wavify.py: bake bucket/endpoint into
# the image, boto3 reads the AWS-style R2 keys from the injected r2-creds secret.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("boto3>=1.34")
    .env(
        {
            "NANO_AUDIO_BUCKET": os.environ.get("NANO_AUDIO_BUCKET", "nano-audio"),
            "NANO_AUDIO_ENDPOINT": os.environ.get("NANO_AUDIO_ENDPOINT", ""),
        }
    )
)

TARGET = 100_000
WAVE_RE = re.compile(r"^wave_(\d+)$")  # exact wave_<digits> -> excludes wave_23_afro etc.
MANIFEST_KEY = "waves/.rebalance_plan.json"
SRC_EXT = ".mp3"


def _wave_num(folder: str) -> int:
    return int(WAVE_RE.match(folder).group(1))


def _list_waves(s3, bucket: str) -> dict:
    """folder -> sorted list of mp3 basenames, for ``waves/wave_<N>/<file>.mp3`` only."""
    from collections import defaultdict

    members: dict = defaultdict(list)
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix="waves/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) != 3 or parts[0] != "waves":  # exactly waves/<folder>/<file>
                continue
            folder, base = parts[1], parts[2]
            if WAVE_RE.match(folder) and base.lower().endswith(SRC_EXT):
                members[folder].append(base)
    for f in members:
        members[f].sort()  # deterministic: lowest basenames stay, tail spills
    return members


def _compute_plan(members: dict, overflow_wave: str, target: int):
    """Return (moves, final_counts, collisions).

    moves: list of {src, dst, base}. Over-target waves spill their sorted tail; the
    spill first fills existing under-target waves (low wave number first), remainder
    goes to ``overflow_wave``.
    """
    waves_sorted = sorted(members, key=_wave_num)
    pool: list = []  # (src_folder, base) for every object above target
    deficits: list = []  # (wave_num, folder, need)
    for f in waves_sorted:
        n = len(members[f])
        if n > target:
            pool.extend((f, b) for b in members[f][target:])
        elif n < target and f != overflow_wave:
            deficits.append((_wave_num(f), f, target - n))
    deficits.sort()

    moves: list = []
    pi = 0
    for _, f, need in deficits:
        for _ in range(min(need, len(pool) - pi)):
            src_f, b = pool[pi]
            pi += 1
            moves.append({"src": f"waves/{src_f}/{b}", "dst": f"waves/{f}/{b}", "base": b})
    while pi < len(pool):
        src_f, b = pool[pi]
        pi += 1
        moves.append({"src": f"waves/{src_f}/{b}", "dst": f"waves/{overflow_wave}/{b}", "base": b})

    # final counts (start from current, apply the moves)
    final = {f: len(members[f]) for f in members}
    for m in moves:
        sf, df = m["src"].split("/")[1], m["dst"].split("/")[1]
        final[sf] = final.get(sf, 0) - 1
        final[df] = final.get(df, 0) + 1

    # collision guard: a moved basename must not already exist in its destination,
    # and no two moves may target the same destination key (would silently overwrite).
    collisions: list = []
    existing = {f: set(bs) for f, bs in members.items()}
    seen_dst: set = set()
    for m in moves:
        df = m["dst"].split("/")[1]
        if m["base"] in existing.get(df, set()):
            collisions.append(f"{m['base']} already in {df}")
        if m["dst"] in seen_dst:
            collisions.append(f"duplicate dst {m['dst']}")
        seen_dst.add(m["dst"])
    return moves, final, collisions


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("r2-creds")],
    timeout=60 * 60 * 6,
)
def rebalance(apply: bool = False, reset: bool = False,
              overflow_wave: str = "wave_22", target: int = TARGET) -> None:
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from botocore.exceptions import ClientError

    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    endpoint = os.environ.get("NANO_AUDIO_ENDPOINT") or None
    s3 = boto3.client("s3", endpoint_url=endpoint)

    def _load_manifest():
        try:
            body = s3.get_object(Bucket=bucket, Key=MANIFEST_KEY)["Body"].read()
            return json.loads(body)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NoSuchKeyError"):
                return None
            raise

    if reset:
        try:
            s3.delete_object(Bucket=bucket, Key=MANIFEST_KEY)
            print(f"reset: deleted stale {MANIFEST_KEY}", flush=True)
        except ClientError:
            pass

    manifest = None if reset else _load_manifest()

    if manifest is None:
        members = _list_waves(s3, bucket)
        moves, final, collisions = _compute_plan(members, overflow_wave, target)
        print(f"\n=== R2 wave rebalance (target {target:,}/wave, overflow -> {overflow_wave}) ===",
              flush=True)
        print("  current -> final:", flush=True)
        for f in sorted(set(members) | {overflow_wave}, key=_wave_num):
            cur = len(members.get(f, []))
            fin = final.get(f, 0)
            flag = "  <-- NEW" if cur == 0 and fin > 0 else ("  <-- still >target" if fin > target else "")
            print(f"    {f}: {cur:,} -> {fin:,}{flag}", flush=True)
        print(f"  moves: {len(moves):,} (server-side copy + delete source)", flush=True)
        if collisions:
            print(f"\n  !! {len(collisions)} COLLISION(S) -- refusing to proceed:", flush=True)
            for c in collisions[:20]:
                print(f"     {c}", flush=True)
            return
        overflow_final = final.get(overflow_wave, 0)
        if overflow_final > target:
            print(f"\n  !! {overflow_wave} would hold {overflow_final:,} > target {target:,} -- "
                  f"need more than one overflow wave; aborting.", flush=True)
            return
    else:
        moves = manifest["moves"]
        print(f"\n=== resuming from manifest {MANIFEST_KEY}: {len(moves):,} planned moves ===",
              flush=True)

    if not apply:
        pending = " (a plan manifest already exists -- resume in progress)" if manifest else ""
        print(f"\nDRY RUN{pending} -- re-run with --apply to copy+delete (server-side).",
              flush=True)
        return

    # persist the plan BEFORE moving so a resume reuses the identical assignment
    if manifest is None:
        s3.put_object(Bucket=bucket, Key=MANIFEST_KEY,
                      Body=json.dumps({"target": target, "overflow_wave": overflow_wave,
                                       "moves": moves}).encode())
        print(f"  wrote plan manifest -> {MANIFEST_KEY}", flush=True)

    # current membership (base -> folders) for idempotent skip / mid-move recovery
    loc: dict = defaultdict(set)
    for f, bs in _list_waves(s3, bucket).items():
        for b in bs:
            loc[b].add(f)

    def _move(m: dict) -> str:
        src, dst, base = m["src"], m["dst"], m["base"]
        src_f, dst_f = src.split("/")[1], dst.split("/")[1]
        here = loc.get(base, set())
        if dst_f in here and src_f not in here:
            return "skip"          # already moved
        if dst_f in here and src_f in here:
            s3.delete_object(Bucket=bucket, Key=src)  # copy done, finish the delete
            return "done"
        if src_f in here:
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src}, Key=dst)
            s3.delete_object(Bucket=bucket, Key=src)
            return "done"
        return "missing"

    n_done = n_skip = n_missing = 0
    with ThreadPoolExecutor(max_workers=64) as pool:
        for i, r in enumerate(pool.map(_move, moves)):
            if r == "done":
                n_done += 1
            elif r == "skip":
                n_skip += 1
            else:
                n_missing += 1
            if (i + 1) % 20_000 == 0:
                print(f"  {i + 1:,}/{len(moves):,}  (moved {n_done:,}, skipped {n_skip:,}, "
                      f"missing {n_missing:,})", flush=True)

    print(f"\ndone: moved {n_done:,}, skipped {n_skip:,}, missing {n_missing:,}.", flush=True)
    if n_missing == 0:
        s3.delete_object(Bucket=bucket, Key=MANIFEST_KEY)
        print(f"  clean -- removed {MANIFEST_KEY}. Every wave is now <= {target:,}.", flush=True)
    else:
        print(f"  {n_missing:,} planned source(s) not found -- kept {MANIFEST_KEY} for inspection.",
              flush=True)


@app.local_entrypoint()
def main(apply: bool = False, reset: bool = False, overflow_wave: str = "wave_22"):
    if apply:
        rebalance.spawn(apply=True, reset=reset, overflow_wave=overflow_wave)
        print("rebalance launched (detached). watch: modal app logs <ap-...> -f")
    else:
        rebalance.remote(apply=False, reset=reset, overflow_wave=overflow_wave)
