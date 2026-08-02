"""Repartition ALL R2 audio into uniform ``waves/wave_base_<N>/`` folders of 100k.

So the v9 pipeline can treat the ENTIRE corpus as waves — one uniform ``--wave-id``
flow. This is a full, deterministic **re-partition** (not just a flat->wave sweep):

  * It considers EVERY ``*.mp3`` in the bucket — the flat top-level objects AND
    everything already under ``waves/`` (existing ``wave_base_*`` plus any other
    wave folder, e.g. a ``wave_missing/`` gap-fill). Nothing is left behind.
  * Files are sorted by basename and assigned ``index // wave_size`` → so wave 0
    holds the first 100k basenames, wave 1 the next 100k, etc. Every wave ends up
    with exactly ``wave_size`` files (the last wave is the remainder).
  * It is a **move**: each misplaced file is copied (server-side ``CopyObject`` —
    no bytes leave R2) to its canonical ``waves/wave_base_<N>/<name>.mp3`` and the
    old object is then deleted. Files already at their canonical spot are skipped.

Because the assignment is deterministic, this is idempotent + resumable: a second
run (after one completes) moves nothing. It also normalizes existing waves —
over-full waves shed their tail, partial waves get filled, stragglers in the
wrong wave get relocated.

Duplicate basenames (same stem, different keys) are NOT overwritten: one canonical
copy is kept and the extras are left in place and reported (let ``r2-dedup`` handle
them). The pipeline keys by stem, so duplicates can't share a wave slot anyway.

Assumes the audio is in the R2 bucket. Set the same env the corpus mount uses:

    export NANO_AUDIO_BUCKET=nano-audio
    export NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
    modal run diskrot/modal_r2_wavify.py                      # dry-run plan
    modal run --detach diskrot/modal_r2_wavify.py --apply     # move (server-side)

After this, run the per-wave v9 pipeline for ``wave_base_0..N`` (see README.v9.md).
"""
import os

import modal

app = modal.App("nano-r2-wavify")

# Bake the bucket/endpoint (read from your shell at `modal run`) into the image so
# the remote fn sees them; r2-creds (AWS-style keys) is the same secret the corpus
# CloudBucketMount uses, and boto3 reads the creds from the injected env.
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

WAVE_PREFIX = "waves/wave_base_"
SRC_EXT = ".mp3"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("r2-creds")],
    timeout=60 * 60 * 6,
)
def wavify(apply: bool = False, wave_size: int = 100_000) -> None:
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import boto3
    from botocore.config import Config

    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    endpoint = os.environ.get("NANO_AUDIO_ENDPOINT") or None
    # max_pool_connections >= the 64 mover threads or they starve waiting for a
    # connection (the original cause of the read timeouts); adaptive retries +
    # a longer read_timeout ride out R2's transient slow CopyObject responses.
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            read_timeout=120,
            connect_timeout=30,
            max_pool_connections=128,
        ),
    )

    # 1. list EVERY *.mp3 anywhere in the bucket (flat + under waves/*).
    by_base: dict[str, list[str]] = defaultdict(list)
    n_objs = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.lower().endswith(SRC_EXT):
                by_base[k.rsplit("/", 1)[-1]].append(k)
                n_objs += 1
    for keys in by_base.values():
        keys.sort()  # deterministic representative = first key

    # 2. deterministic canonical order = sorted basename -> wave = index // wave_size.
    bases = sorted(by_base)
    n = len(bases)
    n_waves = (n + wave_size - 1) // wave_size if n else 0
    n_dupes = n_objs - n

    # 3. classify every basename: already placed / needs move (flat vs wrong-wave).
    moves: list[tuple[str, str]] = []
    per_wave: Counter = Counter()
    already = move_from_flat = move_from_wave = 0
    for i, base in enumerate(bases):
        w = i // wave_size
        per_wave[w] += 1
        target = f"{WAVE_PREFIX}{w}/{base}"
        copies = by_base[base]
        if target in copies:        # canonical copy already in the right place
            already += 1
            continue
        rep = copies[0]             # move the first-sorted copy to its canonical wave
        moves.append((rep, target))
        if "/" in rep:
            move_from_wave += 1
        else:
            move_from_flat += 1

    print(f"\n=== R2 wavify (repartition): {n:,} unique .mp3 -> {n_waves} waves "
          f"of {wave_size:,} ===", flush=True)
    print(f"  total objects: {n_objs:,}  (duplicate basenames left in place: "
          f"{n_dupes:,})", flush=True)
    for w, c in sorted(per_wave.items()):
        print(f"  {WAVE_PREFIX}{w}: {c:,}", flush=True)
    print(f"  already canonical: {already:,}   to move: {len(moves):,} "
          f"(from flat {move_from_flat:,}, from wrong wave {move_from_wave:,})",
          flush=True)
    if not apply:
        print("\nDRY RUN — re-run with --apply to MOVE (server-side copy + delete "
              "old). Duplicates are never overwritten; extras stay put.", flush=True)
        return

    import random
    import time

    def _move(item: tuple[str, str]) -> str:
        src_key, target = item
        for attempt in range(8):  # ride out transient R2/network hiccups
            try:
                s3.copy_object(Bucket=bucket,
                               CopySource={"Bucket": bucket, "Key": src_key},
                               Key=target)
                s3.delete_object(Bucket=bucket, Key=src_key)  # move = copy then drop old
                return "moved"
            except Exception as e:  # NEVER let one move crash the whole repartition
                if attempt == 7:
                    return f"FAILED\t{e}"
                time.sleep(min(2 ** attempt, 30) + random.random())
        return "FAILED"

    def _run(items: list[tuple[str, str]], label: str) -> list[tuple[str, str]]:
        done = 0
        failed: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=64) as pool:
            for i, (item, r) in enumerate(zip(items, pool.map(_move, items))):
                if r == "moved":
                    done += 1
                else:
                    failed.append(item)
                if (i + 1) % 20_000 == 0:
                    print(f"  [{label}] {i + 1:,}/{len(items):,}  "
                          f"({done:,} ok, {len(failed):,} failed)", flush=True)
        return failed

    failed = _run(moves, "pass1")
    if failed:  # one more sweep so a transient blip doesn't force a manual re-run
        print(f"\n{len(failed):,} moves failed pass1 — retrying once more...",
              flush=True)
        failed = _run(failed, "pass2")
    n_done = len(moves) - len(failed)
    print(f"\ndone: moved {n_done:,} into canonical waves "
          f"({already:,} already placed, {n_dupes:,} duplicate extras left).",
          flush=True)
    if failed:
        print(f"WARNING: {len(failed):,} moves still failed after retry — re-run "
              f"(idempotent) to finish them. First few:", flush=True)
        for src, tgt in failed[:10]:
            print(f"  FAILED {src} -> {tgt}", flush=True)
    print("next: run the per-wave v9 pipeline for each wave_base_N (README.v9.md).",
          flush=True)


@app.local_entrypoint()
def main(apply: bool = False, wave_size: int = 100_000):
    if apply:
        # spawn + --detach so a long repartition survives the terminal closing.
        wavify.spawn(apply=True, wave_size=wave_size)
        print("wavify launched (detached). watch: modal app logs <ap-...> -f")
    else:
        wavify.remote(apply=False, wave_size=wave_size)  # dry-run blocks + prints plan
