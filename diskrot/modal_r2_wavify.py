"""Reorganize the flat (non-wave) R2 audio into ``waves/wave_base_<N>/`` folders.

So the v9 pipeline can treat the ENTIRE corpus as waves — one uniform ``--wave-id``
flow, non-destructive to the source audio (it stays in R2, just re-prefixed). Uses
**server-side** S3 ``CopyObject`` (no bytes leave R2), assigns each flat ``*.mp3``
to a wave by sorted index // ``wave_size`` (deterministic + resumable: a re-run
skips objects already under ``wave_base_*``). Dry-run by default.

Assumes the flat audio is in the R2 bucket (``NANO_CORPUS_SOURCE=bucket`` world).
Set the same env the corpus mount uses:

    export NANO_AUDIO_BUCKET=nano-audio
    export NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
    modal run diskrot/modal_r2_wavify.py                              # dry-run plan
    modal run --detach diskrot/modal_r2_wavify.py --apply             # copy (keep sources)
    modal run --detach diskrot/modal_r2_wavify.py --apply --delete-source  # copy + delete flat

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
def wavify(apply: bool = False, delete_source: bool = False, wave_size: int = 100_000) -> None:
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    import boto3

    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    endpoint = os.environ.get("NANO_AUDIO_ENDPOINT") or None
    s3 = boto3.client("s3", endpoint_url=endpoint)

    # 1. list flat source keys (*.mp3 NOT already under waves/) + already-copied basenames
    src: list[str] = []
    done: set[str] = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.startswith(WAVE_PREFIX):
                done.add(k.rsplit("/", 1)[-1])
            elif not k.startswith("waves/") and k.lower().endswith(SRC_EXT):
                src.append(k)
    src.sort()  # deterministic wave assignment (sorted index // wave_size)
    n = len(src)
    n_waves = (n + wave_size - 1) // wave_size if n else 0
    plan = [(k, i // wave_size, k.rsplit("/", 1)[-1]) for i, k in enumerate(src)]

    print(f"\n=== R2 wavify: {n:,} flat .mp3 -> {n_waves} waves of {wave_size:,} "
          f"(already under {WAVE_PREFIX}*: {len(done):,}) ===", flush=True)
    for w, c in sorted(Counter(w for _, w, _ in plan).items()):
        print(f"  {WAVE_PREFIX}{w}: {c:,}", flush=True)
    if not apply:
        print("\nDRY RUN — re-run with --apply to copy (server-side). "
              "Add --delete-source to also remove the flat originals.", flush=True)
        return

    def _copy(item: tuple[str, int, str]) -> str:
        src_key, w, base = item
        if base in done:
            return "skip"
        s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key},
                       Key=f"{WAVE_PREFIX}{w}/{base}")
        if delete_source:
            s3.delete_object(Bucket=bucket, Key=src_key)
        return "done"

    n_done = n_skip = n_err = 0
    with ThreadPoolExecutor(max_workers=64) as pool:
        for i, r in enumerate(pool.map(_copy, plan)):
            if r == "done":
                n_done += 1
            else:
                n_skip += 1
            if (i + 1) % 20_000 == 0:
                print(f"  {i + 1:,}/{n:,}  (copied {n_done:,}, skipped {n_skip:,})", flush=True)
    print(f"\ndone: copied {n_done:,}, skipped {n_skip:,}"
          f"{', deleted flat sources' if delete_source else ''}.", flush=True)
    print("next: run the per-wave v9 pipeline for each wave_base_N (README.v9.md).", flush=True)


@app.local_entrypoint()
def main(apply: bool = False, delete_source: bool = False):
    if apply:
        # spawn + --detach so a long copy survives the terminal closing.
        wavify.spawn(apply=True, delete_source=delete_source)
        print("wavify launched (detached). watch: modal app logs <ap-...> -f")
    else:
        wavify.remote(apply=False)  # dry-run blocks inline + prints the plan
