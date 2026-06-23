"""Build a small throwaway wave by server-side-copying N mp3s from an existing
wave into a new ``waves/wave_<dst>/`` prefix — the calibration/validation wave
for a v9 pipeline dry-run (see README.v9 A8).

The copy is a server-side R2 ``copy_object`` (no bytes transit Modal), so 1000
objects take seconds and cost nothing in egress. Source objects are left intact.

Same image/secret/env contract as the other R2 scripts (``modal_r2_wavify.py`` /
``modal_r2_rebalance_waves.py``): bucket+endpoint bake into the image from your
shell env, boto3 reads the AWS-style R2 keys from the injected ``r2-creds`` secret.

Usage (from a shell with the R2 env exported)::

    export NANO_AUDIO_BUCKET=nano-audio
    export NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
    modal run diskrot/modal_make_test_wave.py --src-wave base_0 --dst-wave test --n 1000

Then ingest it through the normal orchestrator::

    modal run --detach diskrot/modal_ingest_wave.py --wave-id test

Tear down when done (before the real base_0 run)::

    modal run diskrot/modal_make_test_wave.py --dst-wave test --delete
"""
from __future__ import annotations

import os

import modal

app = modal.App("nano-make-test-wave")

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

SRC_EXT = ".mp3"


def _client():
    import boto3

    endpoint = os.environ["NANO_AUDIO_ENDPOINT"]  # required for R2
    return boto3.client("s3", endpoint_url=endpoint), os.environ["NANO_AUDIO_BUCKET"]


@app.function(image=image, secrets=[modal.Secret.from_name("r2-creds")],
              timeout=60 * 30)
def make_wave(src_wave: str, dst_wave: str, n: int) -> None:
    """Copy the first ``n`` mp3s of ``waves/wave_<src_wave>/`` into
    ``waves/wave_<dst_wave>/`` (server-side; sources untouched). Idempotent —
    already-present destination keys are skipped, so a re-run resumes."""
    s3, bucket = _client()
    src_prefix = f"waves/wave_{src_wave}/"
    dst_prefix = f"waves/wave_{dst_wave}/"

    existing_dst = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=dst_prefix):
        for obj in page.get("Contents", []):
            existing_dst.add(obj["Key"].split("/")[-1])

    picked: list[str] = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"].split("/")[-1]
            if name.endswith(SRC_EXT):
                picked.append(name)
                if len(picked) >= n:
                    break
        if len(picked) >= n:
            break

    if not picked:
        raise SystemExit(f"no {SRC_EXT} objects under {src_prefix} — check --src-wave")

    copied = skipped = 0
    for i, name in enumerate(picked, 1):
        if name in existing_dst:
            skipped += 1
            continue
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": src_prefix + name},
            Key=dst_prefix + name,
        )
        copied += 1
        if i % 100 == 0 or i == len(picked):
            print(f"  {i}/{len(picked)} (copied {copied}, skipped {skipped})", flush=True)
    print(f"done: {dst_prefix} now holds {len(picked)} songs "
          f"({copied} copied, {skipped} already present)", flush=True)


@app.function(image=image, secrets=[modal.Secret.from_name("r2-creds")],
              timeout=60 * 30)
def delete_wave(dst_wave: str) -> None:
    """Delete every object under ``waves/wave_<dst_wave>/`` — teardown for the
    throwaway test wave. (Does NOT touch the source wave.)"""
    s3, bucket = _client()
    dst_prefix = f"waves/wave_{dst_wave}/"
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=dst_prefix):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            n += len(keys)
    print(f"deleted {n} objects under {dst_prefix}", flush=True)


@app.local_entrypoint()
def main(src_wave: str = "base_0", dst_wave: str = "test", n: int = 1000,
         delete: bool = False) -> None:
    if not os.environ.get("NANO_AUDIO_ENDPOINT"):
        raise SystemExit("export NANO_AUDIO_ENDPOINT (and NANO_AUDIO_BUCKET) first")
    if delete:
        delete_wave.remote(dst_wave=dst_wave)
    else:
        make_wave.remote(src_wave=src_wave, dst_wave=dst_wave, n=n)
