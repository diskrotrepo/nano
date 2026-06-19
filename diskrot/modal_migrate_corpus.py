"""One-time: copy the existing nano-corpus mp3s into the object-storage bucket.

Unlimited-scale ingestion reads raw audio from an S3/R2 bucket (no inode cap;
see ``modal_common.corpus_mount``). This migrates the mp3s already on the
``nano-corpus`` Volume into the bucket so they're kept for re-derivation and the
Volume can be reclaimed. Resumable: files already present in the bucket are
skipped, so a re-run finishes an interrupted copy.

Run (R2 needs the endpoint; plain AWS S3 omits it)::

    NANO_AUDIO_BUCKET=nano-audio \
    NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com \
    modal run --detach diskrot/modal_migrate_corpus.py

Afterwards run the prep stages with ``NANO_CORPUS_SOURCE=bucket`` (the bucket is
the unlimited-scale raw-audio source).
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

app = modal.App("nano-migrate-corpus")

image = (
    modal.Image.debian_slim(python_version="3.12").add_local_python_source("diskrot")
)

corpus_vol = modal.Volume.from_name("nano-corpus", create_if_missing=True)


def _bucket_mount():
    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    endpoint = os.environ.get("NANO_AUDIO_ENDPOINT")
    kwargs: dict = {"secret": modal.Secret.from_name("r2-creds"), "read_only": False}
    if endpoint:
        kwargs["bucket_endpoint_url"] = endpoint
    return modal.CloudBucketMount(bucket, **kwargs)


bucket_mount = _bucket_mount()


@app.cls(
    image=image, cpu=4.0, timeout=60 * 60, max_containers=20,
    volumes={"/corpus": corpus_vol, "/bucket": bucket_mount},
)
class Copier:
    @modal.method()
    def copy_batch(self, names: list[str]) -> tuple[int, int]:
        """Copy a batch volume->bucket, skipping files already in the bucket.
        Returns (copied, skipped)."""
        import shutil

        copied = skipped = 0
        for name in names:
            dst = Path("/bucket") / name
            if dst.exists():
                skipped += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path("/corpus") / name, dst)
            copied += 1
        return copied, skipped


@app.function(
    image=image, timeout=24 * 60 * 60,
    volumes={"/corpus": corpus_vol, "/bucket": bucket_mount},
)
def migrate(batch_size: int = 500):
    names = [p.name for p in sorted(Path("/corpus").glob("*.mp3"))]
    print(f"{len(names):,} mp3s on nano-corpus -> bucket", flush=True)
    if not names:
        print("nothing to migrate", flush=True)
        return
    chunks = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
    tot_c = tot_s = 0
    for c, s in Copier().copy_batch.map(chunks, order_outputs=False):
        tot_c += c
        tot_s += s
        if (tot_c + tot_s) % 50_000 < batch_size:
            print(f"  {tot_c + tot_s:,}/{len(names):,} (copied {tot_c:,}, "
                  f"skipped {tot_s:,})", flush=True)
    print(f"done: copied {tot_c:,}, skipped {tot_s:,} (already in bucket)", flush=True)


@app.local_entrypoint()
def main(batch_size: int = 500):
    fc = migrate.spawn(batch_size=batch_size)
    print(f"migration launched (detached) — function call id: {fc.object_id}")
    print("monitor: modal app logs nano-migrate-corpus")
