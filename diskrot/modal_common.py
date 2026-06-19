"""Shared Modal helpers for the data-prep entrypoints (tokenize / melody /
auto_tag / transcribe / structure).

``corpus_mount()`` decides where raw audio is read from. By default it's the
``nano-corpus`` Modal Volume (the legacy path). For unlimited-scale wave
ingestion, set ``NANO_CORPUS_SOURCE=bucket`` and it mounts object storage
instead (no inode cap), e.g.::

    NANO_CORPUS_SOURCE=bucket \
    NANO_AUDIO_BUCKET=nano-audio \
    NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com \
    modal run --detach diskrot/modal_tokenize.py --wave-id 17

The bucket branch references the ``r2-creds`` Modal secret (AWS-style keys), so
volume-only users never need it. ``NANO_AUDIO_ENDPOINT`` is required for
Cloudflare R2 / any S3-compatible store; omit it for real AWS S3.
"""
from __future__ import annotations

import os

import modal


def corpus_mount(read_only: bool = True):
    """Return the /corpus mount: the nano-corpus Volume (default) or a
    CloudBucketMount when NANO_CORPUS_SOURCE=bucket. The decision is made
    locally at `modal run` time, so the env vars are read from your shell."""
    if os.environ.get("NANO_CORPUS_SOURCE", "volume").lower() != "bucket":
        return modal.Volume.from_name("nano-corpus", create_if_missing=True)
    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    endpoint = os.environ.get("NANO_AUDIO_ENDPOINT")  # R2/S3-compatible endpoint
    kwargs: dict = {"secret": modal.Secret.from_name("r2-creds"), "read_only": read_only}
    if endpoint:
        kwargs["bucket_endpoint_url"] = endpoint
    return modal.CloudBucketMount(bucket, **kwargs)


def wave_subdir(wave_id: str) -> str:
    """'' (flat legacy layout) or 'waves/wave_<id>' for a wave ingest."""
    return f"waves/wave_{wave_id}" if wave_id else ""
