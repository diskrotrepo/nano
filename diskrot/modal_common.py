"""Shared Modal helpers for the data-prep entrypoints (tokenize / melody /
auto_tag / transcribe / structure).

``corpus_mount()`` returns the raw-audio source: the Cloudflare R2 (``nano-audio``)
bucket, mounted as a ``CloudBucketMount`` (object storage — no inode cap). The
legacy ``nano-corpus`` Modal Volume has been retired; everything lives in R2 under
``waves/wave_<id>/``. Configure with::

    NANO_AUDIO_BUCKET=nano-audio \
    NANO_AUDIO_ENDPOINT=https://<acct>.r2.cloudflarestorage.com \
    modal run --detach diskrot/modal_tokenize.py --wave-id 17

The mount carries the ``r2-creds`` Modal secret (AWS-style keys).
``NANO_AUDIO_ENDPOINT`` is required for Cloudflare R2 / any S3-compatible store;
omit it for real AWS S3.
"""
from __future__ import annotations

import logging
import os

import modal


class _DropHeartbeatNoise(logging.Filter):
    """Drop Modal's per-attempt heartbeat-retry WARNING from container logs.

    On a transient control-plane blip Modal's runtime logs
    "Modal Client → Modal Worker Heartbeat attempt failed (...)" every few
    seconds (modal-client logger, WARNING). On the long fan-out data-prep runs
    that's a flood that buries the real progress lines, making a healthy run
    look stuck. The rarer escalation — "heartbeat attempts have been failing
    for over N minutes ... container will eventually be marked unhealthy" — is
    the one that actually means a container is dying, so it is kept.

    Installed at import time (every fan-out entrypoint imports this module), so
    it attaches in each worker container before the heartbeat loop spams. NOTE:
    the separate "Volume mounted at ... using N% of available inodes" warning is
    injected server-side by Modal's worker runtime, NOT a client log record, so
    it can't be filtered here — clear it by reducing volume inode usage.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Heartbeat attempt failed" not in record.getMessage()


logging.getLogger("modal-client").addFilter(_DropHeartbeatNoise())


def corpus_mount(read_only: bool = True):
    """Return the /corpus mount: the R2 (``nano-audio``) bucket as a
    CloudBucketMount. Env vars are read locally at `modal run` time, so they
    come from your shell. ``NANO_AUDIO_ENDPOINT`` is required for R2."""
    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    endpoint = os.environ.get("NANO_AUDIO_ENDPOINT")  # R2/S3-compatible endpoint
    kwargs: dict = {"secret": modal.Secret.from_name("r2-creds"), "read_only": read_only}
    if endpoint:
        kwargs["bucket_endpoint_url"] = endpoint
    return modal.CloudBucketMount(bucket, **kwargs)


def wave_subdir(wave_id: str) -> str:
    """'' (flat legacy layout) or 'waves/wave_<id>' for a wave ingest."""
    return f"waves/wave_{wave_id}" if wave_id else ""
