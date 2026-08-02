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
import sys
import warnings
from pathlib import Path

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


class _DropAsyncioTeardownNoise(logging.Filter):
    """Drop the asyncio teardown chatter Modal's async client emits when a
    (preemptible) fan-out container exits mid-flight.

    On every container shutdown the loop GC's Modal's still-pending
    ``async_merge().producer()`` coroutine and logs (via the 'asyncio' logger)
    "Task was destroyed but it is pending!" — a flood on a 50-wide preemptible
    fan-out that makes a healthy run look like it is crash-looping. Matched by
    message substring (mirrors _DropHeartbeatNoise), NOT a blanket
    setLevel(CRITICAL): genuine asyncio diagnostics ("Task exception was never
    retrieved", a real Future error) still surface."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Match the asyncio teardown record by its literal message text only. We
        # deliberately do NOT key on the 'async_merge' coroutine name: async_merge
        # is Modal's live .map() result-pump (poll_outputs/check_lost_inputs), not
        # a teardown-only token, so a genuine "Unhandled exception in event loop"
        # rooted in that coroutine must still surface. The "Task was destroyed but
        # it is pending" line already carries the GC'd-coroutine repr, so this
        # catches the teardown flood without that over-broad clause.
        msg = record.getMessage()
        return not (
            "Task was destroyed but it is pending" in msg
            or "Event loop is closed" in msg
        )


# Escape hatch: NANO_LOG_VERBOSE=1 restores the full (noisy) container output.
# These reach the fan-out data-prep containers only — modal_common is imported by
# the diskrot/modal_*.py entrypoints, never by modal_train.py or the inference
# server — so training/serving diagnostics are structurally untouched.
if not os.environ.get("NANO_LOG_VERBOSE"):
    logging.getLogger("asyncio").addFilter(_DropAsyncioTeardownNoise())

    # The "Exception ignored in: <coroutine ... async_merge ...> RuntimeError:
    # Event loop is closed" lines (and modal ClientClosed) are printed by the
    # interpreter's unraisable hook while GC'ing those coroutines after the loop
    # is gone. Drop ONLY that shutdown race; delegate every other unraisable to
    # the original hook so genuine finalizer bugs still surface. This cannot
    # touch the SpectroStream C++ FATAL abort (that never reaches Python).
    try:
        from modal.exception import ClientClosed as _ClientClosed
    except Exception:  # pragma: no cover - modal internal layout drift
        _ClientClosed = ()

    _orig_unraisablehook = sys.unraisablehook

    def _quiet_unraisablehook(unraisable):
        exc = unraisable.exc_value
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        if _ClientClosed and isinstance(exc, _ClientClosed):
            return
        _orig_unraisablehook(unraisable)

    sys.unraisablehook = _quiet_unraisablehook

    # librosa/soundfile mp3 fallback (tokenize._load_audio): soundfile can't open
    # mp3, librosa falls back to audioread and warns on essentially every file.
    # Scoped by message + category so other warnings still print.
    warnings.filterwarnings(
        "ignore",
        message="PySoundFile failed.*",
        category=UserWarning,
    )


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


def r2_env_secret():
    """Secret carrying NANO_AUDIO_BUCKET / NANO_AUDIO_ENDPOINT into the container.

    ``corpus_mount()`` reads these locally to configure the CloudBucketMount, so
    they're baked into the *mount* but NOT present in the container's env. Any
    code that talks to R2 directly (``bulk_delete_r2``'s boto3 client) needs them
    at runtime, so capture the local values at app-build time and inject them.
    Pair with ``modal.Secret.from_name("r2-creds")`` (the AWS-style keys)."""
    d = {k: os.environ[k] for k in ("NANO_AUDIO_BUCKET", "NANO_AUDIO_ENDPOINT")
         if os.environ.get(k)}
    return modal.Secret.from_dict(d)


def bulk_delete_r2(names, *, batch_size: int = 1000, log=print):
    """Delete R2 objects by key via the S3 multi-object ``DeleteObjects`` API.

    ``names`` are keys relative to the bucket root — exactly the paths the
    CloudBucketMount exposes under /corpus (e.g. ``waves/wave_17/song.mp3``).
    Deletes up to 1000 objects per request instead of one FUSE ``unlink()``
    network round-trip per file, turning a serial ~1 hr delete of ~20k objects
    into a few seconds.

    Requires the ``r2-creds`` secret (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    and ``r2_env_secret()`` (bucket + endpoint) attached to the calling function.
    Returns ``(n_deleted, errors)`` where ``errors`` is a list of the S3 error
    dicts (``{Key, Code, Message}``). ``DeleteObjects`` is idempotent, so keys
    that are already gone count as deleted (no FileNotFound bookkeeping needed)."""
    import boto3

    bucket = os.environ.get("NANO_AUDIO_BUCKET", "nano-audio")
    endpoint = os.environ.get("NANO_AUDIO_ENDPOINT")  # R2/S3-compatible endpoint
    # region_name="auto" is R2's convention and avoids boto3's NoRegionError.
    s3 = boto3.client("s3", endpoint_url=endpoint, region_name="auto")

    total = len(names)
    n_deleted = 0
    errors: list = []
    for i in range(0, total, batch_size):
        batch = names[i:i + batch_size]
        # Quiet=True → R2 returns only the failures, not the (large) success list.
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        batch_errors = resp.get("Errors", [])
        errors.extend(batch_errors)
        n_deleted += len(batch) - len(batch_errors)
        for e in batch_errors:
            log(f"failed to delete {e.get('Key')}: {e.get('Code')} {e.get('Message')}")
        log(f"  deleted {min(i + batch_size, total):,}/{total:,}")
    return n_deleted, errors


def wave_subdir(wave_id: str) -> str:
    """'' (flat legacy layout) or 'waves/wave_<id>' for a wave ingest."""
    return f"waves/wave_{wave_id}" if wave_id else ""


def list_wave_mp3s(corpus_root: Path, wave_id: str) -> list[Path]:
    """MP3s a stage should process for *wave_id*.

    '' globs the flat legacy root (non-recursive — empty now that the corpus
    lives under waves/), 'all' sweeps every waves/wave_*/ folder in one pass
    (e.g. a whole-corpus --redo), anything else scopes to waves/wave_<id>.
    """
    if wave_id == "all":
        return sorted(corpus_root.glob("waves/wave_*/*.mp3"))
    return sorted((corpus_root / wave_subdir(wave_id)).glob("*.mp3"))


def assert_stage_produced_output(
    stage: str, n_success: int, n_pending: int, n_failed: int = 0
) -> None:
    """Halt a wave stage that silently produced NOTHING despite having work.

    A worker-fan-out stage whose workers ALL die (e.g. a missing-dependency
    import bug in the worker image) returns normally with 0 successes — and the
    ingest orchestrator's run() marks a stage "done" whenever the call RETURNS
    without raising, so it advances and leaves the corpus missing that whole
    stream. That is exactly how melody (0 chroma) and auto_tag (0 captions)
    silently failed on every wave. Raising here turns those silent no-ops into a
    hard stop with a clear message instead of a "done" stage. Only fires when
    there WAS pending work and NONE of it succeeded (a clean 0-pending re-run is
    fine)."""
    if n_pending > 0 and n_success <= 0:
        raise RuntimeError(
            f"[{stage}] CATASTROPHIC: 0 of {n_pending} pending items succeeded "
            f"(failed={n_failed}). Every worker produced nothing — almost certainly a "
            f"missing-dependency / import bug in the worker image (check the stage app "
            f"logs for the per-worker error). NOT marking the stage done."
        )


# Re-export the shared progress heartbeat so the Modal fan-out orchestrators can
# import it from modal_common alongside the other stage helpers. The class itself
# lives in the dependency-free diskrot.progress (no `modal` import) so the pure
# per-song modules (pack_cache / key_detect / phonemize / filter_lyrics) can use the
# SAME reporter without dragging in modal + this module's log-filter side effects.
from diskrot.progress import ProgressReporter  # noqa: E402,F401  (public re-export)
