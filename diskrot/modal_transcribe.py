"""Modal entrypoint for lyrics transcription (Demucs + Whisper).

Uses multiple GPU containers in parallel via Modal's class pattern.

Run:
    modal run --detach diskrot/modal_transcribe.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

app = modal.App("nano-transcribe")


def _cache_demucs():
    from demucs.pretrained import get_model

    get_model("htdemucs")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "librosa>=0.10",
        "numpy>=1.26",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "demucs",
        "faster-whisper",
    )
    .run_commands("pip install 'protobuf>=4'")
    .run_function(_cache_demucs)
    .add_local_python_source("model", "diskrot")
)

corpus_vol = modal.Volume.from_name("nano-corpus", create_if_missing=True)
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.cls(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    max_containers=50,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Transcriber:
    @modal.enter()
    def load_models(self):
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from faster_whisper import WhisperModel

        self.demucs_model = get_model("htdemucs")
        self.demucs_model.to("cuda")
        self.demucs_model.eval()
        self.apply_fn = apply_model
        self.whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    @modal.method()
    def transcribe_file(self, mp3_name: str) -> tuple[str, dict | None, str | None]:
        """Transcribe a single file. Returns (stem, result_or_None, error_or_None)."""
        from diskrot.transcribe_lyrics import _separate_vocals, _transcribe

        mp3_path = Path("/corpus") / mp3_name
        key = mp3_path.stem
        try:
            vocals = _separate_vocals(self.demucs_model, self.apply_fn, mp3_path, "cuda")
            result = _transcribe(self.whisper_model, vocals)
            return (key, result, None)
        except Exception as e:
            return (key, None, str(e))


LYRICS_DIR = "/tokens/lyrics"
LOCK_PATH = "/tokens/lyrics/.orchestrator.lock"
# A lock older than the orchestrator's own timeout belongs to a dead run and is
# reclaimed; matches the orchestrate() function timeout below.
LOCK_STALE_SEC = 24 * 60 * 60


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def list_pending() -> list[str]:
    """Return mp3 filenames not yet in the sharded lyrics dir."""
    from diskrot.transcribe_lyrics import load_lyrics_shards

    mp3s = sorted(Path("/corpus").glob("*.mp3"))
    existing = load_lyrics_shards(LYRICS_DIR)
    pending = [mp3.name for mp3 in mp3s if mp3.stem not in existing]
    print(f"found {len(mp3s)} total mp3s, {len(existing)} already done, {len(pending)} pending")
    return pending


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def save_results(results: list[tuple[str, dict | None, str | None]]):
    """Merge a batch of results into the sharded lyrics dir.

    Only the shards touched by this batch are read, merged, and atomically
    rewritten — cost is O(batch), not O(corpus), and a kill mid-write can
    corrupt at most one shard (temp+rename), never the whole corpus.
    """
    from diskrot.transcribe_lyrics import (
        _atomic_write_json,
        _lyric_bucket,
        _shard_path,
    )

    # Bucket the incoming results, dropping failures.
    by_bucket: dict[int, dict] = {}
    n_done, n_instrumental, n_failed = 0, 0, 0
    for key, result, error in results:
        if error is not None:
            print(f"FAILED {key}: {error}")
            n_failed += 1
            continue
        by_bucket.setdefault(_lyric_bucket(key), {})[key] = result
        if result is None:
            n_instrumental += 1  # instrumental
        else:
            n_done += 1

    # See commits made by other (warm-reused) save_results containers before
    # read-merge-write, else a stale shard view would overwrite their entries.
    tokens_vol.reload()

    # Read-merge-write only the affected shards.
    n_total = 0
    for bucket, new_entries in by_bucket.items():
        shard = _shard_path(LYRICS_DIR, bucket)
        existing = json.loads(shard.read_text()) if shard.exists() else {}
        existing.update(new_entries)
        _atomic_write_json(shard, existing)
        n_total += len(existing)

    # Retry a transient DataLossError so a storage blip on a flush doesn't
    # propagate back through .remote() and crash the orchestrator mid-run.
    for attempt in range(3):
        try:
            tokens_vol.commit()
            break
        except modal.exception.DataLossError as e:
            if attempt == 2:
                raise
            print(f"commit failed ({e}); retry {attempt + 1}/2", flush=True)
            time.sleep(2.0 * (attempt + 1))
    print(f"\ntranscribed: {n_done}  instrumental: {n_instrumental}  failed: {n_failed}")
    print(f"wrote {len(by_bucket)} shard(s)")


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=24 * 60 * 60,
)
def _acquire_lock() -> bool:
    """Best-effort single-orchestrator lease on the volume.

    Returns False if a fresh lock from another orchestrator is present (refuse to
    start so two runs don't race read-merge-write on the same shard). A lock older
    than ``LOCK_STALE_SEC`` is a dead run and is reclaimed. Best-effort: two runs
    that start within the same reload window can both acquire — that's a rare
    operator double-launch, and the synchronous flush path bounds the damage.
    """
    from diskrot.transcribe_lyrics import _atomic_write_json

    tokens_vol.reload()
    lock = Path(LOCK_PATH)
    if lock.exists():
        try:
            age = time.time() - json.loads(lock.read_text()).get("started_at", 0)
        except Exception:
            age = 0  # unreadable lock — treat as fresh and refuse, to be safe
        if age < LOCK_STALE_SEC:
            print(f"Another orchestrator holds the lock (age {age:.0f}s < "
                  f"{LOCK_STALE_SEC}s) — refusing to start. Stop it first or wait.")
            return False
        print(f"Reclaiming stale lock (age {age:.0f}s).")
    _atomic_write_json(LOCK_PATH, {"started_at": time.time()})
    tokens_vol.commit()
    return True


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def _release_lock() -> None:
    # reload so this (fresh) container sees the lock the acquire container wrote,
    # else the unlink silently misses and the lease lingers for LOCK_STALE_SEC.
    tokens_vol.reload()
    Path(LOCK_PATH).unlink(missing_ok=True)
    tokens_vol.commit()


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=24 * 60 * 60,
)
def orchestrate(flush_every: int = 2000):
    """Dispatch transcription and merge results into the sharded lyrics dir.

    Runs the ``.map()`` collect/flush loop *remotely* (not in local_entrypoint)
    so ``--detach`` truly survives terminal close — the previous version ran
    this loop locally, so closing the terminal killed the result-collector even
    though the worker containers kept running and billing.

    Flushes partial results every ``flush_every`` completions (default 2000).
    The flushing path uses synchronous ``save_results.remote(...)`` so two writers
    never race on a shard — each flush completes (read-merge-write-commit) before
    the next is issued. A larger ``flush_every`` cuts shard write-amplification
    (each flush read-merge-writes whole JSON shards; ~256 shards fill to ~1k
    entries, so frequent flushes rewrite most of the corpus repeatedly). The
    trade-off: on orchestrator death, up to ``flush_every`` unflushed results are
    redone — they stay pending via ``list_pending``, so it's recompute, not data
    loss. A lease lock (see ``_acquire_lock``) refuses a concurrent second run.
    ``list_pending`` skips songs already in the shards, so re-launching resumes.
    """
    if not _acquire_lock.remote():
        return
    try:
        pending = list_pending.remote()
        if not pending:
            print("Nothing to transcribe — all files already transcribed")
            return

        print(f"Dispatching {len(pending)} files across parallel containers "
              f"(flush_every={flush_every})...")
        transcriber = Transcriber()
        batch: list = []
        n_seen = 0
        n_errors = 0
        # order_outputs=False: a preempted file must not head-of-line-block the
        # in-order yield (that idles the other containers while they still bill).
        # Results are flushed into shards by key, so order doesn't matter.
        # return_exceptions=True: a single file raising (e.g. a worker that hard-
        # crashes on a poison file) must NOT crash the orchestrator and lose the
        # run — count it and continue. The file stays pending (not in the shards)
        # and is picked up on the next launch via list_pending.
        for result in transcriber.transcribe_file.map(
            pending, order_outputs=False, return_exceptions=True
        ):
            n_seen += 1
            if isinstance(result, Exception):
                n_errors += 1
                if n_errors <= 20:
                    print(f"FILE FAILED (stays pending, redone next run): "
                          f"{type(result).__name__}: {str(result)[:140]}")
                continue
            batch.append(result)
            if len(batch) >= flush_every:
                print(f"flushing {len(batch)} results ({n_seen}/{len(pending)} done)")
                save_results.remote(batch)
                batch = []
        if batch:
            print(f"final flush of {len(batch)} results ({n_seen}/{len(pending)} done)")
            save_results.remote(batch)
        if n_errors:
            print(f"file errors: {n_errors} "
                  f"(transient — affected files stay pending; re-run to finish them)")
    finally:
        _release_lock.remote()


@app.local_entrypoint()
def main(flush_every: int = 2000):
    """Spawn the remote orchestrator and return immediately.

    Use with ``--detach`` so the run survives terminal close (both pieces are
    required: ``.spawn()`` so the entrypoint exits without blocking, and
    ``--detach`` so the app isn't auto-stopped when the entrypoint completes).
    """
    call = orchestrate.spawn(flush_every)
    print(f"spawned orchestrator: function call id {call.object_id}")
    print("Follow logs in the Modal dashboard; safe to close this terminal "
          "if launched with --detach.")
