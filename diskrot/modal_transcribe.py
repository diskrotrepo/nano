"""Modal entrypoint for lyrics transcription (Whisper on the raw mix).

Demucs-free: large-v3-turbo + VAD transcribes the mono mix directly (separation is
a no-op-to-worse ASR input per arXiv:2506.15514) and vocal gender comes from the
audio-LLM captioner (tags.json), so there is no vocal-isolation or F0 pass.

Uses multiple GPU containers in parallel via Modal's class pattern.

Run:
    modal run --detach diskrot/modal_transcribe.py
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import modal

from diskrot.modal_common import ProgressReporter, corpus_mount, wave_subdir

app = modal.App("nano-transcribe")


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
        # >=1.1.0: BatchedInferencePipeline supports word_timestamps + the vad
        # params (earlier 1.0.x batched mode lacks both).
        "faster-whisper>=1.1.0",
    )
    .run_commands("pip install 'protobuf>=4'")
    # Baked from the deploying shell. NANO_WHISPER_BATCH: VAD segments of one
    # song decode as batches of this size via BatchedInferencePipeline (~1.5-2x
    # L4 throughput); 0 = legacy sequential decode (the instant rollback lever —
    # set both vars and redeploy). NANO_TRANSCRIBE_CONCURRENCY: see the
    # @modal.concurrent comment on the class.
    .env({
        "NANO_WHISPER_BATCH": os.environ.get("NANO_WHISPER_BATCH", "8"),
    })
    # Demucs is gone from transcribe: Whisper runs on the raw mix (large-v3-turbo +
    # VAD makes separation a no-op-to-worse ASR input, arXiv:2506.15514) and vocal
    # gender comes from the audio-LLM captioner, so there is nothing left to isolate.
    # Dropping the demucs dep + the htdemucs weight bake also frees the VRAM that
    # capped @modal.concurrent at 2 — see the class below. (torch stays: it's a
    # top-level import in diskrot.transcribe_lyrics.)
    .add_local_python_source("model", "diskrot")
)

corpus_vol = corpus_mount()  # R2 audio bucket (read-only); see modal_common.corpus_mount
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.cls(
    image=image,
    gpu="L4",
    # 4 cores so the concurrent inputs' CPU work (ffmpeg decode, 44.1→16k resample,
    # Silero VAD) doesn't contend — same pattern as the structure sibling stage.
    # (The pyin gender Viterbi is gone — gender comes from the audio-LLM captioner.)
    cpu=4.0,
    timeout=60 * 60,
    max_containers=50,
    # Retry inputs whose container died under them (guard exits, preemptions,
    # platform cancellations) so they complete in-run instead of staying
    # pending for a manual relaunch sweep (2026-06-11: a ~34k mass input
    # cancellation + poison-guard exits all fell back to "redo next run").
    retries=modal.Retries(max_retries=2, initial_delay=1.0),
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
# Concurrent inputs per container overlap one song's GPU work (Whisper) with the
# others' GPU-idle CPU tails (ffmpeg decode + 44.1→16k resample), raising L4
# utilization at no FLOP cost. Was 4 for sequential batch-1 decode; with
# BatchedInferencePipeline each input now drives real batch-8 GPU work, so 2 is
# enough to keep the L4 fed while the other input decodes — and 4 concurrent
# batch-8 decodes risk VRAM pressure on 24 GB. Env-driven (deploying shell) so
# the NANO_WHISPER_BATCH=0 rollback can restore 4 in the same redeploy.
@modal.concurrent(max_inputs=int(os.environ.get("NANO_TRANSCRIBE_CONCURRENCY", "2")))
class Transcriber:
    @modal.enter()
    def load_models(self):
        # Eagerly import scipy.signal and warm the librosa kernels the per-song
        # path otherwise hits lazily (resample, pyin). A lazy import that dies
        # mid-song (import race / transient memory pressure) leaves sys.modules
        # permanently broken — the container then insta-fails every input with
        # "name '_signal_api' is not defined" and eats the queue at ~3s/song
        # (observed 2026-06-10, ~84% of a run's results dropped).
        import numpy as np
        import scipy.signal  # noqa: F401
        import torch._dynamo.external_utils  # noqa: F401  (third observed lazy-import poison, 2026-06-11)
        import librosa

        # Warm the one librosa kernel the Demucs-free path still hits (44.1→16k
        # resample). pyin is no longer warmed — the F0 gender estimate is gone
        # (gender now comes from the audio-LLM captioner).
        librosa.resample(np.zeros(1600, dtype=np.float32), orig_sr=44100, target_sr=16000)

        from faster_whisper import WhisperModel

        # No Demucs: Whisper runs on the raw mix (see the image comment / class
        # docstring). large-v3-turbo: 4 decoder layers vs 32, ~4-6x faster ASR at near-identical
        # transcription quality. The first ~30k songs were done with large-v3; the
        # transcript mix is fine for training data.
        self.whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        # Batched decode of one song's VAD segments (~1.5-2x GPU throughput at
        # identical weights). batch_size=0 (env, baked at deploy) falls back to
        # the sequential batch-1 path — the rollback lever if the post-wave
        # lyrics-stats parity check drifts.
        self.batch_size = int(os.environ.get("NANO_WHISPER_BATCH", "8"))
        if self.batch_size:
            from faster_whisper import BatchedInferencePipeline
            self.batched = BatchedInferencePipeline(model=self.whisper_model)
        # Warm the full transcribe path INCLUDING the VAD filter: faster-whisper
        # imports onnxruntime lazily on the first vad_filter=True call, and that
        # import sporadically dies under loaded-model memory pressure, leaving
        # the container raising "Applying the VAD filter requires the
        # onnxruntime package" on every input (the 2026-06-11 storm — same
        # lazy-import poisoning as the scipy warmup above). Done here, a broken
        # import fails @enter and the container never takes inputs. When batched
        # decode is on, warm THAT path too — it is the runtime path the poison
        # guards must have exercised.
        list(self.whisper_model.transcribe(
            np.zeros(16000, dtype=np.float32), language="en", vad_filter=True,
        )[0])
        if self.batch_size:
            list(self.batched.transcribe(
                np.zeros(16000, dtype=np.float32), language="en",
                vad_filter=True, batch_size=self.batch_size,
            )[0])
        # Poison guard state: consecutive in-band failures. A healthy container
        # essentially never fails several distinct files in a row (prepare
        # already dropped corrupt/over-long audio), but a container with broken
        # persistent state — sticky CUDA device-side assert, dead cuDNN handle —
        # fails EVERY input in ~4s and eats the queue (observed 2026-06-11:
        # ~146k of 180k results failed in-band; the NameError-only guard below
        # missed it because the storm wasn't a NameError).
        # Guarded with a lock: under @modal.concurrent(max_inputs=2) two inputs
        # run in this container's thread pool, so the read-modify-write of this
        # int would race. The threshold is raised to 5 below (from 3) because up
        # to 2 genuinely-bad files can be in-flight together on a HEALTHY
        # container and both fail — we want more evidence of a persistent fault
        # before exiting (and a spurious exit is only a wasted container: its
        # inputs stay pending and are redone on a healthy one).
        self._fail_lock = threading.Lock()
        self.consecutive_failures = 0

    @modal.method()
    def transcribe_file(self, mp3_name: str) -> tuple[str, dict | None, str | None]:
        """Transcribe a single file. Returns (stem, result_or_None, error_or_None).

        Demucs-free: Whisper runs on the raw mono mix and gender is not estimated
        here (it comes from the audio-LLM captioner / tags.json). large-v3-turbo +
        VAD on the mix is an equal-or-better ASR input (arXiv:2506.15514)."""
        from diskrot.transcribe_lyrics import _load_mix_mono, _transcribe

        mp3_path = Path("/corpus") / mp3_name
        key = mp3_path.stem
        try:
            mix_mono = _load_mix_mono(mp3_path)
            result = _transcribe(
                self.batched if self.batch_size else self.whisper_model,
                mix_mono, estimate_gender=False,
                batch_size=self.batch_size or None)
            with self._fail_lock:
                self.consecutive_failures = 0
            return (key, result, None)
        except NameError as e:
            # Poisoned module state (a lazy import died and left sys.modules
            # broken): every later input in this container would insta-fail the
            # same way. Die so Modal replaces the container; this input stays
            # pending and is redone on a healthy one.
            print(f"poisoned container ({e}); exiting so Modal replaces it", flush=True)
            os._exit(13)
        except Exception as e:
            with self._fail_lock:
                self.consecutive_failures += 1
                n_fail = self.consecutive_failures
            err = f"{type(e).__name__}: {str(e)[:300]}"
            # Print in the WORKER so a failure storm is visible live in any
            # container's logs — save_results only surfaces these at flush time.
            print(f"ERROR (consecutive {n_fail}) {key}: {err}", flush=True)
            if n_fail >= 5:
                # Distinct files don't fail repeatedly on a healthy container;
                # persistent broken state (sticky CUDA assert etc.) does. Same
                # remedy as the NameError guard: die, get replaced, the inputs
                # stay pending and are redone on a healthy container. 5 (not 3)
                # tolerates up to 2 concurrent genuinely-bad files without a
                # spurious exit under @modal.concurrent.
                print("5 consecutive failures — poisoned container; exiting so "
                      "Modal replaces it", flush=True)
                os._exit(13)
            return (key, None, err)


LYRICS_DIR = "/tokens/lyrics"
LOCK_PATH = "/tokens/lyrics/.orchestrator.lock"
# A lock older than the orchestrator's own timeout belongs to a dead run and is
# reclaimed; matches the orchestrate() function timeout below.
LOCK_STALE_SEC = 24 * 60 * 60


def _needs_lang_redo(entry) -> bool:
    """A forced-English (pre-auto-detect) entry: has transcribed words but no
    ``language`` field. ``--redo-missing-language`` re-transcribes exactly these
    (auto-detect overwrites them with the correct language + un-garbled words)."""
    return isinstance(entry, dict) and bool(entry.get("words")) and not entry.get("language")


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    # Globs the full corpus and parses every lyrics shard; at full coverage
    # that's >1 GB of JSON — Modal's 300s default timeout is not enough.
    timeout=30 * 60,
)
def list_pending(wave_id: str = "", redo_missing_language: bool = False) -> list[str]:
    """Return mp3 names (relative to /corpus) not yet in the sharded lyrics dir.
    ``wave_id`` scopes the glob to /corpus/waves/wave_<id>; lyrics shards key on
    the song stem either way. With ``redo_missing_language``, entries that exist
    but lack a ``language`` field (the legacy forced-English transcribe) are ALSO
    pending — a targeted re-do that a plain resume would skip."""
    from diskrot.transcribe_lyrics import load_lyrics_shards

    # The orchestrator now calls this repeatedly (sweep loop) — a warm-reused
    # container must see the shards committed by save_results since it started.
    tokens_vol.reload()
    mp3s = sorted((Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3"))
    existing = load_lyrics_shards(LYRICS_DIR)
    pending = [str(mp3.relative_to("/corpus")) for mp3 in mp3s
               if mp3.stem not in existing
               or (redo_missing_language and _needs_lang_redo(existing.get(mp3.stem)))]
    n_redo = sum(1 for mp3 in mp3s if mp3.stem in existing
                 and redo_missing_language and _needs_lang_redo(existing.get(mp3.stem)))
    extra = f" (incl. {n_redo} language-redo)" if redo_missing_language else ""
    print(f"found {len(mp3s)} total mp3s, {len(existing)} already done, "
          f"{len(pending)} pending{extra}")
    return pending


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    # A 2000-result batch hash-spreads over all 256 shards, so each flush
    # read-merge-rewrites ~the whole lyrics dir; per-flush time grows with
    # corpus coverage and crossed Modal's 300s default overnight 2026-06-10
    # (~100k songs, ~1.6 MB/shard), killing the orchestrator mid-run.
    timeout=30 * 60,
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
def _acquire_lock(call_id: str) -> bool:
    """Best-effort single-orchestrator lease on the volume.

    Returns False if a fresh lock from another orchestrator is present (refuse to
    start so two runs don't race read-merge-write on the same shard). A lock older
    than ``LOCK_STALE_SEC`` is a dead run and is reclaimed. The lock is re-entrant
    by ``call_id``: a restarted orchestrate attempt (same function call, new
    runner) reclaims its own lock instead of refusing and stranding the run.
    Best-effort: two runs that start within the same reload window can both
    acquire — that's a rare operator double-launch, and the synchronous flush
    path bounds the damage.
    """
    from diskrot.transcribe_lyrics import _atomic_write_json

    tokens_vol.reload()
    lock = Path(LOCK_PATH)
    if lock.exists():
        owner = None
        try:
            data = json.loads(lock.read_text())
            owner = data.get("call_id")
            age = time.time() - data.get("started_at", 0)
        except Exception:
            age = 0  # unreadable lock — treat as fresh and refuse, to be safe
        if owner == call_id:
            print("Re-acquiring own lock after orchestrator restart.")
        elif age < LOCK_STALE_SEC:
            print(f"Another orchestrator holds the lock (age {age:.0f}s < "
                  f"{LOCK_STALE_SEC}s) — refusing to start. Stop it first or wait.")
            return False
        else:
            print(f"Reclaiming stale lock (age {age:.0f}s).")
    _atomic_write_json(LOCK_PATH, {"started_at": time.time(), "call_id": call_id})
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
    # The orchestrator is the run's single point of failure: a preempted-and-
    # restarted orchestrate would refuse on its own fresh lock and silently end
    # the run. 3x CPU/mem pricing on one small container is nothing next to the
    # L4 fleet. (GPU workers can't opt out, but their preemption is harmless —
    # the file just stays pending.)
    nonpreemptible=True,
)
def orchestrate(flush_every: int = 5000, chunk_size: int = 3000, wave_id: str = "",
                redo_missing_language: bool = False):
    """Dispatch transcription and merge results into the sharded lyrics dir.
    ``wave_id`` scopes the pass to /corpus/waves/wave_<id>. ``redo_missing_language``
    additionally re-transcribes legacy entries that lack a ``language`` field.

    Runs the ``.map()`` collect/flush loop *remotely* (not in local_entrypoint)
    so ``--detach`` truly survives terminal close — the previous version ran
    this loop locally, so closing the terminal killed the result-collector even
    though the worker containers kept running and billing.

    Dispatches in **chunks** (default 3k ≈ 20-30 min of fleet work) inside a
    **sweep loop** that re-lists pending files until they stop shrinking.
    One giant ``.map()`` held ~190k inputs outstanding for ~20h, and twice on
    2026-06-11 a server-side event cancelled tens of thousands of queued inputs
    en masse (``RemoteError: Function call was cancelled...``) roughly an hour
    into the run. Chunking caps the exposure (a storm can only kill the current
    chunk) and the sweep loop automatically re-queues whatever was cancelled —
    no more manual "final sweep" relaunches. 15k chunks still lost the back
    ~60% of every chunk to the waves (inputs queued ≳1.5h get reaped); 3k
    keeps queue residence under ~30 min, below the observed kill window,
    while paying the end-of-chunk straggler idle 5× less often than 1k would.

    Flushes partial results every ``flush_every`` completions (default 5000).
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
    if not _acquire_lock.remote(modal.current_function_call_id()):
        return
    try:
        transcriber = Transcriber()
        prev_pending: int | None = None
        sweep = 0
        while True:
            sweep += 1
            pending = list_pending.remote(
                wave_id=wave_id, redo_missing_language=redo_missing_language)
            if not pending:
                print("Nothing to transcribe — all files already transcribed")
                return
            if prev_pending is not None and len(pending) >= prev_pending:
                # A sweep that doesn't shrink pending means the remaining files
                # fail deterministically (undecodable etc.) — looping further
                # would re-grind them forever. Leave them pending and stop.
                print(f"sweep {sweep}: pending did not shrink "
                      f"({prev_pending} -> {len(pending)}) — stopping. "
                      f"Remaining files fail deterministically; inspect a few "
                      f"with the single-file diagnostic before re-running.")
                return
            prev_pending = len(pending)
            print(f"sweep {sweep}: {len(pending)} pending, dispatching in "
                  f"chunks of {chunk_size} (flush_every={flush_every}; "
                  f"Demucs-free: Whisper on raw mix)...")

            n_seen = n_errors = n_inband = 0
            # Per-sweep heartbeat (pending is re-listed each sweep and shrinks, so
            # the ETA is for THIS sweep — most work is the first one).
            rep = ProgressReporter(len(pending), "transcribe", unit="files")
            # n_inband: per-file errors (returned, not raised) — these stay
            # pending and are NOT written; surfacing the count here is what
            # makes a poisoned-container storm visible (2026-06-11: 146k of
            # 180k "done" were silent, only visible in save_results logs).
            for start in range(0, len(pending), chunk_size):
                chunk = pending[start : start + chunk_size]
                batch: list = []
                # order_outputs=False: a preempted file must not head-of-line-
                # block the in-order yield (that idles the other containers
                # while they still bill). Results are flushed into shards by
                # key, so order doesn't matter.
                # return_exceptions=True: a single file raising (e.g. a worker
                # that hard-crashes on a poison file) must NOT crash the
                # orchestrator and lose the run — count it and continue. The
                # file stays pending (not in the shards) and is re-queued by
                # the next sweep.
                for result in transcriber.transcribe_file.map(
                    chunk, order_outputs=False, return_exceptions=True
                ):
                    n_seen += 1
                    rep.update(1, extra=f"transcribed {n_seen - n_errors - n_inband:,}, "
                                        f"in-band-fail {n_inband:,}, errored {n_errors:,}")
                    if isinstance(result, Exception):
                        n_errors += 1
                        if n_errors <= 20:
                            print(f"FILE FAILED (re-queued next sweep): "
                                  f"{type(result).__name__}: {str(result)[:140]}")
                        continue
                    batch.append(result)
                    if result[2] is not None:
                        n_inband += 1
                    if len(batch) >= flush_every:
                        print(f"flushing {len(batch)} results "
                              f"({n_seen}/{len(pending)} seen: "
                              f"{n_seen - n_errors - n_inband} transcribed, "
                              f"{n_inband} failed in-band, "
                              f"{n_errors} cancelled/errored)")
                        save_results.remote(batch)
                        batch = []
                if batch:
                    print(f"chunk flush of {len(batch)} results "
                          f"({n_seen}/{len(pending)} seen: "
                          f"{n_seen - n_errors - n_inband} transcribed, "
                          f"{n_inband} failed in-band, "
                          f"{n_errors} cancelled/errored)")
                    save_results.remote(batch)
            rep.done(extra=f"transcribed {n_seen - n_errors - n_inband:,}, "
                           f"in-band-fail {n_inband:,}, errored {n_errors:,}")
            print(f"sweep {sweep} complete: {n_seen} seen, "
                  f"{n_seen - n_errors - n_inband} transcribed, "
                  f"{n_inband} failed in-band, {n_errors} cancelled/errored"
                  + (" — re-sweeping for the remainder" if n_errors or n_inband
                     else ""))
            if not n_errors and not n_inband:
                # Clean sweep — one more list_pending confirms completion (or
                # catches files added meanwhile), then the loop exits above.
                continue
    finally:
        _release_lock.remote()


@app.local_entrypoint()
def main(flush_every: int = 5000, chunk_size: int = 3000, wave_id: str = "",
         redo_missing_language: bool = False):
    """Spawn the remote orchestrator and return immediately.

    Use with ``--detach`` so the run survives terminal close (both pieces are
    required: ``.spawn()`` so the entrypoint exits without blocking, and
    ``--detach`` so the app isn't auto-stopped when the entrypoint completes).
    --wave-id N scopes transcription to /corpus/waves/wave_N.
    --redo-missing-language re-transcribes legacy entries lacking a ``language``
    field (the forced-English 128k) — auto-detect overwrites them.

    Transcribe is Demucs-free: Whisper runs on the raw mix (large-v3-turbo + VAD
    is an equal-or-better ASR input, arXiv:2506.15514) and vocal gender comes from
    the audio-LLM captioner (tags.json), so there is no separation step or F0 pass.
    """
    call = orchestrate.spawn(flush_every, chunk_size, wave_id=wave_id,
                             redo_missing_language=redo_missing_language)
    print(f"spawned orchestrator: function call id {call.object_id}")
    print("Follow logs in the Modal dashboard; safe to close this terminal "
          "if launched with --detach.")
