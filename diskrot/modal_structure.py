"""Modal entrypoint for song-structure analysis (allin1).

Mirrors diskrot/modal_transcribe.py: many GPU containers in parallel, sharded
atomic writes to /tokens/structure, resume-by-skip, remote orchestrator + spawn
so --detach survives terminal close.

Run:
    modal run --detach diskrot/modal_structure.py
    modal run --detach diskrot/modal_structure.py --limit 200   # calibration

NOTE: allin1 pulls torch + demucs + natten; natten wheels are built per
torch/cuda version, so the wheel index below is pinned to torch 2.4.0 / cu121
(natten ships no torch 2.4.1 wheel — that index path 404s).
The ~200-song calibration run (--limit) exists to confirm this image resolves and
to measure seconds/track before the full corpus pass.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from diskrot.modal_common import corpus_mount, wave_subdir

app = modal.App("nano-structure")


def _cache_demucs():
    # allin1 uses demucs internally for source separation; pre-cache it into the
    # image so containers don't each re-download on cold start.
    from demucs.pretrained import get_model

    get_model("htdemucs")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # git: to pip-install madmom from its repo. build-essential: madmom compiles
    # C extensions from Cython at install time and the slim image has no compiler.
    .apt_install("ffmpeg", "libsndfile1", "git", "build-essential")
    # torch pinned to 2.4.0 (not 2.4.1) so it matches the only natten prebuilt
    # wheel available for cu121 — natten's compiled CUDA extension is built per
    # exact torch version, and the index only ships a torch2.4.0 build.
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # natten must match the torch/cuda build above (allin1 depends on it). The
    # prebuilt wheel lives only on shi-labs.com, where (a) the wheel index dir is
    # keyed by torch version — torch2.4.0, NOT 2.4.1 (the 2.4.1 path 404s) — and
    # (b) the TLS cert has expired, so --trusted-host is required to skip the cert
    # check. Without a matching wheel pip falls back to compiling the sdist, which
    # needs cmake + nvcc and yields a useless CPU-only build on the GPU-less
    # builder. The cp312 wheel exists here; keep the image on python 3.12.
    .pip_install(
        "natten==0.17.1",
        find_links="https://shi-labs.com/natten/wheels/cu121/torch2.4.0/",
        extra_options="--trusted-host shi-labs.com",
    )
    # allin1's beat/segment backbone is madmom, which has no PyPI build for
    # py3.12 (last release 0.16.1 predates it) — the official allin1 install is
    # `pip install git+.../madmom`. madmom's setup.py imports numpy + Cython at
    # build time with no pyproject build-system, so default build isolation can't
    # see them: preinstall both and build with --no-build-isolation. numpy pinned
    # <2 (madmom still uses APIs removed in numpy 2.0) and Cython <3 (madmom is
    # pre-Cython-3 source); both pins are re-asserted in the allin1 stage below so
    # the resolver can't bump numpy back to 2.x and break madmom at runtime.
    .pip_install("numpy<2", "cython<3")
    # Pinned to a commit (not a moving branch) so the image is reproducible —
    # madmom's main could change/break under us on any future rebuild.
    .pip_install(
        "madmom @ git+https://github.com/CPJKU/madmom@27f032e8947204902c675e5e341a3faf5dc86dae",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "allin1",
        "librosa>=0.10",
        "numpy>=1.26,<2",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "demucs",
    )
    .run_commands("pip install 'protobuf>=4'")
    .run_function(_cache_demucs)
    .add_local_python_source("model", "diskrot")
)

corpus_vol = corpus_mount()  # nano-corpus Volume, or object storage via NANO_CORPUS_SOURCE=bucket
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.cls(
    image=image,
    # L4, NOT A100: the bottleneck is parallelism, and Modal's A100 supply is scarce
    # — an A100 request only provisioned 1 container (vs 50 on L4), collapsing the
    # fan-out. The GPU work (Demucs + segment inference) is brief and fine on an L4;
    # the real per-track cost is madmom beat tracking, which is CPU-bound (no GPU
    # helps it). So: abundant L4s to actually get 50 containers, and attack the CPU
    # cost with cores + input concurrency below.
    gpu="L4",
    # madmom DBN beat decoding is single-threaded CPU and is the per-track
    # bottleneck (the silent gap after the fast GPU stages). 4 cores cover the two
    # concurrent batches' beat tracking (one core each) plus Demucs/torch CPU work.
    cpu=4.0,
    timeout=60 * 60,
    max_containers=100,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    # allin1 pulls its checkpoint from the HF Hub at runtime; the token lifts the
    # anonymous rate limit that would otherwise throttle/fail a 50-container fan-out
    # (same secret modal_transcribe.py / modal_auto_tag.py already use).
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
# Two BATCHES per container: overlaps one batch's single-threaded madmom beat
# tracking (CPU) with the other's Demucs/segment inference (GPU) without
# over-contending the one L4 GPU — at 4 concurrent the neural step ballooned
# ~4s -> ~20s from GPU contention. Each input is now a chunk of songs handed to
# allin1 in ONE analyze() call, which pays the per-call setup (demucs subprocess
# spawn + htdemucs load + 8-fold harmonix-all ensemble construction, ~15-25s)
# once per chunk instead of once per song.
@modal.concurrent(max_inputs=2)
class Analyzer:
    @modal.enter()
    def load_models(self):
        # Importing allin1 + warming demucs amortizes model load across the many
        # files this container will process.
        import allin1  # noqa: F401
        from demucs.pretrained import get_model

        get_model("htdemucs")
        # Pre-build allin1's segment model once so two concurrent first-calls don't
        # cold-load it simultaneously. Guarded: allin1's internal loader API can
        # differ across versions; a miss just falls back to lazy load on first call.
        try:
            from allin1.models import load_pretrained_model

            load_pretrained_model(model_name="harmonix-all", device="cuda")
        except Exception as e:
            print(f"allin1 model warm skipped: {e}")

    @modal.method()
    def analyze_chunk(self, mp3_names: list[str]) -> list[tuple[str, dict | None, str | None]]:
        """Analyze a chunk of files in one allin1 call.

        Returns one (stem, result_or_None, error_or_None) per input. If the
        batched call fails (one poison file aborts allin1's whole per-file
        loop), fall back to per-file calls so only the bad file loses its
        entry — the rare bad chunk re-pays the per-call setup, nothing else.
        """
        import allin1

        from diskrot.structure import analyze_batch as _analyze_batch
        from diskrot.structure import analyze_file as _analyze_file

        mp3_paths = [Path("/corpus") / name for name in mp3_names]
        try:
            entries = _analyze_batch(allin1.analyze, mp3_paths, "cuda")
            return [(p.stem, entry, None) for p, entry in zip(mp3_paths, entries)]
        except Exception as batch_err:
            print(f"batch of {len(mp3_paths)} failed ({batch_err}); retrying per-file")

        out: list[tuple[str, dict | None, str | None]] = []
        for mp3_path in mp3_paths:
            try:
                out.append((mp3_path.stem, _analyze_file(allin1.analyze, mp3_path, "cuda"), None))
            except Exception as e:
                out.append((mp3_path.stem, None, str(e)))
        return out


STRUCTURE_DIR = "/tokens/structure"


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def list_pending(wave_id: str = "") -> list[str]:
    """Return mp3 names (relative to /corpus) not yet in the sharded structure
    dir. ``wave_id`` scopes the glob to /corpus/waves/wave_<id>; structure shards
    key on the song stem either way."""
    from diskrot.structure import load_structure_shards

    mp3s = sorted((Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3"))
    existing = load_structure_shards(STRUCTURE_DIR)
    pending = [str(mp3.relative_to("/corpus")) for mp3 in mp3s
               if mp3.stem not in existing]
    print(f"found {len(mp3s)} total mp3s, {len(existing)} already done, {len(pending)} pending")
    return pending


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def save_results(results: list[tuple[str, dict | None, str | None]]):
    """Merge a batch of results into the sharded structure dir.

    Only the shards touched by this batch are read, merged, and atomically
    rewritten — O(batch), not O(corpus). A kill mid-write corrupts at most one
    shard (temp+rename), never the whole corpus.
    """
    from diskrot.structure import _atomic_write_json, _shard_path, _struct_bucket

    by_bucket: dict[int, dict] = {}
    n_done, n_empty, n_failed = 0, 0, 0
    for key, result, error in results:
        if error is not None:
            print(f"FAILED {key}: {error}")
            n_failed += 1
            continue
        by_bucket.setdefault(_struct_bucket(key), {})[key] = result
        if result is None:
            n_empty += 1
        else:
            n_done += 1

    # See commits from other warm-reused save_results containers before
    # read-merge-write, else a stale shard view would overwrite their entries.
    tokens_vol.reload()

    for bucket, new_entries in by_bucket.items():
        shard = _shard_path(STRUCTURE_DIR, bucket)
        existing = json.loads(shard.read_text()) if shard.exists() else {}
        existing.update(new_entries)
        _atomic_write_json(shard, existing)

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
    print(f"\nanalyzed: {n_done}  empty: {n_empty}  failed: {n_failed}")
    print(f"wrote {len(by_bucket)} shard(s)")


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=24 * 60 * 60,
    # The driver is a single point of failure: if its container is preempted or a
    # save_results.remote() raises, the whole fan-out is orphaned (containers keep
    # billing with nothing collecting their results). nonpreemptible stops Spot
    # preemption from killing it mid-.map() (a preemption restart cancels every
    # in-flight input — those cancellations were the per-container "errors"); same
    # as run_tokenize / run_auto_tag. retries is the crash backstop, and
    # resume-by-skip (list_pending) makes re-execution idempotent — it just skips
    # the files already committed and continues.
    nonpreemptible=True,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def orchestrate(flush_every: int = 50, limit: int = 0, batch_size: int = 8,
                wave_id: str = ""):
    """Dispatch analysis and merge results into the sharded structure dir.
    ``wave_id`` scopes the pass to /corpus/waves/wave_<id>.

    Runs the ``.map()`` collect/flush loop *remotely* so ``--detach`` survives
    terminal close. ``limit`` (>0) caps the number of pending files dispatched —
    use it for the calibration run. ``list_pending`` skips songs already in the
    shards, so re-launching resumes. ``batch_size`` songs share one allin1 call
    (one demucs subprocess + one ensemble load per chunk instead of per song).
    """
    pending = list_pending.remote(wave_id=wave_id)
    if limit and limit > 0:
        pending = pending[:limit]
    if not pending:
        print("Nothing to analyze — all files already done")
        return

    chunks = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    print(f"Dispatching {len(pending)} files as {len(chunks)} chunks of <= {batch_size} "
          f"across parallel containers (flush_every={flush_every})...")
    analyzer = Analyzer()
    batch: list = []
    n_seen = 0
    n_errors = 0
    # order_outputs=False: a preempted chunk must not head-of-line-block the yield
    # (idling other billing containers); results flush by key so order is moot.
    # return_exceptions=True: a poison chunk must not crash the orchestrator
    # — count it; its files stay pending and are redone on the next launch.
    for result in analyzer.analyze_chunk.map(
        chunks, order_outputs=False, return_exceptions=True
    ):
        if isinstance(result, Exception):
            n_errors += 1
            if n_errors <= 20:
                print(f"CHUNK FAILED (files stay pending, redone next run): "
                      f"{type(result).__name__}: {str(result)[:140]}")
            continue
        n_seen += len(result)
        batch.extend(result)
        if len(batch) >= flush_every:
            print(f"flushing {len(batch)} results ({n_seen}/{len(pending)} done)")
            save_results.remote(batch)
            batch = []
    if batch:
        print(f"final flush of {len(batch)} results ({n_seen}/{len(pending)} done)")
        save_results.remote(batch)
    if n_errors:
        print(f"chunk errors: {n_errors} "
              f"(transient — affected files stay pending; re-run to finish them)")


@app.local_entrypoint()
def main(flush_every: int = 50, limit: int = 0, batch_size: int = 8,
         wave_id: str = ""):
    """Spawn the remote orchestrator and return immediately.

    Use with ``--detach`` (both pieces required: ``.spawn()`` so the entrypoint
    exits without blocking, and ``--detach`` so the app isn't auto-stopped when
    the entrypoint completes). ``--limit 200`` runs the calibration subset.
    --wave-id N scopes analysis to /corpus/waves/wave_N.
    """
    call = orchestrate.spawn(flush_every, limit, batch_size, wave_id=wave_id)
    print(f"spawned orchestrator: function call id {call.object_id}")
    print("Follow logs in the Modal dashboard; safe to close this terminal "
          "if launched with --detach.")
