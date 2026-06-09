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

app = modal.App("nano-structure")


def _cache_demucs():
    # allin1 uses demucs internally for source separation; pre-cache it into the
    # image so containers don't each re-download on cold start.
    from demucs.pretrained import get_model

    get_model("htdemucs")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
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
    .pip_install(
        "allin1",
        "librosa>=0.10",
        "numpy>=1.26",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "demucs",
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
)
class Analyzer:
    @modal.enter()
    def load_models(self):
        # Importing allin1 + warming demucs amortizes model load across the many
        # files this container will process.
        import allin1  # noqa: F401
        from demucs.pretrained import get_model

        get_model("htdemucs")

    @modal.method()
    def analyze_file(self, mp3_name: str) -> tuple[str, dict | None, str | None]:
        """Analyze one file. Returns (stem, result_or_None, error_or_None)."""
        import allin1

        from diskrot.structure import analyze_file as _analyze_file

        mp3_path = Path("/corpus") / mp3_name
        key = mp3_path.stem
        try:
            result = _analyze_file(allin1.analyze, mp3_path, "cuda")
            return (key, result, None)
        except Exception as e:
            return (key, None, str(e))


STRUCTURE_DIR = "/tokens/structure"


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def list_pending() -> list[str]:
    """Return mp3 filenames not yet in the sharded structure dir."""
    from diskrot.structure import load_structure_shards

    mp3s = sorted(Path("/corpus").glob("*.mp3"))
    existing = load_structure_shards(STRUCTURE_DIR)
    pending = [mp3.name for mp3 in mp3s if mp3.stem not in existing]
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
)
def orchestrate(flush_every: int = 200, limit: int = 0):
    """Dispatch analysis and merge results into the sharded structure dir.

    Runs the ``.map()`` collect/flush loop *remotely* so ``--detach`` survives
    terminal close. ``limit`` (>0) caps the number of pending files dispatched —
    use it for the calibration run. ``list_pending`` skips songs already in the
    shards, so re-launching resumes.
    """
    pending = list_pending.remote()
    if limit and limit > 0:
        pending = pending[:limit]
    if not pending:
        print("Nothing to analyze — all files already done")
        return

    print(f"Dispatching {len(pending)} files across parallel containers "
          f"(flush_every={flush_every})...")
    analyzer = Analyzer()
    batch: list = []
    n_seen = 0
    n_errors = 0
    # order_outputs=False: a preempted file must not head-of-line-block the yield
    # (idling other billing containers); results flush by key so order is moot.
    # return_exceptions=True: a single poison file must not crash the orchestrator
    # — count it; the file stays pending and is redone on the next launch.
    for result in analyzer.analyze_file.map(
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


@app.local_entrypoint()
def main(flush_every: int = 200, limit: int = 0):
    """Spawn the remote orchestrator and return immediately.

    Use with ``--detach`` (both pieces required: ``.spawn()`` so the entrypoint
    exits without blocking, and ``--detach`` so the app isn't auto-stopped when
    the entrypoint completes). ``--limit 200`` runs the calibration subset.
    """
    call = orchestrate.spawn(flush_every, limit)
    print(f"spawned orchestrator: function call id {call.object_id}")
    print("Follow logs in the Modal dashboard; safe to close this terminal "
          "if launched with --detach.")
