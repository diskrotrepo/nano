"""Modal entrypoint: per-song global tempo (BPM) estimation -> /tokens/tempo.json.

The cheap, DENSE tempo source. The ``<tempo_*>`` header marker only needs a coarse
14-bucket value (``model.lyric_encoder.bpm_to_id``), so instead of paying the full
allin1 structure pass (Demucs + madmom DBN beat decode on a GPU) for every song, this
runs a librosa global-tempo estimate over the waveform on cheap CPU containers — no
torch GPU, no demucs, no madmom, no allin1. It covers 100% of songs, so it recovers
the tempo coverage lost when the structure pass is sampled (``modal_structure.py
--sample-pct``); the dataset prefers tempo.json over structure bpm where both exist.

Mirrors modal_melody.py (corpus-decode fan-out, ``subdir`` parameter) for the read
side, and modal_structure.save_results (reload -> read-merge-write -> retried commit)
for the single shared ``tempo.json`` output. Resumable: ``list_pending`` skips songs
already in tempo.json, so re-launching (or running per-wave) continues.

Run::

    modal run --detach diskrot/modal_tempo.py                 # full sweep
    modal run --detach diskrot/modal_tempo.py --wave-id 0     # one wave
    modal run --detach diskrot/modal_tempo.py --limit 200     # calibration subset

Monitor::

    modal app logs nano-tempo
"""
# NOTE: no `from __future__ import annotations` — Modal's @app.cls + modal.parameter()
# (``subdir`` below) reads raw annotations and rejects PEP 563 stringified ones (same
# gotcha documented in modal_melody.py).

import json
import time
from pathlib import Path

import modal

from diskrot.modal_common import corpus_mount, wave_subdir

app = modal.App("nano-tempo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4",  # diskrot.melody imports model.codec (constants only, no GPU)
        "librosa>=0.10",
        "numpy>=1.26",
        "soundfile>=0.12",
    )
    .add_local_python_source("model", "diskrot")
)

corpus_vol = corpus_mount()  # R2 audio bucket (read-only); see modal_common.corpus_mount
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)

TEMPO_JSON = "/tokens/tempo.json"


@app.cls(
    image=image,
    cpu=2.0,
    timeout=60 * 60,
    max_containers=50,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
class TempoExtractor:
    # '' = flat legacy layout; 'waves/wave_<id>' reads mp3s from that subdir.
    subdir: str = modal.parameter(default="")

    @modal.method()
    def extract_batch(self, stems: list[str]) -> list[tuple[str, float]]:
        """Estimate tempo for a batch of song stems.

        Returns only the ``(stem, bpm)`` pairs that produced a usable estimate;
        silent/degenerate songs and decode failures are dropped (-> the song gets
        no entry -> ``<unknown_tempo>`` at train time). The orchestrator merges the
        returned pairs into the shared tempo.json centrally (single file, so a
        worker can't safely write it without racing other workers)."""
        from diskrot.melody import SAMPLE_RATE, _load_audio_file
        from diskrot.tempo_detect import estimate_tempo

        corpus_dir = Path("/corpus") / self.subdir
        out: list[tuple[str, float]] = []
        n_skip = n_fail = 0
        for stem in stems:
            mp3 = corpus_dir / f"{stem}.mp3"
            if not mp3.exists():
                n_skip += 1
                continue
            try:
                bpm = estimate_tempo(_load_audio_file(str(mp3)), SAMPLE_RATE)
            except Exception as e:  # noqa: BLE001 — one bad file must not kill the batch
                print(f"FAILED {stem}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                n_fail += 1
                continue
            if bpm is None:
                n_skip += 1
                continue
            out.append((stem, bpm))
        if n_skip or n_fail:
            print(f"batch: {len(out)} ok, {n_skip} no-beat/missing, {n_fail} failed", flush=True)
        return out


@app.function(image=image, volumes={"/corpus": corpus_vol, "/tokens": tokens_vol})
def list_pending(wave_id: str = "", sample_pct: int = 100) -> list[str]:
    """Stems with an mp3 under /corpus/<wave subdir> not yet in tempo.json.
    ``sample_pct`` (<100) is offered for symmetry with the structure pass but
    defaults to 100 — tempo is the DENSE pass and should cover every song."""
    from diskrot.structure import _sample_keep

    mp3s = sorted((Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3"))
    existing = json.loads(Path(TEMPO_JSON).read_text()) if Path(TEMPO_JSON).exists() else {}
    pending = [m.stem for m in mp3s
               if m.stem not in existing and _sample_keep(m.stem, sample_pct)]
    print(f"found {len(mp3s)} mp3s, {len(existing)} already done, {len(pending)} pending")
    return pending


@app.function(image=image, volumes={"/corpus": corpus_vol, "/tokens": tokens_vol})
def save_tempo(results: list[tuple[str, float]]):
    """Merge a batch of (stem, bpm) pairs into the shared /tokens/tempo.json.

    Single shared file (like keys.json), so reload -> read -> update -> atomic
    write -> retried commit. Called SERIALLY by the orchestrator (awaited, not
    spawned) so concurrent writers can't clobber each other's entries."""
    from diskrot.tempo_detect import _atomic_write_json

    # See commits from any earlier save before read-merge-write.
    tokens_vol.reload()
    out = Path(TEMPO_JSON)
    tempo = json.loads(out.read_text()) if out.exists() else {}
    for stem, bpm in results:
        tempo[stem] = {"bpm": float(bpm)}
    _atomic_write_json(tempo, out)

    # Retry a transient DataLossError so a flush blip doesn't crash the orchestrator.
    for attempt in range(3):
        try:
            tokens_vol.commit()
            break
        except modal.exception.DataLossError as e:
            if attempt == 2:
                raise
            print(f"commit failed ({e}); retry {attempt + 1}/2", flush=True)
            time.sleep(2.0 * (attempt + 1))
    print(f"tempo.json now has {len(tempo)} songs (+{len(results)} this flush)", flush=True)


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=24 * 60 * 60,
    nonpreemptible=True,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def orchestrate(batch_size: int = 200, limit: int = 0, wave_id: str = "",
                sample_pct: int = 100, flush_every: int = 2000):
    """Dispatch tempo estimation across CPU containers and merge into tempo.json.

    Runs the ``.map()`` collect/flush loop remotely so ``--detach`` survives
    terminal close. ``limit`` (>0) caps pending files (calibration). Resumes by
    skip via ``list_pending``."""
    tokens_vol.reload()
    pending = list_pending.remote(wave_id=wave_id, sample_pct=sample_pct)
    if limit and limit > 0:
        pending = pending[:limit]
    if not pending:
        print("Nothing to estimate — all songs already in tempo.json")
        return

    chunks = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
    print(f"Dispatching {len(pending)} songs as {len(chunks)} batches of <= {batch_size} "
          f"(flush_every={flush_every})...")
    extractor = TempoExtractor(subdir=wave_subdir(wave_id))
    batch: list = []
    n_seen = 0
    t0 = time.time()
    # order_outputs=False: a slow batch doesn't head-of-line-block. return_exceptions
    # =True: a hard worker crash counts and continues (those stems stay pending).
    for result in extractor.extract_batch.map(
        chunks, order_outputs=False, return_exceptions=True
    ):
        if isinstance(result, Exception):
            print(f"BATCH FAILED (stays pending): {type(result).__name__}: {str(result)[:140]}")
            continue
        n_seen += len(result)
        batch.extend(result)
        if len(batch) >= flush_every:
            save_tempo.remote(batch)  # awaited (serial) — single shared file, no race
            rate = n_seen / max(time.time() - t0, 1e-6)
            print(f"flushed ({n_seen} estimated, {rate:.1f}/s)", flush=True)
            batch = []
    if batch:
        save_tempo.remote(batch)
    print(f"\nDONE: {n_seen} tempo estimates over {len(pending)} pending songs")


@app.local_entrypoint()
def main(batch_size: int = 200, limit: int = 0, wave_id: str = "",
         sample_pct: int = 100):
    """Spawn the remote orchestrator and return (use with --detach).
    --wave-id N scopes to /corpus/waves/wave_N. --limit 200 is the calibration run."""
    call = orchestrate.spawn(batch_size, limit, wave_id=wave_id, sample_pct=sample_pct)
    print(f"spawned orchestrator: function call id {call.object_id}")
    print("monitor with: modal app logs nano-tempo "
          "(safe to close terminal if launched with --detach)")
