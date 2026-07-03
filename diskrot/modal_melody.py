"""Modal entrypoint: extract per-song chromagrams for melody conditioning.

For each tokenized song it computes the 12-bin chroma (``diskrot.melody``) and
writes ``<name>.mel.npy`` to the dedicated **nano-melody** volume (NOT next to
the ``<name>.pt`` on nano-tokens — see the inode note below).
``diskrot.pack_cache`` then folds those into the parallel ``packed_NNN.mel.bin``
sidecar (``--mel-cache-dir /melody``) so the dataset can co-crop melody with the
tokens.

Pipeline order: tokenize → **melody** → pack (tokens+chroma) → train. Melody
needs the song's DAC frame count (read from ``<name>.pt``) to force-align the
chroma frame-for-frame, so it runs AFTER tokenize.

CPU-only and embarrassingly parallel: each ``.mel.npy`` is independent (no
read-merge-write shard contention like lyrics), so workers write their own files
and commit per batch. Resumable — ``list_pending`` skips songs that already have
a ``.mel.npy``.

Note on inodes: this adds one small (~60 KB fp16) file per song. nano-tokens
already holds ~one ``.pt`` per song and is near the 500k-inode volume cap, so the
chroma files live on their **own nano-melody volume** — co-locating them on
nano-tokens would push the loose-file count over the cap mid-run. Both the
``.pt`` and the ``.mel.npy`` are only inputs to ``pack_cache`` and may be pruned
after packing (training reads only the packed shards).

Run::

    modal run --detach diskrot/modal_melody.py

Monitor::

    modal app logs nano-melody
"""
# NOTE: do NOT add `from __future__ import annotations` here. Modal's class
# parameter validation (`@app.cls` + `modal.parameter()`, e.g. `subdir` below)
# reads raw type annotations and rejects them when PEP 563 stringifies them —
# you'll see `KeyError: 'str'` / `AttributeError: 'str' object has no attribute
# '__name__'` at deploy time. Python 3.12 doesn't need the future import anyway
# (`list[str]`, `tuple[int, int, int]` work natively).

import os
import time
from pathlib import Path

import modal

from diskrot.modal_common import assert_stage_produced_output, corpus_mount

app = modal.App("nano-melody")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4",
        "librosa>=0.10",
        "numpy>=1.26",
        "soundfile>=0.12",
    )
    # MUST include "model": the worker imports diskrot.melody, which imports
    # `from model.codec import codec_constants` (for the codec frame rate that
    # aligns the chroma). Without "model" baked in, every batch dies with
    # ModuleNotFoundError("No module named 'model'") -> 0 chroma written, yet the
    # orchestrator still returns "DONE: chroma=0" -> melody silently no-ops on
    # every wave (and /cover trains on nothing). torch is already pip-installed
    # above, so the model.codec top-level `import torch` resolves fine.
    .add_local_python_source("model", "diskrot")
)

corpus_vol = corpus_mount()  # R2 audio bucket (read-only); see modal_common.corpus_mount
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
# Dedicated volume for the per-song chroma sidecars: keeps the ~1 file/song they
# add off nano-tokens, which is already near the 500k-inode cap.
melody_vol = modal.Volume.from_name("nano-melody", create_if_missing=True)

_MEL_EXT = ".mel.npy"  # must match diskrot.pack_cache._MEL_EXT
_BATCH = 200           # songs per worker call (bounds commit frequency)


@app.cls(
    image=image,
    # The decode + chroma_cqt loop is single-core serial, so 1 reserved core is
    # the right size; 100 x 1-core containers = the same peak reservation as the
    # old 50 x 2-core at roughly half the wall-clock.
    cpu=1.0,
    memory=2048,
    timeout=60 * 60,
    max_containers=100,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol, "/melody": melody_vol},
)
class MelodyExtractor:
    # Wave ingestion: '' = flat legacy layout; 'waves/wave_<id>' reads mp3/.pt
    # from and writes .mel.npy into that subdir so a wave's chroma is isolated
    # for pack_append + per-wave cleanup.
    subdir: str = modal.parameter(default="")

    @modal.method()
    def extract_batch(self, stems: list[str]) -> tuple[int, int, int]:
        """Extract chroma for a batch of song stems. Returns (done, missing, failed).

        Each ``<stem>.mel.npy`` is written atomically (temp + os.replace) and the
        whole batch is committed once at the end so a preemption loses at most one
        in-flight batch (those stems stay pending and are redone next run).
        """
        import numpy as np
        import torch

        from diskrot.melody import extract_chroma

        corpus_dir = Path("/corpus") / self.subdir
        pt_dir = Path("/tokens") / self.subdir
        mel_dir = Path("/melody") / self.subdir
        mel_dir.mkdir(parents=True, exist_ok=True)
        n_done = n_missing = n_failed = 0
        for stem in stems:
            mp3 = corpus_dir / f"{stem}.mp3"
            pt = pt_dir / f"{stem}.pt"
            out = mel_dir / f"{stem}{_MEL_EXT}"
            if not mp3.exists() or not pt.exists():
                n_missing += 1
                continue
            try:
                # DAC frame count for exact alignment with the token tensor.
                toks = torch.load(pt, weights_only=True, map_location="cpu")
                n_frames = int(toks.shape[1])
                chroma = extract_chroma(str(mp3), n_frames=n_frames)  # [12, n_frames] f32
                tmp = out.with_suffix(out.suffix + ".tmp")
                # Write via an open handle: np.save appends ".npy" to any path
                # arg that doesn't already end in ".npy", which would silently
                # retarget the temp file and break the os.replace below.
                with open(tmp, "wb") as fh:
                    np.save(fh, chroma.astype(np.float16))
                os.replace(tmp, out)
                n_done += 1
            except Exception as e:  # noqa: BLE001 — one bad file must not kill the batch
                print(f"FAILED {stem}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                n_failed += 1
        melody_vol.commit()
        return (n_done, n_missing, n_failed)


def _wave_subdir(wave_id: str) -> str:
    """'' (flat legacy layout) or 'waves/wave_<id>' for a wave ingest."""
    return f"waves/wave_{wave_id}" if wave_id else ""


@app.function(image=image, volumes={"/tokens": tokens_vol, "/melody": melody_vol})
def list_pending(wave_id: str = "") -> list[str]:
    """Stems with a tokenized ``.pt`` (nano-tokens) but no ``.mel.npy`` yet
    (nano-melody), within the (wave) subdir."""
    sub = _wave_subdir(wave_id)
    pt_stems = {p.stem for p in (Path("/tokens") / sub).glob("*.pt")}
    done = {p.name[: -len(_MEL_EXT)]
            for p in (Path("/melody") / sub).glob(f"*{_MEL_EXT}")}
    pending = sorted(pt_stems - done)
    print(f"{len(pt_stems)} tokenized, {len(done)} with chroma, {len(pending)} pending")
    return pending


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol, "/melody": melody_vol},
    timeout=24 * 60 * 60,
)
def orchestrate(batch: int = _BATCH, wave_id: str = ""):
    """Dispatch chroma extraction across parallel containers (runs remotely so
    ``--detach`` survives terminal close — mirrors modal_transcribe.orchestrate).
    ``wave_id`` scopes the pass to the /…/waves/wave_<id> subdirs."""
    tokens_vol.reload()
    melody_vol.reload()
    sub = _wave_subdir(wave_id)
    pending = list_pending.remote(wave_id=wave_id)
    if not pending:
        print("Nothing to extract — all tokenized songs already have chroma")
        return

    chunks = [pending[i:i + batch] for i in range(0, len(pending), batch)]
    print(f"Dispatching {len(pending)} songs in {len(chunks)} batches of {batch}...")
    extractor = MelodyExtractor(subdir=sub)
    t0 = time.time()
    tot_done = tot_missing = tot_failed = 0
    n_chunks_seen = 0
    # order_outputs=False so a slow batch doesn't head-of-line-block; each batch
    # commits its own files, so order is irrelevant. return_exceptions=True so a
    # hard worker crash counts and continues (those stems stay pending).
    for res in extractor.extract_batch.map(
        chunks, order_outputs=False, return_exceptions=True
    ):
        n_chunks_seen += 1
        if isinstance(res, Exception):
            print(f"BATCH FAILED (stays pending): {type(res).__name__}: {str(res)[:140]}")
            continue
        d, m, f = res
        tot_done += d
        tot_missing += m
        tot_failed += f
        if n_chunks_seen % 20 == 0:
            rate = (tot_done + tot_failed) / max(time.time() - t0, 1e-6)
            print(f"{n_chunks_seen}/{len(chunks)} batches  done={tot_done} "
                  f"missing={tot_missing} failed={tot_failed}  ({rate:.1f}/s)", flush=True)
    print(f"\nDONE: chroma={tot_done}  missing_inputs={tot_missing}  failed={tot_failed}")
    # Hard-stop a silent no-op (the exact failure mode this stage just had: the
    # worker image missing "model" → 0 chroma while marked done).
    assert_stage_produced_output("melody", tot_done, len(pending), tot_failed)


@app.local_entrypoint()
def main(batch: int = _BATCH, wave_id: str = ""):
    """Spawn the remote orchestrator and return (use with --detach).
    --wave-id N scopes to /…/waves/wave_N."""
    call = orchestrate.spawn(batch, wave_id=wave_id)
    print(f"spawned orchestrator: function call id {call.object_id}")
    print("monitor with: modal app logs nano-melody "
          "(safe to close terminal if launched with --detach)")
