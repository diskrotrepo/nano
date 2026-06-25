"""Modal entrypoint: extract per-song stem-token streams for /addstem conditioning.

For each tokenized song it Demucs-separates the 4 stems (drums/bass/vocals/other),
encodes each with the active codec, and writes ``<name>.stems.npy`` (a
``[4, depth, T]`` int16 array, stems in ``STEM_TYPES`` order, force-aligned to the
song's token frame count) to the dedicated **nano-stems** volume.
``diskrot.pack_cache`` then folds those into the parallel ``packed_NNN.stem.bin``
sidecar (``--stem-cache-dir /stems``) so the dataset can co-crop stems with tokens.

Pipeline order: tokenize → **stems** → pack (tokens+chroma+stems) → train. Stems
need the song's token frame count (read from ``<name>.pt``) to align frame-for-frame,
so this runs AFTER tokenize (mirrors melody).

GPU stage (unlike CPU-only melody): Demucs (torch) AND the SpectroStream codec
(TF/JAX) both run on the GPU. They coexist on one A100 with the XLA memory fraction
held low so torch/Demucs has headroom — calibrate with ``--limit`` (a small wave)
before a full ingest, exactly as the plan's scale gate requires.

Inode note: one ``<name>.stems.npy`` per song (~all 4 stems bundled = 1 inode/song,
like melody) on its OWN volume so it doesn't push nano-tokens over the 500k cap.

Cost: this is the most expensive optional prep stage (Demucs + 4x codec encode on
GPU). ``--sample-pct 50`` runs it on a deterministic half of the corpus; the rest
are flagged absent by the packer's present mask and skipped as /addstem targets —
graceful degradation, and stem_prob=0.15 means only ~15% of train batches use stems
anyway. Calibrate $/song with ``--limit`` on one wave first.

Run::

    NANO_CODEC=spectrostream modal run --detach diskrot/modal_stems.py --wave-id <id> --sample-pct 50

Monitor::

    modal app logs nano-stems
"""
# NOTE: do NOT add `from __future__ import annotations` here — Modal's
# @app.cls + modal.parameter() type validation rejects PEP 563-stringified
# annotations (see modal_melody.py / modal_tokenize.py for the gory details).

import hashlib
import os
import time
from pathlib import Path

import modal

from diskrot.modal_common import assert_stage_produced_output, corpus_mount


def _sample_keep(stem: str, sample_pct: int) -> bool:
    """Deterministic membership in the sampled subset, in [0, sample_pct) of 100.

    Stems are the most expensive optional prep stage, and a song WITHOUT stems is
    simply flagged absent by the packer (present mask) and skipped as a stem-add
    target — graceful degradation — so we can run the stage on a fraction of the
    corpus. ``sample_pct >= 100`` keeps all, ``<= 0`` keeps none. Uses sha1 (stable
    across processes, so a re-run never re-decides membership — resume stays
    correct), with a stems-specific ``"stem-sample:"`` salt so the kept set is NOT
    correlated with the structure stage's sample (more songs get some conditioning)."""
    if sample_pct >= 100:
        return True
    if sample_pct <= 0:
        return False
    return int(hashlib.sha1(("stem-sample:" + stem).encode()).hexdigest()[:8], 16) % 100 < sample_pct

app = modal.App("nano-stems")

# Codec chosen LOCALLY at `modal run` time via NANO_CODEC (mirrors modal_tokenize),
# so the image is built for the right codec. The stem stage adds Demucs on top.
_CODEC = os.environ.get("NANO_CODEC", "dac").lower()
_IS_SS = _CODEC in ("spectrostream", "ss")
_MAGENTA_GPU_IMAGE = "us-docker.pkg.dev/brain-magenta/magenta-rt/magenta-rt:gpu"

if _IS_SS:
    # SpectroStream (TF/JAX on GPU) + Demucs (torch on GPU) in one container. Unlike
    # modal_tokenize's SS image (which installs CPU torch), we need CUDA torch so
    # Demucs separates on the GPU — so the XLA memory fraction is held lower to leave
    # room for torch/Demucs alongside the JAX/TF codec.
    image = (
        modal.Image.from_registry(_MAGENTA_GPU_IMAGE)
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install("soundfile>=0.12", "librosa>=0.10", "numpy>=1.26", "tqdm>=4.66")
        .pip_install(
            "torch==2.4.1", "torchaudio==2.4.1",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install("demucs>=4.0")
        .env(
            {
                "NANO_CODEC": "spectrostream",
                "NANO_SS_DEPTH": os.environ.get("NANO_SS_DEPTH", "32"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                # Lower than tokenize's 0.4: leave GPU headroom for torch/Demucs.
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "GRPC_VERBOSITY": "ERROR",
                "GLOG_minloglevel": "2",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HOME": "/cache/hf",
                "HF_HUB_CACHE": "/cache/hf",
                "XDG_CACHE_HOME": "/cache",
            }
        )
        .add_local_python_source("model", "diskrot")
    )
    _GPU = os.environ.get("NANO_SPIKE_GPU", "A100-40GB")
    _cache_vol = modal.Volume.from_name("nano-ss-cache", create_if_missing=True)
else:
    # DAC path (local/legacy): CUDA torch carries both DAC and Demucs, like transcribe.
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install(
            "torch==2.4.1", "torchaudio==2.4.1",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "librosa>=0.10", "descript-audio-codec>=1.0.0", "numpy>=1.26",
            "soundfile>=0.12", "demucs>=4.0",
        )
        .run_commands("pip install 'protobuf>=4'")
        .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        .add_local_python_source("model", "diskrot")
    )
    _GPU = "L4"
    _cache_vol = None

corpus_vol = corpus_mount()  # R2 audio bucket (read-only)
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
# Dedicated volume for the per-song stem sidecars (own volume → keeps the ~1
# file/song they add off nano-tokens and its 500k-inode cap, like nano-melody).
stems_vol = modal.Volume.from_name("nano-stems", create_if_missing=True)

_STEM_EXT = ".stems.npy"  # must match diskrot.pack_cache._STEM_EXT
_BATCH = 100              # songs per worker call (Demucs is heavy → smaller than melody)

_VOLUMES = {"/corpus": corpus_vol, "/tokens": tokens_vol, "/stems": stems_vol}
if _cache_vol is not None:
    _VOLUMES["/cache"] = _cache_vol


@app.cls(
    image=image,
    gpu=_GPU,
    timeout=60 * 60,
    max_containers=50,
    volumes=_VOLUMES,
)
class StemExtractor:
    # '' = flat legacy layout; 'waves/wave_<id>' isolates a wave's stems for
    # pack_append + per-wave cleanup.
    subdir: str = modal.parameter(default="")

    @modal.enter()
    def load_models(self):
        from model.codec import get_codec
        from diskrot.stems import load_demucs

        self.codec = get_codec(device="cuda")
        self.demucs = load_demucs("cuda")

    @modal.method()
    def extract_batch(self, stem_names: list[str]) -> tuple[int, int, int]:
        """Extract stem tokens for a batch of songs. Returns (done, missing, failed).

        Each ``<name>.stems.npy`` is written atomically (temp + os.replace); the
        whole batch is committed once so a preemption loses at most one in-flight
        batch (those songs stay pending and are redone next run)."""
        import numpy as np
        import torch

        from diskrot.stems import extract_stem_array

        corpus_dir = Path("/corpus") / self.subdir
        pt_dir = Path("/tokens") / self.subdir
        out_dir = Path("/stems") / self.subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        n_done = n_missing = n_failed = 0
        for name in stem_names:
            mp3 = corpus_dir / f"{name}.mp3"
            pt = pt_dir / f"{name}.pt"
            out = out_dir / f"{name}{_STEM_EXT}"
            if not mp3.exists() or not pt.exists():
                n_missing += 1
                continue
            try:
                toks = torch.load(pt, weights_only=True, map_location="cpu")
                n_frames = int(toks.shape[1])
                arr = extract_stem_array(
                    str(mp3), self.codec, self.demucs, n_frames=n_frames, device="cuda"
                )  # [4, depth, n_frames] int16
                tmp = out.with_suffix(out.suffix + ".tmp")
                with open(tmp, "wb") as fh:
                    np.save(fh, arr)
                os.replace(tmp, out)
                n_done += 1
            except Exception as e:  # noqa: BLE001 — one bad file must not kill the batch
                print(f"FAILED {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                n_failed += 1
        stems_vol.commit()
        return (n_done, n_missing, n_failed)


def _wave_subdir(wave_id: str) -> str:
    return f"waves/wave_{wave_id}" if wave_id else ""


@app.function(image=image, volumes={"/tokens": tokens_vol, "/stems": stems_vol})
def list_pending(wave_id: str = "", sample_pct: int = 100) -> list[str]:
    """Songs with a tokenized ``.pt`` but no ``.stems.npy`` yet, within the subdir.

    ``sample_pct`` (<100) keeps only the deterministic ``_sample_keep`` subset — the
    cost lever; the non-sampled songs are never queued (no ``.stems.npy`` → the
    packer flags them absent → the dataset skips them as stem-add targets)."""
    sub = _wave_subdir(wave_id)
    pt_stems = {p.stem for p in (Path("/tokens") / sub).glob("*.pt")}
    done = {p.name[: -len(_STEM_EXT)]
            for p in (Path("/stems") / sub).glob(f"*{_STEM_EXT}")}
    pending = sorted(s for s in (pt_stems - done) if _sample_keep(s, sample_pct))
    sampled = "" if sample_pct >= 100 else f" (sampled to {sample_pct}%)"
    print(f"{len(pt_stems)} tokenized, {len(done)} with stems, "
          f"{len(pending)} pending{sampled}")
    return pending


@app.function(
    image=image,
    volumes=_VOLUMES,
    timeout=24 * 60 * 60,
)
def orchestrate(batch: int = _BATCH, wave_id: str = "", limit: int = 0,
                sample_pct: int = 100):
    """Dispatch stem extraction across parallel GPU containers (runs remotely so
    ``--detach`` survives terminal close). ``wave_id`` scopes to the wave subdir;
    ``limit`` (>0) caps the pass to the first N pending songs for cost calibration;
    ``sample_pct`` (<100) runs the stage on only a deterministic fraction of the
    corpus — the cost lever (the rest fall back to no-stem via the present mask)."""
    tokens_vol.reload()
    stems_vol.reload()
    sub = _wave_subdir(wave_id)
    pending = list_pending.remote(wave_id=wave_id, sample_pct=sample_pct)
    if limit and limit > 0:
        pending = pending[:limit]
        print(f"[calibration] limited to first {len(pending)} songs")
    if not pending:
        print("Nothing to extract — all tokenized songs already have stems")
        return

    chunks = [pending[i:i + batch] for i in range(0, len(pending), batch)]
    print(f"Dispatching {len(pending)} songs in {len(chunks)} batches of {batch}...")
    extractor = StemExtractor(subdir=sub)
    t0 = time.time()
    tot_done = tot_missing = tot_failed = 0
    n_chunks_seen = 0
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
        if n_chunks_seen % 10 == 0:
            rate = (tot_done + tot_failed) / max(time.time() - t0, 1e-6)
            print(f"{n_chunks_seen}/{len(chunks)} batches  done={tot_done} "
                  f"missing={tot_missing} failed={tot_failed}  ({rate:.2f}/s)", flush=True)
    print(f"\nDONE: stems={tot_done}  missing_inputs={tot_missing}  failed={tot_failed}")
    assert_stage_produced_output("stems", tot_done, len(pending), tot_failed)


@app.local_entrypoint()
def main(batch: int = _BATCH, wave_id: str = "", limit: int = 0, sample_pct: int = 100):
    """Spawn the remote orchestrator and return (use with --detach).
    --wave-id N scopes to /…/waves/wave_N; --limit N calibrates cost on N songs;
    --sample-pct 50 runs stems on a deterministic 50% of the corpus (cost lever)."""
    call = orchestrate.spawn(batch, wave_id=wave_id, limit=limit, sample_pct=sample_pct)
    print(f"spawned orchestrator: function call id {call.object_id}")
    print("monitor with: modal app logs nano-stems "
          "(safe to close terminal if launched with --detach)")
