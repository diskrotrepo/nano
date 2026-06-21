"""Modal entrypoint for tokenizing mp3s on GPU, fanned out across many containers.

Mirrors the parallelism pattern from modal_transcribe.py: one container per GPU,
each loads DAC once via @modal.enter() and processes files via .map().

Setup (one-time):
    # raw audio lives in the R2 nano-audio bucket, under waves/wave_<id>/
    # (NANO_AUDIO_BUCKET / NANO_AUDIO_ENDPOINT + the r2-creds secret)
    modal volume create nano-tokens

Run tokenization (detached so the local shell can disconnect):
    modal run --detach diskrot/modal_tokenize.py --wave-id <id>

Pull tokens locally (optional):
    modal volume get nano-tokens / ./token_cache/
"""
# NOTE: do NOT add `from __future__ import annotations` here. Modal's class
# parameter validation (`@app.cls` + `modal.parameter()`) reads raw type
# annotations and rejects them when they get stringified by PEP 563 — you'll
# see `KeyError: 'float'` / `AttributeError: 'str' object has no attribute
# '__name__'` at module import time. Python 3.12 doesn't need the future
# import anyway (`list[str]`, `str | None` work natively).

import os
import time
from pathlib import Path

import modal

from diskrot.modal_common import corpus_mount

app = modal.App("nano-tokenize")

# Codec is chosen LOCALLY at `modal run` time via NANO_CODEC (read from your shell),
# so the image + GPU are built for the right codec. "dac" = the legacy mono image;
# "spectrostream"/"ss" = v9 stereo (Magenta RealTime's prebuilt GPU image carrying
# the JAX/TF/T5X stack — same one the codec spike validated).
_CODEC = os.environ.get("NANO_CODEC", "dac").lower()
_IS_SS = _CODEC in ("spectrostream", "ss")
_MAGENTA_GPU_IMAGE = "us-docker.pkg.dev/brain-magenta/magenta-rt/magenta-rt:gpu"


def _cache_dac():
    # Pre-download the DAC 44kHz checkpoint at image build time so 50 containers
    # don't each fetch ~500MB at cold start.
    import dac

    dac.utils.download(model_type="44khz")


if _IS_SS:
    # SpectroStream: TF/JAX codec (the codec compute is TensorFlow on GPU), so
    # torch is CPU-only (just tensor plumbing in model.codec) to avoid contending
    # with JAX/TF for CUDA. NANO_SS_DEPTH=32 = the stored RVQ depth (the model
    # slices to 24). HF weights cache on nano-ss-cache so 50 containers don't each
    # re-download. pyloudnorm powers the loudness-norm in tokenize._load_audio.
    image = (
        modal.Image.from_registry(_MAGENTA_GPU_IMAGE)
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install("soundfile>=0.12", "librosa>=0.10", "pyloudnorm>=0.1", "tqdm>=4.66")
        .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
        .env(
            {
                "NANO_CODEC": "spectrostream",
                "NANO_SS_DEPTH": os.environ.get("NANO_SS_DEPTH", "32"),
                "NANO_LOUDNORM_LUFS": os.environ.get("NANO_LOUDNORM_LUFS", "-14.0"),
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.4",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
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
            "descript-audio-codec>=1.0.0",
            "numpy>=1.26",
            "tqdm>=4.66",
            "soundfile>=0.12",
            "pyloudnorm>=0.1",
        )
        .run_commands("pip install 'protobuf>=4'")  # override descript-audiotools' old pin
        # `expandable_segments:True` switches the PyTorch caching allocator to a
        # strategy that doesn't fragment as the encoder loop grinds through files.
        # Without it, ~20 files into a container the GPU reports 21 GiB used /
        # 22 GiB total even though no single encode needs >5 GiB — fragmentation
        # eats the rest. PyTorch's own CUDA-OOM error suggests setting this.
        .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        .run_function(_cache_dac)
        .add_local_python_source("model", "diskrot")
    )
    _GPU = "L4"
    _cache_vol = None

corpus_vol = corpus_mount()  # R2 audio bucket (read-only); see modal_common.corpus_mount
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
_VOLUMES = {"/corpus": corpus_vol, "/tokens": tokens_vol}
if _cache_vol is not None:
    _VOLUMES["/cache"] = _cache_vol


@app.cls(
    image=image,
    gpu=_GPU,  # L4 for DAC; A100-40GB for the SpectroStream TF/JAX stack
    timeout=60 * 60,
    max_containers=50,
    volumes=_VOLUMES,
)
class Tokenizer:
    # Modal restricts modal.parameter() types to {int, str, bytes, bool} — no
    # float. Integer seconds is fine here since min_frames = min_seconds * 86
    # rounds to int anyway.
    min_seconds: int = modal.parameter(default=20)
    # Wave ingestion: when set (e.g. "waves/wave_17"), read mp3s from
    # /corpus/<subdir>/ and write .pt to /tokens/<subdir>/ so one wave's tokens
    # are isolated for pack_append + per-wave cleanup. Empty = the flat legacy
    # layout (/corpus/*.mp3 -> /tokens/*.pt).
    subdir: str = modal.parameter(default="")

    @modal.enter()
    def load_codec(self):
        from model.codec import get_codec

        # NANO_CODEC (baked into the image env): dac -> DACodec, spectrostream ->
        # SpectroStreamCodec (stereo, stored depth NANO_SS_DEPTH).
        self.codec = get_codec(device="cuda")
        self.min_frames = int(self.min_seconds * self.codec.FRAME_RATE_HZ)

    @modal.method()
    def tokenize_batch(
        self, mp3_names: list[str]
    ) -> list[tuple[str, str, int, str | None]]:
        """Tokenize a batch of files within one container with prefetched audio
        loading and a single batched DAC forward pass. Returns one
        (key, status, frames, error) per input. ``mp3_names`` are basenames
        within the (wave) corpus subdir."""
        from diskrot.tokenize import tokenize_files_streaming

        corpus_dir = Path("/corpus") / self.subdir
        out_dir = Path("/tokens") / self.subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        items = [
            (corpus_dir / name, out_dir / (Path(name).stem + ".pt"))
            for name in mp3_names
        ]
        out: list[tuple[str, str, int, str | None]] = []
        # batch_size=1: encode one file at a time. Batching multiple 5-min
        # chunks into a single forward pass pads them to the longest and
        # multiplies peak GPU memory by the batch size — OOMs on L4 (22 GiB)
        # at batch_size=8 even after prep splits long tracks to ≤10 min.
        # Per-file encode loses ~30% throughput vs batched but is safe.
        for (mp3_path, _), result in zip(
            items,
            tokenize_files_streaming(
                self.codec, items, self.min_frames, batch_size=1
            ),
        ):
            out.append((mp3_path.stem, result.status, result.frames, result.error))
        # One commit per batch — each container writes independent .pt files so
        # commits don't conflict; batching them amortizes commit overhead.
        # Retry a transient DataLossError ("failed to publish commit to server")
        # so a storage blip doesn't fail the whole batch and bubble up through
        # the orchestrator's .map() loop — which would crash run_tokenize and
        # trigger a full restart + fleet re-spawn (see run_tokenize).
        for attempt in range(3):
            try:
                tokens_vol.commit()
                break
            except modal.exception.DataLossError as e:
                if attempt == 2:
                    raise
                print(f"commit failed ({e}); retry {attempt + 1}/2", flush=True)
                time.sleep(2.0 * (attempt + 1))
        return out


# Lightweight image for the orchestrator (no torch/DAC needed — it just
# lists files, dispatches .map(), and tallies results).
orchestrator_image = modal.Image.debian_slim(python_version="3.12")
if _IS_SS:
    # So the in-container summary reports the SpectroStream frame rate (25 Hz).
    orchestrator_image = orchestrator_image.env({"NANO_CODEC": "spectrostream"})


def _wave_subdir(wave_id: str) -> str:
    """'' (flat legacy layout) or 'waves/wave_<id>' for a wave ingest."""
    return f"waves/wave_{wave_id}" if wave_id else ""


@app.function(
    image=orchestrator_image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def list_pending(wave_id: str = "") -> list[str]:
    """Return mp3 basenames not yet tokenized (no matching .pt) within the
    (wave) corpus subdir."""
    sub = _wave_subdir(wave_id)
    mp3s = sorted((Path("/corpus") / sub).glob("*.mp3"))
    existing = {p.stem for p in (Path("/tokens") / sub).glob("*.pt")}
    pending = [mp3.name for mp3 in mp3s if mp3.stem not in existing]
    print(f"found {len(mp3s)} total mp3s, {len(existing)} already tokenized, "
          f"{len(pending)} pending")
    return pending


@app.function(
    image=orchestrator_image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=60 * 60 * 24,  # 24h cap on the whole orchestration
    # Pin the long-lived coordinator to a non-preemptible instance so it never
    # restarts mid-run (each restart costs a full list_pending() rescan). The
    # GPU Tokenizer fan-out stays preemptible — its .map() inputs auto-retry and
    # commit .pt files per batch. retries kept as a transient-failure backstop;
    # a restart still resumes via list_pending() skipping existing .pt files.
    nonpreemptible=True,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def run_tokenize(min_seconds: int, batch_size: int, wave_id: str = "") -> None:
    """Run the full tokenize pass: list pending, fan out to GPU containers,
    aggregate results, print summary. Designed to be `.spawn()`-ed from the
    local entrypoint so the user can launch and walk away. ``wave_id`` scopes
    the pass to /corpus/waves/wave_<id> -> /tokens/waves/wave_<id>."""
    sub = _wave_subdir(wave_id)
    mp3s = sorted((Path("/corpus") / sub).glob("*.mp3"))
    existing = {p.stem for p in (Path("/tokens") / sub).glob("*.pt")}
    pending = [mp3.name for mp3 in mp3s if mp3.stem not in existing]
    if existing:
        print(f"RESUMING: {len(existing)} of {len(mp3s)} already tokenized, "
              f"{len(pending)} still pending", flush=True)
    else:
        print(f"fresh run: {len(mp3s)} total mp3s, 0 already tokenized, "
              f"{len(pending)} pending", flush=True)

    if not pending:
        print("Nothing to tokenize — all files already have .pt cache", flush=True)
        return

    chunks = [pending[i:i + batch_size]
              for i in range(0, len(pending), batch_size)]
    print(f"Dispatching {len(pending)} files in {len(chunks)} batches of "
          f"~{batch_size} across parallel containers...", flush=True)
    tokenizer = Tokenizer(min_seconds=min_seconds, subdir=sub)

    n_done = n_short = n_failed = 0
    n_oom = n_decode = n_other = 0
    n_batch_errors = 0
    total_frames = 0
    # Periodic progress: unlike auto_tag (which flushes a shared tags.json and
    # prints on each flush), tokenize workers commit their own .pt per batch, so
    # the orchestrator has no flush to piggyback on — we print a progress line
    # every PROGRESS_EVERY processed files instead.
    n_seen = 0
    PROGRESS_EVERY = 10_000
    next_report = PROGRESS_EVERY
    # Sample a handful of "other" errors at the start so users can still
    # debug novel failure modes — after that, just count by category.
    OTHER_SAMPLES = 20
    other_samples_shown = 0

    # order_outputs=False: a preempted batch must not head-of-line-block the
    # in-order yield (that idles the other containers while they still bill).
    # Results are tallied independently, so order doesn't matter.
    # return_exceptions=True: a single batch raising (e.g. a transient
    # DataLossError on commit that survived the worker-side retry) must NOT
    # propagate out and crash this orchestrator — that would trigger run_tokenize's
    # retry, re-glob, re-dispatch, and re-spin-up the whole 50-container fleet.
    # Instead we count the failed batch and continue; its files stay uncommitted
    # (pending) and are picked up on the next run via the list_pending skip logic.
    for batch in tokenizer.tokenize_batch.map(
        chunks, order_outputs=False, return_exceptions=True
    ):
        if isinstance(batch, Exception):
            n_batch_errors += 1
            if n_batch_errors <= 20:
                print(f"BATCH FAILED (files stay pending, redone next run): "
                      f"{type(batch).__name__}: {str(batch)[:140]}", flush=True)
            continue
        for key, status, frames, error in batch:
            if status == "failed":
                n_failed += 1
                err_lower = (error or "").lower()
                if "out of memory" in err_lower or "cuda oom" in err_lower:
                    n_oom += 1
                elif "decode" in err_lower or "invalid data" in err_lower:
                    n_decode += 1
                else:
                    n_other += 1
                    if other_samples_shown < OTHER_SAMPLES:
                        short = (error or "").split("\n")[0][:140]
                        print(f"FAILED {key}: {short}")
                        other_samples_shown += 1
                        if other_samples_shown == OTHER_SAMPLES:
                            print(f"  …suppressing further 'other' error details "
                                  f"(category will still be counted)")
            elif status == "skipped_short":
                n_short += 1
            elif status == "done":
                n_done += 1
                total_frames += frames
        n_seen += len(batch)
        if n_seen >= next_report:
            processed = n_done + n_short + n_failed
            pct = 100.0 * processed / len(pending)
            print(f"  progress {processed:,}/{len(pending):,} ({pct:.1f}%) "
                  f"— tokenized {n_done:,}, short {n_short:,}, failed {n_failed:,}",
                  flush=True)
            next_report += PROGRESS_EVERY

    print(f"\ndone:                {n_done}", flush=True)
    print(f"skipped (too short): {n_short}", flush=True)
    print(f"failed:              {n_failed} "
          f"(OOM {n_oom} · decode {n_decode} · other {n_other})", flush=True)
    if n_batch_errors:
        print(f"batch errors:        {n_batch_errors} "
              f"(transient — affected files stay pending; re-run to finish them)",
              flush=True)
    if n_done > 0:
        # Frame rate by codec (avoid importing torch/codec in the slim
        # orchestrator): DAC 44.1kHz hop=512 -> 86 Hz; SpectroStream 48k -> 25 Hz.
        fr = 25 if os.environ.get("NANO_CODEC", "dac").lower() in ("spectrostream", "ss") else 86
        secs = total_frames / fr
        print(f"total audio cached:  {secs/60:.1f} min ({secs/3600:.2f} hours)")


@app.local_entrypoint()
def main(min_seconds: int = 20, batch_size: int = 8, wave_id: str = ""):
    # spawn (not remote) — submit the orchestrator and return immediately.
    # Combined with `modal run --detach`, the app stays alive after the local
    # CLI exits, so the user can close their terminal and walk away.
    # --wave-id N scopes the pass to /corpus/waves/wave_N -> /tokens/waves/wave_N.
    fc = run_tokenize.spawn(
        min_seconds=min_seconds, batch_size=batch_size, wave_id=wave_id)
    print(f"tokenize launched (detached) — function call id: {fc.object_id}")
    print(f"watch:  modal app logs $(modal app list | "
          f"awk '/nano-tokenize.*ephemeral/{{print $2; exit}}') -f")
    print(f"stop:   modal app stop $(modal app list | "
          f"awk '/nano-tokenize.*ephemeral/{{print $2; exit}}') -y")
