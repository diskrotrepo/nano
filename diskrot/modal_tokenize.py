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

from diskrot.modal_common import (
    ProgressReporter,
    assert_stage_produced_output,
    corpus_mount,
)

app = modal.App("nano-tokenize")

# Codec is chosen LOCALLY at `modal run` time via NANO_CODEC (read from your shell),
# so the image + GPU are built for the right codec. "dac" = the legacy mono image;
# "spectrostream"/"ss" = v9 stereo (Magenta RealTime's prebuilt GPU image carrying
# the JAX/TF/T5X stack — same one the codec spike validated).
_CODEC = os.environ.get("NANO_CODEC", "dac").lower()
_IS_SS = _CODEC in ("spectrostream", "ss")
_MAGENTA_GPU_IMAGE = "us-docker.pkg.dev/brain-magenta/magenta-rt/magenta-rt:gpu"


# Pre-download the DAC 44kHz checkpoint at image build time so 50 containers don't
# each fetch ~500MB at cold start.
#
# This is a `run_commands` string, NOT a `run_function`: Modal imports the DEFINING
# MODULE to execute a build function, and this module's top-level
# `from diskrot.modal_common import ...` would then run before
# `add_local_python_source("model", "diskrot")` has been layered in — a hard
# ModuleNotFoundError at build time. Keeping the source layer last is deliberate
# (see the profiling-env note below), so the bake must not depend on it. Mirrors
# how the SpectroStream branch bakes its SavedModels.
_CACHE_DAC_CMD = "python -c 'import dac; dac.utils.download(model_type=\"44khz\")'"


if _IS_SS:
    # SpectroStream: TF/JAX codec (the codec compute is TensorFlow on GPU), so
    # torch is CPU-only (just tensor plumbing in model.codec) to avoid contending
    # with JAX/TF for CUDA. NANO_SS_DEPTH=32 = the stored RVQ depth (the model
    # slices to 24). The SpectroStream SavedModels are baked into the image at
    # build (the prefetch run_commands below) so containers don't re-download them
    # per cold-start. pyloudnorm powers the loudness-norm in tokenize._load_audio.
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
                # Quiet the TF/absl/XLA C++ WARNING flood (convolution-redzone,
                # bfc_allocator GC notices, the absl::InitializeLog preamble) that
                # buries the real progress lines. Level "2" drops INFO+WARNING but
                # KEEPS ERROR+FATAL — so the int32-overflow crash
                # ("F0000 ... gpu_launch_config.h: work_element_count >= 0") that
                # the length guard exists to prevent stays fully visible. Never "3".
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "GRPC_VERBOSITY": "ERROR",
                "GLOG_minloglevel": "2",
                # Kill the per-cold-start huggingface_hub SavedModel download
                # progress-bar redraws (belt-and-suspenders: the SavedModels are
                # baked into the image at build, so there is nothing to download).
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HOME": "/cache/hf",
                "HF_HUB_CACHE": "/cache/hf",
                "XDG_CACHE_HOME": "/cache",
            }
        )
        # Bake the SpectroStream SavedModels (encoder/decoder/quantizer from
        # google/magenta-realtime) into the image so containers don't re-fetch
        # them from HF on every cold-start (the "Downloading from hf:
        # savedmodels/ssv2_48k_stereo/..." tax). Constructing the codec is the
        # EXACT runtime fetch path, so it populates the same HF cache
        # (HF_HOME=/cache/hf, set just above) the codec reads at runtime. Force
        # CPU: the builder has no GPU and we only need the download, not a placed
        # model. depth is a runtime RVQ slice and doesn't change the fetched
        # files. NOTE: the weights now live at the image's /cache/hf, so we must
        # NOT mount the nano-ss-cache volume at /cache below — a mount would
        # shadow the baked dir → re-download.
        .run_commands(
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 "
            "python -c 'from magenta_rt import spectrostream; "
            "spectrostream.SpectroStream(max_rvq_depth=32)'"
        )
        # Stage-0 profiling knob (decode-wait vs encode vs commit split), baked from
        # the local shell like the other NANO_ vars so `NANO_TOKENIZE_PROFILE=1 modal
        # run/deploy ...` reaches the container. Placed AFTER the SavedModel bake so
        # toggling it never invalidates that expensive layer — only the cheap
        # add_local_python_source below rebuilds.
        .env({"NANO_TOKENIZE_PROFILE": os.environ.get("NANO_TOKENIZE_PROFILE", "")})
        .add_local_python_source("model", "diskrot")
    )
    # L40S is the measured cost/perf winner for SS encode: a fixed-file bench
    # (modal_spectrostream_spike.py::bench, 16 real songs) found L40S ~2.8x FASTER
    # than A100-40GB and ~3.0x cheaper per song ($5.4 vs $16.0/1k songs) — SS's
    # FP32 STFT/conv encode doesn't use A100's HBM/tensor-core/FP64 strengths, so the
    # Ada L40S wins on both axes, with 48GB leaving mem headroom. Override with
    # NANO_SPIKE_GPU to re-bench other cards.
    _GPU = os.environ.get("NANO_SPIKE_GPU", "L40S")
    # SS codec weights baked into the image (above) → no runtime cache volume
    # (mounting one at /cache would shadow the baked /cache/hf and re-trigger the
    # per-cold-start HF download this bake exists to kill).
    _cache_vol = None
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
        .run_commands(_CACHE_DAC_CMD)
        # Stage-0 profiling knob, after the DAC weight cache so toggling it doesn't
        # re-run _cache_dac (see the SS image note above).
        .env({"NANO_TOKENIZE_PROFILE": os.environ.get("NANO_TOKENIZE_PROFILE", "")})
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
    gpu=_GPU,  # L4 for DAC; L40S for the SpectroStream TF/JAX stack (bench winner)
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
    # Codec isolation: re-roots the .pt output under /tokens/<data_subdir>/ so a
    # second codec's corpus can be built alongside an existing one without
    # colliding. Empty = /tokens (the original layout, byte-identical behaviour).
    # The corpus side (/corpus/<subdir>) is NEVER re-rooted — both codecs read the
    # same mp3s. See diskrot/modal_train.py:301 for the matching train-side flag.
    data_subdir: str = modal.parameter(default="")

    @modal.enter()
    def load_codec(self):
        import torch

        from model.codec import get_codec

        # NANO_CODEC (baked into the image env): dac -> DACodec, spectrostream ->
        # SpectroStreamCodec (stereo, stored depth NANO_SS_DEPTH).
        self.codec = get_codec(device="cuda")
        self.min_frames = int(self.min_seconds * self.codec.FRAME_RATE_HZ)
        # Warmup encode: pay the TF/XLA graph-build + cuDNN/cuFFT autotune ONCE here
        # instead of on the first real file's critical path. Most effective paired
        # with length bucketing (NANO_SS_BUCKET_FRAMES), which keeps the encoder to a
        # few input shapes so the compiled plans get reused instead of thrashed. Wrapped
        # so a warmup failure never blocks the container from processing real files.
        try:
            sr = int(self.codec.SAMPLE_RATE)
            ch = int(getattr(self.codec, "N_CHANNELS", 1))
            self.codec.encode(torch.zeros((ch, sr * 30), dtype=torch.float32))
            print("[tokenize] codec warmup encode done", flush=True)
        except Exception as e:  # noqa: BLE001 — warmup is best-effort
            print(f"[tokenize] codec warmup skipped ({type(e).__name__}: {e})", flush=True)

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
        out_dir = Path("/tokens") / self.data_subdir / self.subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        items = [
            (corpus_dir / name, out_dir / (Path(name).stem + ".pt"))
            for name in mp3_names
        ]
        out: list[tuple[str, str, int, str | None]] = []
        # batch_size=1: encode one file at a time. For DAC this avoids padding
        # multiple 5-min files into one forward pass, which multiplies peak GPU
        # memory and OOMs L4 (22 GiB) at batch_size=8. For SpectroStream
        # encode_batch just loops per file, so batching buys nothing there either.
        # The throughput win comes from prefetch (NOT batch_size): a background
        # pool keeps the next ~8 files' CPU audio (librosa decode + loudness-norm)
        # ready, so the GPU/TF encode never stalls on librosa — the expensive
        # (A100) SpectroStream GPU would otherwise sit idle through every file's
        # decode.
        # Materialize the generator fully (list()) rather than consuming it via
        # zip(items, gen): zip stops as soon as `items` (its first, equal-length
        # iterable) is exhausted and never resumes the generator past its final
        # yield — so tokenize_files_streaming's end-of-stream NANO_TOKENIZE_PROFILE
        # summary (the decode_wait/encode/save split) would never run. list()
        # drives the generator to StopIteration, which prints that summary. Output
        # is unchanged (tokenize_batch buffers all results into `out` anyway).
        results = list(
            tokenize_files_streaming(
                self.codec, items, self.min_frames, batch_size=1, prefetch=8
            )
        )
        for (mp3_path, _), result in zip(items, results):
            out.append((mp3_path.stem, result.status, result.frames, result.error))
        # One commit per batch — each container writes independent .pt files so
        # commits don't conflict; batching them amortizes commit overhead.
        # Retry a transient DataLossError ("failed to publish commit to server")
        # so a storage blip doesn't fail the whole batch and bubble up through
        # the orchestrator's .map() loop — which would crash run_tokenize and
        # trigger a full restart + fleet re-spawn (see run_tokenize).
        _commit_t0 = time.perf_counter()
        for attempt in range(3):
            try:
                tokens_vol.commit()
                break
            except modal.exception.DataLossError as e:
                if attempt == 2:
                    raise
                print(f"commit failed ({e}); retry {attempt + 1}/2", flush=True)
                time.sleep(2.0 * (attempt + 1))
        # Stage-0 profiling: volume.commit() cost per batch (see
        # tokenize_files_streaming's NANO_TOKENIZE_PROFILE decode/encode split).
        if os.environ.get("NANO_TOKENIZE_PROFILE", "").lower() in ("1", "true", "yes"):
            print(f"[tokenize-profile] commit={time.perf_counter() - _commit_t0:.2f}s "
                  f"for {len(out)} results in batch", flush=True)
        return out


# Lightweight image for the orchestrator (no torch/DAC needed — it just
# lists files, dispatches .map(), and tallies results).
orchestrator_image = modal.Image.debian_slim(python_version="3.12")
if _IS_SS:
    # So the in-container summary reports the SpectroStream frame rate (25 Hz).
    orchestrator_image = orchestrator_image.env({"NANO_CODEC": "spectrostream"})
# Required for `modal deploy` (the ingest orchestrator calls run_tokenize via
# from_name): unlike `modal run`, deploy does NOT auto-mount the entrypoint's
# package, so the module's top-level `from diskrot...` import would crash-loop
# with ModuleNotFoundError without this. Kept as the final layer (after .env).
orchestrator_image = orchestrator_image.add_local_python_source("model", "diskrot")


def _wave_subdir(wave_id: str) -> str:
    """'' (flat legacy layout) or 'waves/wave_<id>' for a wave ingest."""
    return f"waves/wave_{wave_id}" if wave_id else ""


def _token_root(data_subdir: str) -> Path:
    """The .pt root for this codec: /tokens, or /tokens/<data_subdir>."""
    return Path("/tokens") / data_subdir if data_subdir else Path("/tokens")


def _assert_codec_identity(root: Path) -> None:
    """Pin a token tree to ONE codec, permanently.

    Filenames are ``<stem>.pt`` regardless of codec — only the tensor's shape
    differs (DAC 9 codebooks @86 Hz vs SpectroStream 24/32 @25 Hz). So writing a
    second codec's tokens into an existing tree is silent and unrecoverable: the
    packer would mix incompatible depths, and nothing downstream re-derives the
    codec from the data.

    NANO_CODEC is baked into the image at `modal deploy`/`modal run` time, so the
    realistic failure is deploying a stage app under the wrong codec and only
    noticing 14 waves later. Stamp it on first write and refuse any mismatch.
    """
    import json

    codec = os.environ.get("NANO_CODEC", "dac").lower()
    is_ss = codec in ("spectrostream", "ss")
    want = {
        "codec": "spectrostream" if is_ss else "dac",
        "frame_rate_hz": 25 if is_ss else 86,
        "stored_n_codebooks": int(os.environ.get("NANO_SS_DEPTH", "32")) if is_ss else 9,
    }
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / "codec.json"
    if not stamp.exists():
        tmp = stamp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(want, indent=2))
        os.replace(tmp, stamp)
        tokens_vol.commit()
        print(f"[tokenize] stamped {stamp} = {want}", flush=True)
        return
    try:
        got = json.loads(stamp.read_text())
    except Exception:  # noqa: BLE001 — a corrupt stamp must not be silently trusted
        raise SystemExit(f"CODEC GUARD: {stamp} exists but is unreadable — refusing to write.")
    if got != want:
        raise SystemExit(
            f"CODEC GUARD: {root} was built with {got}, but this container is "
            f"running {want}. Writing would silently mix incompatible token "
            f"shapes into one tree.\n"
            f"Fix: re-deploy the stage apps with the right NANO_CODEC "
            f"(e.g. `NANO_CODEC={want['codec']} modal deploy diskrot/modal_tokenize.py`), "
            f"or point --data-subdir at the correct tree."
        )


@app.function(
    image=orchestrator_image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def list_pending(wave_id: str = "", data_subdir: str = "") -> list[str]:
    """Return mp3 basenames not yet tokenized (no matching .pt) within the
    (wave) corpus subdir."""
    sub = _wave_subdir(wave_id)
    mp3s = sorted((Path("/corpus") / sub).glob("*.mp3"))
    existing = {p.stem for p in (_token_root(data_subdir) / sub).glob("*.pt")}
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
def run_tokenize(min_seconds: int, batch_size: int, wave_id: str = "",
                 data_subdir: str = "") -> None:
    """Run the full tokenize pass: list pending, fan out to GPU containers,
    aggregate results, print summary. Designed to be `.spawn()`-ed from the
    local entrypoint so the user can launch and walk away. ``wave_id`` scopes
    the pass to /corpus/waves/wave_<id> -> /tokens/<data_subdir>/waves/wave_<id>."""
    sub = _wave_subdir(wave_id)
    root = _token_root(data_subdir)
    _assert_codec_identity(root)
    out_dir = root / sub
    print(f"[tokenize] corpus=/corpus/{sub}  ->  out={out_dir}", flush=True)
    mp3s = sorted((Path("/corpus") / sub).glob("*.mp3"))
    existing = {p.stem for p in out_dir.glob("*.pt")}
    pending = [mp3.name for mp3 in mp3s if mp3.stem not in existing]
    if existing:
        print(f"RESUMING: {len(existing)} of {len(mp3s)} already tokenized, "
              f"{len(pending)} still pending", flush=True)
    else:
        print(f"fresh run: {len(mp3s)} total mp3s, 0 already tokenized, "
              f"{len(pending)} pending", flush=True)

    # "0 mp3s found" and "all already tokenized" are very different: the first is
    # a path bug (wrong wave_id, un-uploaded wave, wrong corpus mount) that would
    # otherwise sail through the ingest as a green stage, the second is a legit
    # resume no-op. Only the latter may return quietly.
    if not mp3s:
        raise RuntimeError(
            f"tokenize found 0 mp3s at /corpus/{sub} — refusing to report success. "
            f"Check --wave-id and that the wave exists in the corpus bucket."
        )
    if not pending:
        print("Nothing to tokenize — all files already have .pt cache", flush=True)
        return

    chunks = [pending[i:i + batch_size]
              for i in range(0, len(pending), batch_size)]
    print(f"Dispatching {len(pending)} files in {len(chunks)} batches of "
          f"~{batch_size} across parallel containers...", flush=True)
    tokenizer = Tokenizer(min_seconds=min_seconds, subdir=sub, data_subdir=data_subdir)

    n_done = n_short = n_failed = 0
    n_oom = n_decode = n_other = 0
    n_too_long = 0  # subset of n_short: skipped pre-encode by the int32 length guard
    n_batch_errors = 0
    total_frames = 0
    # Periodic progress: unlike auto_tag (which flushes a shared tags.json and
    # prints on each flush), tokenize workers commit their own .pt per batch, so
    # the orchestrator has no flush to piggyback on — we print a progress line
    # every PROGRESS_EVERY processed files instead.
    # Uniform progress heartbeat, shared across every pipeline stage (see
    # diskrot.progress.ProgressReporter): % complete, files/min, elapsed, ETA.
    rep = ProgressReporter(len(pending), "tokenize", unit="files")
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
                if (error or "").startswith("too_long:"):
                    n_too_long += 1
            elif status == "done":
                n_done += 1
                total_frames += frames
        rep.update(len(batch),
                   extra=f"tokenized {n_done:,}, short {n_short:,}, failed {n_failed:,}")
    rep.done(extra=f"tokenized {n_done:,}, short {n_short:,}, failed {n_failed:,}")

    print(f"\ndone:                {n_done}", flush=True)
    print(f"skipped (too short): {n_short} "
          f"(incl. {n_too_long} over the int32 length cap — likely corrupt-metadata mp3s)",
          flush=True)
    print(f"failed:              {n_failed} "
          f"(OOM {n_oom} · decode {n_decode} · other {n_other})", flush=True)
    if n_batch_errors:
        print(f"batch errors:        {n_batch_errors} "
              f"(transient — affected files stay pending; re-run to finish them)",
              flush=True)
    assert_stage_produced_output("tokenize", n_done, len(pending), n_failed)
    if n_done > 0:
        # Frame rate by codec (avoid importing torch/codec in the slim
        # orchestrator): DAC 44.1kHz hop=512 -> 86 Hz; SpectroStream 48k -> 25 Hz.
        fr = 25 if os.environ.get("NANO_CODEC", "dac").lower() in ("spectrostream", "ss") else 86
        secs = total_frames / fr
        print(f"total audio cached:  {secs/60:.1f} min ({secs/3600:.2f} hours)")


@app.local_entrypoint()
def main(min_seconds: int = 20, batch_size: int = 64, wave_id: str = "",
         data_subdir: str = ""):
    # spawn (not remote) — submit the orchestrator and return immediately.
    # Combined with `modal run --detach`, the app stays alive after the local
    # CLI exits, so the user can close their terminal and walk away.
    # --wave-id N scopes the pass to /corpus/waves/wave_N -> /tokens/waves/wave_N.
    fc = run_tokenize.spawn(
        min_seconds=min_seconds, batch_size=batch_size, wave_id=wave_id,
        data_subdir=data_subdir)
    print(f"tokenize launched (detached) — function call id: {fc.object_id}")
    print(f"watch:  modal app logs $(modal app list | "
          f"awk '/nano-tokenize.*ephemeral/{{print $2; exit}}') -f")
    print(f"stop:   modal app stop $(modal app list | "
          f"awk '/nano-tokenize.*ephemeral/{{print $2; exit}}') -y")
