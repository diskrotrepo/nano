"""Modal entrypoint for the nano inference server.

Serves the existing FastAPI app (``server/main.py``) on a GPU container, loading
the checkpoint from the ``nano-ckpts`` volume. The same /health, /generate and
/extend endpoints are exposed over a public HTTPS URL.

Setup (one-time):
    pip install modal
    modal token new
    # a trained checkpoint must already be on the nano-ckpts volume. Prefer the
    # slim, inference-only export (optimizer state stripped, fp16) — far faster
    # to load on cold start:
    modal run diskrot/modal_export_ckpt.py --src v10_dac_2b/best.pt

Develop (hot-reloading public URL, torn down on Ctrl-C):
    modal serve diskrot/modal_serve.py

Deploy (persistent URL):
    modal deploy diskrot/modal_serve.py

Both bring up TWO web endpoints: the inference API (`serve`, GPU) and the
Flutter web UI (`ui`, CPU static files) at sibling URLs:
    https://<workspace>--nano-serve-serve[-dev].modal.run   # API
    https://<workspace>--nano-serve-ui[-dev].modal.run      # UI
The UI is mounted from the local `webapp/build/web` bundle, so build it first:
    cd webapp && flutter build web
(The UI pre-fills its server-url field with the sibling API URL.)

Point at a different checkpoint on the volume:
    NANO_CKPT=/ckpts/v10_dac_2b/latest.pt modal serve diskrot/modal_serve.py

Register MULTIPLE switchable checkpoints (the request's `model` field picks one;
the others lazy-load on first use; GET /models lists them). Keeps the big model
available alongside the fast distilled student:
    NANO_MODELS="fast=/ckpts/v8_distill/best_inference.pt,full=/ckpts/v8_sing4/best_inference.pt" \\
      NANO_DEFAULT_MODEL=fast modal deploy diskrot/modal_serve.py

Quantize the weights to shrink the model + speed up the memory-bound decode
(CUDA only; mirrors NANO_MLX_BITS on the Apple-Silicon path). Default is fp16:
    NANO_BITS=8 modal serve diskrot/modal_serve.py   # int8 weight-only (~1.5GB)
    NANO_BITS=4 modal serve diskrot/modal_serve.py   # int4 weight-only (~0.75GB)

Smoke test once it's up (URL is printed by modal serve/deploy):
    curl https://<your-app>.modal.run/health
    curl -X POST https://<your-app>.modal.run/generate \\
      -F seconds=10 -F prompt="punchy techno with groovy synth bass" \\
      -F per_cb_temperature="0.9,0.9,0.7,0.7,0.5,0.5,0.4,0.4,0.3" \\
      --output out.mp3
"""
from __future__ import annotations

import os

import modal

app = modal.App("nano-serve")

# Default checkpoint location on the nano-ckpts volume (mounted at /ckpts).
# Prefer the slim inference export; fall back to the raw training checkpoints.
# Override with the NANO_CKPT env var at `modal serve`/`modal deploy` time.
# (v7_1500m checkpoints predate the fused output head and cannot load on this
# code — don't list them as fallbacks.) DAC is the live path again as of the
# v10 migration (2026-07-21), so the default branch points at the v10 run; the
# SpectroStream branch keeps the (abandoned) v9 run so that image stays
# launchable.
DEFAULT_CKPT_CANDIDATES = [
    "/ckpts/v9_stereo/best_inference.pt",
    "/ckpts/v9_stereo/best.pt",
    "/ckpts/v9_stereo/latest.pt",
] if os.environ.get("NANO_CODEC", "").strip().lower() == "spectrostream" else [
    "/ckpts/v10_dac_2b/best_inference.pt",
    "/ckpts/v10_dac_2b/best.pt",
    "/ckpts/v10_dac_2b/latest.pt",
]

# Local shell env does NOT cross into the container — bake the documented
# overrides into the image at `modal serve`/`modal deploy` time. (This module
# is also imported inside the container, where these are absent; the dict is
# empty there and the .env() layer is a no-op.)
_FORWARDED_ENV = {
    k: v for k in ("NANO_CKPT", "NANO_MODELS", "NANO_DEFAULT_MODEL", "NANO_BITS",
                   "NANO_WARMUP_SECONDS", "TORCH_LOGS", "NANO_COMPILE", "NANO_WARMUP_CFG",
                   # v9/SpectroStream serving: the codec choice must reach the
                   # container (it also selects the SS image variant below).
                   "NANO_CODEC", "NANO_SS_DEPTH")
    if (v := os.environ.get(k))
}


def _prefetch_weights() -> None:
    """Bake DAC + CLAP + sweetener (Qwen) weights into the image so a cold
    container never blocks on a Hugging Face / DAC CDN download at first request.
    (msclap's CLAP() ctor has no timeout — a stalled CDN connection would hang
    forever.) The sweetener is on by default (server/main.py), so its Qwen model
    must be cached too, or the first /generate would stall ~1-2 min downloading
    it."""
    import dac
    from msclap import CLAP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Must match prompt_sweetener.DEFAULT_MODEL. Hardcoded (not imported) because
    # this prefetch runs at image-build time, before add_local_python_source()
    # mounts the `server` package.
    sweetener_model = "Qwen/Qwen2.5-1.5B-Instruct"

    dac.utils.download(model_type="44khz")
    CLAP(version="2023", use_cuda=False)
    AutoTokenizer.from_pretrained(sweetener_model)
    AutoModelForCausalLM.from_pretrained(sweetener_model)


def _prefetch_g2p() -> None:
    """Bake g2p_en's nltk corpora + model into the image (mirrors
    modal_train._prefetch_g2p). The lyric path phonemizes request lyrics with
    g2p_en at inference; its first G2p() call needs CMUdict + the POS tagger."""
    import nltk

    for res in ("averaged_perceptron_tagger_eng", "cmudict", "averaged_perceptron_tagger"):
        nltk.download(res, quiet=True)
    from g2p_en import G2p

    G2p()("warm up the cache")


# --- SpectroStream (v9) image variant -------------------------------------
# model/codec.py imports magenta_rt on the SS path, which only exists in the
# magenta-rt GPU image (TF + JAX stack). Selecting NANO_CODEC=spectrostream at
# `modal serve`/`deploy` time swaps the image base; the DAC image below stays
# the default. Same base + CuDNN reconciliation as diskrot/modal_stems.py —
# see the long comment there for the 9.1-vs-9.3 abort trap.
_IS_SS = os.environ.get("NANO_CODEC", "").strip().lower() == "spectrostream"
_MAGENTA_GPU_IMAGE = "us-docker.pkg.dev/brain-magenta/magenta-rt/magenta-rt:gpu"


def _prefetch_weights_ss() -> None:
    """SS-image twin of _prefetch_weights: CLAP + sweetener only. The SS codec
    SavedModels are baked by a run_commands step (the exact runtime fetch
    path); DAC isn't installed on this image."""
    from msclap import CLAP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Must match prompt_sweetener.DEFAULT_MODEL (see _prefetch_weights).
    sweetener_model = "Qwen/Qwen2.5-1.5B-Instruct"

    CLAP(version="2023", use_cuda=False)
    AutoTokenizer.from_pretrained(sweetener_model)
    AutoModelForCausalLM.from_pretrained(sweetener_model)


_ss_image = (
    modal.Image.from_registry(_MAGENTA_GPU_IMAGE)
    # espeak-ng: the v9 multilingual lyric path phonemizes request lyrics via
    # the `phonemizer` lib (espeak backend) — without it every lyrics= request
    # 500s (g2p_en is the retired v8 English-only path).
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng", "libespeak-ng1")
    # torch pinned to the stems-proven combo (cu121); newer torch re-breaks the
    # CuDNN dance below. torchao pinned to a torch-2.4-compatible release.
    .pip_install(
        "torch==2.4.1", "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "librosa>=0.10", "numpy>=1.26", "soundfile>=0.12",
        "fastapi>=0.115", "python-multipart>=0.0.12",
        # HARD-pinned: the magenta base preinstalls transformers 4.57.1, whose
        # top-level no longer resolves GPT2LMHeadModel (msclap's CLAP text
        # encoder imports it) — a range pin is "already satisfied" and won't
        # downgrade. 4.46.3 has the export and supports the Qwen2.5 sweetener.
        "msclap", "transformers==4.46.3", "torchao==0.7.0", "g2p_en==2.1.0",
        "phonemizer>=3.2",
    )
    # CuDNN reconciliation: torch+cu121 pins nvidia-cudnn-cu12==9.1.0.70 and
    # downgrades the 9.3 wheel magenta-rt's TF was compiled against, which
    # aborts every codec decode. Force 9.3 back (satisfies both).
    .run_commands(
        "pip install --no-deps --force-reinstall 'nvidia-cudnn-cu12==9.3.0.75'"
    )
    .env(
        {
            "NANO_CODEC": "spectrostream",
            "NANO_SS_DEPTH": os.environ.get("NANO_SS_DEPTH", "32"),
            # TF/JAX must share the GPU with the ~4GB torch model + KV cache:
            # no preallocation, modest fraction, growth on demand.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.3",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "GRPC_VERBOSITY": "ERROR",
            "GLOG_minloglevel": "2",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            # Pin the HF cache to a stable baked path BEFORE the SS SavedModel
            # bake below, so the runtime codec reads the baked copy instead of
            # re-downloading at cold start (mirrors modal_stems; no volume is
            # mounted at /cache in this app, so the baked dir survives).
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "XDG_CACHE_HOME": "/cache",
            # The sweetener/CLAP run on torch — transformers must not touch the
            # image's TensorFlow.
            "USE_TF": "0",
        }
    )
    # Bake the SpectroStream SavedModels into the image so cold containers
    # don't re-fetch them from HF (same fetch path the codec uses at runtime;
    # CPU-only — the builder has no GPU).
    .run_commands(
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 "
        "python -c 'from magenta_rt import spectrostream; "
        "spectrostream.SpectroStream(max_rvq_depth=32)'"
    )
    .run_function(_prefetch_weights_ss)
    .run_function(_prefetch_g2p)
    .env(_FORWARDED_ENV)
    .add_local_python_source("model", "diskrot", "server")
) if _IS_SS else None

_dac_image = (
    modal.Image.debian_slim(python_version="3.12")
    # espeak-ng: v10's lyric path phonemizes request lyrics via the `phonemizer`
    # lib (espeak backend) to IPA — without it every lyrics= request 500s
    # (g2p_en is the retired v8 English-only path, kept only as a fallback).
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng", "libespeak-ng1")
    .pip_install(
        "torch>=2.4",
        "torchaudio>=2.4",
        "librosa>=0.10",
        "descript-audio-codec>=1.0.0",
        "numpy>=1.26",
        "soundfile>=0.12",
        "fastapi>=0.115",
        "python-multipart>=0.0.12",
        "msclap",
        # transformers powers the opt-in `sweeten` prompt rewriter; harmless to
        # include even when unused (the model is lazy-loaded on first sweeten).
        "transformers>=4.35",
        # weight-only int8/int4 quantization for the CUDA path (NANO_BITS=8|4);
        # only imported when NANO_BITS selects a quantized mode.
        "torchao>=0.7",
        # v10 lyric conditioning: espeak-backed IPA phonemizer (primary path)
        # plus g2p_en as the retired-v8 fallback.
        "phonemizer>=3.2",
        "g2p_en==2.1.0",
    )
    # Match modal_train.py's protobuf intersection (descript-audiotools pins
    # <3.20, but msclap/transformers want newer). 4.x satisfies everyone.
    .run_commands("pip install 'protobuf>=4.25,<5'")
    .run_function(_prefetch_weights)
    .run_function(_prefetch_g2p)
    .env(_FORWARDED_ENV)
    .add_local_python_source("model", "diskrot", "server")
)

# NANO_CODEC=spectrostream at serve/deploy time selects the SS variant.
image = _ss_image if _IS_SS else _dac_image

ckpts_vol = modal.Volume.from_name("nano-ckpts", create_if_missing=True)
# Every generation is also persisted here (server/main.py:_save_output via
# NANO_OUTPUT_DIR). Pull files with: modal volume get nano-output /<name>.mp3 .
output_vol = modal.Volume.from_name("nano-output", create_if_missing=True)


# Horizontal scale-out: how many GPU containers Modal may run in parallel, so a
# burst of N requests (e.g. the webapp's "generate 8") fans out to N containers
# instead of serializing on one. Each container still runs ONE generation at a
# time (max_inputs=1, KV-cache safety) — true server-side batching (8 clips in
# one batched forward on a single GPU) is the cheaper companion via POST
# /generate_batch. Set NANO_MIN_CONTAINERS=1 to keep one always warm (kills the
# cold-start tax on the first burst, at the cost of idle GPU time).
_MAX_CONTAINERS = int(os.environ.get("NANO_MAX_CONTAINERS", "8"))
_MIN_CONTAINERS = int(os.environ.get("NANO_MIN_CONTAINERS", "0"))


@app.function(
    image=image,
    gpu="H100",  # autoregressive decode is memory-bandwidth-bound: each token
    # streams all ~2B weights from HBM, so the H100's ~3.35 TB/s (~11x the L4's
    # ~300 GB/s) is both far faster AND cheaper per generation. (L4/A10G fit the
    # model fine but are bandwidth-starved on long 90s gens.)
    volumes={"/ckpts": ckpts_vol, "/outputs": output_vol},
    # Keep a warm container for 5 min after the last request so back-to-back
    # generations don't each pay the model-load cold start.
    scaledown_window=300,
    min_containers=_MIN_CONTAINERS,
    max_containers=_MAX_CONTAINERS,
    timeout=60 * 10,  # a 90s single-shot gen on L4 is minutes, not seconds.
)
# Generation is GPU-bound and the engine's static KV cache is not safe to share
# across concurrent generations — serialize requests onto each container (and let
# max_containers handle concurrency by spreading them across containers).
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def serve():
    # Resolve the checkpoint before the FastAPI lifespan builds the engine.
    # InferenceEngine reads NANO_CKPT; NANO_DEVICE forces CUDA on the GPU box
    # (its default detection only knows MPS/CPU locally).
    os.environ.setdefault("NANO_DEVICE", "cuda")
    os.environ.setdefault("NANO_OUTPUT_DIR", "/outputs")
    # Decode runs ~2x faster compiled (fusion, mode="default") and the decode loop
    # falls back to eager on any compile error (identical tokens), so it's the safe
    # production default. NOT "graphs" — CUDA-graph capture intermittently NaNs on
    # this model. An explicit local NANO_COMPILE (forwarded into the image) wins.
    os.environ.setdefault("NANO_COMPILE", "default")
    # NANO_MODELS ("fast=/ckpts/v8_distill/best_inference.pt,full=/ckpts/v8_sing4/
    # best_inference.pt") registers several switchable checkpoints; the request's
    # `model` field picks one, the rest lazy-load. When it's set the server reads
    # it directly — skip the single-checkpoint resolution below. Otherwise fall
    # back to NANO_CKPT (explicit) or the default-candidate search.
    if "NANO_MODELS" not in os.environ and "NANO_CKPT" not in os.environ:
        for cand in DEFAULT_CKPT_CANDIDATES:
            if os.path.exists(cand):
                os.environ["NANO_CKPT"] = cand
                break
        else:
            raise FileNotFoundError(
                "No checkpoint found on the nano-ckpts volume at any of "
                f"{DEFAULT_CKPT_CANDIDATES}. Export one with "
                "`modal run diskrot/modal_export_ckpt.py --src v10_dac_2b/best.pt`, "
                "or set NANO_CKPT / NANO_MODELS to its path(s)."
            )
    _sel = os.environ.get("NANO_MODELS") or os.environ.get("NANO_CKPT")
    print(f"[serve] device={os.environ['NANO_DEVICE']} models={_sel}")

    from server.main import app as fastapi_app

    return fastapi_app


# --- Flutter web UI ----------------------------------------------------------
# A CPU-only sibling endpoint serving the pre-built `webapp/build/web` bundle.
# Kept off the GPU function on purpose: `serve` runs max_inputs=1 (KV-cache
# safety), so page loads there would queue behind generations and wake an L4.

_WEBAPP_BUILD = os.path.join(os.path.dirname(os.path.dirname(__file__)), "webapp", "build", "web")

ui_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi>=0.115")
    .add_local_dir(_WEBAPP_BUILD, remote_path="/web")
)


@app.function(image=ui_image)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    if not os.path.exists("/web/index.html"):
        raise FileNotFoundError(
            "webapp/build/web has no index.html — build the Flutter bundle "
            "before serving: `cd webapp && flutter build web`."
        )

    static_app = FastAPI()
    # html=True serves index.html at / (the Flutter app is a single page).
    static_app.mount("/", StaticFiles(directory="/web", html=True), name="ui")
    return static_app
