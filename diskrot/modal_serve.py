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
    modal run diskrot/modal_export_ckpt.py --src v7_1500m/best.pt

Develop (hot-reloading public URL, torn down on Ctrl-C):
    modal serve diskrot/modal_serve.py

Deploy (persistent URL):
    modal deploy diskrot/modal_serve.py

Point at a different checkpoint on the volume:
    NANO_CKPT=/ckpts/v7_1500m/latest.pt modal serve diskrot/modal_serve.py

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
DEFAULT_CKPT_CANDIDATES = [
    "/ckpts/v7_1500m/best_inference.pt",
    "/ckpts/v7_1500m/best.pt",
    "/ckpts/v7_1500m/latest.pt",
]


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


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
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
    )
    # Match modal_train.py's protobuf intersection (descript-audiotools pins
    # <3.20, but msclap/transformers want newer). 4.x satisfies everyone.
    .run_commands("pip install 'protobuf>=4.25,<5'")
    .run_function(_prefetch_weights)
    .add_local_python_source("model", "diskrot", "server")
)

ckpts_vol = modal.Volume.from_name("nano-ckpts", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",  # the ~1.5B model in fp16 fits comfortably; bump to "A10G"/"H100"
    # for lower latency on long (90s) generations.
    volumes={"/ckpts": ckpts_vol},
    # Keep a warm container for 5 min after the last request so back-to-back
    # generations don't each pay the model-load cold start. Set min_containers=1
    # to keep one always warm (costs idle GPU time).
    scaledown_window=300,
    timeout=60 * 10,  # a 90s single-shot gen on L4 is minutes, not seconds.
)
# Generation is GPU-bound and the engine's static KV cache is not safe to share
# across concurrent generations — serialize requests onto each container.
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def serve():
    # Resolve the checkpoint before the FastAPI lifespan builds the engine.
    # InferenceEngine reads NANO_CKPT; NANO_DEVICE forces CUDA on the GPU box
    # (its default detection only knows MPS/CPU locally).
    os.environ.setdefault("NANO_DEVICE", "cuda")
    if "NANO_CKPT" not in os.environ:
        for cand in DEFAULT_CKPT_CANDIDATES:
            if os.path.exists(cand):
                os.environ["NANO_CKPT"] = cand
                break
        else:
            raise FileNotFoundError(
                "No checkpoint found on the nano-ckpts volume at any of "
                f"{DEFAULT_CKPT_CANDIDATES}. Export one with "
                "`modal run diskrot/modal_export_ckpt.py --src v7_1500m/best.pt`, "
                "or set NANO_CKPT to its path."
            )
    print(f"[serve] device={os.environ['NANO_DEVICE']} ckpt={os.environ['NANO_CKPT']}")

    from server.main import app as fastapi_app

    return fastapi_app
