---
name: serve-model
description: >-
  Serve nano inference locally or on Modal and exercise the generation endpoints.
  Use this skill when the user wants to run the server, generate or extend audio
  from a checkpoint, cover a hummed melody, deploy the inference API, or test
  /generate /extend /cover with text, lyrics, melody, or style conditioning.
allowed-tools: Read, Bash
---

# Serve nano inference

## Get a deployable checkpoint first

The server loads `GPTConfig` from the checkpoint's `cfg` dict, so any checkpoint
works. It **prefers the slim `best_inference.pt`** export. Produce and download one
via the **train-model** skill (extract + download step):

```bash
modal run diskrot/modal_export_ckpt.py --src v7_1500m/best.pt
modal volume get nano-ckpts /v7_1500m/best_inference.pt ./checkpoints/latest.pt --force
```

## Local server

```bash
uv sync                                                       # first time: build the .venv from uv.lock
uv run python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```
Loads `./checkpoints/latest.pt` by default (override with `NANO_CKPT`). On Apple
silicon the 1.5B model runs on `mps`; loading the checkpoint takes ~30s before the
`Uvicorn running` line. If port 8000 is busy it's usually a stale server — check
`lsof -iTCP:8000 -sTCP:LISTEN` and `curl :8000/health` (the response reports
`model_params`, so you can tell the 1.5B from the retired 287M model).

## Modal server

```bash
modal serve diskrot/modal_serve.py     # dev: hot-reload, ephemeral public URL (Ctrl-C tears down)
modal deploy diskrot/modal_serve.py    # persistent public URL
```
On the volume it prefers `/ckpts/v7_1500m/best_inference.pt`, falling back to
`best.pt` then `latest.pt`. Override with `NANO_CKPT=/ckpts/custom.pt`.

## Weight quantization (optional)

Shrinks the ~3GB fp16 model and speeds up the memory-bound decode. **Weight-only**
(activations/KV cache stay fp16), so quality cost is small at int8 and more
aggressive at int4 — A/B before shipping. Set at serve time; no new checkpoint
needed (quantization happens at load).

- **CUDA** (Modal serve, via torchao): `NANO_BITS=8` (int8, ~1.5GB) or `NANO_BITS=4`
  (int4, ~0.75GB). Default fp16. int4 runs the compute in bf16 (needs sm80+; the L4
  is fine).
  ```bash
  NANO_BITS=8 modal serve diskrot/modal_serve.py
  ```
- **Apple Silicon** (local MLX backend): `NANO_MLX_BITS=8|4` (default 8). Force the
  PyTorch-MPS path instead with `NANO_MLX=0`.

Both skip the embeddings and the lyric encoder (small + intelligibility-sensitive)
and quantize the big attention/MLP/head Linears. The startup log prints the active
weight format (`weights=int8 weight-only (fp16 compute)`).

## Endpoints

`GET /health`, `POST /generate`, `POST /extend`, `POST /cover`. `/generate` and
`/extend` accept optional `prompt` (tags), `lyrics`, `style_audio` (file), and
`style_weight`.

**Sampling fields (HTTP form):** `temperature` / `top_k` / `top_p` are **scalars
only** — passing a list (`[0.9,...]`) returns HTTP 422. For per-codebook control use
the **separate** `per_cb_temperature` / `per_cb_top_k` / `per_cb_top_p` fields, each
a **bare comma-separated list of 9** (no brackets), which override the scalar. A
**decreasing** ladder (coarse codebooks hot, fine DAC-residual codebooks cold)
usually sounds better than one flat temperature.

```bash
curl -X POST http://localhost:8000/generate \
  -F prompt="lofi hip hop, mellow, chill, jazzy electric piano, vinyl crackle, boom bap drums, warm bass, slow tempo" \
  -F seconds=30 \
  -F per_cb_temperature="0.9,0.9,0.7,0.7,0.5,0.5,0.4,0.4,0.3" \
  -F top_p=0.95 \
  --output out.mp3
```

Other `/generate` knobs: `cfg_scale` (default 3.0), `negative_prompt`, `sweeten`
(default on). From-scratch generation always seeds from a random DAC column.

`/extend` continues a clip forward from a point in time. It takes a multipart
`audio=@file.mp3` plus `add_seconds`, `overlap_seconds` (seed window before the cut
point), and optional `from_seconds` (the cut point T — keep the original up to T,
regenerate after; defaults to the clip's tail = seamless append). Chain `/extend` to
grow clips past the ~95s single-shot limit.

`/cover` re-renders a hummed/uploaded melody in the prompt's timbre (hum → solo
violin). It takes a **required** multipart `melody_audio=@hum.mp3` — only the clip's
chromagram conditions generation; its audio/tokens never appear in the output. The
hum's length sets the output length. Use `prompt` (tags) for timbre/instrumentation
and `lyrics` for words; `melody_cfg_scale` (>0) pushes melody adherence with its own
guidance scale, independent of `cfg_scale`. Same per-cb sampling fields as above.
**Requires a melody-trained checkpoint** (`use_melody_conditioning`) — otherwise it
returns HTTP 400.

```bash
curl -X POST http://localhost:8000/cover \
  -F melody_audio=@hum.mp3 \
  -F prompt="solo violin, warm, expressive, legato" \
  -F cfg_scale=3.0 -F melody_cfg_scale=2.0 \
  -F per_cb_temperature="0.9,0.9,0.7,0.7,0.5,0.5,0.4,0.4,0.3" \
  --output cover.mp3
```

See the **Inference** section of [README.md](../../../README.md) for the full curl
set (style blending, extend).

## Comes from

- **train-model** / **eval-checkpoint** — produce and validate the checkpoint you serve.
