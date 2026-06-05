# Training on RTX 5090

Local training on a single NVIDIA RTX 5090 (32 GB VRAM). Suitable for the full pipeline without cloud costs. Linux + CUDA 12.x assumed.

All time estimates below are quoted **per 1,000 songs** on a small experimentation corpus — enough to validate the pipeline locally, but not enough to produce a musical model (see [README.modal.md](README.modal.md) for recommended corpus sizing). Local training runs the **same ~1.5B model** as Modal: the local `diskrot.train` CLI has no architecture flags, so it uses the `GPTConfig` defaults ([model/nano_audio_gpt.py](model/nano_audio_gpt.py)) — d_model=2048, 22 layers, 30s segments, gradient checkpointing on. **At 1.5B this will not fit in 32 GB** even at `--batch-size 1`, so on a 5090 this path is realistically for **validating the pipeline end-to-end**, not full training. To actually train locally, hand-edit the `GPTConfig` defaults to a smaller shape (e.g. the old d_model=1024 / 16-layer / d_ff=4096 ~287M config) first. There's also no sharded-mmap path locally (the whole token cache loads into RAM). For the full 1.5B model on the full corpus, use [README.modal.md](README.modal.md).

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- NVIDIA driver with CUDA 12.x support
- A directory of MP3 files to train on
- ffmpeg (for MP3 decoding)

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 1. Tokenize

Encodes every MP3 into DAC tokens. Clips shorter than 20 seconds are skipped. Takes about 30–45 minutes per 1,000 songs on a 5090.

```bash
python -m diskrot.tokenize --corpus /path/to/mp3s --out ./token_cache --device cuda
```

## 2. Auto-tag (optional, needed for text conditioning)

Takes about 20–30 minutes per 1,000 songs.

```bash
python -m diskrot.auto_tag --corpus /path/to/mp3s --out ./token_cache/tags.json --device cuda
```

## 3. Transcribe lyrics (optional)

Takes about 3–4 hours per 1,000 songs (single GPU, sequential).

```bash
python -m diskrot.transcribe_lyrics --corpus /path/to/mp3s --out ./token_cache/lyrics --device cuda
```

## 4. Train

Trains the ~1.5B model on 30-second segments. The CLI defaults to 125K steps (lr 2.5e-4, patience 15); cut `--steps` for a quick experimental run. **The default 1.5B shape will OOM on a 32 GB 5090** even at `--batch-size 1` — for real local training, first shrink the `GPTConfig` defaults (see the note at the top). The command below is shown as a pipeline-validation smoke test.

```bash
python -m diskrot.train \
  --device cuda \
  --steps 20000 \
  --batch-size 2 \
  --cache-dir ./token_cache \
  --ckpt-dir ./checkpoints \
  --tags-path ./token_cache/tags.json \
  --lyrics-path ./token_cache/lyrics
```

`--batch-size 2` only fits once you've shrunk `GPTConfig` to a smaller shape; the default 1.5B model won't fit at any batch size on 32 GB. Drop `--tags-path` / `--lyrics-path` if you skipped steps 2 and 3.

Checkpoints land in `./checkpoints/`. Then follow the [inference instructions](README.md#inference) in the main README.

## Resetting (start fresh)

```bash
rm -rf ./token_cache ./checkpoints
```

Then re-run steps 1-4.
