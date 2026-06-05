# Training on M4 Max

Local training on an Apple Silicon M4 Max via PyTorch's MPS backend. Workable for small corpora and experimentation; for full-scale runs prefer [Modal](README.modal.md) or [a 5090](README.5090.md).

All time estimates below are quoted **per 1,000 songs** on a small experimentation corpus — enough to validate the pipeline locally, but not enough to produce a musical model (see [README.modal.md](README.modal.md) for recommended corpus sizing). Local training runs the **same ~1.5B model** as Modal: the local `diskrot.train` CLI has no architecture flags, so it uses the `GPTConfig` defaults ([model/nano_audio_gpt.py](model/nano_audio_gpt.py)) — d_model=2048, 22 layers, 30s segments, gradient checkpointing on. **At 1.5B this needs a large-memory M4 Max (64/128 GB) just to fit, and MPS training is extremely slow at that size** — so on Apple Silicon this path is realistically for **validating the pipeline**, not full training. To actually train locally, hand-edit the `GPTConfig` defaults to a smaller shape (e.g. the old d_model=1024 / 16-layer ~287M config) first. It loads the whole token cache into RAM (no sharded-mmap path locally). For the full 1.5B model on the full corpus, use [README.modal.md](README.modal.md).

## Prerequisites

- macOS 14+
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A directory of MP3 files to train on
- ffmpeg (`brew install ffmpeg`)

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Verify MPS is available:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## 1. Tokenize

Encodes every MP3 into DAC tokens. Clips shorter than 20 seconds are skipped. Takes about 60–85 minutes per 1,000 songs on MPS.

```bash
python -m diskrot.tokenize --corpus /path/to/mp3s --out ./token_cache --device mps
```

## 2. Auto-tag (optional, needed for text conditioning)

Takes about 45–65 minutes per 1,000 songs on MPS.

```bash
python -m diskrot.auto_tag --corpus /path/to/mp3s --out ./token_cache/tags.json --device mps
```

## 3. Transcribe lyrics (optional)

Demucs and faster-whisper run on CPU on macOS (no MPS support for these models). Takes about 6–8 hours per 1,000 songs.

```bash
python -m diskrot.transcribe_lyrics --corpus /path/to/mp3s --out ./token_cache/lyrics
```

## 4. Train

Trains the ~1.5B model on 30-second segments. The CLI defaults to 125K steps (lr 2.5e-4, patience 15); cut `--steps` for a quick experimental run. **The default 1.5B shape needs a 64/128 GB M4 Max just to fit and is impractically slow on MPS** — for real local training, first shrink the `GPTConfig` defaults (see the note at the top). The command below is shown as a pipeline-validation smoke test.

```bash
python -m diskrot.train \
  --device mps \
  --steps 20000 \
  --batch-size 2 \
  --cache-dir ./token_cache \
  --ckpt-dir ./checkpoints \
  --tags-path ./token_cache/tags.json \
  --lyrics-path ./token_cache/lyrics
```

`--batch-size 2` only fits once you've shrunk `GPTConfig` to a smaller shape; the default 1.5B model is too heavy to train at any usable speed on MPS. Drop `--tags-path` / `--lyrics-path` if you skipped steps 2 and 3.

Checkpoints land in `./checkpoints/`. Then follow the [inference instructions](README.md#inference) in the main README.

## Resetting (start fresh)

```bash
rm -rf ./token_cache ./checkpoints
```

Then re-run steps 1-4.
