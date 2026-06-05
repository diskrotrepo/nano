---
name: train-model
description: >-
  Launch, monitor, resume, and pull checkpoints for nano training — local
  pipeline-validation runs and Modal 8xH100 DDP full runs. Use this skill when the
  user wants to train, start or resume a training run, kick off Modal training,
  read or diagnose training logs, check tok/s or per-codebook loss, or extract and
  download a checkpoint from the volume.
allowed-tools: Read, Bash
---

# Train the nano model

Assumes a packed token cache already exists on `nano-tokens` (see the
**add-songs** skill).

## Pick your path

| Path | Command | Use when |
|---|---|---|
| **Modal 8×H100 DDP** | `modal run --detach diskrot/modal_train.py --n-gpus 8` | **The real path.** Full ~1.5B model. |
| Local | `python -m diskrot.train ...` | Pipeline validation only — the local CLI has **no** architecture flags, so it trains the full ~1.5B `GPTConfig` shape and won't fit on consumer GPUs. |
| Modal single-GPU | `modal run --detach diskrot/modal_train.py` | Not recommended — each rank pays the CLAP precompute and memory is tight. |

`DEFAULTS` in [diskrot/modal_train.py](../../../diskrot/modal_train.py) is the
source of truth for the model shape. Changing architecture requires fresh
checkpoints (old weights are incompatible).

## Modal launch (the real path)

```bash
modal volume create nano-ckpts        # first time only
modal run --detach diskrot/modal_train.py --n-gpus 8
```
DDP auto-picks per-rank batch 8 → global 64, matching the tuned LR. Common flags
(all optional — defaults come from `DEFAULTS`):

```
--steps 400000   --lr 3.0e-4   --warmup-steps 5000   --patience 20
--ckpt-subdir v7_1500m
--d-model 2048   --n-layers 22   --n-heads 16   --d-ff 8192
--text-conditioned True           # set False to disable text conditioning
--segment-seconds 30.0   --max-seq-len 8192
--wandb-project NAME   --wandb-run-name NAME    # needs WANDB_API_KEY secret
```

## Local launch (validation)

```bash
python -m diskrot.train --device cuda --cache-dir ./token_cache --ckpt-dir ./checkpoints
# flags: --steps  --batch-size  --lr  --patience
#        --tags-path ./tags.json        (text conditioning)
#        --lyrics-path ./lyrics         (lyric conditioning)
```
For device-specific setup see [README.5090.md](../../../README.5090.md) (CUDA /
RTX 5090) and [README.m4max.md](../../../README.m4max.md) (Apple MPS).

## Monitor

```bash
modal app list | grep nano-train                 # find the ephemeral app id
modal app logs <app-id> -f                        # follow logs
modal app stop <app-id> -y                        # stop it
```
**Reading the logs** — phase ordering, expected silence before the first `step`,
healthy `tok/s`, per-codebook `cb[...]` losses, and stall diagnosis are covered in
[references/reading-logs.md](references/reading-logs.md) (which links the full
[README.logs.md](../../../README.logs.md)).

## Resume

Automatic from `latest.pt` — just relaunch the **same** command. Modal fan-out is
crash-safe; a preemption mid-run resumes from the last checkpoint.

## Extract a slim model + download the best checkpoint

After training, export an inference-only checkpoint (optimizer stripped, fp16 —
keeps `model`, `cfg`, `step`, `best_val_loss`, `text_proj`) and pull it locally:

```bash
# 1. extract just the model into a slim best_inference.pt on the volume
modal run diskrot/modal_export_ckpt.py --src v7_1500m/best.pt
#    flags: --dst PATH   --half {True|False}   (default: dst auto = best_inference.pt, fp16)

# 2. download the best version
modal volume get nano-ckpts /v7_1500m/best_inference.pt ./checkpoints/latest.pt --force
#    fall back to /v7_1500m/best.pt if you skipped the export
```
Inspect the checkpoint trajectory on the volume:
```bash
modal run diskrot/modal_inspect_ckpts.py --prefix v7_1500m
```

## Next step

- **eval-checkpoint** — score the downloaded checkpoint.
- **serve-model** — serve it (the server prefers `best_inference.pt`).
