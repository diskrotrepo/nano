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
**add-songs** skill). For **melody conditioning** (the `/cover` capability) the pack
must carry the chroma sidecar — run add-songs' melody step then repack. Without it
the model trains the null path only (a `WARNING: ... pack has NO chroma sidecar` line
is logged at startup) — harmless, just no melody signal.

## Pick your path

| Path | Command | Use when |
|---|---|---|
| **Modal 4×B200 DDP** | `modal run --detach diskrot/modal_train.py --n-gpus 4` | **The real path.** Full ~1.5B model, from scratch or full fine-tune. |
| **LoRA (local or Modal 1×H100)** | `… --init-from <ckpt> --lora …` | Adapt a trained checkpoint on new data — the frozen base makes the 1.5B trainable on a single GPU / Apple Silicon. See [README.finetune.md](../../../README.finetune.md). |
| Local from-scratch | `python -m diskrot.train ...` | Pipeline validation only — the local CLI has **no** architecture flags, so a from-scratch run trains the full ~1.5B `GPTConfig` shape and won't fit on consumer GPUs. (With `--init-from`, the architecture comes from the checkpoint instead.) |
| Modal single-GPU from-scratch | `modal run --detach diskrot/modal_train.py` | Not recommended — each rank pays the CLAP precompute and memory is tight. (Single-GPU **is** the recommended LoRA path, though.) |

`DEFAULTS` in [diskrot/modal_train.py](../../../diskrot/modal_train.py) is the
source of truth for the model shape. Changing architecture requires fresh
checkpoints (old weights are incompatible).

## Modal launch (the real path)

```bash
modal volume create nano-ckpts        # first time only
modal run --detach diskrot/modal_train.py --n-gpus 4
```
DDP auto-picks per-rank batch 8 → global 32, matching the tuned LR. Common flags
(all optional — defaults come from `DEFAULTS`):

```
--steps 400000   --lr 3.0e-4   --warmup-steps 5000   --patience 20
--ckpt-subdir v7_1500m
--d-model 2048   --n-layers 22   --n-heads 16   --d-ff 8192
--text-conditioned True           # drives tags + lyrics + melody together (set False to disable)
--segment-seconds 30.0   --max-seq-len 8192
--wandb-project NAME   --wandb-run-name NAME    # needs WANDB_API_KEY secret
```

## Local launch (validation)

```bash
python -m diskrot.train --device cuda --cache-dir ./token_cache --ckpt-dir ./checkpoints
# flags: --steps  --batch-size  --lr  --patience
#        --tags-path ./tags.json        (text conditioning)
#        --lyrics-path ./lyrics         (lyric conditioning)
#        --melody                       (melody conditioning; needs a chroma-packed cache)
```
For device-specific setup see [README.5090.md](../../../README.5090.md) (CUDA /
RTX 5090) and [README.m4max.md](../../../README.m4max.md) (Apple MPS).

## Fine-tune from an existing checkpoint

Full walkthrough (modes, corpus prep, merge flow, rules of thumb):
[README.finetune.md](../../../README.finetune.md). The short version — always a
**new** `--ckpt-subdir`/`--ckpt-dir` (an existing `latest.pt` there resumes
instead; precedence is `latest.pt` > `--init-from` > scratch):

```bash
# Full fine-tune (every weight, fresh optimizer/step, lower LR)
modal run --detach diskrot/modal_train.py --n-gpus 4 \
  --init-from v8_sing/best.pt --ckpt-subdir v8_ft --lr 5e-5
# optional: --data-subdir my_corpus  (pack the fine-tune corpus under /tokens/my_corpus)
```

## LoRA training

Freezes the base, trains ~21.6M low-rank adapters (defaults r=16, alpha=32);
requires `--init-from`. Checkpoints are **adapter-only** (~100 MB) and must be
**merged** before serving — inference is unchanged after the merge.

```bash
# Modal (single H100 is the recommended LoRA path)
modal run --detach diskrot/modal_train.py \
  --init-from v8_sing/best.pt --lora --ckpt-subdir v8_lora_x

# Local (Apple Silicon / single GPU)
python -m diskrot.train --device mps \
  --init-from ./checkpoints/latest.pt --lora --ckpt-dir ./checkpoints/ft_run \
  --cache-dir ./token_cache --tags-path ./tags.json --lyrics-path ./lyrics

# Merge when done, then pull (drop-in standard checkpoint)
modal run diskrot/modal_merge_lora.py --base v8_sing/best.pt --lora v8_lora_x/best.pt
modal volume get nano-ckpts /v8_lora_x/merged_inference.pt ./checkpoints/latest.pt --force
#   local merge: python -m diskrot.merge_lora --base ... --lora ... --out ...
```

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
