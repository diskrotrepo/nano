# Fine-tuning Plan (at a glance)

How to continue training the model from an existing checkpoint instead of from
scratch — **what each mode means, what gates what, and the exact order.** For
the from-scratch data-prep and training pipeline, see [README.plan.md](README.plan.md).

## What "fine-tune" means here (two modes)

| | Full fine-tune (`--init-from`) | LoRA (`--init-from --lora`) |
|---|---|---|
| What trains | **Every** weight, continued from the checkpoint | The base is **frozen**; small low-rank adapters (~21.6M params, ~1.4% of 1.5B) train on top |
| Optimizer / step | Fresh — new optimizer, step 0, fresh warmup + cosine schedule | Fresh, and only the adapters carry optimizer state |
| Checkpoints | Standard full checkpoints (multi-GB) | **Adapter-only** (~100 MB) — must be **merged** into the base before serving |
| Hardware | 8×H100 (same as a full run) | A **single H100 — or your local GPU / Apple Silicon**; freezing the base removes the optimizer-state memory that makes local 1.5B training impractical |
| Pick it when | Large new corpus; you want to shift the whole model (lower `--lr`, e.g. 5e-5) | Smaller corpus, style/domain adaptation, or you want to train locally |

Both modes take the **architecture from the checkpoint** (`--d-model` etc.
become advisory), so a fine-tune always matches the base. Neither changes the
data pipeline: a fine-tune corpus goes through the **same prep waves** as the
main corpus (README.plan.md), just packed under its own directory.

## The graph

```
(optional) new corpus ──► README.plan.md waves ──► /tokens/<data-subdir>/…
                                                          │
base checkpoint (v8_sing/best.pt) ──► fine-tune / LoRA train ──┬─ full FT ─► standard ckpt ─► (slim-export) ─► serve/eval
                                                               └─ LoRA ───► adapter ckpt ──► MERGE ─────────► serve/eval
```

## Step by step

**1. Prepare the fine-tune corpus** (skip to reuse the main pack):
run the [README.plan.md](README.plan.md) waves into a subdirectory — the pack and
every conditioning artifact live under `/tokens/<data-subdir>/` with the usual
layout (`packed/`, `tags.json`, `lyrics/`, `phonemes/`, …), then pass
`--data-subdir <data-subdir>` at launch. Without it, training reads the main pack.

**2. Pick the base checkpoint:**

```bash
modal run diskrot/modal_inspect_ckpts.py --prefix v8_sing   # on the volume
# local runs can point --init-from at any local .pt (full or *_inference slim)
```

**3. Launch** — always a **new** `--ckpt-subdir` / `--ckpt-dir` (an existing
`latest.pt` there would resume instead of starting the fine-tune):

```bash
# Modal LoRA (single H100 is plenty — recommended LoRA path)
modal run --detach diskrot/modal_train.py \
  --init-from v8_sing/best.pt --lora --ckpt-subdir v8_lora_x \
  [--data-subdir my_corpus] [--lora-r 16 --lora-alpha 32]

# Local LoRA (Apple Silicon / single GPU — the headline local use)
python -m diskrot.train --device mps \
  --init-from ./checkpoints/latest.pt --lora --ckpt-dir ./checkpoints/ft_run \
  --cache-dir ./token_cache --tags-path ./tags.json --lyrics-path ./lyrics

# Modal full fine-tune (8×H100, lower LR than the from-scratch 2.1e-4)
modal run --detach diskrot/modal_train.py --n-gpus 8 \
  --init-from v8_sing/best.pt --ckpt-subdir v8_ft --lr 5e-5
```

**4. Monitor / resume** — exactly like training: same log format, and a re-run
of the **same command** resumes from the run dir's `latest.pt` (a LoRA resume
rebuilds the frozen base from the recorded path and restores the adapters).

**5. Merge (LoRA only)** — fold the adapters into the base to get a standard
inference checkpoint (full-FT checkpoints are already standard; use
`modal_export_ckpt.py` to slim them as usual):

```bash
modal run diskrot/modal_merge_lora.py --base v8_sing/best.pt --lora v8_lora_x/best.pt
# local: python -m diskrot.merge_lora --base ... --lora ... --out ./checkpoints/merged_inference.pt
```

**6. Pull + serve/eval** — unchanged from a normal run; the merged file is a
drop-in standard checkpoint:

```bash
modal volume get nano-ckpts /v8_lora_x/merged_inference.pt ./checkpoints/latest.pt --force
```

## Rules of thumb

- **Startup precedence:** `latest.pt` in the run dir **>** `--init-from` **>**
  scratch. That's why a fine-tune needs a fresh run dir, and why re-running the
  same command is always the resume.
- **`--lora` requires `--init-from`** — adapters train on top of a pretrained
  base, never from scratch.
- **LoRA defaults:** r=16, alpha=32, targets = every attention + MLP Linear in
  the decoder blocks. Frozen: output head, token embeddings, norms, the
  lyric/melody encoders, and the CLAP tag projection
  (`--lora-train-text-proj` opts the projection in).
- **Checkpoint sizes:** adapter ckpts ~100 MB; full ckpts multi-GB. An adapter
  checkpoint has no `model` key — it records the base's path and is useless
  without it (and without merging, for inference).
- **LR:** full fine-tune wants a lower LR than from-scratch (think 5e-5 vs
  2.1e-4); LoRA tolerates the default-range LR since only adapters move.
- **Nothing here touches the main run or the prepared data** — the v8 launch
  command, checkpoint schema, and the data pipeline are unchanged; fine-tuning
  only ever writes to its own `--ckpt-subdir`.
