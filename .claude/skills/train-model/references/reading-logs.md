# Reading nano training logs (cheat-sheet)

Fast in-run triage. For the full phase table, per-prefix decoding, and deep
stall diagnosis, read [README.logs.md](../../../../README.logs.md) — this is just
the one-screen version.

## Phase order (8×H100 DDP)

`[parent] v2 sharded layout detected` → `[bundle]` train/val split → mmap open →
`[tags] loaded N entries` → `[clap-parent] encoding N unique tags` (the long one)
→ workers spawn → `DDP active: world_size=8` + `model: 1514.xx M params` →
`torch.compile` → first `step` line → training.

## The one thing that trips everyone up

**Expect a long silence before the first `step` line — upwards of an hour at large
corpus sizes.** The CLAP text precompute (one-time, GPU 0 only) dominates setup.
No `step` line yet ≠ broken. Only worry if it's been **> 3 h**.

## Healthy numbers

| Metric | Healthy | Trouble |
|---|---|---|
| Train/val "load" | seconds (mmap open) | minutes → you fell back to the in-RAM path |
| CLAP precompute | ~50 tags/s | < 20 tags/s → CLAP not on GPU |
| First `step` | within ~95 min | > 3 h → silently stuck |
| `tok/s` on 8×H100 | ~6–7× single-H100 | ~1× → DDP broken |

Per-step line: `step N/400000 loss X lr X tok/s X cb[c0 c1 ... c8]`. Per-codebook
losses `cb[...]` converge **in order** (codebook 0 first). Random chance per
codebook is `ln(1024) ≈ 6.93`. Watch codebooks 2–4 — most perceptually important.

## Monitor commands

```bash
modal app list | grep nano-train      # should show ephemeral (detached), Tasks: 1
modal app logs <app-id> -f            # follow
modal app stop <app-id> -y            # stop
```

If logs are silent, confirm the container is alive and peek at process state /
GPUs per the "Diagnosing silent runs" section of
[README.logs.md](../../../../README.logs.md).

## Benign noise (ignore)

`[modal-client] ... Heartbeat attempt failed` (local CLI hiccup), mpg123/id3 and
`PySoundFile failed` warnings (tokenize only), `weights_only=False` FutureWarning
(DAC loader, pinned/safe).
