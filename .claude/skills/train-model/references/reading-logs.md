# Reading nano training logs (cheat-sheet)

Fast in-run triage. For the full phase table, per-prefix decoding, and deep
stall diagnosis, read [README.logs.md](../../../../README.logs.md) — this is just
the one-screen version.

## Phase order (8×H100 DDP)

`[parent] v2 sharded layout detected` → `[bundle]` train/val split → mmap open →
`[tags] loaded N entries` → `[clap-parent] sharding N tags across 8 GPUs` (~90 s)
→ workers spawn → `DDP active: world_size=8` + `model: 2013.xx M params` →
`torch.compile` → first `step` line → training.

## The one thing that trips everyone up

**Expect a few minutes of silence before the first `step` line.** The CLAP text
precompute is 8-way GPU-sharded (~90 s at full corpus scale — measured 241,603
tags in 89 s on 2026-06-12); the rest is NCCL init + torch.compile (~2 min).
No `step` line yet ≠ broken. Only worry past **~30 min**.

## Healthy numbers

| Metric | Healthy | Trouble |
|---|---|---|
| Train/val "load" | seconds (mmap open) | minutes → you fell back to the in-RAM path |
| CLAP precompute | ~2.5k tags/s per GPU (8-way shard, ~90 s total) | minutes-long per-GPU ETAs → CLAP not on GPU |
| First `step` | within ~10 min | > 30 min → silently stuck |
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

`[modal-client] ... Heartbeat attempt failed` (local CLI hiccup), `terminate
called without an active exception` right after `[clap-parent] done` (CLAP shard
worker teardown — parent survives; only worry if the task count drops),
`find_unused_parameters=True ... did not find any unused parameters` (false
positive — the flag is REQUIRED for CFG-dropout steps; never turn it off),
`Profiler record function ... will be ignored` (torch.compile noise),
mpg123/id3 and `PySoundFile failed` warnings (tokenize only),
`weights_only=False` FutureWarning (DAC loader, pinned/safe).
