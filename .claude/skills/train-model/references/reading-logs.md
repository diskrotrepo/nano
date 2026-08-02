# Reading nano training logs (cheat-sheet)

Fast in-run triage. For the full phase table, per-prefix decoding, and deep
stall diagnosis, read [README.logs.md](../../../../README.logs.md) — this is just
the one-screen version.

## Phase order (multi-GPU DDP — B200:4 on the current path; 8×H100 historically)

`[parent] v2 sharded layout detected` → `[bundle]` train/val split → mmap open →
`[tags] loaded N entries` → `[clap-parent] sharding N tags across <n_gpus> GPUs`
→ workers spawn → `DDP active: world_size=<n_gpus>` (4 on B200:4) +
`model: XXXX.xx M params` (~2.0B; the exact number moves with K and the enabled
conditioning modules — v8 measured 2013.8M) → `EMA enabled (decay=0.999) —
best.pt saves EMA weights` (when `use_ema`, the v9 default) → `torch.compile` →
first `step` line → training.

## The one thing that trips everyone up

**Expect a few minutes of silence before the first `step` line.** The CLAP text
precompute is GPU-sharded n_gpus-wide (measured 241,603 tags in 89 s 8-way on
2026-06-12; expect ~2× that 4-way); the rest is NCCL init + torch.compile
(~2 min). No `step` line yet ≠ broken. Only worry past **~30 min**.

## Healthy numbers

| Metric | Healthy | Trouble |
|---|---|---|
| Train/val "load" | seconds (mmap open) | minutes → you fell back to the in-RAM path |
| CLAP precompute | ~2.5k tags/s per GPU | minutes-long per-GPU ETAs → CLAP not on GPU |
| First `step` | within ~10 min | > 30 min → silently stuck |
| `tok/s` steady state | v9 on B200:4 ≈ 2.95–3.05M (measured 2026-07-12); v8 on 8×H100 ≈ 1.6–1.7M | single-GPU-sized tok/s on a multi-GPU app → DDP broken |

`tok/s` counts global-batch × K × frames — NOT comparable across codecs/shapes;
recalibrate the healthy band whenever either changes.

Per-step line: `step N/400000 loss X lr X grad X tok/s X cb[c0 c1 ... c(K-1)]`
(24 `cb` entries on v9 SpectroStream, 9 on DAC). Per-codebook losses converge
**in order** (codebook 0 first). Random chance per codebook is `ln(1024) ≈ 6.93`
(vocab 1024 on both codecs). The shallow codebooks carry the most perceptual
content (on DAC: cb2–4); a flat `grad` is healthy, a ramp precedes divergence —
the eval-training-run skill owns the judgement bands.

## Monitor commands

```bash
modal app list | grep nano-train      # should show ephemeral (detached), Tasks: 1
modal app logs <app-id> -f            # follow
modal app stop <app-id> -y            # stop
```

`modal app logs` may return only a recent **tail** of the history (server-side
truncation — observed 2026-07-12: startup lines and 11 of 13 checkups gone on a
same-day run). Missing early lines ≠ missing events; reconstruct checkup
history from the `beat previous best (X)` fields.

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
