---
name: eval-training-run
description: >-
  Evaluate the health of a LIVE nano training run from its Modal logs — loss
  trajectory vs the random floor, per-codebook convergence order, val-checkup
  trend and strikes, throughput stability, ETA, and infra noise triage. Use this
  skill when the user asks "how is training going", "is the run healthy",
  "evaluate the training run", "is it diverging", or wants an ETA to a step
  count. For evaluating a finished CHECKPOINT (sample quality, WER, overfitting)
  use eval-checkpoint instead; for launching/resuming runs use train-model.
allowed-tools: Read, Bash
---

# Evaluate a live training run

Produces a structured health report from the run's logs. Read
[train-model/references/reading-logs.md](../train-model/references/reading-logs.md)
for line formats and the benign-noise list; this skill is the judgement layer on
top.

## Gather

```bash
modal app list | grep nano-train                  # find the app id + state
APP=ap-...
modal app logs $APP 2>&1 | grep -aE "step +[0-9]+/" > /tmp/steps.txt
modal app logs $APP 2>&1 | grep -aE "checkup at step|early stopping" > /tmp/checkups.txt
modal app logs $APP 2>&1 | grep -acE "Timed out .* waiting for clients|Traceback|CUDA out of memory"
```

(Full-log queries can hit a server-side resource limit once CLAP progress spam
accumulates — always pipe through a grep filter, never page raw logs.)

## Judge — in this order

1. **Reference numbers**: random-chance loss is `ln(1024) ≈ 6.93` per codebook.
   Train loss above 6.93 after warmup = confidently wrong, not "still learning".
2. **Val trajectory** (the verdict): checkups every 1000 steps must trend DOWN.
   Two+ consecutive strikes while LR is still climbing through warmup = the
   2026-06-12 divergence signature — stop the run, don't wait out patience.
   Healthy v8 reference (the successful bf16+qk-norm launch): 6.75@1k,
   6.69@2k, 5.11@8k, 4.93@10k (warmup end, full LR) — every checkup a new best.
   Diverged v8 reference: 7.17 → 7.87 → 8.70 by step 3000.
   The step line's `grad` field is the leading indicator: healthy ≈ flat
   0.3–2.5 band (clip=1.0 occasionally touched is fine); a ramp across windows
   precedes divergence by thousands of steps.
3. **Per-codebook order**: cb0 must be the LOWEST (it converges first; docs:
   "converge in order"). cb0 highest and climbing = the divergence canary —
   it led the 2026-06 explosion by ~800 steps. All nine equal after thousands
   of steps = delay-pattern suspicion.
4. **Throughput**: steady-state `tok/s` ≈ 1.6–1.7M on 8×H100 with the
   bf16+qk-norm recipe (60s segments, global batch 32, ~1.1 steps/s; the old
   fp16 run measured ~2.0M — qk-norm/bf16 cost ~20%). Isolated low windows in
   the first few thousand steps = late first-hit variant compiles (finite,
   self-quiescing). Persistently low/erratic after that = a starving rank.
   Ignore the step-25 line (compile-polluted average).
5. **ETA**: steps/s from the spacing of recent step lines (each line = 25
   steps), then `(target - current) / steps_per_s`. State it with the caveat
   that val-driven early stopping (patience 20) may end the run first.
6. **Infra events**: count NCCL rendezvous timeouts ("waiting for clients") —
   one is a transient (Modal auto-retries; resume is checkpoint-safe), repeats
   on the same app = bad node, stop + relaunch for fresh placement. Check the
   benign-noise list before flagging anything else.
7. **Checkpoint state**: `modal volume ls nano-ckpts <subdir>/` — `best.pt`
   updates on checkup improvements, `latest.pt`/`step_*.pt` every 5000 steps.

## Report

Lead with the one-line verdict (healthy / watch / diverging / stalled / dead),
then: latest checkup vs best, last train loss + cb spread, steps/s + ETA to the
user's target (default: next checkpoint-compat milestone or 100k), and any
infra events. Compare against the references above so "healthy" is a claim
about THIS model's known-good trajectory, not vibes.
