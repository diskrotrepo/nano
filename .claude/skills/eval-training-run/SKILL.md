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
top. (Hardware/throughput numbers in that doc are v8-era — the bands below
supersede them.)

Two vintages of reference numbers appear below; pick by the run's codec +
shape: **v9-class** (SpectroStream, K=24, 25 Hz, 180s segments, B200:4, EMA
on) vs **v8-class** (DAC, K=9, 86 Hz, 60s segments, 8×H100, no EMA). If
unsure, count the entries in a step line's `cb[...]` — 24 = v9-class.

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

`modal app logs` may also return only a recent **tail** of the history
(server-side truncation — observed 2026-07-12: a run 13 checkups in returned
just the last 2). Missing early checkup lines do NOT mean "no checkups ran";
reconstruct the earlier trajectory from each surviving checkup's
`beat previous best (X)` field, which carries the prior best score.

## Judge — in this order

1. **Reference numbers**: random-chance loss is `ln(vocab) = ln(1024) ≈ 6.93`
   per codebook — vocab is 1024 for both DAC and SpectroStream, so the floor
   survives the codec switch (re-derive it if vocab ever changes). Train loss
   above 6.93 after warmup = confidently wrong, not "still learning".
2. **Val trajectory** (the verdict): checkups every 1000 steps must trend DOWN.
   With EMA on (v9 default, `use_ema=True`) the checkup score — and `best.pt`
   — are the **EMA weights**, not the live ones: (a) EMA smooths noise away,
   so consecutive worse checkups are a *stronger* divergence signal than they
   were on v8; (b) don't compare the step-line train loss (live weights)
   against the checkup score (EMA) as if they were the same basis.
   Two+ consecutive strikes while LR is still climbing through warmup = the
   2026-06-12 divergence signature — stop the run, don't wait out patience.
   Healthy v9_stereo reference (B200:4, launched 2026-07-12): every checkup a
   new best through 13k — 5.2321@11k, 5.1777@12k, 5.1300@13k (just past the
   10k warmup end, full lr 1.5e-4). Healthy v8 reference (bf16+qk-norm):
   6.75@1k, 6.69@2k, 5.11@8k, 4.93@10k (warmup end) — every checkup a new
   best. Diverged v8 reference: 7.17 → 7.87 → 8.70 by step 3000. Scores are
   NOT comparable across vintages (different codec + K).
   The step line's `grad` field is the leading indicator: healthy is FLAT —
   v9_stereo sits ~0.10–0.20 at full LR; the v8 recipe ran a 0.3–2.5 band
   (clip=1.0 occasionally touched is fine on either). A ramp across windows
   precedes divergence by thousands of steps.
3. **Per-codebook order**: shallow codebooks converge first, so a checkup's
   `cb[...]` spread must be ordered low→high from cb0 (healthy v9 @13k:
   cb0 1.88 rising monotonically to cb23 6.46). Adjacent levels swapping in a
   single noisy 25-step train line is normal; judge ordering on checkup lines
   (averaged over 50 eval batches). cb0/the shallow levels HIGHEST or climbing
   = the divergence canary — cb0 led the 2026-06 explosion by ~800 steps. All
   K equal after thousands of steps = delay-pattern suspicion.
4. **Throughput**: reported `tok/s` counts global-batch × K × frames, so it is
   NOT comparable across codecs or shapes. v9_stereo steady state ≈
   **2.95–3.05M tok/s on B200:4** (the hardwired multi-GPU config; ~3.46M
   tokens/step → ~0.86 steps/s, ~29s per 25-step line). v8-era: ~1.6–1.7M on
   8×H100 (K=9, 60s). Isolated low windows in the first few thousand steps =
   late first-hit variant compiles (finite, self-quiescing). Persistently
   low/erratic after that = a starving rank. Ignore the step-25 line
   (compile-polluted average).
5. **ETA**: steps/s from the spacing of recent step lines (each line = 25
   steps), then `(target - current) / steps_per_s`. State it with the caveat
   that val-driven early stopping (patience 20) may end the run first.
6. **Infra events**: count NCCL rendezvous timeouts ("waiting for clients") —
   one is a transient (Modal auto-retries; resume is checkpoint-safe), repeats
   on the same app = bad node, stop + relaunch for fresh placement. Check the
   benign-noise list before flagging anything else.
7. **Checkpoint state**: `modal volume ls nano-ckpts <subdir>/` — `best.pt`
   updates on checkup improvements (EMA weights when EMA is on),
   `latest.pt`/`step_*.pt` every 5000 steps.

## Report

Lead with the one-line verdict (healthy / watch / diverging / stalled / dead),
then: latest checkup vs best, last train loss + cb spread, steps/s + ETA to the
user's target (default: next checkpoint-compat milestone or 100k), and any
infra events. Compare against the references above so "healthy" is a claim
about THIS model's known-good trajectory, not vibes — and say which vintage
(v9/v8) the comparison used.
