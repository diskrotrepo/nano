# v9 Distillation Plan (teacher → student)

How to distill the trained v9 base (`v9_stereo`, ~2B) into a small, fast student
for serving. **The pipeline already exists** (`--distill-from`, landed 2026-06-17,
tested, never yet trained) — this doc is the runbook for using it once the v9 base
run (README.v9.md A9) has plateaued. For continuing/adapting the *same-size* model
see [README.finetune.md](README.finetune.md); distillation is the opposite trade —
a **new, smaller model** that imitates the big one.

**Why:** decode speed scales with student size. A ~⅓-size student cuts per-step
compute ~3× (wall-clock gain is somewhat less — decode is bandwidth-bound; measure,
don't assume). This stacks with the other inference wins (25 Hz codec, cross-KV
cache, compiled decode) rather than replacing them.

## What distillation means here

| | |
|---|---|
| Student | A **fresh, smaller** `NanoAudioGPT` trained from scratch — shape set by `--d-model/--n-layers/--n-heads/--d-ff` at launch. Its lyric/melody encoders are its own (they scale with the student's `d_model`). |
| Teacher | A frozen checkpoint on nano-ckpts (`--distill-from v9_stereo/best.pt`). Loaded eval + `requires_grad_(False)` + bf16, one replica per rank, never DDP-wrapped/compiled, never saved ([diskrot/train.py](diskrot/train.py) `_build_teacher`). |
| Loss | `total = alpha·CE + (1-alpha)·KD`, `KD = tau²·KL(teacher‖student)` over the soft per-codebook logits, computed in fp32, same pad mask as CE (`_distill_loss`). Defaults `--distill-alpha 0.5 --distill-tau 2.0` (tau 1.5–2.0 typical). |
| Conditioning | The teacher sees the **identical batch and conditioning** as the student — one shared RNG draw for the tag CFG-drop, shared raw CLAP chunk cache (only the projection differs), identical lyric/melody keep gates. CFG dropout is preserved, so the student learns the uncond states too → **`cfg_scale`/`lyric_cfg_scale` work unchanged at inference.** |
| Val / best.pt | Plain CE on the student (`_evaluate` never runs the teacher), EMA-smoothed as usual — so the student's `best_val_loss` is **directly comparable to the teacher's**. |
| Checkpoints | Standard schema (nothing LoRA-like) — slim-export, serving, eval, MLX all work unchanged. |

## The graph

```
v9 base run (README.v9.md A9) ─► v9_stereo/best.pt  (teacher — EMA weights, has text_proj)
                                        │
packed v9 corpus (unchanged) ──► distill train (student from scratch, teacher frozen)
                                        │
                               v9_distill/best.pt ─► slim export ─► NANO_MODELS dual serve
                                                                    (fast=student, full=teacher)
```

No new data prep: the distill run reads the same pack / tags / lyrics / phonemes /
melody as the base run.

## Step by step

**0. Prereq — the teacher exists.** The v9 base run has plateaued and
`v9_stereo/best.pt` is the keeper (with `use_ema` on, `best.pt` already stores the
EMA-smoothed weights as `model`, which is exactly what you want to distill from —
**not** `latest.pt`, which holds the live weights).

```bash
modal run diskrot/modal_inspect_ckpts.py --prefix v9_stereo
```

**1. Launch** — same env gotcha as A8/A9 (`NANO_CODEC` bakes the frame rate), a
**fresh** `--ckpt-subdir` (an existing `latest.pt` there resumes instead), and the
student shape on the flags. The starting recipe (the shape documented at the
`--distill-from` flag in [diskrot/modal_train.py](diskrot/modal_train.py)):

```bash
export NANO_CODEC=spectrostream NANO_SS_DEPTH=32
modal run --detach diskrot/modal_train.py --n-gpus 4 --ckpt-subdir v9_distill \
  --distill-from v9_stereo/best.pt \
  --d-model 1280 --n-layers 16 --n-heads 10 --d-ff 5120 \
  --steps 150000 --lr 3.0e-4
```

- d1280 / L16 / h10 (head_dim 128, RoPE-safe) / ff5120 → a ~⅓-scale decoder; at
  v9's K=24 the head + embeddings add back some params, so expect roughly a
  600M-class student — **read the measured count off the launch line**, don't quote
  a number from here.
- `--n-gpus 4` matches the hardwired `B200:4` (8 trips the device-count assert);
  per-rank batch auto-selects 8 (global 32). The higher LR (3.0e-4 vs the base run's
  1.5e-4) is right for the smaller model; warmup/patience/EMA ride the DEFAULTS.
- Treat `--steps 150000` as a **cap** — patience=20 on the EMA val stops it at
  plateau, same discipline as the base run.

**2. Monitor** — same logs as any run (`modal app logs nano-train`, the
eval-training-run skill applies). Distill runs add `ce <x> kd <y>` to the step
line (and `train/ce_loss` / `train/kd_loss` to wandb): CE tracks hard-label fit,
KD tracks teacher agreement. Both should descend together; val checkups are plain
CE and comparable 1:1 against the teacher's `best_val_loss`.

**3. Slim-export + pull** (standard schema, standard tooling):

```bash
modal run diskrot/modal_export_ckpt.py --src v9_distill/best.pt
modal volume get nano-ckpts /v9_distill/best_inference.pt ./checkpoints/v9_distill.pt
```

**4. Serve — student default, teacher on request.** `NANO_MODELS` boots the server
multi-model (see [server/main.py](server/main.py) `_parse_models_env`); every
endpoint takes a `model` param routed through `_get_engine`, `GET /models` lists
what's loaded:

```bash
NANO_MODELS="fast=./checkpoints/v9_distill.pt,full=./checkpoints/v9_stereo.pt" \
NANO_DEFAULT_MODEL=fast  uvicorn server.main:app ...
# per request: POST /generate {"model": "full", ...} to fall back to the teacher
```

**5. Eval the gap before switching the default.** Same-prompt/same-seed A/B against
the teacher: `scripts/eval_checkpoint.py` for sample quality, the eval-checkpoint
skill's sampling sweep, and specifically **lyric intelligibility** (eval-lyrics →
the model-side WER eval) — fine-codebook detail and sung-word clarity are where a
student degrades first. If the gap is audible, the next levers in order: more steps
(raise the cap), a bigger student (d1536/L18), lower `--distill-alpha` (0.3 leans
harder on the teacher signal).

## Constraints (checked by asserts at startup)

- **The student cannot change the token geometry.** `n_codebooks`, vocab, and
  `pad_id` must equal the teacher's, and teacher `max_seq_len` must cover the
  student's ([diskrot/train.py](diskrot/train.py) startup asserts). A K=16 "lite"
  variant is therefore **not** a distillation product — that's a fresh base train
  (the pack stores depth 32, so it needs no re-tokenize, just a new run).
- `--distill-from` is **incompatible with `--lora`** (LoRA freezes a base;
  distillation trains a fresh student — pick one).
- The teacher checkpoint must carry `text_proj` for the tag path (`best.pt` does;
  a missing one only earns a startup warning and silently un-teaches tags — don't).
- FIM / stem conditioning follow the student's own flags (both off in v9 DEFAULTS,
  matching the teacher).

## Cost & memory notes

- The teacher adds a frozen 2B bf16 forward per step — expect a distill step to
  cost roughly **~2× a solo student step** (the teacher forward is comparable to
  the student's fwd+bwd). Still far cheaper per step than the base run, and the
  run is much shorter.
- The KD term materializes two fp32 log-softmaxes over `[B, K, T, V]` — at 180 s
  crops that's a multi-GB transient in the loss. Fine on B200:4 at per-rank 8; if
  it OOMs, drop `--batch-size` (per-rank) before touching anything else.
- Resume works like any run: re-running the same command resumes `v9_distill`'s
  `latest.pt` (which carries the EMA shadow for continuation).

## What distillation does **not** buy

- No quality above the teacher — it transfers, it doesn't improve. Quality work
  (data, re-caption, cb0 weighting) belongs in the base run.
- No step-count reduction — the student is still AR at 25 Hz, one step per frame.
  Fewer/faster steps come from the codec (banked), cross-KV cache, compiled decode,
  and CFG stage-batching, all orthogonal to this plan.
