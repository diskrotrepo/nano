---
name: eval-checkpoint
description: >-
  Evaluate a nano checkpoint and pick the right diagnostic for the question. Use
  this skill when the user wants to evaluate a checkpoint, score sample quality
  (CLAP / spectral / silence / onset), check overfitting (train vs val), decide
  whether lyric conditioning is worth the compute, sweep sampling parameters,
  check genre coverage, or sanity-check DAC codec fidelity.
allowed-tools: Read, Bash
---

# Evaluate a nano checkpoint

The value of this skill is **routing to the right eval**. Pick the row that
matches the question.

## Which eval do I want?

| Question | Command |
|---|---|
| Are the samples any good? (audio metrics + WAVs) | `python -m scripts.eval_checkpoint --ckpt ./checkpoints/best.pt --out ./eval/run` (add `--sweep` for a sampling sweep) |
| Am I overfitting / at capacity? | `modal run scripts/eval_train_vs_val.py` |
| Is lyric conditioning earning its compute? (val-loss ablation) | `modal run scripts/eval_lyrics_ablation.py` |
| Can a listener make out the words? (intelligibility / WER) | `modal run scripts/eval_lyric_wer.py` (tune `--cfg-scale`, `--lyric-cfg-scale`) |
| Best lyric/cfg settings, with clips to listen to? (real path) | `modal run scripts/eval_lyric_sweep.py` (grid: cfg × lyric_cfg × ladder × {lyrics, instrumental}; sweetener ON; WER + one MP3 per cell) |
| What sampling params are best? | `python -m eval.sweep.run_sweep --stage 1` then `--stage 2 --top-n 4` |
| How well is each genre covered? | `python -m eval.genre_sweep` (gaps: `python eval/genre_gap_eval.py`) |
| Is the DAC codec itself fine? | `python scripts/dac_roundtrip.py` |

## Details

### Sample quality — `eval_checkpoint`
```bash
python -m scripts.eval_checkpoint --ckpt ./checkpoints/best.pt --out ./eval/run --sweep
# flags: --seconds 15.0  --device  --prompts prompts.json  --top-n 5
```
Writes `report.md` (configs ranked by composite score), `metrics.json`, and sample
WAVs under `--out`. Metrics: CLAP text↔audio score, spectral flatness, silence
ratio, onset density. The score **filters silence and white-noise failure modes**
before ranking, so a high rank means actually-musical output.

### Overfitting — `eval_train_vs_val` (Modal, H100)
```bash
modal run scripts/eval_train_vs_val.py
# flags: --n-batches 100  --batch-size 64  --val-ratio 0.12  --seed 42
```
`train_loss << val_loss` → memorizing, add data. `train_loss ≈ val_loss` → at
capacity, go bigger.

### Lyric ablation — `eval_lyrics_ablation` (Modal, H100)
```bash
modal run scripts/eval_lyrics_ablation.py
# flags: --n-batches 400  --batch-size 64  --val-ratio 0.12  --segment-seconds 10.0
```
Compares val loss on lyric-bearing samples under full (tags+phoneme lyrics) vs
tags_only (lyric stream dropped) vs uncond. If the lyrics delta ≈ 0, the lyric
conditioner isn't being used — but for *intelligibility* prefer the WER eval below
(val loss can improve without the words being recoverable).

### Lyric intelligibility / WER — `eval_lyric_wer` (Modal, H100)
```bash
modal run scripts/eval_lyric_wer.py
# flags: --ckpt-path /ckpts/v8_sing/best.pt  --n-clips 30  --cfg-scale 3  --lyric-cfg-scale 0
```
The **primary success metric** for "does it sing the words." Generates audio
conditioned on held-out lyric lines, transcribes the output with the same
Demucs+Whisper pipeline used to build the training lyrics, and reports WER vs the
input lyric. Gibberish ≈ 1.0; falling WER across checkpoints == emerging
intelligibility. Raise `--lyric-cfg-scale` (e.g. 5–6) to push the lyric axis harder.

### Realistic lyric sweep — `eval_lyric_sweep` (Modal, H100)
```bash
modal run scripts/eval_lyric_sweep.py --ckpt-path /ckpts/v8_sing2/best_inference.pt
# flags: --n-clips 5  --seconds 12  --cfg-scales 3.0,5.0  --lyric-cfgs 0.0,3.0,6.0
```
Like `eval_lyric_wer` but through the **exact `/generate` path** — `InferenceEngine`
+ the prompt **sweetener ON** + per-cb ladder — so the clips match what users hear.
Grid = `cfg × lyric_cfg × ladder(HOT_FLAT/COLD_LADDER) × {lyrics, instrumental}` =
16 cells (12 lyric + 4 instrumental controls), `--n-clips` held-out lyrics each.
Per cell: WER (lyric cells) or leaked-word count (instrumental cells should be ≈0 —
confirms the `<instrumental>` marker) **plus one MP3 saved** to nano-output
(`/lyric_sweep/<tag>/`) to listen. Pull: `modal volume get nano-output /lyric_sweep/<tag> ./sweep_clips --force`.
Use `eval_lyric_wer` for the clean isolated WER number; use this for picking
serving settings + hearing them.

### Sampling sweep — `eval.sweep.run_sweep`
```bash
SMOKE=1 python -m eval.sweep.run_sweep                  # 1 quick 3s clip, sanity
python -m eval.sweep.run_sweep --stage 1               # coarse: 24 settings × 3 prompts
python -m eval.sweep.run_sweep --stage 2 --top-n 4     # fine: re-rank top-N, 2 clips each
NANO_CKPT=checkpoints/best_1500m.pt python -m eval.sweep.run_sweep --stage 1
```
Samples land in `eval/sweep/samples/<ckpt_tag>/`, results in
`eval/sweep/results/<ckpt_tag>/run_manifest.json` (cumulative).

### Genre coverage / gaps
```bash
python -m eval.genre_sweep              # ranked hit counts → eval/genre_report.txt (reads a local eval/tags.json)
python eval/genre_gap_eval.py           # gap diagnostics; auto-pulls latest tags.json off nano-tokens
#   genre_gap_eval flags: --local  --tags-path PATH  --no-clap  --clap-sample 30000  --seed 0
```
Both run from the repo root and need a `tags.json` (the auto-tag step). `genre_sweep`
reads `eval/tags.json` locally; `genre_gap_eval` pulls the latest off `nano-tokens`
unless you pass `--local` / `--tags-path`.

## Prereqs

- A **local checkpoint** at `./checkpoints/best.pt` (or pass `--ckpt` / set
  `NANO_CKPT`). Get one via the **train-model** skill's extract + download step.
- Genre evals need `tags.json` on `nano-tokens`.

> Local CLAP audio scoring can crash on systems with ffmpeg 8 (torchcodec
> incompatibility); generation/decode/wav-write are unaffected — if so, rank by the
> librosa metrics and skip CLAP.

## Next step

A checkpoint that scores well → **serve-model**.
