---
name: eval-lyrics
description: >-
  Audit and evaluate nano's lyric data and lyric conditioning. Use this skill
  when the user wants to analyze the lyric dataset, check transcribe/phonemize
  coverage or backfill progress, measure the Whisper-hallucination rate, see
  the vocal-gender or word-count distribution, decide whether the lyric filter
  should run, or evaluate whether a checkpoint actually sings the words.
allowed-tools: Read, Bash
---

# Evaluate lyrics — data health and model intelligibility

Two distinct questions live here. Route first:

| Question | Command |
|---|---|
| **Data**: how healthy is the lyric corpus right now? (coverage, hallucination rate, gender, word counts) | `modal run scripts/lyrics_audit.py` |
| **Data**: exactly what would the hallucination filter null? | `modal run diskrot/modal_filter_lyrics.py` (dry-run report) |
| **Model**: does a checkpoint sing intelligible words? (WER) | `modal run scripts/eval_lyric_wer.py` — details in the **eval-checkpoint** skill |
| **Model**: is lyric conditioning earning its compute? (val-loss ablation) | `modal run scripts/eval_lyrics_ablation.py` — details in the **eval-checkpoint** skill |

The model-side rows are owned by **eval-checkpoint**; this skill owns the
data side.

## The data audit — `scripts/lyrics_audit.py`

One CPU container, read-only sweep of `nano-tokens` + the corpus listing.
**Safe to run while transcribe/filter is in flight** — it's a snapshot of disk.
Reports per-stream coverage of the packed corpus (tags / structure / keys /
lyrics / phonemes), the lyric breakdown, gender + word-count distributions
among vocal-ready songs, and the top detected keys.

### Reading the lyric breakdown

Entry semantics in `lyrics/` (the audit reports all four buckets):

- **null** = transcribed, no vocals found → trains as `<instrumental>`
- **has words, flagged** = Whisper hallucination (a caption like "Thank you."
  invented over instrumental audio) → the filter will null it
- **has words, clean** = **vocal-ready**, the songs that teach singing
- **ABSENT** = never transcribed → trains as `<unknown_vocals>`

Expected rates from the 2026-06 full-corpus sweeps: ~48% instrumental(null),
~25% of with-words entries hallucinated, ~38% of transcribed songs
vocal-ready. A hallucination rate far above ~25% or a vocal-ready share far
below ~⅓ deserves investigation before training.

`gender: unset` means a pre-gender (v7-era) entry — the field is written by
the transcribe pass itself, so only re-running transcribe adds it.

### Acting on the audit

- Lyrics/phonemes coverage low → transcribe backfill incomplete (check
  `modal app list` for a running `nano-transcribe` app) or phonemize hasn't
  run since the last transcribe.
- Hallucinated count > 0 → run the filter, **but only after the transcribe
  fleet has fully finished** — its orchestrator holds shard contents in
  memory and the next flush would clobber concurrent edits. Dry-run is
  read-only and always safe; `--apply` is the dangerous one.
- Required order: transcribe → `modal run diskrot/modal_filter_lyrics.py
  --apply` → `modal run --detach diskrot/modal_phonemize.py` → re-audit →
  train. Phonemize after the filter, so junk never enters the phoneme store.

## Prereqs

- The packed corpus must exist on `nano-tokens` (the audit measures coverage
  *of the packed set*; see the **add-songs** skill for the pipeline).
- Model-side evals need a checkpoint — see **eval-checkpoint** for flags,
  baselines, and interpretation.

## Next step

Data healthy and fully phonemized → **train-model**. Checkpoint in hand →
**eval-checkpoint** (WER row) to track intelligibility across steps.
