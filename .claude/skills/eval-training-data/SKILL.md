---
name: eval-training-data
description: >-
  Audit the health of nano's training data before a run — corpus size vs the
  scale rails, per-stream conditioning coverage (tags / structure / keys /
  lyrics / phonemes / melody), song-duration distribution, and tag/genre
  spread. Use this skill when the user wants to check whether the data is
  ready to train, see how many songs have each conditioning stream, check
  melody or phoneme coverage, audit corpus size, or find what data prep is
  still missing. For the lyric corpus specifically use eval-lyrics; for a
  trained checkpoint use eval-checkpoint.
allowed-tools: Read, Bash
---

# Evaluate the training data

Answers one question: **is the packed corpus healthy and ready to train?**
Coverage of every conditioning stream, corpus scale, duration spread, and
genre distribution — measured against the *packed (trainable)* set, not the
raw corpus, because that's what the dataset actually serves.

This is the data-side counterpart to **eval-checkpoint** (which evaluates a
trained model). It's a superset of **eval-lyrics** on the coverage axis; that
skill owns the deep lyric-quality detail (hallucination filter, gender,
language).

## Which eval do I want?

| Question | Command |
|---|---|
| **Is everything covered + at scale?** (the full audit — start here) | `modal run scripts/lyrics_audit.py` |
| What would the lyric hallucination filter null? | `modal run --detach diskrot/modal_filter_lyrics.py` (dry-run report) — see **eval-lyrics** |
| **What's the genre mix / are the gaps closed?** | `python -m eval.genre_gap_eval` (auto-pulls `tags.json`; regex + CLAP, CLOSED/OPEN verdict) — see **Genre mix** below |
| Is the DAC codec itself faithful? (token quality, not coverage) | `python scripts/dac_roundtrip.py` — see **eval-checkpoint** |

The lyric detail is owned by **eval-lyrics**; this skill owns the
**whole-corpus readiness** question, which includes running the genre gap eval
as a second step.

## The audit — `scripts/lyrics_audit.py`

One CPU container, read-only sweep of `nano-tokens` + `nano-melody` + the
corpus listing. **Safe to run while any data-prep pass is in flight** (melody,
transcribe, structure, …) — it's a snapshot of disk. Despite the historical
name it covers *all* streams, not just lyrics.

```bash
modal run scripts/lyrics_audit.py        # prints the report to the logs
```

### What it reports

1. **Corpus + scale check** — packed (trainable) song count with a verdict vs
   the rails in [CLAUDE.md](../../../CLAUDE.md): below ~10k = noise
   (pipeline-validation only), ~50k = recommended floor for coherent output,
   ~500k = a soft reference scale (the corpus is in R2 — no inode ceiling). More
   of the same data helps; variety does not — don't read a genre skew as a
   problem to curate away.
2. **Per-stream coverage of trainable songs** — tags / structure / keys /
   lyrics / phonemes / **melody**, each as `n / trainable (pct)`.
3. **Melody source vs packed** — `.mel.npy` on nano-melody (real contour
   extracted) vs `.mel.bin` folded into a `has_melody` pack shard. **Both**
   must hold for melody to train. Source-without-packed → re-pack with
   `--mel-cache-dir`. Packed-but-zero-filled → in a melody shard but missing
   source chroma (silent rows).
4. **Song-duration distribution** — from the packed offsets, plus the share
   shorter than the 60s Modal training crop.
5. **Model settings vs data scale** — the default 400k-step run translated into
   **effective epochs of unique audio** (`steps × batch × 60s ÷ total trainable
   audio`), with an under/over-training verdict. nano is *one* fixed-shape ~2.0B
   net (no family of sizes — the `DEFAULTS` dict in `diskrot/modal_train.py` is
   the source of truth), so the step count is the lever that has to match the
   trainable-song scale; this is where you confirm it does.
6. **Lyric breakdown / gender / word-count / language / keys** — the lyric
   detail; **eval-lyrics** is the skill that interprets these in depth.

The audit container is slim and torch-free, so it deliberately does **not**
compute the genre mix — run the dedicated tool as a second step (below).

### Reading the coverage numbers

- **Optional streams** (tags / structure / keys / melody) under 100% are not
  blockers — the model trains with `<unknown_*>` / null fallbacks for missing
  rows. They're a *completeness* signal: a stream you intended to use should be
  near-full before you rely on it.
- **phonemes coverage** is measured vs *all* trainable songs, so it looks low
  whenever the corpus is mostly instrumental — that's expected, not a gap.
  What matters is phonemes ≈ vocal-ready lyrics (every singable song
  phonemized); the **eval-lyrics** skill explains this distinction.
- A stream you just ran (e.g. melody, mid-`modal_melody.py`) climbing toward
  full across re-runs = the pass is progressing; flat = it stalled (check
  `modal app list`).
- **Effective epochs** is the run sweeping the corpus end-to-end (random crops,
  so it's a content-coverage ratio, not literal passes). Healthy ~2–15×: under
  ~2× the net is undertrained for a corpus this size (add data or raise
  `steps`); over ~15× invites repetition/memorization (cut `steps` or add data).
  Because the model shape is fixed, the only knob the verdict points at is the
  step count in `DEFAULTS` — adjust there, not the architecture.
## Genre mix — `eval/genre_gap_eval.py`

The genre picture is a **second step**, not part of the Modal audit: it needs
CLAP (torch), runs locally, and pulls the latest `tags.json` off nano-tokens
itself. Run it after the audit for the authoritative distribution.

```bash
python -m eval.genre_gap_eval            # regex + CLAP zero-shot (default)
python -m eval.genre_gap_eval --no-clap  # regex only (fast, no torch/CLAP)
```

What it gives you (writes `eval/genre_gap_report.txt` + a
`eval/genre_gap_snapshot.json` for run-over-run deltas):

- Per **gap genre** (country / latin / jazz / blues / soul-r&b / funk-disco /
  reggae / gospel / afro / indian / east-asian / mediterranean) a **regex %**
  and a **CLAP≈ advisory %**, against the floor (≥4% of corpus **and** ≥8k
  distinct songs) → a **CLOSED ✅ / OPEN ❌** verdict.
- The **verdict is regex-driven** (conservative, never invents a genre); CLAP is
  a coarse zero-shot **upper bound** shown alongside — a big `clap ≫ regex` gap
  flags a genre worth a manual look, it does not flip the verdict.
- Exits **non-zero if any gap is OPEN**, so it can gate a launch.

Read it for *gap coverage*, not a full distribution — the dominant
electronic/rock/pop captions are CLAP **distractor** anchors, not reported
buckets. Per the scale principle, an OPEN gap is a *data* signal (add more of
that kind via **add-songs**); settings can't close it, and you should **not**
curate the overall skew away — more of the same data is the goal, breadth of
genre is not.

### Acting on the audit

| Symptom | Fix |
|---|---|
| Trainable count below ~50k | Add data — **add-songs** skill (more of the same kind). |
| Effective epochs under ~2× | Raise `steps` in `DEFAULTS` (or add data) — re-audit. |
| Effective epochs over ~15× | Lower `steps` in `DEFAULTS` (or add data) to avoid memorization. |
| tags / structure / keys / lyrics low | Run the missing prep stage — **add-songs**. |
| melody source low | Let `modal run --detach diskrot/modal_melody.py` finish. |
| melody source high but packed low | Re-pack with `--mel-cache-dir` (**add-songs** pack step) so the chroma reaches the dataset. |
| phonemes ≪ vocal-ready lyrics | Phonemize hasn't run since the last transcribe — **eval-lyrics** / **add-songs**. |
| hallucinated lyrics > 0 | Run the filter — **eval-lyrics** owns the order (transcribe → filter → phonemize). |
| Many songs < 60s | Expected for short clips; only worrying if it's most of the corpus (crops get padded/wrapped). |
| Genre gap OPEN (`genre_gap_eval`) | Add more of that genre — **add-songs** (a data gap; settings can't close it). |

## Prereqs

- The corpus must be **packed** on `nano-tokens` — the audit measures coverage
  *of the packed set*. If `packed/packed_index.json` is absent, run the pack
  step first (**add-songs**).
- Genre distribution needs `tags.json` on `nano-tokens` (the auto-tag step).

## Next step

All streams you intend to use are near-full, the corpus is at scale, effective
epochs land in the healthy band, and the genre gaps you care about are CLOSED
(or knowingly accepted) → **train-model**. A stream is short → the fix table
above, then re-audit. Lyric data needs deeper triage → **eval-lyrics**. Already
have a checkpoint → **eval-checkpoint**.
