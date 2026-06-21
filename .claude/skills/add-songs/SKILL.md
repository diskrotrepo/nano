---
name: add-songs
description: >-
  Add MP3s to the nano training corpus and run them through the full data-prep
  pipeline (upload, prepare, tokenize, optional melody/auto-tag/transcribe/
  structure/key-detect, phonemize, pack). Use this skill when the user wants to
  add training data, ingest songs, build or grow the corpus, prepare data for
  training, run tokenize/melody/pack/tag/transcribe/structure/phonemize, or asks
  how to get their MP3s into the model.
allowed-tools: Read, Bash
---

# Add songs to the nano corpus

Gets raw MP3s into a train-ready token cache. This is the data side of the
project — everything before `train-model`.

## Mental model

nano is **one bespoke model trained at scale on one kind of data**. More of the
same data helps; variety does not — do **not** curate for genre/style diversity.

- **Recommended corpus size:** ~50k songs minimum for coherent output. Below ~10k
  the model produces noise (pipeline-validation only). No hard ceiling — the raw
  audio lives in R2 object storage (the old ~500k figure was the retired
  `nano-corpus` Volume's inode cap).
- For the cost-per-1,000-songs table and end-to-end walkthrough, read
  [README.modal.md](../../../README.modal.md) — don't restate the numbers here.
- For the per-stage data shapes, see the **Data Flow** section of
  [CLAUDE.md](../../../CLAUDE.md).

## Storage

Raw audio lives in **Cloudflare R2** (the `nano-audio` bucket, under
`waves/wave_<id>/`), mounted via `modal_common.corpus_mount()`. The legacy
`nano-corpus` Volume is retired. Everything else is on Modal Volumes:

| Store | Holds |
|---|---|
| `nano-audio` (R2) | Raw MP3 files under `waves/wave_<id>/` |
| `nano-tokens` | `.pt` token files, `packed/` shards (incl. `.mel.bin`), `tags.json`, `lyrics/`, `structure/`, `keys.json`, `phonemes/` |
| `nano-melody` | `<name>.mel.npy` chroma sidecars (own volume — keeps nano-tokens under its inode cap) |
| `nano-ckpts` | Training checkpoints |

All Modal fan-out steps below are launched with `--detach` and are **resumable** —
re-run the same command to continue; it's safe to close your terminal.

## Pipeline (run in order)

### 1. Upload your MP3s
Raw audio goes to the R2 `nano-audio` bucket under a wave prefix (S3-compatible
upload — rclone / `aws s3 cp` / the Cloudflare UI). One-time R2 + `r2-creds` +
`NANO_AUDIO_*` setup is in [README.waves.md](../../../README.waves.md#one-time-setup).
```bash
rclone copy /path/to/mp3s/ r2:nano-audio/waves/wave_0/
```
(No crawler — you supply your own MP3s. See [README.waves.md](../../../README.waves.md)
for the wave-by-wave ingestion model.)

### 2. Prepare — validate, dedupe, drop
Dry-run first (no deletions), inspect the report, then apply:
```bash
modal volume create nano-tokens      # first time only
modal run --detach diskrot/modal_prepare.py          # dry-run + report
modal run --detach diskrot/modal_prepare.py --apply  # actually delete
```
Drops files that fail `ffprobe`, byte-identical duplicates (SHA-256), clips <20s,
and files >5:30. **Why the length cap:** long DJ mixes / album rips OOM the L4
tokenizer and distort the per-file crop sampler. Resumable via
`/tokens/prepare_manifest.json`.

### 3. Tokenize — MP3 → DAC tokens
Modal (fan-out, the scale path):
```bash
modal run --detach diskrot/modal_tokenize.py
```
Local (validation / small corpora):
```bash
python -m diskrot.tokenize --corpus /path/to/mp3s --out ./token_cache
# flags: --device {cuda|mps|cpu}  --min-seconds 20.0  --batch-size 4
```
Output: per-song int16 `.pt` files (`[9, T]`) on `nano-tokens` (or `./token_cache`).

### 4. Extract melody (chroma) — *optional*, needed for melody conditioning / `/cover`
```bash
modal run --detach diskrot/modal_melody.py
```
Writes a per-song `<name>.mel.npy` (12-bin chromagram, forced to the song's DAC
frame count) to the dedicated **`nano-melody`** volume. **Run after tokenize** (it
reads each `.pt` on `nano-tokens` for the frame count) and **before pack**. CPU,
cheap, resumable (skips songs that already have chroma). There is no local CLI for
this step (use `diskrot.melody.extract_chroma` programmatically for a local smoke
corpus). Skip it if you won't use melody conditioning — the rest of the pipeline
works unchanged (tags+lyrics only).

> Why its own volume: this adds one small file per song. `nano-tokens` already
> holds ~one `.pt` per song and sits near the 500k-inode volume cap, so co-locating
> the chroma there would push it over mid-run — hence `nano-melody`. The loose `.pt`
> / `.mel.npy` are only inputs to pack — prunable after packing (training reads only
> the shards).

### 5. Pack — `.pt` files (+ chroma) → sharded mmap layout
Required once before training. Auto-detected by the trainer.
```bash
modal run --detach diskrot/modal_pack_cache.py
# or locally:
python -m diskrot.pack_cache --cache-dir ./token_cache --mel-cache-dir ./token_cache
# flags: --out-dir  --shard-target-songs 5000  --mel-cache-dir (chroma dir)
```
Writes `packed/packed_NNN.bin` + per-shard JSON + `packed_index.json` on
`nano-tokens`. The Modal wrapper mounts `nano-melody` and **auto-detects**
`*.mel.npy` there, writing the parallel `packed_NNN.mel.bin` chroma sidecar (at the
same offsets) back onto `nano-tokens`; the local packer needs `--mel-cache-dir` to
do so. Shards are written atomically and a re-run skips complete-and-valid shards.

### 6. Auto-tag — *optional*, needed for text conditioning
```bash
modal run --detach diskrot/modal_auto_tag.py
# or locally:
python -m diskrot.auto_tag --corpus /path/to/mp3s --out ./tags.json
# flags: --device  --limit N
```
LP-MusicCaps writes a natural-language description per song into `tags.json`.
Re-running only processes new files.

### 7. Transcribe lyrics — *optional*, needed for lyric conditioning, **expensive**
```bash
modal run --detach diskrot/modal_transcribe.py
# or locally:
python -m diskrot.transcribe_lyrics --corpus /path/to/mp3s --out ./lyrics
# flags: --device
```
Demucs (vocal isolation) → Whisper, into a sharded `lyrics/` dir. This is by far
the costliest step — **skip it unless you will actually use lyric conditioning at
inference.**

### 7b. Filter hallucinated lyrics — *recommended* after step 7 completes
```bash
modal run --detach diskrot/modal_filter_lyrics.py          # dry-run report first
modal run --detach diskrot/modal_filter_lyrics.py --apply  # then rewrite shards
# or locally: python -m diskrot.filter_lyrics --lyrics-dir ./lyrics [--apply]
```
Nulls Whisper-invented captions over instrumentals ("Thank you." etc. — ~24% of
with-words entries) so they train as `<instrumental>`, not `<vocals>` with
garbage words. CPU, seconds, idempotent. **Only after the transcribe fleet has
fully finished** (its orchestrator's in-memory flush clobbers concurrent edits),
and before phonemize.

### 8. Structure — *optional*, needed for section markers (`[chorus]` etc.)
```bash
modal run --detach diskrot/modal_structure.py     # --limit 200 first to calibrate
```
allin1 (Demucs + joint beat/segment model) → sharded `structure/` dir with
per-song sections + bpm (the tempo marker source). Loaded at train time, not
packed — a partial pass just yields `<no_section>`. Expensive (L4 fan-out).

### 9. Phonemize — *recommended* if you ran transcribe (step 7)
```bash
modal run --detach diskrot/modal_phonemize.py
# or locally: python -m diskrot.phonemize --lyrics-path ./lyrics --out-dir ./phonemes
```
Pre-runs g2p per song into a sharded `phonemes/` dir so the DataLoader doesn't
pay ~20–200 ms/song of live g2p at train time (which can starve the 8×H100
step). CPU, ~$1, resumable. Re-run after any re-transcribe.

### 10. Key detect — *optional*, needs the melody-packed shards (steps 4+5)
```bash
modal run --detach diskrot/modal_key_detect.py
# or locally: python -m diskrot.key_detect --cache-dir ./token_cache
```
Krumhansl key estimate over the packed chroma → `keys.json`, the `<key_*>`
header-marker source (enables "[a minor]" prompts). CPU, ~$1, resumable; songs
without an estimate get `<unknown_key>`.

## Decision points

- **Need tags?** Only if you'll train/serve text-conditioned (the default). Run step 6.
- **Need lyrics?** Only if you'll use lyric conditioning. Run step 7 (expensive),
  then 7b (filter hallucinations — cheap) and step 9 (phonemize — cheap, protects
  training throughput).
- **Need melody / `/cover`?** Run step 4 (melody) then repack (step 5) so the chroma
  sidecar lands. Cheap (CPU) — worth it if you want the hum→re-render capability.
  With the sidecar packed, step 10 (key detect) is ~free and adds key control.
- **Need section markers (`[verse]`/`[chorus]`)?** Run step 8 (expensive).
- **Local vs Modal?** Modal for real fan-out scale; local for a smoke corpus to
  exercise the pipeline.

## Verify the cache is train-ready

On `nano-tokens` you should have `packed/packed_index.json` (required); if you ran
the melody step, `packed/` also has `packed_NNN.mel.bin` (and `packed_index.json`
reports `"has_melody": true`); and per optional pass: `tags.json` (step 6),
`lyrics/` (7), `structure/` (8), `phonemes/` (9), `keys.json` (10). Quick check:
```bash
modal volume ls nano-tokens
modal volume ls nano-tokens packed | head
```

## Next step

Cache is packed → go to the **train-model** skill. A fast pre-flight that exercises
the data path on synthetic tokens (no corpus needed): `pytest -m "not benchmark"`
(see the **run-tests** skill).
