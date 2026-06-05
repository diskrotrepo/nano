---
name: add-songs
description: >-
  Add MP3s to the nano training corpus and run them through the full data-prep
  pipeline (upload, prepare, tokenize, pack, optional auto-tag and transcribe).
  Use this skill when the user wants to add training data, ingest songs, build or
  grow the corpus, prepare data for training, run tokenize/pack/tag/transcribe, or
  asks how to get their MP3s into the model.
allowed-tools: Read, Bash
---

# Add songs to the nano corpus

Gets raw MP3s into a train-ready token cache. This is the data side of the
project — everything before `train-model`.

## Mental model

nano is **one bespoke model trained at scale on one kind of data**. More of the
same data helps; variety does not — do **not** curate for genre/style diversity.

- **Recommended corpus size:** ~50k songs minimum for coherent output. Below ~10k
  the model produces noise (pipeline-validation only). Ceiling is ~500k files
  (the `nano-corpus` volume's inode limit).
- For the cost-per-1,000-songs table and end-to-end walkthrough, read
  [README.modal.md](../../../README.modal.md) — don't restate the numbers here.
- For the per-stage data shapes, see the **Data Flow** section of
  [CLAUDE.md](../../../CLAUDE.md).

## Volumes

| Volume | Holds |
|---|---|
| `nano-corpus` | Raw MP3 files |
| `nano-tokens` | `.pt` token files, `packed/` shards, `tags.json`, `lyrics/` |
| `nano-ckpts` | Training checkpoints |

All Modal fan-out steps below are launched with `--detach` and are **resumable** —
re-run the same command to continue; it's safe to close your terminal.

## Pipeline (run in order)

### 1. Upload your MP3s
```bash
modal volume create nano-corpus      # first time only
modal volume put nano-corpus /path/to/mp3s/ /
```
Trailing `/` matters. (No crawler — you supply your own MP3s.)

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

### 4. Pack — `.pt` files → sharded mmap layout
Required once before training. Auto-detected by the trainer.
```bash
modal run --detach diskrot/modal_pack_cache.py
# or locally:
python -m diskrot.pack_cache --cache-dir ./token_cache
# flags: --out-dir  --shard-target-songs 5000
```
Writes `packed/packed_NNN.bin` + per-shard JSON + `packed_index.json`. Shards are
written atomically and a re-run skips complete-and-valid shards.

### 5. Auto-tag — *optional*, needed for text conditioning
```bash
modal run --detach diskrot/modal_auto_tag.py
# or locally:
python -m diskrot.auto_tag --corpus /path/to/mp3s --out ./tags.json
# flags: --device  --limit N
```
LP-MusicCaps writes a natural-language description per song into `tags.json`.
Re-running only processes new files.

### 6. Transcribe lyrics — *optional*, needed for lyric conditioning, **expensive**
```bash
modal run --detach diskrot/modal_transcribe.py
# or locally:
python -m diskrot.transcribe_lyrics --corpus /path/to/mp3s --out ./lyrics
# flags: --device
```
Demucs (vocal isolation) → Whisper, into a sharded `lyrics/` dir. This is by far
the costliest step — **skip it unless you will actually use lyric conditioning at
inference.**

## Decision points

- **Need tags?** Only if you'll train/serve text-conditioned (the default). Run step 5.
- **Need lyrics?** Only if you'll use lyric conditioning. Run step 6 (expensive).
- **Local vs Modal?** Modal for real fan-out scale; local for a smoke corpus to
  exercise the pipeline.

## Verify the cache is train-ready

On `nano-tokens` you should have `packed/packed_index.json` (required) and, if you
ran step 5, `tags.json`. Quick check:
```bash
modal volume ls nano-tokens
modal volume ls nano-tokens packed | head
```

## Next step

Cache is packed → go to the **train-model** skill. A fast pre-flight that exercises
the data path on synthetic tokens (no corpus needed): `pytest -m "not benchmark"`
(see the **run-tests** skill).
