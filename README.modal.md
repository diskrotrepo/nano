# Training on Modal

End-to-end pipeline using Modal's cloud GPUs. Best path if you don't have local hardware or want to scale tagging/transcription across many containers.

This is a bespoke model: it trains on one kind of data at scale. You supply your own MP3s — more of the same data helps, but variety does not, so the corpus is not curated for genre/style diversity.

**Recommended corpus size:** at least **~50,000 songs** for coherent musical output. Below **~10,000 songs** the model mostly produces noise — useful only for validating the pipeline end-to-end, not for a real model. More of the same kind of data keeps helping, so larger is better; the practical ceiling is **~500,000 files** (the `nano-corpus` volume's inode limit). A tiny "smoke" corpus is still handy for exercising the pipeline, but it will not produce musical output.

## Cost summary

Cost for the data-prep steps scales roughly linearly with corpus size, so the table below quotes them **per 1,000 songs** — multiply by however many thousands of songs you bring. Training is the exception: its cost is driven by `steps` and model size, **not** corpus size, so it's a flat range. All cost estimates are against Modal's published GPU prices ([modal.com/pricing](https://modal.com/pricing)).

| Step | GPU | Per 1,000 songs |
|---|---|---|
| 1. Upload | — | uplink-bound (~0.7 GB/min at 100 Mbps) |
| 2. Prepare | CPU × 20 | ~20 s, <$0.01 |
| 3. Tokenize | L4 × 50 | ~1.5 min, ~$0.07–0.10 |
| 4. Auto-tag (optional) | L4 × 20 | ~0.5 min, ~$0.02 |
| 4b. Transcribe lyrics (optional) | L4 × 50 | ~14 min, ~$5–7 |
| 4c. Filter lyrics (recommended, after 4b) | CPU | negligible (seconds, one container) |
| 4d. Phonemize (recommended, after 4c) | CPU × 16 | negligible (~$1 full corpus) |
| 5. Pack (sharded mmap) | CPU | negligible |
| 5b. Key detect (optional, after 5) | CPU × 4 | negligible (~$1 full corpus) |
| 6. Train (flat — corpus-independent) | H100 × 8 DDP | a few thousand USD for the full 400k-step run |

GPU rates used above (as of 2026-05): H100 ≈ $5.92/hr, A100-40 ≈ $3.10/hr, L4 ≈ $0.30/hr, debian_slim CPU ≈ $0.10/hr.

Modal bills per second of actual compute, and the tokenize/tag/transcribe/train steps are all detached, so wall-clock time doesn't tie up your terminal. The two expensive steps are transcribe (skip it unless you actually plan to use lyric conditioning at inference) and train. Training has early stopping with `patience=20`, so a typical run finishes 30–50% sooner than the full-step worst case.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A [Modal](https://modal.com) account
- A directory of MP3 files to train on

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
modal token new
```

### Hugging Face token (optional, recommended)

Steps 3 and 4 download models from Hugging Face Hub. Without a token you may hit rate limits. Create a Modal secret with your HF token:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_YOUR_TOKEN_HERE
```

Get a token at https://huggingface.co/settings/tokens (a read-only token is sufficient).

## 1. Get your corpus onto Modal

Create a Modal volume and upload your own MP3s (trailing `/` is important):

```bash
modal volume create nano-corpus
modal volume put nano-corpus /path/to/mp3s/ /
```

Upload runs at your connection speed (~0.7 GB/min at 100 Mbps).


## 2. Prepare (required for tokenize on L4)

Cheap CPU pass that validates, dedupes, and filters the uploaded corpus. Catches files that would otherwise crash or be silently skipped downstream, and drops long mixes (≥`MAX_DURATION_S`) so the L4 tokenizer doesn't OOM on hour-long files. The constants live at [modal_prepare.py:64-71](diskrot/modal_prepare.py#L64-L71) if you need to tune.

What it does:

- Runs `ffprobe` on every MP3 — files that fail to decode are marked for deletion.
- Drops files shorter than 20s (the tokenizer's silent-skip threshold) so the count surfaces in the report instead of disappearing later.
- Drops files longer than `MAX_DURATION_S` (5:30) — these are DJ mixes / full-album rips / hour-long streams, not songs, and they distort the per-file crop sampler.
- SHA-256s file bytes and removes byte-identical duplicates.
- Writes a manifest to `/tokens/prepare_manifest.json` so re-runs only process new files.

Dry-run first (default — no deletions):

```bash
modal volume create nano-tokens   # if you haven't yet
modal run --detach diskrot/modal_prepare.py
```

Like the other steps, this spawns the orchestrator and returns immediately — the stats report (what would be deleted, plus the kept corpus's duration / codec / bitrate distribution) streams to the orchestrator's logs, not your terminal. The launch line prints the `modal app logs ... -f` command to watch it. Apply when you're happy with the report:

```bash
modal run --detach diskrot/modal_prepare.py --apply
```

Runs across ~20 CPU containers in parallel; throughput is roughly ~3,000 songs/min once the fan-out is saturated. Resumable — kill it and re-run anytime, validated files are remembered via the manifest.

## 3. Tokenize

Encodes every MP3 into DAC tokens on L4 GPUs, fanned out across up to 50 containers ([modal_tokenize.py:62-68](diskrot/modal_tokenize.py#L62-L68)). Clips shorter than 20 seconds are skipped. The `.pt` files are heavily compressed int16 token tensors (~140 KB per song on average).

```bash
modal volume create nano-tokens
modal run --detach diskrot/modal_tokenize.py
```

Tokens are persisted in the `nano-tokens` volume. Throughput is roughly ~600 songs/min on L4 × 50.

> **L4 caveat — tokenize relies on prep's length cap.** Each L4 has 22 GiB of GPU memory and DAC's full-sequence encoder peaks at activation memory ~ proportional to audio length. The workarounds for L4 fit are: prep drops anything longer than 5:30, [modal_tokenize.py](diskrot/modal_tokenize.py) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the image to avoid fragmentation, and the per-container batch is forced to `batch_size=1` so 8 files don't get padded together. Even with all that, expect ~5% of files to OOM on the upper-bound lengths (single 4-5 GiB layer allocations exceeding what L4 has free). If you need 100% completion, change `gpu="L4"` → `gpu="A100"` at [modal_tokenize.py:70](diskrot/modal_tokenize.py#L70), revert the workarounds, and accept ~3× the per-GPU-hour cost.

## 4. Auto-tag (optional, needed for text conditioning)

Captions each MP3 with a natural-language description (genre, mood, instruments) using the vendored LP-MusicCaps BART captioner, fanned out across up to 20 L4 containers ([modal_auto_tag.py:68-70](diskrot/modal_auto_tag.py#L68-L70)):

```bash
modal run --detach diskrot/modal_auto_tag.py
```

Writes `tags.json` to the `nano-tokens` volume. Re-running skips files already present in `tags.json` — only new MP3s are processed. Throughput is roughly ~1,700 songs/min across 20 L4 containers.

## 4b. Transcribe lyrics (optional, needed for lyric conditioning)

Isolates vocals with Demucs (`htdemucs`), then transcribes them with faster-whisper (`large-v3`, word-level timestamps), fanned out across up to 50 L4 containers ([modal_transcribe.py](diskrot/modal_transcribe.py)):

```bash
modal run --detach diskrot/modal_transcribe.py
```

Writes a sharded `lyrics/` dir (`lyrics_NNN.json`, 256 shards keyed by a stable hash of the stem) to the `nano-tokens` volume; each entry is `{stem: {"text": ..., "words": [{word, start, end}, ...]}}` and instrumental tracks map to `null`. Sharding keeps each flush O(batch) instead of rewriting one giant JSON, and writes are atomic (temp+rename) so a kill can corrupt at most one shard. The `.map()` collect/flush loop runs in a spawned remote function, so `--detach` survives terminal close. Re-running skips files already present. **This is the single most expensive step** (~$5–7 per 1,000 songs) — skip it unless you actually plan to use lyric conditioning at inference.

## 4c. Filter hallucinated lyrics (recommended after transcribe)

Whisper invents captions over instrumental audio — "Thank you.", "Thanks for watching!", "We'll be right back." — and any entry with usable words trains as `<vocals>` with those words attended, so each one mislabels an instrumental song *and* feeds it garbage lyrics (~24% of with-words entries in the 2026-06 full-corpus sweep). This pass nulls them back to the transcribed-but-wordless convention (trains as `<instrumental>`) ([modal_filter_lyrics.py](diskrot/modal_filter_lyrics.py)):

```bash
modal run --detach diskrot/modal_filter_lyrics.py            # dry-run report
modal run --detach diskrot/modal_filter_lyrics.py --apply    # rewrite shards
```

Flags entries with fewer than 6 valid words, or a known caption-artifact phrase ("thank you for watching", "subscribe", …) in a transcript under 30 words — long real lyrics that merely mention such a phrase survive. Rewrites are atomic per shard and the pass is idempotent. **Run it only after the transcribe fleet has fully finished** (the transcribe orchestrator holds shard contents in memory and its next flush would clobber concurrent edits), and before phonemize so junk never enters the phoneme store.

To see the overall state of the lyric data at any point — per-stream coverage of the packed corpus, instrumental/hallucinated/vocal-ready breakdown, gender and word-count distributions — run the read-only audit ([scripts/lyrics_audit.py](scripts/lyrics_audit.py), safe even mid-transcribe): `modal run scripts/lyrics_audit.py`.

## 4d. Phonemize (recommended after transcribe + filter)

Pre-runs g2p over every transcribed song and writes the per-word phoneme-id groups to a sharded `phonemes/` dir ([modal_phonemize.py](diskrot/modal_phonemize.py)):

```bash
modal run --detach diskrot/modal_phonemize.py
```

Why: the dataset otherwise phonemizes lazily inside the DataLoader workers, and OOV-heavy Whisper transcripts cost ~20–200 ms/song there — at corpus scale that can starve the 8×H100 training step. This pass is a ~$1 CPU one-shot that turns it into a pure lookup. Resumable (per-song skip, atomic shard rewrites); training falls back to live g2p for any song not covered, so partial is safe. Re-run it after any re-transcribe (stale entries are detected by word count and redone).

## 5. Pack (sharded mmap layout)

At scale the token corpus no longer fits in RAM as one shared-memory tensor, so we write a sharded, mmap-friendly layout and let the kernel page tokens in on demand. See [diskrot/pack_cache.py](diskrot/pack_cache.py) for the format (`packed/packed_NNN.bin` + per-shard JSON + a global `packed_index.json`).

Run it on Modal (it mounts the `nano-tokens` volume and packs in place):

```bash
modal run --detach diskrot/modal_pack_cache.py
```

Or run the packer directly in any environment that has the `nano-tokens` volume mounted (locally after `modal volume get`, or inside an interactive Modal shell):

```bash
python -m diskrot.pack_cache --cache-dir /tokens
```

The packer is **deterministic** (same `.pt` set → byte-identical shards) and **resumable**: each shard is written atomically (temp file → `os.replace`, with the per-shard `.json` written last as a commit marker) and committed to the volume as it lands, and a re-run **skips shards already complete-and-valid on disk** rather than rebuilding them. So a kill or Modal worker preemption mid-pack costs only the in-flight shard — just re-run (or let the `retries` on `pack_remote` restart it) and it continues from the last committed shard. A truncated or half-written shard is detected (size/membership check) and rebuilt; adding files shifts shards in the alphabetical tail, and only the shifted shards are rebuilt. It is still **not an in-place update** — re-running validates every shard — but it is no longer "rebuild from scratch", and an interrupted run never has to start over. The trainer's parent process auto-detects the `packed/` dir and switches to `load_mmap_bundle`; a cache layout without `packed/` falls back to the legacy in-RAM path (only viable on a tiny smoke-test corpus).

## 5b. Key detect (optional, after pack)

Estimates each song's musical key from the packed chroma sidecar (mean chroma → Krumhansl-Schmuckler) and writes `keys.json` — the source of the `<key_*>` header marker that enables "generate in A minor" prompts ([modal_key_detect.py](diskrot/modal_key_detect.py)):

```bash
modal run --detach diskrot/modal_key_detect.py
```

CPU-only, one container, ~$1; needs the pack to carry the melody sidecar (`packed_NNN.mel.bin`). Resumable; songs without an estimate just get `<unknown_key>` at train time.

## 6. Train

Trains the ~1.5B parameter transformer (d_model=2048, n_layers=22, n_heads=16, d_ff=8192) over 30-second segments with RoPE (`max_seq_len=8192`) so inference can extrapolate to ~95s single-shot generation. Defaults to 400K steps with early stopping at `patience=20`, gradient checkpointing on, and checkpoints under `/ckpts/v8_sing/` — the `DEFAULTS` dict in [diskrot/modal_train.py](diskrot/modal_train.py) is the source of truth. Text conditioning is enabled by default (requires step 4 auto-tagging).

This is an 8×H100 DDP job. Single-GPU is technically possible but not recommended — the per-GPU memory footprint of the 1.5B model at 30s segments + bf16 + grad-checkpointing is tight on 80 GB H100s even at the per-rank batch of 8:

```bash
modal volume create nano-ckpts
modal run --detach diskrot/modal_train.py --n-gpus 8
```

The DDP path auto-picks `batch_size = DDP_PER_RANK_BATCH = 8` per rank → global = 64, matching the tuned LR (lr=3.0e-4, warmup=5000). Single-GPU runs use the full `batch_size=64`. Pass `--batch-size N` to override (per-rank in DDP mode); if you do, sqrt-scale the LR proportionally. Any field can be overridden on the CLI (`--d-model`, `--steps`, `--ckpt-subdir`, …).

Expect considerably more wall-clock and cost than the old 287M model — the 1.5B is ~5× the per-step compute. Rough order: a few thousand USD on 8×H100 for the full 400k steps (treat as an estimate, not a quote); early stopping (`patience=20`) commonly cuts this once the val loss plateaus. Training cost is independent of corpus size — only `steps` and model size drive it. Checkpoints land in `/ckpts/v7_1500m/` every ~5000 steps.

The trainer auto-switches to the sharded mmap dataset whenever `cache_dir/packed/packed_index.json` exists, so the launch line above just works after step 5 (pack) completes.

### DDP path notes

The DDP path went through several rounds of bug-fixing during its first real end-to-end run. The fixes are now in code; this section documents what was wrong, what it now does, and where to look if you make changes.

1. **CFG dropout & `find_unused_parameters=True`.** With `cfg_dropout=0.1`, 10% of training steps set `text_emb=None` and skip cross-attention entirely — those params get no gradient that step. DDP would crash with "Expected to have finished reduction in the prior iteration..." without `find_unused_parameters=True`. Now set at [train.py](diskrot/train.py) where DDP is constructed. Cost: ~5% per-step overhead from the extra autograd-graph traversal. Alternative if you ever need to disable the flag: feed a learned-zero embedding during CFG dropout to keep the param graph dense (also requires matching change in inference's unconditional CFG branch).

2. **`text_encoder.proj` lives outside the DDP wrapper.** The CLAP→d_model projection ([train.py](diskrot/train.py) — `CLAPTextEncoder.proj`) is a separate `nn.Linear(1024, d_model)` (1024→1024, ~1M params, no bias) that isn't part of `NanoAudioGPT`, so DDP doesn't sync it. The code now explicitly handles both halves: `_broadcast_module_params(text_encoder.proj, src=0)` after construction makes initial weights identical across ranks; `_all_reduce_module_grads(text_encoder.proj)` between `scaler.unscale_(optim)` and `optim.step()` averages per-rank grads each step. See helpers in [diskrot/train.py](diskrot/train.py).

3. **Batch-size auto-default.** Without `--batch-size`, `main()` resolves the right per-rank value: `DDP_PER_RANK_BATCH` (8) when `--n-gpus > 1`, the full `batch_size` (64) otherwise. A startup line prints which value was picked so it's not silent. Override with any positive `--batch-size N`; sentinel `0` (the default) means auto.

4. **`model.cfg` doesn't passthrough DDP/compile wrappers.** `_evaluate()` previously read `model.cfg.pad_id`; on DDP runs that raised `AttributeError: 'DistributedDataParallel' object has no attribute 'cfg'` at the first checkup (step 1000). Now reads `cfg.model.pad_id` from the passed `TrainConfig` instead. If you add new DDP-mode code paths, prefer `cfg.model.<x>` over `model.<x>` for any `<x>` that isn't an `nn.Module` method.

5. **Setup phase is silent for a long time at the full corpus.** Before `mp.spawn` fires the workers, the parent process opens the mmap shards and encodes CLAP for every unique tag ([diskrot/dataset.py](diskrot/dataset.py), [diskrot/modal_train.py:_precompute_clap_cache](diskrot/modal_train.py)). At large corpus sizes the CLAP precompute dominates and the first `step` line can be over an hour out. It prints progress every 1000 files / 500 tags. If you see no `[train]`, `[bundle]`, or `[clap-parent]` lines for several minutes after launch, see [README.logs.md](README.logs.md) for diagnostics — don't kill the run just because it hasn't started training yet.

## 7. Pull the checkpoint

```bash
mkdir -p checkpoints
modal volume get nano-ckpts /v7_1500m/best.pt ./checkpoints/latest.pt --force
```

Then follow the [inference instructions](README.md#inference) in the main README. The server reads `GPTConfig` from the checkpoint's `cfg` dict, so any checkpoint works without client-side flag changes.

Or skip the download and serve straight from the volume: `modal serve diskrot/modal_serve.py` (dev) / `modal deploy diskrot/modal_serve.py` (persistent) brings up **two** endpoints — the inference API (`...-serve[-dev].modal.run`, L4) and the Flutter web UI (`...-ui[-dev].modal.run`, CPU static files; run `cd webapp && flutter build web` first). The UI pre-fills its server-url field with the sibling API URL.

## Resetting (start fresh)

Delete tags, lyrics, packed shards, and checkpoints to retrain from scratch:

```bash
modal volume delete nano-corpus
modal volume create nano-corpus
modal volume rm nano-tokens tags.json
modal volume rm -r nano-tokens lyrics
modal volume rm -r nano-tokens packed
modal volume delete nano-ckpts && modal volume create nano-ckpts
```
