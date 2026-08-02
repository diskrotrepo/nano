# Training on Modal

End-to-end pipeline using Modal's cloud GPUs. Best path if you don't have local hardware or want to scale tagging/transcription across many containers.

This is a bespoke model: it trains on one kind of data at scale. You supply your own MP3s — more of the same data helps, but variety does not, so the corpus is not curated for genre/style diversity.

**Recommended corpus size:** at least **~50,000 songs** for coherent musical output. Below **~10,000 songs** the model mostly produces noise — useful only for validating the pipeline end-to-end, not for a real model. More of the same kind of data keeps helping, so larger is better; the raw audio lives in the R2 `nano-audio` bucket (object storage — no inode cap, so no hard file ceiling). A tiny "smoke" corpus is still handy for exercising the pipeline, but it will not produce musical output.

> **⚠️ #1 footgun — the codec is selected by an env var that Modal does NOT forward.** v9 tokenizes the corpus with **SpectroStream** (Magenta RealTime's codec: 48 kHz, joint stereo, **24 codebooks** predicted, **25 Hz** frame rate), selected with `export NANO_CODEC=spectrostream`. Modal doesn't forward your local shell env to remote containers, so you MUST set it at **both `modal deploy` and `modal run` time** — every stage that tokenizes or reads tokens (tokenize, stems, train, serve) has to agree on the codec, and a mismatch silently corrupts the pipeline. The code default is still **DAC** (44.1 kHz, **mono**, 9 codebooks, 86 Hz) — that's the fallback, not the v9 path. (SpectroStream's 25 Hz frame rate is also why single-shot generation reaches ~5.4 min instead of DAC's ~95 s.)

## Cost summary

Cost for the data-prep steps scales roughly linearly with corpus size, so the table below quotes them **per 1,000 songs** — multiply by however many thousands of songs you bring. Training is the exception: its cost is driven by `steps` and model size, **not** corpus size, so it's a flat range. All cost estimates are against Modal's published GPU prices ([modal.com/pricing](https://modal.com/pricing)).

| Step | GPU | Per 1,000 songs |
|---|---|---|
| 1. Upload | — | uplink-bound (~0.7 GB/min at 100 Mbps) |
| 2. Prepare (+ optional `--quality-gate`) | CPU × 20 | ~20 s, <$0.01 |
| 2b. Audio-dedup (optional, after prepare) | CPU × 50 | negligible |
| 3. Tokenize | L4 × 50 | ~1.5 min, ~$0.07–0.10 |
| 3b. Melody (optional, after tokenize — for `/cover`) | CPU × 50 | negligible |
| 3c. Stems (optional, after tokenize — for `/addstem`) | GPU × 50 | most expensive optional stage (Demucs; `--sample-pct 50` halves it) |
| 4. Auto-tag (optional) | A100 × 50 | audio-LLM captioner (Qwen2-Audio-7B) |
| 4b. Transcribe lyrics (optional) | L4 × 50 | ~$5–7 (Demucs-free now, much cheaper than before) |
| 4c. Filter lyrics (recommended, after 4b) | CPU | negligible (seconds, one container) |
| 4d. Align lyrics (optional, after filter) | L4 × 50 | forced alignment, refinement-only |
| 4e. Phonemize (recommended, after 4c) | CPU × 16 | negligible (~$1 full corpus) |
| 5. Pack (sharded mmap) | CPU | negligible |
| 5b. Key detect (optional, after 5) | CPU × 4 | negligible (~$1 full corpus) |
| 5c. Tempo (optional, dense tempo markers) | CPU | negligible |
| 6. Train (flat — corpus-independent) | B200 × 4 DDP | a few thousand USD for the full 400k-step run |

GPU rates used above (as of 2026-05): B200 ≈ $6.25/hr, H100 ≈ $5.92/hr, A100-40 ≈ $3.10/hr, L4 ≈ $0.30/hr, debian_slim CPU ≈ $0.10/hr.

Modal bills per second of actual compute, and the tokenize/tag/transcribe/train steps are all detached, so wall-clock time doesn't tie up your terminal. The expensive steps are auto-tag (the audio-LLM captioner runs on A100), stems (the most expensive *optional* stage — GPU Demucs; sampled to 50% by default), transcribe (skip it unless you actually plan to use lyric conditioning at inference — but it's much cheaper now that Demucs is gone), and train. Training has early stopping with `patience=20`, so a typical run finishes 30–50% sooner than the full-step worst case.

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

## 1. Get your corpus into R2

Raw audio lives in a Cloudflare R2 bucket (`nano-audio`), not a Modal Volume —
object storage has no inode cap, so the corpus can grow without a hard ceiling.
One-time R2 + Modal-secret setup (the `r2-creds` secret, the `NANO_AUDIO_*` env)
is in [README.waves.md](README.waves.md#one-time-setup). Then upload your MP3s
under a wave prefix (S3-compatible — rclone / `aws s3 cp` / the Cloudflare UI):

```bash
rclone copy /path/to/mp3s/ r2:nano-audio/waves/wave_0/
```

Upload runs at your connection speed (~0.7 GB/min at 100 Mbps). For the
unlimited-scale, wave-by-wave ingestion model, see [README.waves.md](README.waves.md).


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

Encodes every MP3 into codec tokens on L4 GPUs, fanned out across up to 50 containers ([modal_tokenize.py:62-68](diskrot/modal_tokenize.py#L62-L68)). Clips shorter than 20 seconds are skipped. The `.pt` files are heavily compressed int16 token tensors (~140 KB per song on average). For v9 the codec is **SpectroStream** — remember `export NANO_CODEC=spectrostream` at both deploy and run time (see the footgun note above); the default is DAC.

```bash
modal volume create nano-tokens
export NANO_CODEC=spectrostream    # v9 codec — required at deploy AND run time
modal run --detach diskrot/modal_tokenize.py
```

Tokens are persisted in the `nano-tokens` volume. Throughput is roughly ~600 songs/min on L4 × 50.

> **L4 caveat — tokenize relies on prep's length cap.** Each L4 has 22 GiB of GPU memory and the codec's full-sequence encoder peaks at activation memory ~ proportional to audio length. The workarounds for L4 fit are: prep drops anything longer than 5:30, [modal_tokenize.py](diskrot/modal_tokenize.py) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the image to avoid fragmentation, and the per-container batch is forced to `batch_size=1` so 8 files don't get padded together. Even with all that, expect ~5% of files to OOM on the upper-bound lengths (single 4-5 GiB layer allocations exceeding what L4 has free). If you need 100% completion, change `gpu="L4"` → `gpu="A100"` at [modal_tokenize.py:70](diskrot/modal_tokenize.py#L70), revert the workarounds, and accept ~3× the per-GPU-hour cost.

## 3b. Melody (optional, needed for `/cover`)

Extracts a time-aligned 12-bin chromagram per song and writes it as `<name>.mel.npy` to the dedicated **nano-melody** volume ([modal_melody.py](diskrot/modal_melody.py)). This is the octave-invariant melodic contour the `/cover` path conditions on (a hummed/uploaded melody re-rendered in the prompt's timbre). CPU-only and embarrassingly parallel across up to 50 containers:

```bash
modal run --detach diskrot/modal_melody.py
```

Runs **after** tokenize (it forces the chroma to the song's token frame count) and **before** pack (the packer folds `<name>.mel.npy` into a parallel `packed_NNN.mel.bin` sidecar via `--mel-cache-dir`). The chroma lives on its own volume so the ~1 file/song it adds doesn't push nano-tokens over its ~500k-inode cap. A song without a `.mel.npy` is zero-filled at pack time (membership stays identical), so a partial pass is safe — but `/cover` only works if the corpus was largely melody-covered.

## 3c. Stems (optional, needed for `/addstem`)

Demucs-separates each song into 4 stems (drums/bass/vocals/other), codec-tokenizes each, and writes `<name>.stems.npy` (`[4, depth, T]` int16) to the dedicated **nano-stems** volume ([modal_stems.py](diskrot/modal_stems.py)). This is what the generative `/addstem` path conditions on. It's a **GPU** stage (Demucs + the codec) and the **most expensive optional stage**, so `--sample-pct` (the ingest default is **50**) runs it on a deterministic fraction — non-sampled songs are flagged absent by the packer's present mask and just skipped as stem-add targets:

```bash
export NANO_CODEC=spectrostream    # the stems are codec-tokenized — must match the corpus
modal run --detach diskrot/modal_stems.py --wave-id <id> --sample-pct 50
```

Runs **after** tokenize (needs the token frame count to align) and **before** pack (folded into `packed_NNN.stem.bin` via `--stem-cache-dir`). Calibrate cost with `--limit` on one wave first. Stem-add is off by default in the v9 checkpoint (`use_stem_conditioning=False`), so this stage is only worth running if you plan to train stem conditioning on.

## 4. Auto-tag (optional, needed for text conditioning)

Captions each MP3 with a rich, multi-facet natural-language description (genre/mood, drums, bass, instruments, vocals, production, arc) using the **audio-LLM captioner (Qwen2-Audio-7B-Instruct)** over the whole song, fanned out across up to 50 **A100** containers ([modal_auto_tag.py](diskrot/modal_auto_tag.py)). Each caption also carries a vocal-gender tag and per-stem lines — the gender marker source (so transcribe no longer needs Demucs) and the `/addstem` per-stem caption source. Set `NANO_CAPTIONER=bart` (+ the matching image) for the legacy single-window LP-MusicCaps BART captioner on L4.

```bash
modal run --detach diskrot/modal_auto_tag.py             # caption missing songs
modal run --detach diskrot/modal_auto_tag.py --redo      # upgrade legacy short captions (resumable)
modal run --detach diskrot/modal_auto_tag.py --limit 50  # calibrate image/cost first
```

Writes `tags.json` to the `nano-tokens` volume. Each entry is stamped with a captioner marker (`CAPTIONER_MARKER = "audio_llm_v5"`), so a bare run captions only missing songs and `--redo` re-captions only legacy/non-current entries (skips already-upgraded ones → resumable). Calibrate cost with a small `--limit` run before the full corpus — the audio-LLM is heavy (~7B, A100).

## 4b. Transcribe lyrics (optional, needed for lyric conditioning)

Transcribes each song with faster-whisper (`large-v3-turbo` + Silero VAD, word-level timestamps) directly on the **raw mono mix**, fanned out across up to 50 L4 containers ([modal_transcribe.py](diskrot/modal_transcribe.py)). The stage is **Demucs-free**: vocal isolation was a no-op-to-worse ASR input (arXiv:2506.15514) and was the dominant former cost, so it's gone — vocal gender now comes from the audio-LLM captioner (step 4), not an F0/Demucs pass.

```bash
modal run --detach diskrot/modal_transcribe.py
```

Writes a sharded `lyrics/` dir (`lyrics_NNN.json`, 256 shards keyed by a stable hash of the stem) to the `nano-tokens` volume; each entry is `{stem: {"text": ..., "words": [{word, start, end}, ...]}}` and instrumental tracks map to `null`. Sharding keeps each flush O(batch) instead of rewriting one giant JSON, and writes are atomic (temp+rename) so a kill can corrupt at most one shard. The `.map()` collect/flush loop runs in a spawned remote function, so `--detach` survives terminal close. Re-running skips files already present. This is one of the more expensive optional steps (~$5–7 per 1,000 songs, down substantially now that Demucs is gone) — skip it unless you actually plan to use lyric conditioning at inference.

## 4c. Filter hallucinated lyrics (recommended after transcribe)

Whisper invents captions over instrumental audio — "Thank you.", "Thanks for watching!", "We'll be right back." — and any entry with usable words trains as `<vocals>` with those words attended, so each one mislabels an instrumental song *and* feeds it garbage lyrics (~24% of with-words entries in the 2026-06 full-corpus sweep). This pass nulls them back to the transcribed-but-wordless convention (trains as `<instrumental>`) ([modal_filter_lyrics.py](diskrot/modal_filter_lyrics.py)):

```bash
modal run --detach diskrot/modal_filter_lyrics.py            # dry-run report
modal run --detach diskrot/modal_filter_lyrics.py --apply    # rewrite shards
```

Flags entries with fewer than 6 valid words, or a known caption-artifact phrase ("thank you for watching", "subscribe", …) in a transcript under 30 words — long real lyrics that merely mention such a phrase survive. Rewrites are atomic per shard and the pass is idempotent. **Run it only after the transcribe fleet has fully finished** (the transcribe orchestrator holds shard contents in memory and its next flush would clobber concurrent edits), and before phonemize so junk never enters the phoneme store.

To see the overall state of the lyric data at any point — per-stream coverage of the packed corpus, instrumental/hallucinated/vocal-ready breakdown, gender and word-count distributions — run the read-only audit ([scripts/lyrics_audit.py](scripts/lyrics_audit.py), safe even mid-transcribe): `modal run scripts/lyrics_audit.py`.

## 4d. Align lyrics (optional, after filter)

Sharpens Whisper's loose word timestamps with a CTC forced aligner (torchaudio MMS_FA, optionally on Demucs-isolated vocals) and rewrites the refined onsets into the sharded `lyrics/` store in place ([modal_align_lyrics.py](diskrot/modal_align_lyrics.py)). It's idempotent (an `aligned` stamp) and refinement-only — any failure keeps the original timestamp. Dry-run reports eligibility cheaply (no GPU); `--apply` runs the L4 fan-out. Tighter onsets help the sung-alignment learning and the dataset's vocal-crop biasing.

```bash
modal run --detach diskrot/modal_align_lyrics.py --wave-id N              # dry-run report
modal run --detach diskrot/modal_align_lyrics.py --wave-id N --apply      # rewrite shards
```

Runs **after** transcribe + filter, before (or around) phonemize.

## 4e. Phonemize (recommended after transcribe + filter)

Pre-runs g2p over every transcribed song and writes the per-word phoneme-id groups to a sharded `phonemes/` dir ([modal_phonemize.py](diskrot/modal_phonemize.py)):

```bash
modal run --detach diskrot/modal_phonemize.py
```

Why: the dataset otherwise phonemizes lazily inside the DataLoader workers, and OOV-heavy Whisper transcripts cost ~20–200 ms/song there — at corpus scale that can starve the 4×B200 training step. This pass is a ~$1 CPU one-shot that turns it into a pure lookup. Resumable (per-song skip, atomic shard rewrites); training falls back to live g2p for any song not covered, so partial is safe. Re-run it after any re-transcribe (stale entries are detected by word count and redone).

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

## 5c. Tempo (optional, dense tempo markers)

Estimates each song's tempo and writes dense `<tempo_*>` markers to `tempo.json` — the source of the tempo header/inline markers the lyric stream carries (the dataset prefers this over the allin1 per-song bpm). CPU-only, negligible cost. Songs without an estimate just get `<unknown_tempo>` at train time.

## 6. Train

Trains the ~2.08B parameter transformer (d_model=2048, n_layers=22, n_heads=16, d_ff=8192) over 180-second (full-song) segments with RoPE (`max_seq_len=8192`) so inference reaches ~5.4 min single-shot generation (at SpectroStream's 25 Hz frame rate — a full song fits in one shot). Defaults to 400K steps with early stopping at `patience=20`, gradient checkpointing on, EMA on (`use_ema=True`, so `best.pt` is selected on the EMA's smoother val loss), and checkpoints under `/ckpts/v9_stereo/` — the `DEFAULTS` dict in [diskrot/modal_train.py](diskrot/modal_train.py) is the source of truth. Text conditioning is enabled by default (requires step 4 auto-tagging). Don't forget `export NANO_CODEC=spectrostream` at deploy and run time.

This is a 4×B200 DDP job (the only multi-GPU function is hardwired `gpu="B200:4"`; `--n-gpus 8` would trip its `torch.cuda.device_count()` assert). Single-GPU is technically possible but not recommended — each rank pays the CLAP precompute and memory is tighter:

```bash
modal volume create nano-ckpts
export NANO_CODEC=spectrostream
modal run --detach diskrot/modal_train.py --n-gpus 4
```

The DDP run reports `DDP active: world_size=4, per-rank batch=8, global batch=32`, matching the tuned LR (lr=1.5e-4, warmup=10000). Single-GPU runs use the full `batch_size=32` (`DEFAULTS["batch_size"]`). Pass `--batch-size N` to override (per-rank in DDP mode); if you do, sqrt-scale the LR proportionally. Any field can be overridden on the CLI (`--d-model`, `--steps`, `--ckpt-subdir`, …). Note that infill (`use_fim`) and stem-add (`use_stem_conditioning`) are both deferred (off) in the v9 defaults.

Expect considerably more wall-clock and cost than the old 287M model — the ~2B is roughly an order of magnitude more per-step compute. Rough order: a few thousand USD on 4×B200 for the full 400k steps (treat as an estimate, not a quote); early stopping (`patience=20`) commonly cuts this once the val loss plateaus. Training cost is independent of corpus size — only `steps` and model size drive it. Checkpoints land in `/ckpts/v9_stereo/` every ~5000 steps.

The trainer auto-switches to the sharded mmap dataset whenever `cache_dir/packed/packed_index.json` exists, so the launch line above just works after step 5 (pack) completes.

### DDP path notes

The DDP path went through several rounds of bug-fixing during its first real end-to-end run. The fixes are now in code; this section documents what was wrong, what it now does, and where to look if you make changes.

1. **CFG dropout & `find_unused_parameters=True`.** With `cfg_dropout=0.1`, 10% of training steps set `text_emb=None` and skip cross-attention entirely — those params get no gradient that step. DDP would crash with "Expected to have finished reduction in the prior iteration..." without `find_unused_parameters=True`. Now set at [train.py](diskrot/train.py) where DDP is constructed. Cost: ~5% per-step overhead from the extra autograd-graph traversal. Alternative if you ever need to disable the flag: feed a learned-zero embedding during CFG dropout to keep the param graph dense (also requires matching change in inference's unconditional CFG branch).

2. **`text_encoder.proj` lives outside the DDP wrapper.** The CLAP→d_model projection ([train.py](diskrot/train.py) — `CLAPTextEncoder.proj`) is a separate `nn.Linear(1024, d_model)` (1024→1024, ~1M params, no bias) that isn't part of `NanoAudioGPT`, so DDP doesn't sync it. The code now explicitly handles both halves: `_broadcast_module_params(text_encoder.proj, src=0)` after construction makes initial weights identical across ranks; `_all_reduce_module_grads(text_encoder.proj)` between `scaler.unscale_(optim)` and `optim.step()` averages per-rank grads each step. See helpers in [diskrot/train.py](diskrot/train.py).

3. **Batch-size auto-default.** Without `--batch-size`, `main()` resolves the right per-rank value: `DDP_PER_RANK_BATCH` (8 per rank → global 32 on 4 ranks) when `--n-gpus > 1`, the full `batch_size` (32, `DEFAULTS["batch_size"]`) otherwise. A startup line prints which value was picked so it's not silent. Override with any positive `--batch-size N`; sentinel `0` (the default) means auto.

4. **`model.cfg` doesn't passthrough DDP/compile wrappers.** `_evaluate()` previously read `model.cfg.pad_id`; on DDP runs that raised `AttributeError: 'DistributedDataParallel' object has no attribute 'cfg'` at the first checkup (step 1000). Now reads `cfg.model.pad_id` from the passed `TrainConfig` instead. If you add new DDP-mode code paths, prefer `cfg.model.<x>` over `model.<x>` for any `<x>` that isn't an `nn.Module` method.

5. **Setup phase is silent for a long time at the full corpus.** Before `mp.spawn` fires the workers, the parent process opens the mmap shards and encodes CLAP for every unique tag ([diskrot/dataset.py](diskrot/dataset.py), [diskrot/modal_train.py:_precompute_clap_cache](diskrot/modal_train.py)). At large corpus sizes the CLAP precompute dominates and the first `step` line can be over an hour out. It prints progress every 1000 files / 500 tags. If you see no `[train]`, `[bundle]`, or `[clap-parent]` lines for several minutes after launch, see [README.logs.md](README.logs.md) for diagnostics — don't kill the run just because it hasn't started training yet.

## 7. Pull the checkpoint

```bash
mkdir -p checkpoints
modal volume get nano-ckpts /v9_stereo/best.pt ./checkpoints/latest.pt --force
```

Then follow the [inference instructions](README.md#inference) in the main README. The server reads `GPTConfig` from the checkpoint's `cfg` dict, so any checkpoint works without client-side flag changes.

Or skip the download and serve straight from the volume: `modal serve diskrot/modal_serve.py` (dev) / `modal deploy diskrot/modal_serve.py` (persistent) brings up **two** endpoints — the inference API (`...-serve[-dev].modal.run`, L4) and the Flutter web UI (`...-ui[-dev].modal.run`, CPU static files; run `cd webapp && flutter build web` first). The UI pre-fills its server-url field with the sibling API URL.

## Resetting (start fresh)

Delete the derived artifacts (tags, lyrics, packed shards, checkpoints) to retrain
from scratch. The raw audio in R2 is the source — leave it in place (re-derive from
it); only clear the R2 bucket via `rclone`/`aws s3` if you truly want to discard the
corpus.

```bash
modal volume rm nano-tokens tags.json
modal volume rm -r nano-tokens lyrics
modal volume rm -r nano-tokens packed
modal volume delete nano-ckpts && modal volume create nano-ckpts
```
