# nano — Project Guide

Audio generation model: a decoder-only transformer trained on DAC-tokenized audio with optional CLAP text / phoneme-lyric / melody conditioning. ~2.0B params (d_model=2048, n_layers=22, n_heads=16, d_ff=8192; ~1.14B with text conditioning off).

## Project principles

- **One bespoke model, not a family of tiers.** `DEFAULTS` in [diskrot/modal_train.py](diskrot/modal_train.py) is the single source of truth for its shape. (v4/v5/v6 framing is retired; `ckpt_subdir="v7_1500m"` is just a directory name.)
- **Scale of training data matters; variety does not.** ~50k songs minimum (below ~10k is noise, pipeline-validation only); no hard ceiling (corpus lives in R2, no inode cap). More of the same kind of data helps — do **not** curate for genre/style diversity.

## Architecture

```
MP3 corpus → DAC tokenizer → [9 codebooks, 1024 vocab, 86 Hz] → Transformer → audio
                                                                      ↑
                                              tags (pooled CLAP) + lyrics (phonemes)
                                              + melody (chromagram)
```

- **Model**: decoder-only transformer, 9 codebook heads. Trains on 60s segments (Modal; 30s local). Max single-shot generation ~95s (max_seq_len=8192, RoPE).
- **Delay pattern**: MusicGen-style — codebook k shifted right by k positions. All 9 codebook embeddings summed as input; one fused output Linear (`model.head`, d_model→9×vocab) predicts the next token per codebook, so later codebooks condition on earlier ones at the same frame.

**Conditioning axes** — each travels a *separate path*; each drops independently 10% of the time for classifier-free guidance (`cfg_scale` jointly, or per-axis `lyric_cfg_scale`/`melody_cfg_scale`/`stem_cfg_scale`). Encoders are submodules of `NanoAudioGPT`, saved in the `model` state_dict.

- **Tags** (genre/timbre/vibe): a natural-language caption per song (`auto_tag` → `tags.json`), split into ≤77-token chunks, each pooled by frozen CLAP → sequence of vectors the decoder cross-attends to (`encode_chunked`, breaks CLAP's 77-token wall).
- **Lyrics** (words to sing): a **phoneme-ID sequence** (g2p_en ARPABET), NOT CLAP. A trainable `LyricEncoder` (bidirectional transformer) feeds a per-block lyric cross-attention; the model learns its own near-monotonic alignment. The phoneme stream also carries **marker tokens** (`PHONEME_VOCAB`, size 129): structure (`<verse>`/`<chorus>`…), gender, tempo, key, vocal-presence (`<vocals>`/`<instrumental>`). Every stream opens with a dense header `BOS <gender> <tempo> <key> <vocals> <section>`. Train injector (`TokenDataset._get_segment_lyric_ids`) and inference parser (`text_with_markers_to_phoneme_ids`, `[female] [120bpm] [a minor] [instrumental] [chorus]` brackets) must build byte-identical streams — `tests/test_structure_markers.py` guards it. Any marker change alters `PHONEME_VOCAB_SIZE` → checkpoint-incompatible.
- **Melody** (`/cover`): time-aligned 12-bin chromagram (`chroma_cqt` at 86 Hz; `diskrot/melody.py:extract_chroma` is the train==inference contract). Dense + frame-aligned, so conditioned **additively** at the cb0 anchor via `MelodyEncoder`, not cross-attention. Packed into a `packed_NNN.mel.bin` sidecar co-cropped with tokens.
- **Infill** (`/infill`, `use_fim`): FIM frame-domain reorder into `prefix <SUF> suffix <MID> middle` before the delay pattern (causal/delay logic untouched). Drops lyrics, keeps tags + co-reordered melody. **Deferred (v8+v9): `use_fim=False`** — no checkpoint trains the control ids, so `/infill` won't work yet.
- **Stem** (`/addstem`, `use_stem_conditioning`): generate a new isolated stem that fits a song. `StemEncoder` embeds the other stems' full codec tokens + a target-stem-type id, added at the cb0 anchor. **Deferred (v9): `use_stem_conditioning=False`** — code complete, needs the stems prep stage.

Marker/vocab changes and encoder additions are checkpoint-incompatible.

## Key Files

### Model (`model/`)
- `nano_audio_gpt.py` — GPTConfig, NanoAudioGPT, KV cache, self/cross attention. The core model.
- `codec.py` — codec facade (`get_codec()`, env `NANO_CODEC`): `DACodec` (44.1kHz mono, 9×1024 @86Hz — **current v10 codec**) and `SpectroStreamCodec` (48kHz stereo, ≤64×1024 @25Hz — v9, abandoned). SS falls back to `spectrostream_mlx.py` on Apple Silicon.
- `delay_pattern.py` — apply_delay / revert_delay / build_train_inputs (MusicGen delay logic).
- `text_encoder.py` — CLAPTextEncoder (frozen CLAP + learned projection). `encode_chunked()` is the production tag path; `encode_audio()` for style refs.
- `lyric_encoder.py` — `PHONEME_VOCAB` + `text_to_phoneme_ids` (train==inference id map) + marker mappers + `text_with_markers_to_phoneme_ids` (bracket parser) + `LyricEncoder`.
- `melody_encoder.py` — `MelodyEncoder` (12→d_model, additive at cb0 anchor).
- `stem_encoder.py` — `StemEncoder` + `STEM_TYPES` (drums/bass/vocals/other id order). `tests/test_stem.py` guards it.
- `fim.py` — `fim_reorder_batch` (train) + `build_fim_prompt` (inference); `tests/test_fim.py` guards equivalence.
- `lora.py` — hand-rolled LoRA (`LoRALinear` subclasses `nn.Linear`; adapters purely additive). `inject_lora` targets decoder-block Linears; defaults r=16/alpha=32 → ~21.6M trainable.
- `audio_llm_captioner.py` — **default captioner**: Qwen2-Audio whole-song → rich caption + `GENDER:` tag (+ v5 per-stem lines for `/addstem`). Heavy (~7B, A100).
- `captioner.py` — legacy LP-MusicCaps BART fallback (`NANO_CAPTIONER=bart`).

### Training (`diskrot/`)
- `train.py` — TrainConfig + train_run(). Device-agnostic loop, cosine LR + warmup, AdamW, early stopping. Fine-tune/LoRA entry: `--init-from <ckpt>` (weights+cfg from checkpoint, fresh optimizer), `--lora` (freeze base, train adapters). Precedence: `latest.pt` resume > `--init-from` > scratch.
- `modal_train.py` — Modal multi-GPU DDP entrypoint; `DEFAULTS` is the model's source of truth. `modal run --detach diskrot/modal_train.py --n-gpus 4`.
- `dataset.py` — TokenDataset (mmap-backed sharded). `__getitem__` → `(tokens, tags, lyric_ids, melody)` co-cropped; `collate_lyrics` pads/masks. Reads phonemes/, keys.json.
- `melody.py` — `extract_chroma()` (train==inference chromagram).
- `stems.py` — `extract_stem_tokens`/`extract_stem_array` (Demucs 4 stems → codec tokens; train==inference contract).
- `audio_io.py` — `decode_pcm()` shared ffmpeg PCM decoder (single decode contract for tokenize + inference).
- `tokenize.py` — MP3 → codec .pt files. `NANO_TOKENIZE_PROFILE=1` prints decode/encode/commit split.
- `auto_tag.py` — corpus captioning → `tags.json` (`--redo` re-captions).
- `transcribe_lyrics.py` — Whisper transcription on the raw mix (Modal stage is Demucs-free; gender comes from the captioner). Demucs helpers stay for `/stem`.
- `filter_lyrics.py` — nulls Whisper-hallucinated entries ("Thank you." etc.). Run after transcribe, before phonemize; `--apply` to rewrite.
- `align_lyrics.py` — CTC forced-alignment (torchaudio MMS_FA) to sharpen word timestamps; idempotent, refinement-only.
- `structure.py` — allin1 → functional sections → sharded `structure/`. Loaded at train time (not packed).
- `key_detect.py` — packed chroma → Krumhansl key → `keys.json`. After pack.
- `phonemize.py` — offline g2p → sharded `phonemes/` (avoids in-DataLoader g2p stall). After transcribe/filter.
- `pack_cache.py` — .pt (+ `.mel.npy`, + `.stems.npy`) → sharded mmap `packed/` (+ `.mel.bin`/`.stem.bin` sidecars). Resumable.
- `audio_quality.py` / `audio_dedup.py` — content-quality gate / chromaprint near-dup dedup (prepare-adjacent).
- `merge_lora.py` — fold a LoRA adapter into its base → standard slim checkpoint.
- `backup.py` — laptop-bundle helpers for the backup stage.
- **Modal wrappers** (`modal_*.py`): thin fan-out wrappers around each stage above (`modal_tokenize`, `modal_auto_tag`, `modal_transcribe`, `modal_filter_lyrics`, `modal_align_lyrics`, `modal_structure`, `modal_melody`, `modal_stems`, `modal_key_detect`, `modal_phonemize`, `modal_prepare`, `modal_audio_dedup`, `modal_pack_cache`, `modal_merge_lora`, `modal_backup`, `modal_inspect_ckpts`). Up to 50 containers each. `--limit` to calibrate cost.

### Server (`server/`)
- `main.py` — FastAPI: GET /health, POST /generate, /extend, /cover, /infill, /stem, /addstem.
  - `/generate`,`/extend` — optional text, lyrics, style_audio, style_weight (`/extend` continues from `from_seconds`).
  - `/cover` — required `melody_audio` hum, conditions on its chromagram (needs melody-trained ckpt).
  - `/infill` — `before_audio`+`after_audio`+`gap_seconds` → bridge (needs FIM ckpt).
  - `/stem` — **pure Demucs, no model**: `remove`/`keep` stem lists → mixdown (any ckpt).
  - `/addstem` — generative add: `audio`+`target_stem`+`prompt` → new stem (needs stem-trained ckpt).
- `inference.py` — InferenceEngine: generate_audio/extend_audio/cover_audio/infill_audio/separate_stems/add_stem.

### Scripts (`scripts/`)
- `dac_roundtrip.py` — DAC encode/decode sanity check.
- `eval_checkpoint.py` / `eval_train_vs_val.py` — checkpoint loss/gen check / overfitting check.
- `lyrics_audit.py` — whole-corpus health audit (all streams, despite the name). `modal run scripts/lyrics_audit.py`.

### Claude skills (`.claude/skills/`)
Runbooks: `add-songs` (data prep), `train-model`, `eval-training-data`, `eval-checkpoint`, `eval-lyrics`, `run-tests`, `serve-model`. User-facing prompting guide: [README.prompting.md](README.prompting.md).

## Training Defaults

Local defaults in [diskrot/train.py](diskrot/train.py); Modal in `DEFAULTS` ([diskrot/modal_train.py](diskrot/modal_train.py)).

| Parameter | Local | Modal |
|---|---|---|
| batch_size | 8 | 32 (global) |
| lr | 2.5e-4 | 2.1e-4 |
| steps | 125,000 | 400,000 |
| segment_seconds | 30 | 60 |
| patience | 15 | 20 |
| warmup_steps | 1,500 | 5,000 |

The local CLI has **no** architecture flags — it always trains the `GPTConfig` ~2.0B shape, so local full training is impractical (use it for pipeline validation only). Modal is a multi-GPU DDP job.

## Data Flow

1. **Add corpus** → R2 `nano-audio` bucket under `waves/wave_<id>/`.
2. **Prepare** (`modal_prepare`) — ffprobe validate + SHA-256 dedupe + drop <20s + drop >5:30 (OOM guard). `--quality-gate` drops clipped/silent/dead/low-bitrate.
   - 2b. **Audio-dedup** (optional) — chromaprint SimHash near-dup grouping.
3. **Tokenize** — MP3 → DAC → int16 `[9, T]` .pt.
4. **Melody** (optional) — chromagram → `<name>.mel.npy` on **nano-melody**. After tokenize, before pack.
   - 4b. **Stems** (optional) — Demucs → codec tokens → `<name>.stems.npy` on **nano-stems** (GPU stage, `--sample-pct`).
5. **Pack** — .pt (+ mel/stems sidecars) → sharded mmap `packed/`.
6. **Caption** (optional) — audio-LLM → `tags.json`.
7. **Transcribe** (optional) — Whisper on raw mix → sharded `lyrics/`.
8. **Filter lyrics** (recommended) — null hallucinated entries. After transcribe, before phonemize.
   - 8b. **Forced-align** (optional) — sharpen word timestamps.
9. **Structure** (optional) — allin1 sections → `structure/`. Loaded at train time.
10. **Key detect** (optional) — packed chroma → `keys.json`. After pack.
11. **Phonemize** (recommended) — offline g2p → `phonemes/`. After transcribe/filter.
12. **Train** — packed shards + tags + lyrics/structure/keys/phonemes → delayed sequence → per-codebook CE.
13. **Inference** — checkpoint → autoregressive gen (KV cache) → DAC decode → MP3.

The `add-songs` skill walks the full ingest end-to-end.

## Storage

Raw audio is in **Cloudflare R2** (`nano-audio`, `CloudBucketMount`, no inode cap). Everything else is on Modal Volumes:

| Store | Contents |
|---|---|
| `nano-audio` (R2) | Raw MP3s under `waves/wave_<id>/` |
| nano-tokens | .pt files, `packed/` (incl. `.mel.bin`/`.stem.bin`), tags.json, lyrics/, structure/, keys.json, phonemes/ |
| nano-melody | `<name>.mel.npy` chroma sidecars (own volume for inode cap) |
| nano-stems | `<name>.stems.npy` stem sidecars (own volume) |
| nano-ckpts | Checkpoints (step_*, latest.pt, best.pt) |
| nano-output | Inference-server generations |
| `nano-backup` (R2) | Backups: `tokens/`, `ckpts/` (opt-in), `local/` (laptop bundle) |

**Backup**: `modal run --detach diskrot/modal_backup.py` mirrors GPU-expensive derived data (incremental rclone; `--verify` audits). Re-run after each wave ingest and after any `--redo`/`--apply` sweep. Raw audio, mel/stem intermediates, caches, and `step_*` history are NOT backed up.

## Checkpoints

Dicts with keys: `model` (state_dict), `optim`, `step`, `cfg` (GPTConfig dict), `best_val_loss`, `evals_without_improvement`, optional `text_proj`. Fine-tune runs add an inert `init_from` key.

**LoRA runs** use an adapter-only schema: no `model` key; `lora = {config, state, base_ckpt}` + adapter-only `optim`. Merge before serving (`diskrot.merge_lora`).

Inference loads GPTConfig from the checkpoint's `cfg`, so architecture changes are picked up automatically. Old checkpoints are incompatible if weight shapes change.

## Common Tasks

- **Add training data**: re-run prepare → tokenize → melody/stems → pack → tag → transcribe → filter → structure → key-detect → phonemize → re-train (the `add-songs` skill). More of the same kind of data — don't curate for variety.
- **Change model size**: edit `DEFAULTS` (Modal) and/or `GPTConfig` defaults (local). Start fresh (delete checkpoints).
- **Change segment length**: `segment_seconds` in TrainConfig; also `max_seq_len` in GPTConfig (≥ segment_frames + n_codebooks − 1).
- **Disable text conditioning**: `--text-conditioned false` (Modal) or omit tags/lyrics paths (local).
- **Fine-tune**: `--init-from <ckpt>` + a **fresh** `--ckpt-subdir`, lower LR (~5e-5). See [README.finetune.md](README.finetune.md).
- **LoRA-train**: add `--lora` to an `--init-from` launch; merge before serving. See [README.finetune.md](README.finetune.md).
- **Distill a fast student**: `--distill-from <teacher>` + student shape flags + fresh `--ckpt-subdir` (student keeps teacher n_codebooks/vocab). See [README.distill.md](README.distill.md).
