# nano — Project Guide

Audio generation model. Decoder-only transformer trained on DAC-tokenized audio with optional CLAP text/audio conditioning. ~1.5B params with text-conditioning cross-attention on (the default; ~1.14B without it) — d_model=2048, n_layers=22, n_heads=16, d_ff=8192.

## Project principles

- **This is a bespoke model — there is one model, not a family of tiers.** The `DEFAULTS` dict in [diskrot/modal_train.py](diskrot/modal_train.py) is the single source of truth for its shape. (Earlier "v4/v5/v6" framing is retired; the only surviving artifact is the `ckpt_subdir="v7_1500m"` constant, which is just a directory name — the model scaled from ~287M to ~1.5B.)
- **Scale of training data matters; variety does not.** Recommended corpus size is **~50k songs minimum** (below ~10k produces noise — pipeline-validation only) up to the `nano-corpus` volume's ~500k-file ceiling. More of the same kind of data helps; do **not** optimize tagging or curation for genre/style diversity. Breadth of genre is not a goal.

## Architecture

```
MP3 corpus → DAC tokenizer → [9 codebooks, 1024 vocab, 86 Hz] → Transformer → audio
                                                                      ↑
                                                        optional CLAP conditioning
                                                        (tags, lyrics, style audio)
```

**Model**: Decoder-only transformer with 9 codebook prediction heads. Trains on 30-second segments. Max single-shot generation ~95 seconds (max_seq_len=8192, RoPE). Shape is configured at launch via `GPTConfig` — `DEFAULTS` in [diskrot/modal_train.py](diskrot/modal_train.py) is the source of truth (currently d_model=2048, n_layers=22, n_heads=16, d_ff=8192 → ~1.5B).

**Delay pattern**: MusicGen-style — codebook k is shifted right by k positions. At each step, all 9 codebook embeddings are summed as input; 9 separate linear heads predict the next token per codebook. This lets later codebooks condition on earlier codebooks at the same frame.

**Text conditioning** — tags and lyrics travel **separate paths** (don't conflate them):

- **Tags** (genre/timbre/"vibe"): `auto_tag` runs the LP-MusicCaps captioner to write a natural-language description per song into `tags.json`; at train/inference frozen Microsoft CLAP (1024-dim) encodes that string and a learned linear layer projects to d_model — a single pooled vector at cross-attention position 0.
- **Lyrics** (the actual words to sing): conditioned as a **phoneme-ID sequence**, NOT through CLAP. `text_to_phoneme_ids` (g2p_en) converts the lyric string to ARPABET phonemes; a trainable `LyricEncoder` ([model/lyric_encoder.py](model/lyric_encoder.py)) — a small bidirectional transformer living *inside* `NanoAudioGPT` — encodes them into a sequence, and each transformer block has a **separate lyric cross-attention** over it. The model learns its own (near-monotonic) alignment, so no inference-time timestamps/duration model are needed. This is what makes the model sing intelligible words; a single pooled CLAP "vibe" vector structurally cannot.

Each stream drops independently for classifier-free guidance (10% each during training); at inference `cfg_scale` guides them jointly, or an optional `lyric_cfg_scale` pushes the lyric axis harder via composed guidance. The phoneme `LyricEncoder` is a submodule of the model, so it saves/restores in the `model` state_dict (no sidecar key like CLAP's `text_proj`). Checkpoints from before the lyric encoder (v7) are incompatible — `ckpt_subdir` is now `v8_sing`.

## Key Files

### Model (`model/`)
- `nano_audio_gpt.py` — GPTConfig, NanoAudioGPT, StaticLayerKVCache, CausalSelfAttention, CrossAttention. The core model.
- `codec.py` — DACodec wrapper. Constants: SAMPLE_RATE=44100, N_CODEBOOKS=9, VOCAB_SIZE=1024, FRAME_RATE_HZ=86. encode() and decode() methods.
- `delay_pattern.py` — apply_delay(), revert_delay(), build_train_inputs(). MusicGen delay logic.
- `text_encoder.py` — CLAPTextEncoder. Frozen CLAP + learned projection. encode() for **tags**, encode_audio() for style references. (Lyrics no longer go through CLAP.)
- `lyric_encoder.py` — phoneme lyric conditioning. Frozen `PHONEME_VOCAB` (ARPABET) + `text_to_phoneme_ids`/`text_to_word_phoneme_groups` (g2p_en, the single train==inference id mapping) + `LyricEncoder` (small bidirectional transformer, a submodule of NanoAudioGPT). The token-sequence path that lets the model sing words.
- `captioner.py` — Vendored LP-MusicCaps (BART-based audio captioner). load_captioner() for inference.

### Training (`diskrot/`)
- `train.py` — TrainConfig + train_run(). Device-agnostic training loop. Cosine LR with warmup, AdamW, early stopping, per-codebook loss logging. CLI: `python -m diskrot.train`.
- `dataset.py` — TokenDataset. Mmap-backed sharded dataset via `load_mmap_bundle()`. Serves random 30s crops. `__getitem__` returns `(tokens, tags, lyric_ids)` (time-aligned phoneme ids for the crop window, BOS-seeded); `collate_lyrics` pads the ragged lyric sequences + builds the mask. g2p runs once per song (cached, sliced by word).
- `tokenize.py` — Converts MP3 corpus to cached DAC .pt files. CLI: `python -m diskrot.tokenize`.
- `auto_tag.py` — LP-MusicCaps audio captioning. Generates natural-language descriptions from audio. CLI: `python -m diskrot.auto_tag`.
- `transcribe_lyrics.py` — Demucs vocal isolation + Whisper transcription. CLI: `python -m diskrot.transcribe_lyrics`.
- `pack_cache.py` — Packs .pt token files into the sharded mmap layout (`packed/packed_NNN.bin` + `packed_index.json`). Deterministic and **resumable**: shards are written atomically and a re-run skips shards already complete-and-valid on disk, so a kill/preemption mid-pack only loses the in-flight shard (not an in-place update — it still validates every shard). `pack()` takes an optional `commit_cb` so the Modal wrapper can commit each shard to the volume. CLI: `python -m diskrot.pack_cache`.
- `modal_train.py` — Modal H100/DDP training entrypoint. `DEFAULTS` here is the model's source of truth. `modal run --detach diskrot/modal_train.py --n-gpus 4`.
- `modal_tokenize.py` — Modal L4 tokenization (up to 50 containers). `modal run --detach diskrot/modal_tokenize.py`.
- `modal_auto_tag.py` — Modal L4 audio captioning (up to 20 containers). `modal run --detach diskrot/modal_auto_tag.py`.
- `modal_transcribe.py` — Modal L4 parallel transcription (up to 50 containers). `modal run --detach diskrot/modal_transcribe.py`.
- `modal_prepare.py` — CPU pass that validates (ffprobe), dedupes (SHA-256), drops <20s clips, and drops long files (>5:30) so the L4 tokenizer doesn't OOM. `modal run --detach diskrot/modal_prepare.py [--apply]`.
- `modal_pack_cache.py` — Modal wrapper around `pack_cache.py`.
- `modal_inspect_ckpts.py` — Inspect a checkpoint trajectory on the volume. `modal run diskrot/modal_inspect_ckpts.py --prefix v7_1500m`.

### Server (`server/`)
- `main.py` — FastAPI app with endpoints: GET /health, POST /generate, POST /continue, POST /extend. All generation endpoints accept optional text, lyrics, style_audio, and style_weight params.
- `inference.py` — InferenceEngine. Loads checkpoint, handles float16 casting and torch.compile. continue_audio(), extend_audio(), generate_audio() methods.

### Scripts (`scripts/`)
- `dac_roundtrip.py` — DAC encode/decode sanity check. Writes orig + reconstructed WAVs.
- `eval_checkpoint.py` — Evaluate a checkpoint (loss / generation sanity check).
- `eval_train_vs_val.py` — Compare train vs val loss for a checkpoint (overfitting check).

### Claude skills (`.claude/skills/`)
Task runbooks that orchestrate the CLIs above (link the READMEs, don't duplicate them):
- `add-songs` — full data-prep pipeline (upload → prepare → tokenize → pack → tag → transcribe).
- `train-model` — launch/monitor/resume training; extract a slim model + download the best checkpoint. Includes `references/reading-logs.md`.
- `eval-checkpoint` — route to the right diagnostic (sample quality, overfitting, lyric ablation, sampling sweep, genre coverage, codec fidelity).
- `run-tests` — pytest suite, benchmark marker, `synth_tokens_dir` fixture.
- `serve-model` — local/Modal inference server and the generation endpoints.

## Training Defaults

Local defaults live in [diskrot/train.py](diskrot/train.py); Modal defaults in `DEFAULTS` in [diskrot/modal_train.py](diskrot/modal_train.py).

| Parameter | Local | Modal (H100) |
|---|---|---|
| batch_size | 8 | 64 |
| lr | 2.5e-4 | 3.0e-4 |
| steps | 125,000 | 400,000 |
| segment_seconds | 30 | 30 |
| patience | 15 | 20 |
| warmup_steps | 1,500 | 5,000 |

(Local values are the `diskrot.train` CLI defaults; note the CLI's `--patience` default is 15 even though the `TrainConfig` dataclass default is 20. The local CLI has **no** architecture flags — it always trains the `GPTConfig` defaults, i.e. the same ~1.5B shape as Modal, so local full training is impractical on consumer GPUs; use it for pipeline validation. Modal is an 8×H100 DDP job; the per-rank batch is 8, global 64.)

## Data Flow

1. **Add corpus**: upload your own MP3s → `nano-corpus` volume (`modal volume put`)
2. **Prepare**: validate (ffprobe) + dedupe (SHA-256) + drop <20s + drop long files (>5:30) via `diskrot.modal_prepare` — required so the L4 tokenizer doesn't OOM
3. **Tokenize**: MP3 → librosa (44.1kHz mono) → DAC encode → int16 tensor [9, T_frames] saved as .pt
4. **Pack**: .pt files → sharded mmap layout (`packed/packed_NNN.bin` + JSON sidecars) via `diskrot.pack_cache`
5. **Caption** (optional): MP3 → LP-MusicCaps (16 kHz mel → BART) → natural-language description → tags.json
6. **Transcribe** (optional): MP3 → Demucs (vocal isolation) → Whisper (word-level timestamps) → sharded `lyrics/` dir (`lyrics_NNN.json`, 256 hash-keyed shards, atomic writes; the `.map()` loop runs in a spawned remote fn so `--detach` survives terminal close)
7. **Train**: packed shards + tags.json + lyrics/ → TokenDataset (mmap-backed random 30s crops) → delayed sequence → cross-entropy loss per codebook
8. **Inference**: checkpoint → NanoAudioGPT → autoregressive generation with KV cache → DAC decode → MP3 via ffmpeg

## Modal Volumes

| Volume | Contents |
|---|---|
| nano-corpus | Raw MP3 files |
| nano-tokens | .pt token files, packed/ shards, tags.json, lyrics.json |
| nano-ckpts | Training checkpoints (step_*.pt, latest.pt, best.pt) |

## Checkpoints

Saved as dicts with keys: `model` (state_dict), `optim`, `step`, `cfg` (GPTConfig as dict), `best_val_loss`, `evals_without_improvement`, optional `text_proj` (CLAP projection weights).

Inference loads GPTConfig from checkpoint's `cfg` dict, so model architecture changes are automatically picked up. Old checkpoints are incompatible if weight shapes change.

## Common Tasks

- **Add more training data**: Add MP3s to corpus, re-run prepare + tokenize + pack + auto-tag + transcribe, then re-train (the `add-songs` skill walks this end-to-end). More of the same kind of data is the goal — don't curate for variety.
- **Change model size**: Edit the `DEFAULTS` dict in `diskrot/modal_train.py` (Modal) and/or `GPTConfig` defaults in `model/nano_audio_gpt.py` (local, since the local CLI has no architecture flags). Must start fresh (delete checkpoints).
- **Change segment length**: Edit `segment_seconds` in TrainConfig (`diskrot/train.py`). Also update `max_seq_len` in GPTConfig if needed (must be >= segment_frames + n_codebooks - 1).
- **Disable text conditioning**: Pass `--text-conditioned false` to modal_train.py or omit `--tags-path` / `--lyrics-path` from train.py.
