# nano — Project Guide

Audio generation model. Decoder-only transformer trained on DAC-tokenized audio with optional CLAP text/audio conditioning. ~2.0B params with text-conditioning on (the default; 2013.8M measured at v8 startup — the ~1.5B decoder plus per-block lyric cross-attention ~369M and the lyric/melody encoders ~130M; ~1.14B with text conditioning off) — d_model=2048, n_layers=22, n_heads=16, d_ff=8192.

## Project principles

- **This is a bespoke model — there is one model, not a family of tiers.** The `DEFAULTS` dict in [diskrot/modal_train.py](diskrot/modal_train.py) is the single source of truth for its shape. (Earlier "v4/v5/v6" framing is retired; the only surviving artifact is the `ckpt_subdir="v7_1500m"` constant, which is just a directory name — the model scaled from ~287M to ~2.0B.)
- **Scale of training data matters; variety does not.** Recommended corpus size is **~50k songs minimum** (below ~10k produces noise — pipeline-validation only) up to the `nano-corpus` volume's ~500k-file ceiling. More of the same kind of data helps; do **not** optimize tagging or curation for genre/style diversity. Breadth of genre is not a goal.

## Architecture

```
MP3 corpus → DAC tokenizer → [9 codebooks, 1024 vocab, 86 Hz] → Transformer → audio
                                                                      ↑
                                              tags (pooled CLAP) + lyrics (phonemes)
                                              + melody (time-aligned chromagram)
```

**Model**: Decoder-only transformer with 9 codebook prediction heads. Trains on 60-second segments (Modal default; the local CLI default is 30s). Max single-shot generation ~95 seconds (max_seq_len=8192, RoPE). Shape is configured at launch via `GPTConfig` — `DEFAULTS` in [diskrot/modal_train.py](diskrot/modal_train.py) is the source of truth (currently d_model=2048, n_layers=22, n_heads=16, d_ff=8192 → ~2.0B with all conditioning modules).

**Delay pattern**: MusicGen-style — codebook k is shifted right by k positions. At each step, all 9 codebook embeddings are summed as input; a single fused output Linear (`model.head`, d_model → 9×vocab, viewed back to per-codebook logits — one matmul instead of 9 launches) predicts the next token per codebook. This lets later codebooks condition on earlier codebooks at the same frame.

**Text conditioning** — tags and lyrics travel **separate paths** (don't conflate them):

- **Tags** (genre/timbre/"vibe"): `auto_tag` runs the LP-MusicCaps captioner to write a natural-language description per song into `tags.json`; at train/inference frozen Microsoft CLAP (1024-dim) encodes that string and a learned linear layer projects to d_model — a single pooled vector at cross-attention position 0.
- **Lyrics** (the actual words to sing): conditioned as a **phoneme-ID sequence**, NOT through CLAP. `text_to_phoneme_ids` (g2p_en) converts the lyric string to ARPABET phonemes; a trainable `LyricEncoder` ([model/lyric_encoder.py](model/lyric_encoder.py)) — a small bidirectional transformer living *inside* `NanoAudioGPT` — encodes them into a sequence, and each transformer block has a **separate lyric cross-attention** over it. The model learns its own (near-monotonic) alignment, so no inference-time timestamps/duration model are needed. This is what makes the model sing intelligible words; a single pooled CLAP "vibe" vector structurally cannot.

  **Markers in the lyric stream** — the phoneme stream also carries non-word **marker tokens** the decoder cross-attends to (frozen in `PHONEME_VOCAB`, size 129): *song-structure* markers (`<verse>`, `<chorus>`, …, from the allin1 analyzer), *vocal-gender* markers (`<male>`/`<female>`/`<unknown_gender>`, from an F0 heuristic on the Demucs vocal stem — see `transcribe_lyrics.estimate_vocal_gender`), *tempo* markers (`<tempo_0>`…`<tempo_13>`/`<unknown_tempo>`, the allin1 per-song `bpm` bucketed on a 10-BPM grid by `TEMPO_BPM_EDGES`), *key* markers (`<key_c_major>`…/`<unknown_key>`, 24 keys from a Krumhansl-Schmuckler estimate over the packed chroma — `diskrot/key_detect.py` writes `keys.json`), and *vocal-presence* markers (`<vocals>`/`<instrumental>`/`<unknown_vocals>`, derived from the transcription pass: usable words → vocals, transcribed-but-wordless → instrumental, never transcribed → unknown — this is what lets a user *request* no vocals). Tempo rides the stream because chroma is octave-invariant pitch and structurally can't carry beat rate; key rides it because the additive chroma leaves the key implicit. Every stream opens with a compact dense header `BOS <gender> <tempo> <key> <vocals> <section>` (unknown fallbacks per slot so no row is fully padded); structure boundaries inside a crop are injected inline (gender/tempo/key/vocals are per-song, prefix-only). The train-time injector (`TokenDataset._get_segment_lyric_ids`) and the inference parser (`text_with_markers_to_phoneme_ids`, which accepts `[female] [120bpm] [a minor] [instrumental] [chorus] …` brackets — bpm as numeric/`tempo:NNN`/`slow`/`medium`/`fast`, key as `Am`/`a minor`/`key:Am`/`f# major`; the vocal slot defaults to `<vocals>` when words are present) MUST build byte-identical streams — `tests/test_structure_markers.py` is the guard. Adding/removing any marker changes `PHONEME_VOCAB_SIZE` and is checkpoint-incompatible.

**Melody conditioning** (the `/cover` path) — a third axis, separate again from tags and lyrics. A hummed/uploaded melody is encoded as a **time-aligned 12-bin chromagram** (octave-invariant `chroma_cqt` at the 86 Hz DAC frame rate; `diskrot/melody.py:extract_chroma` is the single train==inference contract). Because the signal is dense and frame-aligned (unlike the ragged phoneme sequence), it is conditioned **additively, not via cross-attention**: `MelodyEncoder` ([model/melody_encoder.py](model/melody_encoder.py), a submodule of `NanoAudioGPT`) projects the chroma and the decoder adds it to the per-frame token-sum input at the **cb0 anchor** (delayed position p ↔ frame p; the K-1 delay tail and any prompt prefix get a learned `null`). This lets the model regenerate a melodic contour in whatever timbre the tags ask for (e.g. hum → solo violin) while the octave stays free. Per-song chroma is extracted once to a `<name>.mel.npy` on the **nano-melody** volume (kept off nano-tokens for its inode cap), then the packer folds it into a parallel `packed_NNN.mel.bin` sidecar at the same offsets as the tokens, so the dataset co-crops it with the identical crop window — `tests/test_melody.py` guards the equivalence.

**Infill conditioning** (the `/infill` path) — fill-in-the-middle, a structurally different axis: not a side conditioning stream but a **frame-domain reorder of the audio tokens themselves** (canonical FIM, Bavarian et al.). When `use_fim` is on, a fraction (`fim_prob`, ~0.15) of training batches are rearranged into `prefix · <SUF> · suffix · <MID> · middle` ([diskrot/train.py](diskrot/train.py) calls [model/fim.py](model/fim.py):`fim_reorder_batch` **before** the delay pattern), so the **causal attention and delay logic are untouched** — the decoder just learns that, shown a prefix and the suffix that follows the gap (separated by two new per-codebook control ids `<SUF>`/`<MID>`), it should generate the bridging middle. Because the reorder scrambles frame order, a FIM batch **drops the lyric stream** (its near-monotonic sung alignment can't survive) but keeps tags (order-invariant) and the **co-reordered melody** (chroma permuted identically, zero rows at the two sentinel frames — in-distribution per the packer's zero-fill, so `MelodyEncoder` is unchanged). At inference, `/infill` reuses `generate()` verbatim: it builds the `prefix <SUF> suffix <MID>` prompt (`build_fim_prompt`, kept byte-identical to the train reorder by `tests/test_fim.py`) and asks for `gap_frames` new frames — those frames *are* the middle, stitched back as `before + middle + after`. Enabling `use_fim` grows the per-codebook vocab by 2, so it's checkpoint-incompatible.

Each stream drops independently for classifier-free guidance (10% each during training; melody drops to its learned null); at inference `cfg_scale` guides them jointly, or an optional `lyric_cfg_scale` / `melody_cfg_scale` pushes that axis harder via composed guidance. The `LyricEncoder` and `MelodyEncoder` are submodules of the model, so they save/restore in the `model` state_dict (no sidecar key like CLAP's `text_proj`). Checkpoints from before the lyric encoder (v7) are incompatible; v8 (`ckpt_subdir="v8_sing"`) ships the lyric encoder, the melody encoder, the expanded marker header (key/vocal-presence/finer tempo), **and** the fused output head in one fresh start — each of those is checkpoint-incompatible on its own, and melody requires the pack to carry the chroma sidecar. **FIM is deferred for v8** (`use_fim=False` in `DEFAULTS`): the code path exists, but the v8 checkpoint trains without the `<SUF>`/`<MID>` control ids, so `/infill` won't work on it — enabling FIM later means another fresh start.

## Key Files

### Model (`model/`)
- `nano_audio_gpt.py` — GPTConfig, NanoAudioGPT, StaticLayerKVCache, CausalSelfAttention, CrossAttention. The core model.
- `codec.py` — DACodec wrapper. Constants: SAMPLE_RATE=44100, N_CODEBOOKS=9, VOCAB_SIZE=1024, FRAME_RATE_HZ=86. encode() and decode() methods.
- `delay_pattern.py` — apply_delay(), revert_delay(), build_train_inputs(). MusicGen delay logic.
- `text_encoder.py` — CLAPTextEncoder. Frozen CLAP + learned projection. encode() for **tags**, encode_audio() for style references. (Lyrics no longer go through CLAP.)
- `lyric_encoder.py` — phoneme lyric conditioning. Frozen `PHONEME_VOCAB` (4 specials + structure + gender + tempo + key + vocal-presence markers + ARPABET, 129 ids) + `text_to_phoneme_ids`/`text_to_word_phoneme_groups` (g2p_en, the single train==inference id mapping) + the per-family mappers (`structure_label_to_id`, `gender_label_to_id`, `bpm_to_id`, `key_label_to_id`, `vocal_label_to_id` and their `is_*`/`parse_*` companions) + `text_with_markers_to_phoneme_ids` (inference `[chorus]`/`[female]`/`[120bpm]`/`[a minor]`/`[instrumental]` bracket parser) + `LyricEncoder` (small bidirectional transformer, a submodule of NanoAudioGPT). The token-sequence path that lets the model sing words (and carries the markers it conditions on).
- `melody_encoder.py` — chromagram melody conditioning. `MelodyEncoder` (12→d_model conv/linear stack + a learned `null`), a submodule of NanoAudioGPT. Unlike tags/lyrics (cross-attention), melody is **dense and time-aligned**, so it's added to the decoder's per-frame input at the cb0 anchor. The additive path that lets a hummed melody be re-rendered in the prompt's timbre (`/cover`).
- `lora.py` — hand-rolled LoRA (no PEFT dependency). `LoRALinear` subclasses `nn.Linear`, so the frozen base `weight` keeps its state-dict key and `lora_A`/`lora_B` are purely additive — adapter extraction is a key-suffix filter and `merge_state_dicts` reproduces the base key layout exactly (a merged checkpoint is indistinguishable from a normally-trained one; inference/eval/MLX/quantization need zero changes). `inject_lora` targets the decoder-block Linears (self-attn qkv/proj, both cross-attns, MLP); head/embeddings/norms/lyric+melody encoders stay frozen. Injection happens after the base load (the weight Parameter is reused, not copied), before DDP/compile. Defaults r=16, alpha=32 → ~21.6M trainable (~1% of 2.0B). `tests/test_lora.py` guards injection/merge/startup-precedence.
- `fim.py` — fill-in-the-middle (infill) reorder. `fim_reorder_batch` (train: rearrange a [B,K,T] crop + its chroma into `prefix <SUF> suffix <MID> middle`, frame-domain, before the delay pattern) + `build_fim_prompt` (inference: the byte-identical `prefix <SUF> suffix <MID>` prompt). Gated by `GPTConfig.use_fim`, which adds the `<SUF>`/`<MID>` control ids; `tests/test_fim.py` guards train==inference layout.
- `captioner.py` — Vendored LP-MusicCaps (BART-based audio captioner). load_captioner() for inference.

### Training (`diskrot/`)
- `train.py` — TrainConfig + train_run(). Device-agnostic training loop. Cosine LR with warmup, AdamW, early stopping, per-codebook loss logging. CLI: `python -m diskrot.train`. Also the fine-tune/LoRA entry: `--init-from <ckpt>` loads pretrained weights + GPTConfig **from the checkpoint** (architecture flags become advisory) with a fresh optimizer/step/schedule; `--lora` additionally freezes the base and trains adapters only. Startup precedence (`_resolve_startup`): `{ckpt_dir}/latest.pt` resume > `--init-from` > scratch — so fine-tunes need a fresh ckpt dir, and re-running the same command is always the resume. LoRA checkpoints are adapter-only (no `"model"` key; `"lora"={config,state,base_ckpt}`; `"cfg"` stays a pure GPTConfig dict) and must be merged before serving. See [README.finetune.md](README.finetune.md).
- `dataset.py` — TokenDataset. Mmap-backed sharded dataset via `load_mmap_bundle()`. Serves random `segment_seconds` crops (60s on Modal). `__getitem__` returns `(tokens, tags, lyric_ids, melody)` — time-aligned phoneme ids (BOS-seeded) and the chroma crop (`[seg,12]` or `None`) co-cropped with the identical window; `collate_lyrics` pads the ragged lyric sequences + builds the mask + stacks the melody. Per-word phoneme groups come from the pre-phonemized `phonemes/` store when present (`phonemes_path`), else live g2p (cached per song, sliced by word; a stale store entry falls back to g2p). Also loads `keys.json` (`keys_path`) and derives the instrumental set from transcribed-but-wordless lyrics entries for the key/vocal header markers.
- `melody.py` — `extract_chroma()`: the shared train==inference chromagram extractor (octave-invariant `chroma_cqt`, hop=512 → 86 Hz, L2-normalized, frame count forced to the DAC `ceil(samples/512)`). Used by the packer (train) and the server cover path (inference).
- `tokenize.py` — Converts MP3 corpus to cached DAC .pt files. CLI: `python -m diskrot.tokenize`.
- `auto_tag.py` — LP-MusicCaps audio captioning. Generates natural-language descriptions from audio. CLI: `python -m diskrot.auto_tag`.
- `transcribe_lyrics.py` — Demucs vocal isolation + Whisper transcription + F0 vocal-gender labeling (`estimate_vocal_gender` writes a `gender` field per song). The `gender` field is written **inside this pass**, so a v7 lyrics dir (pre-gender) has none — re-run transcribe to add it. CLI: `python -m diskrot.transcribe_lyrics`.
- `filter_lyrics.py` — nulls Whisper-hallucinated lyric entries in the sharded `lyrics/` dir (invented captions over instrumental audio: "Thank you.", "Thanks for watching!", … — ~24% of with-words entries in the 2026-06 sweep). Flags entries with <6 valid words, or a known caption-artifact phrase in a <30-word transcript; nulled entries train as `<instrumental>` instead of `<vocals>`-with-garbage. Dry-run by default, `--apply` to rewrite (atomic per shard, idempotent). Runs **after** transcribe completes (the orchestrator's in-memory flush would clobber concurrent edits), **before** phonemize. CLI: `python -m diskrot.filter_lyrics --lyrics-dir ./lyrics [--apply]`.
- `structure.py` — allin1 song-structure analysis (Demucs + joint beat/segment model) → functional sections (`intro/verse/chorus/bridge/outro/break/inst/solo`) with timestamps → sharded `structure/structure_NNN.json` (mirrors `transcribe_lyrics`: 256 hash-keyed shards, atomic writes, resume-by-skip). The dataset injects these markers by timestamp into the phoneme lyric stream (`_get_segment_lyric_ids`); a missing/partial pass just yields `<no_section>`. CLI: `python -m diskrot.structure`.
- `key_detect.py` — per-song key estimation: streams the packed `packed_NNN.mel.bin` chroma, averages over time, Krumhansl-Schmuckler correlation → `keys.json` (`{name: {"key": "a_minor"}}`), the `<key_*>` header marker source. Resumable (atomic rewrite per shard, skip-by-name). Runs **after** pack (reads the chroma sidecar). CLI: `python -m diskrot.key_detect`.
- `phonemize.py` — pre-phonemizes the lyrics corpus: g2p per song offline → sharded `phonemes/phonemes_NNN.json` (per-word phoneme-id groups, mirrors the lyrics sharding). Exists because lazy in-DataLoader g2p costs ~20-200 ms/song on OOV-heavy Whisper transcripts and the LRU misses ~99% at corpus scale — enough to starve 8×H100. Train-time loader validates group count vs word count and falls back to live g2p, so partial/stale is safe. Runs **after** transcribe. CLI: `python -m diskrot.phonemize`.
- `pack_cache.py` — Packs .pt token files into the sharded mmap layout (`packed/packed_NNN.bin` + `packed_index.json`). Deterministic and **resumable**: shards are written atomically and a re-run skips shards already complete-and-valid on disk, so a kill/preemption mid-pack only loses the in-flight shard (not an in-place update — it still validates every shard). `pack()` takes an optional `commit_cb` so the Modal wrapper can commit each shard to the volume. With `--mel-cache-dir` it also writes a parallel `packed_NNN.mel.bin` (12×fp16 chroma) at the **same per-song offsets** from the `<name>.mel.npy` files (missing/mismatched chroma is zero-filled, never dropped, so membership stays identical). CLI: `python -m diskrot.pack_cache`.
- `modal_train.py` — Modal H100/DDP training entrypoint. `DEFAULTS` here is the model's source of truth. `modal run --detach diskrot/modal_train.py --n-gpus 4`. Fine-tune/LoRA flags: `--init-from v8_sing/best.pt` (path on nano-ckpts), `--lora` (+ `--lora-r/--lora-alpha`; single GPU is the recommended LoRA path), `--data-subdir` (train on a corpus packed under `/tokens/<subdir>`), always with a fresh `--ckpt-subdir`.
- `merge_lora.py` — fold a LoRA adapter checkpoint into its base → a standard slim-inference checkpoint (the `modal_export_ckpt` schema), so serving/eval are unchanged. CLI: `python -m diskrot.merge_lora --base ... --lora ... --out ...`.
- `modal_merge_lora.py` — the same merge ON the nano-ckpts volume (the multi-GB base never leaves Modal). `modal run diskrot/modal_merge_lora.py --base v8_sing/best.pt --lora v8_lora/best.pt`.
- `modal_tokenize.py` — Modal L4 tokenization (up to 50 containers). `modal run --detach diskrot/modal_tokenize.py`.
- `modal_auto_tag.py` — Modal L4 audio captioning (up to 20 containers). `modal run --detach diskrot/modal_auto_tag.py`.
- `modal_transcribe.py` — Modal L4 parallel transcription (up to 50 containers). `modal run --detach diskrot/modal_transcribe.py`.
- `modal_filter_lyrics.py` — Modal CPU wrapper around `filter_lyrics.py` (one container, /tokens/lyrics sweep). Dry-run by default: `modal run --detach diskrot/modal_filter_lyrics.py [--apply]` (report prints to the remote logs).
- `modal_structure.py` — Modal L4 parallel song-structure analysis (allin1, up to 50 containers); writes the sharded `structure/` dir. Runs any time after the corpus exists, before train (loaded at train time, not packed). `modal run --detach diskrot/modal_structure.py` (`--limit 200` to calibrate the image first).
- `modal_melody.py` — Modal CPU parallel chromagram extraction (up to 50 containers); writes per-song `<name>.mel.npy` to the dedicated **nano-melody** volume (its own volume so the ~1-file-per-song it adds doesn't push nano-tokens over its ~500k-inode cap). Runs **after** tokenize (needs the token frame count to align), **before** pack (which reads `/melody` via `--mel-cache-dir`). `modal run --detach diskrot/modal_melody.py`.
- `modal_prepare.py` — CPU pass that validates (ffprobe), dedupes (SHA-256), drops <20s clips, and drops long files (>5:30) so the L4 tokenizer doesn't OOM. `modal run --detach diskrot/modal_prepare.py [--apply]`.
- `modal_pack_cache.py` — Modal wrapper around `pack_cache.py`.
- `modal_key_detect.py` — Modal CPU wrapper around `key_detect.py` (one container, sequential mmap sweep over /tokens/packed → /tokens/keys.json). Runs after pack. `modal run --detach diskrot/modal_key_detect.py`.
- `modal_phonemize.py` — Modal CPU wrapper around `phonemize.py` (one container, process-parallel across shard buckets, /tokens/lyrics → /tokens/phonemes). Runs after transcribe. `modal run --detach diskrot/modal_phonemize.py`.
- `modal_inspect_ckpts.py` — Inspect a checkpoint trajectory on the volume. `modal run diskrot/modal_inspect_ckpts.py --prefix v7_1500m`.

### Server (`server/`)
- `main.py` — FastAPI app with endpoints: GET /health, POST /generate, POST /extend, POST /cover, POST /infill. /generate and /extend accept optional text, lyrics, style_audio, and style_weight params. (`/extend` continues a clip forward from a cut point `from_seconds`, defaulting to the clip tail.) `/cover` takes a required `melody_audio` upload (the hum), conditions on its chromagram (the audio never appears in the output), and uses the prompt for timbre — needs a melody-trained checkpoint; supports `melody_cfg_scale`. `/infill` takes required `before_audio` + `after_audio` uploads and a `gap_seconds`, and generates the bridge between them (returns `before | middle | after`); `prompt` drives timbre, an optional `melody_audio` guides the gap's contour, lyrics are unused — needs a FIM-trained checkpoint (`use_fim`).
- `inference.py` — InferenceEngine. Loads checkpoint, handles float16 casting and torch.compile. generate_audio(), extend_audio(), cover_audio(), infill_audio() methods (cover_audio + `_build_melody` for the chroma path; infill_audio + `build_fim_prompt`/`_crossfade_concat` for the gap-fill path).

### Scripts (`scripts/`)
- `dac_roundtrip.py` — DAC encode/decode sanity check. Writes orig + reconstructed WAVs.
- `eval_checkpoint.py` — Evaluate a checkpoint (loss / generation sanity check).
- `eval_train_vs_val.py` — Compare train vs val loss for a checkpoint (overfitting check).
- `lyrics_audit.py` — Training-data health audit (Modal CPU, read-only): packed corpus size + scale check, per-stream coverage of the packed corpus (tags/structure/keys/lyrics/phonemes/melody), melody source-vs-packed coverage, song-duration distribution, instrumental/hallucinated/vocal-ready breakdown, gender + word-count + language distributions, top keys. Despite the name it covers all streams, not just lyrics. Safe mid-prep. `modal run scripts/lyrics_audit.py`.

### Claude skills (`.claude/skills/`)
Task runbooks that orchestrate the CLIs above (link the READMEs, don't duplicate them):
- `add-songs` — full data-prep pipeline (upload → prepare → tokenize → pack → tag → transcribe).
- `train-model` — launch/monitor/resume training; extract a slim model + download the best checkpoint. Includes `references/reading-logs.md`.
- `eval-training-data` — whole-corpus readiness before a run (scale check + per-stream coverage of tags/structure/keys/lyrics/phonemes/melody + melody source-vs-packed + duration spread); routes to eval-lyrics/eval-checkpoint for the deep lyric/genre detail.
- `eval-checkpoint` — route to the right diagnostic (sample quality, overfitting, lyric ablation, sampling sweep, genre coverage, codec fidelity).
- `eval-lyrics` — lyric data health (coverage audit, hallucination filter, gender/word-count distributions) + routing to the model-side lyric evals.
- `run-tests` — pytest suite, benchmark marker, `synth_tokens_dir` fixture.
- `serve-model` — local/Modal inference server and the generation endpoints.

## Training Defaults

Local defaults live in [diskrot/train.py](diskrot/train.py); Modal defaults in `DEFAULTS` in [diskrot/modal_train.py](diskrot/modal_train.py).

| Parameter | Local | Modal (H100) |
|---|---|---|
| batch_size | 8 | 32 (global) |
| lr | 2.5e-4 | 2.1e-4 |
| steps | 125,000 | 400,000 |
| segment_seconds | 30 | 60 |
| patience | 15 | 20 |
| warmup_steps | 1,500 | 5,000 |

(Local values are the `diskrot.train` CLI defaults; note the CLI's `--patience` default is 15 even though the `TrainConfig` dataclass default is 20. The local CLI has **no** architecture flags — it always trains the `GPTConfig` defaults, i.e. the same ~2.0B shape as Modal, so local full training is impractical on consumer GPUs; use it for pipeline validation. Modal is an 8×H100 DDP job; the per-rank batch is 4, global 32 — halved from the v7-era 64 because 60s doubles the sequence (tokens/step unchanged), with lr sqrt-scaled to match.)

## Data Flow

1. **Add corpus**: upload your own MP3s → `nano-corpus` volume (`modal volume put`)
2. **Prepare**: validate (ffprobe) + dedupe (SHA-256) + drop <20s + drop long files (>5:30) via `diskrot.modal_prepare` — required so the L4 tokenizer doesn't OOM
3. **Tokenize**: MP3 → librosa (44.1kHz mono) → DAC encode → int16 tensor [9, T_frames] saved as .pt
4. **Melody** (optional): MP3 → `chroma_cqt` (forced to the song's token frame count) → per-song `<name>.mel.npy` on the **nano-melody** volume via `diskrot.modal_melody` — runs after tokenize, before pack (own volume so it doesn't blow nano-tokens' inode cap)
5. **Pack**: .pt files (+ `<name>.mel.npy` from nano-melody via `--mel-cache-dir`) → sharded mmap layout (`packed/packed_NNN.bin` + parallel `packed_NNN.mel.bin` + JSON sidecars) via `diskrot.pack_cache`
6. **Caption** (optional): MP3 → LP-MusicCaps (16 kHz mel → BART) → natural-language description → tags.json
7. **Transcribe** (optional): MP3 → Demucs (vocal isolation) → Whisper (word-level timestamps) + F0 vocal-gender estimate → sharded `lyrics/` dir (`lyrics_NNN.json`, 256 hash-keyed shards, atomic writes; each per-song entry also carries a `gender` field — written **in this pass**, so a pre-gender lyrics dir must be regenerated by re-running transcribe; the `.map()` loop runs in a spawned remote fn so `--detach` survives terminal close)
8. **Filter lyrics** (recommended if you ran transcribe): sharded `lyrics/` → null out Whisper-hallucinated entries (invented captions over instrumentals — "Thank you." etc.) via `diskrot.modal_filter_lyrics --apply` — **after** transcribe fully completes, **before** phonemize; dry-run by default
9. **Structure** (optional): MP3 → allin1 (Demucs + joint beat/segment model) → functional sections with timestamps → sharded `structure/` dir (`structure_NNN.json`, mirrors lyrics) via `diskrot.modal_structure` — any time after the corpus exists, before train; loaded at train time (not packed), so a partial pass just yields `<no_section>`
10. **Key detect** (optional): packed chroma sidecar → mean chroma → Krumhansl key estimate → `keys.json` via `diskrot.modal_key_detect` — after pack, before train; a partial/absent pass just yields `<unknown_key>`
11. **Phonemize** (recommended): lyrics/ → per-word phoneme-id groups → sharded `phonemes/` dir via `diskrot.modal_phonemize` — after transcribe (and the lyric filter), before train; a partial/absent pass falls back to (slow) live g2p in the DataLoader workers
12. **Train**: packed shards (+ chroma sidecar) + tags.json + lyrics/ + structure/ + keys.json + phonemes/ → TokenDataset (mmap-backed random `segment_seconds` crops — 60s on Modal) → delayed sequence → cross-entropy loss per codebook
13. **Inference**: checkpoint → NanoAudioGPT → autoregressive generation with KV cache → DAC decode → MP3 via ffmpeg (the `/cover` path additionally feeds the uploaded hum's chromagram)

## Modal Volumes

| Volume | Contents |
|---|---|
| nano-corpus | Raw MP3 files |
| nano-tokens | .pt token files, packed/ shards (incl. `.mel.bin`), tags.json, lyrics/, structure/, keys.json, phonemes/ |
| nano-melody | `<name>.mel.npy` chroma sidecars (own volume — keeps nano-tokens under its ~500k-inode cap) |
| nano-ckpts | Training checkpoints (step_*.pt, latest.pt, best.pt) |
| nano-output | Generations from the Modal inference server (every /generate /extend /cover /infill result, written via `NANO_OUTPUT_DIR=/outputs`) |

## Checkpoints

Saved as dicts with keys: `model` (state_dict), `optim`, `step`, `cfg` (GPTConfig as dict), `best_val_loss`, `evals_without_improvement`, optional `text_proj` (CLAP projection weights). Fine-tune runs add an inert `init_from` provenance key.

**LoRA runs save a different, adapter-only schema**: no `model` key; instead `lora = {config, state, base_ckpt}` (the frozen base is referenced by path, not copied) and an adapter-only `optim`. `cfg` stays a pure GPTConfig dict in both schemas. A LoRA checkpoint cannot be served directly — merge it first (`diskrot.merge_lora`), which produces a standard slim-inference checkpoint.

Inference loads GPTConfig from checkpoint's `cfg` dict, so model architecture changes are automatically picked up. Old checkpoints are incompatible if weight shapes change.

## Common Tasks

- **Add more training data**: Add MP3s to corpus, re-run prepare + tokenize + melody + pack + auto-tag + transcribe + filter-lyrics + structure + key-detect + phonemize, then re-train (the `add-songs` skill walks this end-to-end). More of the same kind of data is the goal — don't curate for variety.
- **Change model size**: Edit the `DEFAULTS` dict in `diskrot/modal_train.py` (Modal) and/or `GPTConfig` defaults in `model/nano_audio_gpt.py` (local, since the local CLI has no architecture flags). Must start fresh (delete checkpoints).
- **Change segment length**: Edit `segment_seconds` in TrainConfig (`diskrot/train.py`). Also update `max_seq_len` in GPTConfig if needed (must be >= segment_frames + n_codebooks - 1).
- **Disable text conditioning**: Pass `--text-conditioned false` to modal_train.py or omit `--tags-path` / `--lyrics-path` from train.py.
- **Fine-tune the model**: `--init-from <ckpt>` (Modal: a nano-ckpts path like `v8_sing/best.pt`; local: any `.pt`) with a **fresh** `--ckpt-subdir`/`--ckpt-dir` — weights and GPTConfig come from the checkpoint, optimizer/step/LR schedule start fresh; lower the LR (think 5e-5). Precedence: an existing `latest.pt` in the run dir always resumes instead. Full walkthrough: [README.finetune.md](README.finetune.md).
- **LoRA-train (local or Modal)**: add `--lora` to an `--init-from` launch — base frozen, ~21.6M adapters train (single H100 or local Apple Silicon is enough). Checkpoints are adapter-only; fold into a standard checkpoint with `diskrot.merge_lora` (local) or `diskrot/modal_merge_lora.py` (volume) before serving — inference/eval/MLX/quant need zero changes. See [README.finetune.md](README.finetune.md).
