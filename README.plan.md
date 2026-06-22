# Training Plan (at a glance)

The dependency map for a full training run — **what gates what, and what can run at the same time.** For the actual commands and per-step detail, see [README.modal.md](README.modal.md).

Melody and lyrics are mandatory here. The structure and key-detect passes are
optional; phonemize is cheap and strongly recommended (it removes a real
training-throughput bottleneck).

## The graph

```
Upload ──► Prepare ──► Tokenize ──► Melody ──► Pack ──┬──────────► Train
                          │                      ▲    │              ▲
                          └──────────────────────┘    └► Key-detect ─┤
   (after Prepare, in parallel with Tokenize): ──────────────────────┤
       Auto-tag   (corpus only) ─────────► tags.json ────────────────┤
       Transcribe (corpus only) ► lyrics/ ► Filter ► Phonemize ──────┤
       Structure  (corpus only, OPTIONAL) ► structure/ ──────────────┘
```

## Critical path

The longest mandatory chain — everything else fits inside its shadow:

1. **Upload** — MP3s → R2 `nano-audio` bucket (under `waves/wave_<id>/`).
2. **Prepare** — validate / dedupe / drop too-short and too-long files. Gates everything downstream.
3. **Tokenize** — MP3 → DAC tokens (`.pt`).
4. **Melody** — chroma sidecar (`.mel.npy`). Waits on Tokenize (needs the token frame count to align).
5. **Pack** — fold tokens + melody into the sharded mmap layout. Waits on Tokenize **and** Melody.
6. **Train** — the join point. Waits on Pack **and** every conditioning pass below.

## What runs in parallel

| Stage | Hardware | Waits on | Can run alongside |
|---|---|---|---|
| Prepare | CPU | Upload | — |
| Tokenize | L4 GPU | Prepare | Auto-tag, Transcribe, Structure |
| Melody | CPU | Tokenize | Auto-tag, Transcribe, Structure |
| Pack | CPU | Tokenize + Melody | Auto-tag, Transcribe, Structure |
| Auto-tag | L4 GPU | Prepare (corpus only) | the entire Tokenize → Melody → Pack chain |
| Transcribe | L4 GPU | Prepare (corpus only) | the entire Tokenize → Melody → Pack chain |
| Structure | L4 GPU | Prepare (corpus only) | the entire Tokenize → Melody → Pack chain |
| Key-detect | CPU | Pack (reads the packed chroma) | Auto-tag, Transcribe, Structure, Phonemize |
| Filter-lyrics | CPU | Transcribe **fully complete** (rewrites lyrics/ in place) | the entire token chain, Key-detect |
| Phonemize | CPU | Filter-lyrics (reads the cleaned lyrics/) | the entire token chain, Key-detect |
| Train | 8× H100 | Pack + tags.json + lyrics/ + phonemes/ (+ structure/ + keys.json) | — |

Auto-tag, Transcribe, and Structure read the **corpus only**, so they can all kick off the moment Prepare finishes and run concurrently with the Tokenize → Melody → Pack chain. Don't run them serially — you'd idle expensive GPU time.

## Phase view (with commands)

All Modal steps are detached (`--detach`) — they return immediately and run in the cloud, so you can launch the Wave 1 fan-out back-to-back without waiting. Monitor any of them with `modal app logs <app-name> -f`. First-time setup (volumes, token, HF secret) is in [README.modal.md](README.modal.md).

**Wave 0 — serial gate** (each waits on the previous):

```bash
# Upload to R2 (see README.waves.md for the one-time bucket + r2-creds setup)
rclone copy /path/to/mp3s/ r2:nano-audio/waves/wave_0/

# Prepare — dry-run first, then --apply
modal run --detach diskrot/modal_prepare.py
modal run --detach diskrot/modal_prepare.py --apply
```

**Wave 1 — fan out (launch all of these after Prepare):**

```bash
# Token chain: Tokenize → Melody → Pack (run in this order; each waits on the prior)
modal run --detach diskrot/modal_tokenize.py
modal run --detach diskrot/modal_melody.py        # waits on Tokenize
modal run --detach diskrot/modal_pack_cache.py    # waits on Tokenize + Melody; auto-detects the chroma sidecars

# Conditioning passes — corpus-only, run concurrently with the whole token chain above
modal run --detach diskrot/modal_auto_tag.py      # → tags.json
modal run --detach diskrot/modal_transcribe.py    # → lyrics/
modal run --detach diskrot/modal_structure.py     # → structure/  (optional)
```

**Wave 1.5 — cheap CPU derivations** (each waits only on its input, runs alongside everything else):

```bash
modal run --detach diskrot/modal_key_detect.py    # → keys.json   (waits on Pack; optional)
modal run --detach diskrot/modal_filter_lyrics.py --apply  # nulls Whisper-hallucinated captions in lyrics/ (recommended; dry-run without --apply)
modal run --detach diskrot/modal_phonemize.py     # → phonemes/   (waits on Filter-lyrics; recommended)
```

⚠️ Filter-lyrics is the one Wave 1.5 step with a hard ordering constraint: it must wait until the Transcribe fleet has **fully finished** — the transcribe orchestrator holds shard contents in memory and its next flush would clobber concurrent edits. Run it, then Phonemize, as a serial pair after Transcribe drains.

**Wave 2 — join** (starts only once Pack, `tags.json`, `lyrics/`, and `phonemes/` are done — plus `structure/`/`keys.json` if you ran them):

```bash
modal volume create nano-ckpts
modal run --detach diskrot/modal_train.py --n-gpus 4
```

Pull the trained checkpoint when it's done:

```bash
modal volume get nano-ckpts /v8_sing/best.pt ./checkpoints/latest.pt --force
```

## Mandatory vs optional

- **Mandatory:** Upload, Prepare, Tokenize, Melody, Pack, Auto-tag (tags), Transcribe (lyrics), Train.
- **Recommended:** Filter-lyrics and Phonemize. Filter-lyrics nulls Whisper-hallucinated captions over instrumentals ("Thank you." etc. — ~24% of with-words entries in the 2026-06 sweep) so they train as `<instrumental>` instead of vocal songs with garbage words. Phonemize: without it the lyric stream is identical, but every DataLoader cache miss runs live g2p (~20–200 ms/song on OOV-heavy transcripts) — at corpus scale that can starve the 8×H100 step. A few CPU dollars buys both back.
- **Optional:** Structure (falls back to `<no_section>` markers), Key-detect (falls back to `<unknown_key>`). No other change without them.
