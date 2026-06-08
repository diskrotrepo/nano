# Training Plan (at a glance)

The dependency map for a full training run — **what gates what, and what can run at the same time.** For the actual commands and per-step detail, see [README.modal.md](README.modal.md).

Melody and lyrics are mandatory here. The structure pass is optional.

## The graph

```
Upload ──► Prepare ──► Tokenize ──► Melody ──► Pack ──► Train
                          │                      ▲        ▲
                          └──────────────────────┘        │
   (after Prepare, in parallel with Tokenize): ───────────┤
       Auto-tag   (corpus only) ─────────► tags.json ─────┤
       Transcribe (corpus only) ─────────► lyrics/   ─────┤
       Structure  (corpus only, OPTIONAL) ► structure/ ───┘
```

## Critical path

The longest mandatory chain — everything else fits inside its shadow:

1. **Upload** — MP3s → `nano-corpus`.
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
| Train | 8× H100 | Pack + tags.json + lyrics/ (+ structure/) | — |

Auto-tag, Transcribe, and Structure read the **corpus only**, so they can all kick off the moment Prepare finishes and run concurrently with the Tokenize → Melody → Pack chain. Don't run them serially — you'd idle expensive GPU time.

## Phase view (with commands)

All Modal steps are detached (`--detach`) — they return immediately and run in the cloud, so you can launch the Wave 1 fan-out back-to-back without waiting. Monitor any of them with `modal app logs <app-name> -f`. First-time setup (volumes, token, HF secret) is in [README.modal.md](README.modal.md).

**Wave 0 — serial gate** (each waits on the previous):

```bash
# Upload
modal volume create nano-corpus
modal volume put nano-corpus /path/to/mp3s/ /

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

**Wave 2 — join** (starts only once Pack, `tags.json`, and `lyrics/` are all done — plus `structure/` if you ran it):

```bash
modal volume create nano-ckpts
modal run --detach diskrot/modal_train.py --n-gpus 8
```

Pull the trained checkpoint when it's done:

```bash
modal volume get nano-ckpts /v7_1500m/best.pt ./checkpoints/latest.pt --force
```

## Mandatory vs optional

- **Mandatory:** Upload, Prepare, Tokenize, Melody, Pack, Auto-tag (tags), Transcribe (lyrics), Train.
- **Optional:** Structure. Without it the lyric stream just falls back to `<no_section>` markers — no other change.
