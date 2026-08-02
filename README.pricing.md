# Pricing for your corpus (322,530 songs)

What a full Modal run costs **at your actual corpus size** — 322,530 MP3s in the R2 `nano-audio` bucket (as of 2026-06-09). For the dependency graph see [README.plan.md](README.plan.md); for per-step detail see [README.modal.md](README.modal.md).

> **Point-in-time snapshot.** The corpus count (322,530) and the Modal rate table below are dated **2026-06-09** — the corpus has since grown well past this, and the pipeline is now the v9/SpectroStream path. Treat the corpus size and rates as a fixed snapshot; the per-stage *facts* (which GPU, which model, what's optional) below track the **current v9 code**. Rescale linearly for a bigger corpus (see the [Rescale](#rescale-the-estimate) section).

> **Read this if you only read one thing:** a full *mandatory* run is roughly **$4,500–5,600** (transcribe is now cheaper than shown — see below), dominated by **transcribe** and **train (~$1,000–1,650)**. The optional **structure** pass adds **~$1,500**. The optional **stems** pass (default OFF in v9) is now the single most expensive line at **~$7.6k @ 50% sampling / ~$15k @ 100%**. Skip transcribe entirely if you don't need lyric conditioning and the data-prep side drops to a few hundred dollars.

## Current Modal rates

Fetched from [modal.com/pricing](https://modal.com/pricing) on 2026-06-09. Modal bills **per second of container uptime** (reserved time, not just active compute).

| Resource | Per second | Per hour |
|---|---|---|
| H100 | $0.001097 | $3.95 |
| A100 80GB | $0.000694 | $2.50 |
| L4 | $0.000222 | $0.80 |
| T4 | $0.000164 | $0.59 |
| CPU (physical core) | $0.0000131 | $0.047 |
| Memory | $0.00000222 / GiB | $0.008 / GiB |

Starter plan includes **$30/month** free credits (Team: $100/month).

> **These rates differ from [README.modal.md](README.modal.md)'s cost table**, which was written against older rates (L4 $0.30/hr, H100 $5.92/hr). L4 is now **~2.7× more expensive** and H100 **~33% cheaper** — so GPU-bound prep steps (tokenize, transcribe, structure) cost more than that table implies, while training costs less.

## What it costs for 322,530 songs

Estimates use **reserved container-time × current rate**, which is what Modal actually bills. Per-stage throughput is derived from the documented per-1,000-song wall-clock and container counts in [README.modal.md](README.modal.md). Treat these as order-of-magnitude — real cost moves with packing efficiency, container startup, and the ~5% of files that OOM-retry on L4.

| Stage | Hardware | Throughput basis | Compute (full corpus) | Est. cost |
|---|---|---|---|---|
| Upload | — | ~2.5 TB of MP3s | bandwidth-bound | egress-in is free; **storage billed separately** (see below) |
| Prepare *(+quality-gate)* | CPU × 20 | ~20 s / 1k | ~36 core-hr | **~$2** |
| Audio-dedup *(recommended)* | CPU × 50 | chromaprint fingerprint + one grouping pass | ~tens of core-hr | **~$1–2** |
| Tokenize | L4 × 50 | ~1.5 min / 1k | ~400 GPU-hr | **~$320** |
| Melody | CPU × 50 (2 core) | ~2 min / 1k (est.) | ~1,100 core-hr | **~$50–80** |
| **Stems** *(optional, default OFF)* | GPU (Demucs + SpectroStream codec) × 50 | GPU-bound, most expensive prep line | — | **~$7.6k @ 50% sampling / ~$15k @ 100%** (code figure) |
| Pack | CPU (8 core) | one-shot, ~3–5 h | ~30–40 core-hr | **~$2–5** |
| Auto-tag | A100 (Qwen2-Audio-7B) | ~1 whole-song pass / song | — | **re-measure** (was ~$45 on L4×20 BART; A100 is pricier/hr) |
| Transcribe | L4 × 50 | Whisper large-v3-turbo + VAD (Demucs removed) | — | **re-measure** (was ~$3,000 with Demucs; Demucs was ~45–60% of it) |
| Align-lyrics *(optional)* | L4 | CTC forced alignment (MMS_FA), refined shards only | ~small | **~$1–few** |
| Tempo *(optional)* | CPU | dense `<tempo_*>` markers, full coverage | one-shot | **~$1** |
| Train | B200 × 4 DDP | flat — 30–50 h wall | ~120–200 B200-hr | **~$1,000–1,650** |
| **Structure** *(optional)* | L4 × 50 | ~21 GPU-s / song (measured) | ~1,900 GPU-hr | **~$1,500** |
| Key-detect *(optional)* | CPU (4 core) | one-shot mmap sweep of chroma, ~0.5–2 h | ~2–8 core-hr | **~$1** |
| Phonemize *(recommended)* | CPU (16 core) | ~20–200 ms/song g2p, one-shot | ~5–20 core-hr | **~$1** |

### Totals

| Scenario | Est. cost |
|---|---|
| **Mandatory run** (upload → prepare → tokenize → melody → pack → auto-tag → transcribe → train) | **~$4,500–5,600** |
| **+ Structure** (optional section markers) | **~$6,000–7,100** |
| **+ Stems** (optional `/addstem` path, default OFF) | **+~$7.6k @ 50% / +~$15k @ 100%** |
| **Lyric-free run** (skip transcribe *and* structure) | **~$1,500–2,150** |

> The mandatory total leans on the **stale ~$3,000 transcribe line** (Demucs-dominated) and the **~$45 L4 auto-tag line** — both changed in v9 (transcribe went Demucs-free → cheaper; auto-tag moved to A100 Qwen2-Audio → different cost). Both **need re-measurement**, so the mandatory total should skew *lower* on the transcribe side once re-measured. Do not treat $4,500–5,600 as a v9 quote.

Training has early stopping (`patience=20`, on the EMA val loss), so a real run commonly finishes 30–50% sooner than the 400k-step worst case — knock ~$300–800 off the train line when the val loss plateaus. Training cost is **flat in corpus size** (driven by `steps` + model size); every other line scales linearly with song count. The train stage is now **4×B200 DDP** (was 8×H100); per-GPU B200 is ~2× an H100 but there are half as many, so the flat-cost framing holds.

## Where the money goes

The optional stages dominate the bill, and every one of them is skippable:

- **Stems (~$7.6k @ 50% sampling / ~$15k @ 100%)** — GPU Demucs 4-stem separation + SpectroStream codec-tokenization per song, for the `/addstem` generative-stem path. This is now **the single most expensive stage in the whole pipeline**, prep or train (the figure is an authoritative code comment in `diskrot/modal_train.py`'s `DEFAULTS`). It is **OFF by default in v9** (`use_stem_conditioning=False`, and at <100% sampling a stem-add batch only fires when every song in the per-rank batch has stems — ~0.4% of batches — so the feature barely trains anyway). Only enable it on a fresh start if you're committing to the `/addstem` feature *and* ~100% stem coverage.
- **Transcribe (needs re-measurement — was ~$3,000)** — Whisper large-v3-turbo + VAD per song. **v9's transcribe is Demucs-free**: the Demucs vocal-isolation pass (which was ~45–60% of the old ~$3,000 estimate, and the dominant cost) was removed as a no-op-to-worse ASR input, and vocal gender now comes from the audio-LLM captioner. So the ~$3,000 figure is **stale and too high** — the stage is meaningfully cheaper now but has not been re-measured on the current path. Only needed if you want the model to **sing intelligible words**; skip it (and the `lyrics/` dir) to drop lyric conditioning.
- **Structure (~$1,500)** — allin1 (Demucs + a joint beat/segment model). Fully optional: without it the lyric stream just falls back to `<no_section>` markers, no other change. Measured at **~2.4 songs/sec aggregate across 50 L4s (~21 GPU-s/song, ~21.5 h wall for 182k songs)** from a live run on 2026-06-09 — still a four-figure line item; only run it if section-aware generation matters to you.
- **Auto-tag (needs re-measurement)** — the default captioner is now the **audio-LLM Qwen2-Audio-7B on A100** (one whole-song pass producing a rich multi-facet caption + vocal-gender tag + per-stem lines), replacing the old **LP-MusicCaps BART on L4×20** (~$45). A100 is pricier per hour than L4 but it's a single whole-song pass rather than a windowed sweep, so the net cost has changed and hasn't been re-measured. (`NANO_CAPTIONER=bart` still selects the cheaper legacy L4 path.)

Everything mandatory-and-cheap combined (prepare + audio-dedup + tokenize + melody + pack + key-detect + phonemize) is **~$420**, and training is **~$1,000–1,650** regardless of corpus size. The two newest CPU passes (key-detect → `keys.json` for the `<key_*>` marker, phonemize → `phonemes/` for the pre-phonemized lyric cache) are rounding errors (~$1 each) — phonemize in particular *pays for itself immediately* by keeping the training step from stalling on in-DataLoader g2p ($3.95/hr-era GPUs make even a few percent of dataloader stall worth ~$10+/day).

## Storage (recurring, not in the totals above)

The corpus lives in Cloudflare R2 and its derived artifacts on Modal volumes; both bill **monthly** while they exist — separate from the one-shot compute above:

- `nano-audio` (R2 bucket) — ~2.5 TB of MP3s (322,530 files ≤ 5:30). R2 storage is billed by Cloudflare, not Modal.
- `nano-tokens` — codec tokens + packed shards (incl. `.mel.bin` + `.stem.bin` sidecars) + tags + lyrics + structure + keys.json + tempo.json + phonemes (~100–155 GB; phonemes/ adds ~1–2 GB, keys.json ~15 MB).
- `nano-melody` — chroma sidecars (~140 GB).
- `nano-stems` — stem-token sidecars (`<name>.stems.npy`, `[4,depth,T]` int16; its own volume for the same inode reason as nano-melody). Only populated when the optional stems stage runs (default OFF in v9).
- `nano-ckpts` — checkpoints (every ~5k steps; tens of GB).

> **Codec caveat.** The sizes above are DAC-era figures (44.1 kHz mono, 9 codebooks, 86 Hz). v9 trains on **SpectroStream** (stereo, 48 kHz, 25 Hz frame rate, 24 codebooks), so per-song token and chroma footprints differ — the lower frame rate cuts frame count but the higher codebook depth and stereo add it back. Treat the GB figures as an order-of-magnitude DAC snapshot, not a SpectroStream measurement.

Check the storage rate on [modal.com/pricing](https://modal.com/pricing) and multiply by ~2.8 TB. Delete the corpus volume once tokenize + melody are packed if you won't re-tokenize — the packed shards are all training needs.

## Rescale the estimate

Every prep line is linear in song count. To re-price for a different corpus size `N`:

```
stage_cost(N) ≈ stage_cost(322,530) × N / 322,530
```

Or from scratch for any stage: `cost = (per-1k wall minutes × containers / 60) × (N / 1000) × hourly_rate`.

## Caveats

- **Reserved-time billing.** Costs assume containers stay busy for their wall-clock slice. With ragged input sizes, the tail of a `.map()` fan-out can leave a few containers idle-but-alive — real cost can run modestly higher than the table.
- **L4 OOM retries.** ~5% of files exceed L4's 22 GiB on the upper-bound lengths and auto-retry; switching tokenize to A100 for 100% completion costs ~3× the per-GPU-hour rate.
- **Rates move.** Modal has changed GPU prices before (L4 nearly tripled vs this repo's older docs). Re-fetch [modal.com/pricing](https://modal.com/pricing) before committing to a budget.
- **Estimates, not quotes.** Structure is measured from a live run (2026-06-09); melody throughput is estimated (not measured per-1k in the repo); the rest are derived from documented per-1,000-song numbers.
