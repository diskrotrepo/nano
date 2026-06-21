# Pricing for your corpus (322,530 songs)

What a full Modal run costs **at your actual corpus size** — 322,530 MP3s in the R2 `nano-audio` bucket (as of 2026-06-09). For the dependency graph see [README.plan.md](README.plan.md); for per-step detail see [README.modal.md](README.modal.md).

> **Read this if you only read one thing:** a full *mandatory* run is roughly **$4,500–5,600**, dominated by **transcribe (~$3,000)** and **train (~$1,000–1,650)**. The optional **structure** pass adds **~$1,500**. Skip transcribe entirely if you don't need lyric conditioning and the data-prep side drops to a few hundred dollars.

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
| Prepare | CPU × 20 | ~20 s / 1k | ~36 core-hr | **~$2** |
| Tokenize | L4 × 50 | ~1.5 min / 1k | ~400 GPU-hr | **~$320** |
| Melody | CPU × 50 (2 core) | ~2 min / 1k (est.) | ~1,100 core-hr | **~$50–80** |
| Pack | CPU (8 core) | one-shot, ~3–5 h | ~30–40 core-hr | **~$2–5** |
| Auto-tag | L4 × 20 | ~0.5 min / 1k | ~54 GPU-hr | **~$45** |
| Transcribe | L4 × 50 | ~14 min / 1k | ~3,760 GPU-hr | **~$3,000** |
| Train | H100 × 8 DDP | flat — 30–50 h wall | ~250–420 H100-hr | **~$1,000–1,650** |
| **Structure** *(optional)* | L4 × 50 | ~21 GPU-s / song (measured) | ~1,900 GPU-hr | **~$1,500** |
| Key-detect *(optional)* | CPU (4 core) | one-shot mmap sweep of ~140 GB chroma, ~0.5–2 h | ~2–8 core-hr | **~$1** |
| Phonemize *(recommended)* | CPU (16 core) | ~20–200 ms/song g2p, one-shot | ~5–20 core-hr | **~$1** |

### Totals

| Scenario | Est. cost |
|---|---|
| **Mandatory run** (upload → prepare → tokenize → melody → pack → auto-tag → transcribe → train) | **~$4,500–5,600** |
| **+ Structure** (optional section markers) | **~$6,000–7,100** |
| **Lyric-free run** (skip transcribe *and* structure) | **~$1,500–2,150** |

Training has early stopping (`patience=20`), so a real run commonly finishes 30–50% sooner than the 400k-step worst case — knock ~$300–800 off the train line when the val loss plateaus. Training cost is **flat in corpus size** (driven by `steps` + model size); every other line scales linearly with song count.

## Where the money goes

Two stages are ~85% of the data-prep bill, and both are skippable:

- **Transcribe (~$3,000)** — Demucs vocal isolation + Whisper per song. Only needed if you want the model to **sing intelligible words**. Skip it (and the `lyrics/` dir) and you lose lyric conditioning but save ~$3k.
- **Structure (~$1,500)** — allin1 (Demucs again + a joint beat/segment model). Fully optional: without it the lyric stream just falls back to `<no_section>` markers, no other change. Measured at **~2.4 songs/sec aggregate across 50 L4s (~21 GPU-s/song, ~21.5 h wall for 182k songs)** from a live run on 2026-06-09 — cheaper than first projected, but still a four-figure line item; only run it if section-aware generation matters to you.

Everything else combined (prepare + tokenize + melody + pack + auto-tag + key-detect + phonemize) is **~$420**, and training is **~$1,000–1,650** regardless of corpus size. The two newest passes (key-detect → `keys.json` for the `<key_*>` marker, phonemize → `phonemes/` for the pre-phonemized lyric cache) are CPU-only rounding errors (~$1 each) — phonemize in particular *pays for itself immediately* by keeping the 8×H100 step from stalling on in-DataLoader g2p ($3.95/hr × 8 GPUs makes even a few percent of dataloader stall worth ~$10+/day).

## Storage (recurring, not in the totals above)

The corpus lives in Cloudflare R2 and its derived artifacts on Modal volumes; both bill **monthly** while they exist — separate from the one-shot compute above:

- `nano-audio` (R2 bucket) — ~2.5 TB of MP3s (322,530 files ≤ 5:30). R2 storage is billed by Cloudflare, not Modal.
- `nano-tokens` — DAC tokens + packed shards + tags + lyrics + structure + keys.json + phonemes (~100–155 GB; phonemes/ adds ~1–2 GB, keys.json ~15 MB).
- `nano-melody` — chroma sidecars (~140 GB).
- `nano-ckpts` — checkpoints (every ~5k steps; tens of GB).

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
