"""Crop-supervision audit: does a training crop actually carry sung words?

Quantifies the two data-path problems the v8_sing4 vocal fixes target, by
Monte-Carlo sampling crops over the packed corpus on the nano-tokens volume
(read-only, one CPU container):

  1. Zero-word <vocals> crops. A 60s crop of a vocal-ready song is sampled
     uniformly within the song, so it often lands on the instrumental
     intro/solo/outro and contains NO transcribed words — yet still trains with
     the <vocals> header, starving the lyric cross-attention. Reported as the
     share of vocal-song crops with zero in-window words, UNIFORM vs the new
     word-BIASED sampler (diskrot/dataset.py _choose_crop_start), so you can see
     the fix move the number.

  2. Truncated lyric streams. The phoneme stream is capped at max_lyric_len and
     truncated whole-word when it overflows (lyric_encoder.append_unit_capped),
     silently dropping the tail of dense crops. Reported as the share of
     word-bearing crops whose (uncapped) stream length exceeds 256 vs 512 — i.e.
     what the old cap clipped and what the new 512 cap keeps.

The crop-start math mirrors diskrot/dataset.py::_choose_crop_start and the
stream-length math mirrors _get_segment_lyric_ids (header = BOS + 5 markers; each
in-window word adds one word-boundary separator + its phonemes). Keep them in
sync if those change. Inline section markers are ignored, so the length is a
slight UNDER-estimate of truncation (truncation is at least what's reported).

Read-only; safe to run any time. Run:
  modal run scripts/crop_supervision_audit.py
  modal run scripts/crop_supervision_audit.py --n-samples 50000 --seg-seconds 60
"""
from __future__ import annotations

import modal

app = modal.App("nano-crop-supervision-audit")

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy>=1.26")

tokens_vol = modal.Volume.from_name("nano-tokens")


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
)
def audit(n_samples: int = 30_000, seg_seconds: float = 60.0):
    import json
    import random
    from pathlib import Path

    import numpy as np

    random.seed(0)

    # Constants mirrored from the production code — keep in sync.
    FRAME_RATE_HZ = 86                  # model/codec.py
    HEADER_LEN = 6                      # BOS + dense 5-marker header (dataset.py)
    BIAS_P = 0.9                        # dataset._VOCAL_CROP_BIAS_PROB
    seg_frames = int(seg_seconds * FRAME_RATE_HZ)

    def is_valid_word(w):
        return (
            isinstance(w, dict)
            and isinstance(w.get("start"), (int, float)) and not isinstance(w["start"], bool)
            and isinstance(w.get("end"), (int, float)) and not isinstance(w["end"], bool)
        )

    def load_shards(d, pattern):
        merged = {}
        for shard in sorted(Path(d).glob(pattern)):
            merged.update(json.loads(shard.read_text()))
        return merged

    # --- per-song frame counts from the packed offsets (cumulative frames) ---
    packed = Path("/tokens/packed")
    frames: dict[str, int] = {}
    index = json.loads((packed / "packed_index.json").read_text())
    for sh in index["shards"]:
        meta = json.loads((packed / f"packed_{sh['shard_id']:03d}.json").read_text())
        names, offs = meta["names"], meta.get("offsets")
        if isinstance(offs, list) and len(offs) == len(names) + 1:
            for i, n in enumerate(names):
                frames[n] = offs[i + 1] - offs[i]
    print(f"packed (trainable) songs:        {len(frames)}")

    lyrics = load_shards("/tokens/lyrics", "lyrics_*.json")
    phon = load_shards("/tokens/phonemes", "phonemes_*.json")  # {name: list[list[int]]}

    # Vocal-eligible pool: has >=1 valid word AND long enough to fill a crop
    # (the dataset split index drops songs shorter than segment_frames).
    pool: list[str] = []
    words_by: dict[str, list] = {}
    too_short = 0
    for n, entry in lyrics.items():
        if n not in frames:
            continue
        if not isinstance(entry, dict):
            continue
        ws = [w for w in (entry.get("words") or []) if is_valid_word(w)]
        if not ws:
            continue
        if frames[n] < seg_frames:
            too_short += 1
            continue
        pool.append(n)
        words_by[n] = ws

    vocal_share = len(pool) / max(len(frames), 1)
    print(f"vocal-ready & crop-fitting songs:{len(pool):>8}  ({vocal_share:.1%} of trainable)")
    if too_short:
        print(f"  (excluded {too_short} vocal songs shorter than the {seg_seconds:g}s crop)")
    if not pool:
        print("no vocal-ready songs that fit a crop — nothing to sample.")
        return

    def in_window_indices(ws, start):
        s = start / FRAME_RATE_HZ
        e = (start + seg_frames) / FRAME_RATE_HZ
        return [i for i, w in enumerate(ws) if w["end"] > s and w["start"] < e]

    def uniform_start(max_start):
        return 0 if max_start <= 0 else random.randint(0, max_start)

    def biased_start(ws, max_start):
        # Byte-for-byte mirror of dataset._choose_crop_start's biased branch.
        if max_start <= 0:
            return 0
        if random.random() < BIAS_P:
            w = random.choice(ws)
            seg_sec = seg_frames / FRAME_RATE_HZ
            lo = max(0.0, float(w["end"]) - seg_sec)
            hi = min(max_start / FRAME_RATE_HZ, float(w["start"]))
            if lo <= hi:
                st = int(round(random.uniform(lo, hi) * FRAME_RATE_HZ))
                return max(0, min(st, max_start))
        return random.randint(0, max_start)

    # --- Monte-Carlo over uniformly-chosen vocal songs (matches the dataset's
    # per-index uniform song sampling) ---
    zero_uniform = zero_biased = 0
    trunc_lengths: list[int] = []   # uncapped stream length under the biased sampler
    store_hits = 0
    for _ in range(n_samples):
        n = random.choice(pool)
        ws = words_by[n]
        max_start = frames[n] - seg_frames

        if not in_window_indices(ws, uniform_start(max_start)):
            zero_uniform += 1
        bstart = biased_start(ws, max_start)
        bidx = in_window_indices(ws, bstart)
        if not bidx:
            zero_biased += 1

        # Truncation: needs per-word phoneme counts from the store, aligned to
        # the word list (stale/missing entry -> skip, counted as a store miss).
        groups = phon.get(n)
        if isinstance(groups, list) and len(groups) == len(ws) and bidx:
            store_hits += 1
            length = HEADER_LEN + sum(1 + len(groups[i]) for i in bidx)  # +1 WB/word
            trunc_lengths.append(length)

    print(f"\nsamples: {n_samples}")
    print("zero-word <vocals> crops (lower is better):")
    print(f"  uniform sampler:   {zero_uniform / n_samples:.1%}")
    print(f"  biased  sampler:   {zero_biased / n_samples:.1%}   "
          f"(bias_vocal_crops, p={BIAS_P})")

    if trunc_lengths:
        L = np.array(trunc_lengths)
        gt256 = int((L > 256).sum()) / len(L)
        gt512 = int((L > 512).sum()) / len(L)
        print(f"\nlyric-stream length on word-bearing biased crops "
              f"(store-aligned n={len(L)}):")
        print(f"  mean {L.mean():.0f}  p50 {np.percentile(L, 50):.0f}  "
              f"p95 {np.percentile(L, 95):.0f}  max {L.max()}")
        print(f"  truncated at cap 256 (old): {gt256:.1%}")
        print(f"  truncated at cap 512 (new): {gt512:.1%}")
        print(f"  -> the 256->512 bump rescues ~{gt256 - gt512:.1%} of word-bearing crops")
    else:
        print("\n(no phoneme-store-aligned word-bearing crops sampled — "
              "run phonemize, or raise --n-samples)")


@app.local_entrypoint()
def main(n_samples: int = 30_000, seg_seconds: float = 60.0):
    audit.remote(n_samples=n_samples, seg_seconds=seg_seconds)
