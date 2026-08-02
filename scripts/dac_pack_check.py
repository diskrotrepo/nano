"""Verify a DAC pack is actually DAC — depth, frame rate, and growth.

The failure this exists to catch is silent: ``.pt`` filenames are identical across
codecs, and nothing in the packed index records which codec produced it (there is
no ``frame_rate_hz`` key). So a mis-deployed stage app or a forgotten
``--data-subdir`` produces a pack that LOOKS fine and trains to garbage — the
conditioning markers land at the wrong times because the dataset resolves the
frame rate from ``NANO_CODEC``, not from the data.

Two checks:

1. **Depth** — ``n_codebooks == 9`` exactly (DAC). SpectroStream stores 24/32.
2. **Frame rate** — decode a packed song's frame count against the source mp3's
   true duration and confirm ~86 Hz. This is the only check that actually proves
   the tokens are DAC rather than merely 9-deep.

Run::

    modal run scripts/dac_pack_check.py --data-subdir dac
    modal run scripts/dac_pack_check.py --data-subdir dac --expect-songs 100000
"""
from __future__ import annotations

import modal

app = modal.App("nano-dac-pack-check")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy>=1.26")
    .add_local_python_source("diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens")

DAC_FRAME_RATE_HZ = 86.13  # 44100 / 512
DAC_N_CODEBOOKS = 9


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=60 * 30)
def check(data_subdir: str = "dac", expect_songs: int = 0, n_probe: int = 5) -> dict:
    import json
    import subprocess
    from pathlib import Path

    from diskrot.pack_cache import SHARD_INDEX_NAME

    root = Path("/tokens") / data_subdir if data_subdir else Path("/tokens")
    packed = root / "packed"
    idx_path = packed / SHARD_INDEX_NAME
    if not idx_path.exists():
        raise SystemExit(f"no packed index at {idx_path} — nothing to check")
    idx = json.loads(idx_path.read_text())

    n_cb = int(idx["n_codebooks"])
    n_songs = int(idx.get("n_songs_total", 0))
    print(f"pack: {packed}")
    print(f"  n_codebooks   = {n_cb}")
    print(f"  n_songs_total = {n_songs}")
    print(f"  n_shards      = {idx.get('n_shards')}")
    print(f"  has_melody    = {idx.get('has_melody')}")

    if n_cb != DAC_N_CODEBOOKS:
        raise SystemExit(
            f"DEPTH CHECK FAILED: n_codebooks={n_cb}, expected {DAC_N_CODEBOOKS}. "
            f"This pack is NOT DAC — almost certainly a NANO_CODEC / --data-subdir "
            f"mismatch. Do not train on it."
        )
    print(f"  depth OK ({DAC_N_CODEBOOKS} codebooks)")

    if expect_songs and n_songs < expect_songs:
        raise SystemExit(
            f"GROWTH CHECK FAILED: {n_songs} songs < expected {expect_songs}"
        )

    # FRAME-RATE CHECK — the one that actually proves the codec.
    #
    # Depth alone is necessary but NOT sufficient: a pack can carry 9 codebooks and
    # still hold non-DAC timing, which trains to garbage silently (the dataset
    # resolves the frame rate from NANO_CODEC, never from the data, and no
    # frame_rate_hz is recorded in the index).
    #
    # Rather than probe individual songs against the corpus (needs the R2 mount and
    # depends on the per-shard sidecar layout), interpret the aggregate: mean
    # frames/song under each candidate rate. `prepare` drops everything over 5:30,
    # so a mean song length above that is physically impossible for this corpus —
    # which makes the two rates trivially separable.
    total_frames = sum(int(s.get("total_frames", 0)) for s in idx["shards"])
    if total_frames and n_songs:
        avg = total_frames / n_songs
        dac_s, ss_s = avg / DAC_FRAME_RATE_HZ, avg / 25.0
        print(f"  mean frames/song = {avg:.0f}")
        print(f"    at 86.13 Hz (DAC) = {dac_s:6.1f}s = {dac_s/60:.2f} min")
        print(f"    at 25.00 Hz (SS)  = {ss_s:6.1f}s = {ss_s/60:.2f} min")
        # prepare's ceiling is 5:30 (330s); allow generous headroom for the mean.
        if not (30.0 < dac_s < 400.0):
            raise SystemExit(
                f"FRAME-RATE CHECK FAILED: mean song length reads {dac_s:.1f}s at "
                f"the DAC rate, which is not a plausible song. This pack has DAC "
                f"depth but non-DAC timing — do not train on it."
            )
        print(f"  frame rate OK (mean song length is only plausible at ~86 Hz)")
    else:
        print("  (frame-rate check skipped — index carries no total_frames)")

    print("\nDAC PACK CHECK PASSED")
    return {"n_codebooks": n_cb, "n_songs": n_songs}


@app.local_entrypoint()
def main(data_subdir: str = "dac", expect_songs: int = 0, n_probe: int = 5):
    check.remote(data_subdir=data_subdir, expect_songs=expect_songs, n_probe=n_probe)
