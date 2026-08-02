"""Quick CPU audit: corpus song-duration distribution + clip-length survival.

Answers the v9 "full-song" clip-length decision: the dataset drops any song
shorter than the training clip, so for each candidate clip length we report how
many songs survive (train) vs are dropped. Durations are real seconds (codec-
independent), so this is valid for the 25 Hz SpectroStream run even though it
reads the existing 86 Hz DAC pack.

Reads only the pack shard METAS (names + offsets) on nano-tokens — no token data,
no torch — so it's fast and ~free.

    modal run diskrot/modal_duration_audit.py
"""
import modal

app = modal.App("nano-duration-audit")

image = modal.Image.debian_slim(python_version="3.12")
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)

# Candidate training clip lengths (seconds) to evaluate survival at.
CANDIDATES = [60, 90, 120, 150, 180, 210, 240, 270, 300]
FRAME_RATE_HZ = 86  # the EXISTING DAC pack; durations come out in real seconds


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=60 * 20)
def audit() -> None:
    import json
    from pathlib import Path

    packed = Path("/tokens/packed")
    index_path = packed / "packed_index.json"
    if not index_path.exists():
        print(f"no pack index at {index_path}")
        return
    index = json.loads(index_path.read_text())

    durations: list[float] = []
    for shard_entry in index["shards"]:
        meta = json.loads((packed / f"packed_{shard_entry['shard_id']:03d}.json").read_text())
        names = meta["names"]
        offsets = meta.get("offsets")
        if isinstance(offsets, list) and len(offsets) == len(names) + 1:
            durations.extend(
                (offsets[i + 1] - offsets[i]) / FRAME_RATE_HZ for i in range(len(names))
            )

    n = len(durations)
    if not n:
        print("no durations found (offsets missing from shard metas)")
        return
    durations.sort()

    def pct(p: float) -> float:
        return durations[min(n - 1, int(p / 100.0 * n))]

    total_hours = sum(durations) / 3600.0
    print(f"\n=== corpus duration distribution ({n:,} packed songs, {total_hours:,.0f} h) ===")
    print(f"min {durations[0]:.0f}s  p10 {pct(10):.0f}s  p25 {pct(25):.0f}s  "
          f"median {pct(50):.0f}s  p75 {pct(75):.0f}s  p90 {pct(90):.0f}s  "
          f"max {durations[-1]:.0f}s")
    print(f"mean {sum(durations)/n:.0f}s")

    print(f"\n=== clip-length survival (songs >= clip length are trained) ===")
    print(f"{'clip':>6} {'survive':>10} {'%kept':>7} {'dropped':>10} {'%lost':>7}  train-hours")
    for L in CANDIDATES:
        survive = sum(1 for d in durations if d >= L)
        kept = 100.0 * survive / n
        # training hours = survivors * clip length (each survivor yields L-second crops)
        train_h = survive * L / 3600.0
        print(f"{L:>5}s {survive:>10,} {kept:>6.1f}% {n - survive:>10,} "
              f"{100 - kept:>6.1f}% {train_h:>11,.0f}")
    print("\n(survive = songs the dataset keeps at that clip length; the rest are "
          "dropped unless we add short-song padding.)", flush=True)


@app.local_entrypoint()
def main() -> None:
    audit.remote()
