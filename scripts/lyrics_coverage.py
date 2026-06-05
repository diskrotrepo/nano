"""Diagnostic: how much of the trainable corpus actually has usable lyrics?

Reads the sharded ``lyrics/`` dir on the nano-tokens volume and the packed
index (the set of songs the trainer actually sees) and reports:

  - trainable songs (packed) vs lyrics entries present
  - has-words / instrumental(null) / empty-words / missing breakdown
  - with-lyrics fraction of the trainable corpus
  - word-count distribution among lyric-bearing songs

Caveat: transcription forces ``language="en"`` (transcribe_lyrics.py), so
non-English vocals are mis-transcribed into noise. There's no cheap reliable
language detector here, but the word-count buckets surface the tell — a heavy
tail of 1-5 word "songs" is usually instrumental bleed or garbled output.

Run:  modal run scripts/lyrics_coverage.py
"""
from __future__ import annotations

import modal

app = modal.App("nano-lyrics-coverage")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26")
    .add_local_python_source("model", "diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens")


def _summarize(lyrics: dict, trainable: set[str] | None) -> None:
    """Print coverage stats. Pure logic so it can run on any loaded dict."""
    has_words = [k for k, v in lyrics.items() if isinstance(v, dict) and v.get("words")]
    empty_words = [k for k, v in lyrics.items() if isinstance(v, dict) and not v.get("words")]
    instrumental = [k for k, v in lyrics.items() if v is None]

    print(f"lyrics entries:      {len(lyrics)}")
    print(f"  has words:         {len(has_words)}")
    print(f"  empty words:       {len(empty_words)}")
    print(f"  instrumental(null):{len(instrumental)}")

    if trainable is not None:
        present = set(lyrics)
        missing = trainable - present
        usable = set(has_words) & trainable
        print(f"\ntrainable songs (packed): {len(trainable)}")
        print(f"  missing from lyrics:    {len(missing)}")
        print(f"  with usable lyrics:     {len(usable)}  "
              f"({len(usable) / max(len(trainable), 1):.1%} of trainable)")

    # Word-count distribution among lyric-bearing songs.
    counts = sorted(len(lyrics[k]["words"]) for k in has_words)
    if counts:
        import numpy as np
        arr = np.array(counts)
        buckets = [(0, 5), (6, 20), (21, 50), (51, 150), (151, 400), (401, 10**9)]
        print("\nword-count distribution (lyric-bearing songs):")
        for lo, hi in buckets:
            n = int(((arr >= lo) & (arr <= hi)).sum())
            label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print(f"  {label:>8} words: {n:>7}  ({n / len(arr):.1%})")
        print(f"  mean {arr.mean():.0f}  median {int(np.median(arr))}  "
              f"max {arr.max()}")
        tiny = int((arr <= 5).sum())
        print(f"\n  ⚠ {tiny} songs ({tiny / len(arr):.1%}) have ≤5 words — "
              f"likely instrumental bleed or garbled non-English transcription.")


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=60 * 20)
def coverage():
    from pathlib import Path
    from diskrot.transcribe_lyrics import load_lyrics_shards

    lyrics = load_lyrics_shards("/tokens/lyrics")
    if not lyrics:
        raise SystemExit("no lyrics found at /tokens/lyrics")

    trainable: set[str] | None = None
    packed = Path("/tokens/packed")
    if (packed / "packed_index.json").exists():
        from diskrot.pack_cache import iter_all_names
        trainable = {name for _, _, name, _ in iter_all_names(packed)}

    _summarize(lyrics, trainable)


@app.local_entrypoint()
def main():
    coverage.remote()
