"""Modal entrypoint: null Whisper-hallucinated entries in /tokens/lyrics.

Thin CPU wrapper around diskrot.filter_lyrics (see its docstring for the
flagging rules). Pure JSON sweep over the 256 lyric shards, one container,
seconds-to-minutes of work.

Ordering: run AFTER the transcribe fleet has fully finished (its orchestrator
holds shard contents in memory and would clobber concurrent edits on its next
flush) and BEFORE phonemize (so junk never enters the phoneme store).

Run::

    modal run diskrot/modal_filter_lyrics.py            # dry-run report
    modal run diskrot/modal_filter_lyrics.py --apply    # rewrite shards

Monitor::

    modal app logs nano-filter-lyrics
"""
from __future__ import annotations

import modal

app = modal.App("nano-filter-lyrics")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",  # transcribe_lyrics import chain (is_valid_word only, no GPU)
        "numpy>=1.26",
        "tqdm",
    )
    .add_local_python_source("diskrot", "model")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
corpus_vol = modal.Volume.from_name("nano-corpus", create_if_missing=True)


@app.function(
    image=image,
    cpu=2.0,
    memory=4 * 1024,
    timeout=60 * 30,
    volumes={"/tokens": tokens_vol, "/corpus": corpus_vol},
)
def filter_remote(apply: bool = False) -> dict:
    from pathlib import Path

    from diskrot.filter_lyrics import filter_lyrics

    stats = filter_lyrics(
        "/tokens/lyrics", apply=apply, verbose=True,
        commit_cb=tokens_vol.commit if apply else None,
    )
    if apply:
        tokens_vol.commit()  # idempotent safety net
    # Corpus coverage: songs with no shard entry at all were never transcribed
    # (they train as <unknown_vocals>, NOT <instrumental> — different marker).
    n_mp3s = sum(1 for _ in Path("/corpus").glob("*.mp3"))
    stats["corpus_mp3s"] = n_mp3s
    stats["never_transcribed"] = n_mp3s - stats["total_transcribed"]
    print(f"[filter]   never transcribed:          {stats['never_transcribed']:>7} "
          f"(of {n_mp3s} corpus mp3s -> <unknown_vocals> at train time)", flush=True)
    return stats


@app.local_entrypoint()
def main(apply: bool = False):
    stats = filter_remote.remote(apply=apply)
    print(stats)
    if not apply:
        print("dry run only — re-run with --apply to null the flagged entries")
