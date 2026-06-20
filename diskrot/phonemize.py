"""Pre-phonemize the lyrics corpus -> sharded phonemes/ dir (train-time cache).

The dataset phonemizes each song's words lazily inside the DataLoader workers
(``TokenDataset._get_segment_lyric_ids``). g2p is cheap for dictionary words but
falls back to a neural seq2seq for OOV words — and Whisper song transcripts are
full of those ("oooh", slang, misspellings), so a real song costs ~20-200 ms.
With the per-worker LRU capped far below corpus size, the miss rate at full
scale is ~99%, which can starve an 8xH100 step. This pass runs g2p ONCE per
song offline and stores the per-word phoneme-id groups; the dataset then reads
them as a pure lookup (with a live-g2p fallback for songs not covered, so a
partial/absent pass is always safe).

Output mirrors the lyrics sharding (same 256 stem-hash buckets, atomic writes,
resume by per-song skip): ``phonemes/phonemes_NNN.json`` mapping
``{name: [[phoneme ids of word 0], [word 1], ...]}``. The group count is
validated against the song's word count at train time, so a re-transcribed song
whose lyrics changed simply falls back to live g2p until this pass re-runs.

The ids come from ``model.lyric_encoder.text_to_word_phoneme_groups`` — the
single train==inference g2p contract — so the stored stream is byte-identical
to what live g2p would produce (guarded by tests/test_phonemize.py).

CLI::

    python -m diskrot.phonemize --lyrics-path ./lyrics --out-dir ./phonemes
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PHONEMES_DIR_NAME = "phonemes"


def _shard_path(out_dir: str | Path, bucket: int) -> Path:
    return Path(out_dir) / f"phonemes_{bucket:03d}.json"


def load_phoneme_shards(phonemes_dir: str | Path) -> dict[str, list[list[int]]]:
    """Read all ``phonemes_*.json`` shards into one merged dict."""
    phonemes_dir = Path(phonemes_dir)
    merged: dict[str, list[list[int]]] = {}
    if not phonemes_dir.exists():
        return merged
    for shard in sorted(phonemes_dir.glob("phonemes_*.json")):
        merged.update(json.loads(shard.read_text()))
    return merged


def _load_lyric_words(lyrics_path: str | Path) -> dict[str, tuple[list[str], str | None]]:
    """{name: (clean word list, detected language)} for every song with usable
    transcribed words.

    Mirrors ``dataset._load_lyrics``'s cleaning exactly (same
    ``transcribe_lyrics.is_valid_word`` predicate), so the stored group count
    always equals the word count the train-time loader sees — the staleness
    contract. The language (transcribe stores it) is carried so each song is
    phonemized in ITS language (v9 multilingual) — byte-identical to the dataset's
    live fallback, which passes the same per-song language. Deliberately does NOT
    import diskrot.dataset: that module pulls in model.codec (librosa/dac), which
    the slim phonemize image doesn't carry."""
    from diskrot.transcribe_lyrics import is_valid_word, load_lyrics_shards

    lp = Path(lyrics_path)
    if not lp.exists():
        return {}
    raw = load_lyrics_shards(lp) if lp.is_dir() else json.loads(lp.read_text())
    out: dict[str, tuple[list[str], str | None]] = {}
    for name, val in raw.items():
        if not isinstance(val, dict):
            continue
        words = val.get("words")
        clean = [w["word"] for w in words if is_valid_word(w)] if isinstance(words, list) else []
        if clean:
            out[name] = (clean, val.get("language"))
    return out


def _phonemize_bucket(args: tuple[int, str, dict[str, list[str]]]) -> tuple[int, int, int]:
    """Worker: phonemize one bucket's songs and atomically write its shard.

    ``args`` is (bucket, out_dir, {name: words}). Returns (bucket, n_new, n_total).
    Runs in its own process — g2p_en's model loads once per process and is
    GIL-bound, so processes (not threads) are required for real parallelism.
    """
    from diskrot.transcribe_lyrics import _atomic_write_json
    from model.lyric_encoder import text_to_word_phoneme_groups

    bucket, out_dir, songs = args
    path = _shard_path(out_dir, bucket)
    existing: dict[str, list[list[int]]] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    n_new = 0
    for name, (words, language) in songs.items():
        prior = existing.get(name)
        if prior is not None and len(prior) == len(words):
            continue  # resume-by-skip (word-count match = not stale)
        existing[name] = text_to_word_phoneme_groups(words, language=language)
        n_new += 1
    if n_new:
        _atomic_write_json(path, existing)
    return bucket, n_new, len(existing)


def phonemize_corpus(
    lyrics_path: str | Path,
    out_dir: str | Path,
    n_workers: int = 8,
    verbose: bool = True,
    commit_cb=None,
) -> Path:
    """Phonemize every lyric'd song under ``lyrics_path`` into ``out_dir`` shards.

    ``lyrics_path`` is the sharded lyrics dir (or a single legacy lyrics.json).
    Buckets are processed in parallel processes; each bucket is skipped per-song
    on resume and written atomically, so a kill loses at most the in-flight
    buckets. ``commit_cb`` (e.g. a Modal volume commit) runs after each batch of
    completed buckets."""
    from diskrot.transcribe_lyrics import _lyric_bucket

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lyric_words = _load_lyric_words(lyrics_path)
    if verbose:
        print(f"[phonemize] {len(lyric_words)} songs with usable words from "
              f"{lyrics_path}", flush=True)

    buckets: dict[int, dict[str, list[str]]] = {}
    for name, words in lyric_words.items():
        buckets.setdefault(_lyric_bucket(name), {})[name] = words

    # Warm g2p in the parent BEFORE the pool forks: G2p() triggers nltk data
    # downloads when the corpora are missing, and N workers doing that
    # concurrently race on the same zip file (BadZipFile). After this line the
    # data exists (and fork-start workers inherit the built G2p outright).
    from model.lyric_encoder import text_to_word_phoneme_groups
    text_to_word_phoneme_groups(["warmup"])

    t0 = time.time()
    n_done = 0
    tasks = [(b, str(out_dir), songs) for b, songs in sorted(buckets.items())]
    if n_workers <= 1:
        # Inline path — no process pool, so it also works where spawn can't
        # re-import __main__ (e.g. ad-hoc stdin scripts).
        results = map(_phonemize_bucket, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=n_workers)
        results = pool.map(_phonemize_bucket, tasks)
    try:
        for i, (bucket, n_new, n_total) in enumerate(results):
            n_done += n_new
            if commit_cb is not None and n_new:
                commit_cb()
            if verbose and (n_new or (i + 1) % 32 == 0):
                print(f"[phonemize] bucket {bucket:03d}: +{n_new} (shard total "
                      f"{n_total}); {i + 1}/{len(tasks)} buckets, "
                      f"{time.time() - t0:.0f}s", flush=True)
    finally:
        if n_workers > 1:
            pool.shutdown()
    if verbose:
        print(f"[phonemize] done: {n_done} songs newly phonemized -> {out_dir} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lyrics-path", type=str, required=True,
                   help="sharded lyrics dir (or legacy lyrics.json)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="output directory for sharded phonemes_NNN.json files")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    phonemize_corpus(args.lyrics_path, args.out_dir, n_workers=args.workers)
