"""Lyric-data health audit: coverage + quality of every conditioning stream.

One CPU container sweeps the nano-tokens volume (and the corpus listing) and
reports:

  - per-stream coverage of the packed (trainable) corpus: tags, structure,
    keys, lyrics, phonemes
  - lyric breakdown: instrumental(null) / hallucinated (what the filter
    would null, same rules as diskrot/filter_lyrics.py) / vocal-ready /
    never-transcribed
  - vocal-gender distribution among vocal-ready songs ('unset' = pre-gender
    v7 entries — re-run transcribe to add the field)
  - word-count distribution among vocal-ready songs
  - language breakdown of vocal-ready transcripts (py3langid on the stored
    text)
  - top detected keys (sanity check on key_detect output)

Read-only, so it's safe to run while a transcribe/filter pass is in flight —
the numbers are just a snapshot of what's on disk right now.

Caveat: transcription forces ``language="en"`` (transcribe_lyrics.py), so
non-English vocals decode into English-looking word salad rather than their
source language. The language breakdown is therefore a LOWER BOUND on
non-English contamination — it only flags transcripts where enough source
language leaked through for a text detector to see; the fully-anglicized
salad still counts as "en".

Run:  modal run scripts/lyrics_audit.py
"""
from __future__ import annotations

import modal

app = modal.App("nano-lyrics-audit")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy>=1.26", "py3langid>=0.3"
)

tokens_vol = modal.Volume.from_name("nano-tokens")
corpus_vol = modal.Volume.from_name("nano-corpus")


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol, "/corpus": corpus_vol},
    timeout=60 * 30,
    # Modal's spontaneous platform cancellations (the same waves that hit the
    # big .map fleets) can kill this single .remote() call mid-sweep — retry.
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
)
def audit():
    import json
    from collections import Counter
    from pathlib import Path

    import numpy as np

    # diskrot.filter_lyrics / pack_cache import torch transitively, which this
    # slim image doesn't carry — inline the few pure-JSON helpers needed here.
    # Keep MIN_WORDS / JUNK_PHRASES / JUNK_MAX_WORDS in sync with
    # diskrot/filter_lyrics.py.
    MIN_WORDS, JUNK_MAX_WORDS = 6, 30
    JUNK_PHRASES = (
        "thank you for watching", "thanks for watching", "we'll be right back",
        "see you next time", "see you in the next video", "subscribe",
        "subtitles", "copyright", "transcript", "www.", ".com",
    )

    def is_valid_word(w):
        return (
            isinstance(w, dict)
            and isinstance(w.get("word"), str)
            and isinstance(w.get("start"), (int, float)) and not isinstance(w["start"], bool)
            and isinstance(w.get("end"), (int, float)) and not isinstance(w["end"], bool)
        )

    def hallucination_reason(entry):
        if not isinstance(entry, dict):
            return None
        n_words = sum(1 for w in entry.get("words", ()) if is_valid_word(w))
        if n_words < MIN_WORDS:
            return "short"
        if n_words < JUNK_MAX_WORDS:
            text = (entry.get("text") or "").lower()
            for phrase in JUNK_PHRASES:
                if phrase in text:
                    return f"junk:{phrase}"
        return None

    def load_lyrics_shards(lyrics_dir):
        merged = {}
        for shard in sorted(Path(lyrics_dir).glob("lyrics_*.json")):
            merged.update(json.loads(shard.read_text()))
        return merged

    # --- corpus / packed (trainable) ---
    corpus = {p.stem for p in Path("/corpus").glob("*.mp3")}
    print(f"corpus mp3s:              {len(corpus)}")

    trainable: set[str] = set()
    packed = Path("/tokens/packed")
    if (packed / "packed_index.json").exists():
        index = json.loads((packed / "packed_index.json").read_text())
        for shard_entry in index["shards"]:
            meta = json.loads(
                (packed / f"packed_{shard_entry['shard_id']:03d}.json").read_text()
            )
            trainable.update(meta["names"])
    print(f"packed (trainable) songs: {len(trainable)}")

    # --- per-stream coverage vs trainable ---
    tags_path = Path("/tokens/tags.json")
    tags = json.loads(tags_path.read_text()) if tags_path.exists() else {}
    keys_path = Path("/tokens/keys.json")
    keys = json.loads(keys_path.read_text()) if keys_path.exists() else {}

    structure: set[str] = set()
    for shard in sorted(Path("/tokens/structure").glob("structure_*.json")):
        structure.update(json.loads(shard.read_text()).keys())

    phonemes: set[str] = set()
    ph_dir = Path("/tokens/phonemes")
    if ph_dir.exists():
        for shard in sorted(ph_dir.glob("phonemes_*.json")):
            phonemes.update(json.loads(shard.read_text()).keys())

    lyrics = load_lyrics_shards("/tokens/lyrics")

    def cov(name: str, have: set[str]) -> None:
        n = len(have & trainable) if trainable else len(have)
        pct = n / max(len(trainable), 1)
        print(f"  {name:<12} {n:>7} / {len(trainable)}  ({pct:.1%})")

    print("\ncoverage of trainable songs:")
    cov("tags", set(tags))
    cov("structure", structure)
    cov("keys", set(keys))
    cov("lyrics", set(lyrics))
    cov("phonemes", phonemes)

    # --- lyric breakdown (null=instrumental, ABSENT=never transcribed) ---
    instrumental = [k for k, v in lyrics.items() if v is None]
    has_words = {k: v for k, v in lyrics.items()
                 if isinstance(v, dict) and v.get("words")}
    empty = len(lyrics) - len(instrumental) - len(has_words)

    halluc = [k for k, v in has_words.items() if hallucination_reason(v)]
    halluc_set = set(halluc)
    vocal_ready = {k: v for k, v in has_words.items() if k not in halluc_set}

    print(f"\nlyrics entries:           {len(lyrics)}")
    print(f"  instrumental (null):    {len(instrumental)}  "
          f"({len(instrumental) / max(len(lyrics), 1):.1%})")
    print(f"  empty-words dict:       {empty}")
    print(f"  hallucinated (filter):  {len(halluc)}  "
          f"({len(halluc) / max(len(lyrics), 1):.1%})")
    print(f"  vocal-ready:            {len(vocal_ready)}  "
          f"({len(vocal_ready) / max(len(lyrics), 1):.1%})")
    print(f"  never transcribed:      {len(corpus - set(lyrics))} of corpus")

    # --- gender distribution among vocal-ready ---
    g = Counter((v.get("gender") or "unset") for v in vocal_ready.values())
    print("\nvocal gender (vocal-ready songs):")
    for label, n in g.most_common():
        print(f"  {label:<8} {n:>7}  ({n / max(len(vocal_ready), 1):.1%})")
    n_with_gender = sum(n for l, n in g.items() if l != "unset")
    print(f"  (gender field present on {n_with_gender}/{len(vocal_ready)} — "
          f"'unset' = pre-gender v7 entries)")

    # --- word-count distribution among vocal-ready ---
    counts = np.array(sorted(len(v["words"]) for v in vocal_ready.values()))
    if len(counts):
        print("\nword-count distribution (vocal-ready):")
        for lo, hi in [(6, 20), (21, 50), (51, 150), (151, 400), (401, 10**9)]:
            n = int(((counts >= lo) & (counts <= hi)).sum())
            label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print(f"  {label:>8} words: {n:>7}  ({n / len(counts):.1%})")
        print(f"  mean {counts.mean():.0f}  median {int(np.median(counts))}  "
              f"max {int(counts.max())}")

    # --- language ID on vocal-ready transcripts (text-based, LOWER BOUND) ---
    # Transcribe forces language="en", so a non-English song usually decodes
    # into English-looking word salad the detector reads as "en". Anything
    # flagged non-English here is the unambiguous tail (source language leaked
    # through); the true non-English share is higher. Audio-based language ID
    # would need another Whisper pass (language=None) — see transcribe_lyrics.
    if vocal_ready:
        import py3langid as langid

        lang_counts = Counter()
        for v in vocal_ready.values():
            text = (v.get("text") or "").strip()[:1500]
            if not text:
                lang_counts["(empty)"] += 1
                continue
            lang, _ = langid.classify(text)
            lang_counts[lang] += 1
        n_en = lang_counts.get("en", 0)
        print(f"\nlanguage of vocal-ready transcripts (text-based — a lower "
              f"bound on non-English; forced-en salad still reads as 'en'):")
        for label, n in lang_counts.most_common(10):
            print(f"  {label:<8} {n:>7}  ({n / len(vocal_ready):.1%})")
        rest = len(vocal_ready) - sum(n for _, n in lang_counts.most_common(10))
        if rest:
            print(f"  (other)  {rest:>7}  ({rest / len(vocal_ready):.1%})")
        print(f"  non-en (detectable): {len(vocal_ready) - n_en} "
              f"({(len(vocal_ready) - n_en) / len(vocal_ready):.1%})")

    # --- key distribution (sanity that key-detect output is plausible) ---
    if keys:
        kc = Counter(v.get("key", "?") for v in keys.values())
        print(f"\nkeys.json: {len(keys)} songs, top 8 keys:")
        for label, n in kc.most_common(8):
            print(f"  {label:<12} {n:>7}  ({n / len(keys):.1%})")


@app.local_entrypoint()
def main():
    audit.remote()
