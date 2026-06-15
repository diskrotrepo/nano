"""Training-data health audit: coverage + quality of every conditioning stream.

One CPU container sweeps the nano-tokens volume (+ nano-melody + the corpus
listing) and reports:

  - packed (trainable) corpus size with a scale check vs the ~10k/~50k/~500k
    rails (see CLAUDE.md "Scale of training data")
  - per-stream coverage of the packed (trainable) corpus: tags, structure,
    keys, lyrics, phonemes, melody
  - melody source (.mel.npy on nano-melody) vs packed (.mel.bin) coverage —
    both must hold for a contour to actually train
  - song-duration distribution from the packed offsets (and the share shorter
    than the 60s training crop)
  - model settings vs data scale: the default 400k-step run translated into
    effective epochs of unique audio, with an under/over-training verdict
    (nano is one fixed-shape ~2.0B net, so step count is the lever that has to
    match the trainable-song scale)
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
# Source chromagrams live on their own volume (kept off nano-tokens for its
# inode cap). Read-only here; create_if_missing so the audit still runs before
# any melody pass has populated it.
melody_vol = modal.Volume.from_name("nano-melody", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol, "/corpus": corpus_vol, "/melody": melody_vol},
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

    # DAC frame rate (model/codec.py FRAME_RATE_HZ); inlined to keep the slim
    # image free of torch. seconds = frames / 86.
    FRAME_RATE_HZ = 86
    # Corpus health rails (see CLAUDE.md "Scale of training data"): below ~10k is
    # noise (pipeline-validation only), ~50k is the recommended floor for
    # coherent output, ~500k is the nano-corpus inode ceiling.
    FLOOR_NOISE, FLOOR_COHERENT, CEILING = 10_000, 50_000, 500_000

    # --- corpus / packed (trainable) ---
    corpus = {p.stem for p in Path("/corpus").glob("*.mp3")}
    print(f"corpus mp3s:              {len(corpus)}")

    trainable: set[str] = set()
    durations: list[float] = []          # per-song seconds (from packed offsets)
    melody_packed: set[str] = set()      # songs in a shard packed WITH chroma
    packed = Path("/tokens/packed")
    if (packed / "packed_index.json").exists():
        index = json.loads((packed / "packed_index.json").read_text())
        for shard_entry in index["shards"]:
            meta = json.loads(
                (packed / f"packed_{shard_entry['shard_id']:03d}.json").read_text()
            )
            names = meta["names"]
            trainable.update(names)
            # offsets are cumulative frame counts (len == n_songs + 1), so a
            # song's frame length is the gap between consecutive offsets.
            offsets = meta.get("offsets")
            if isinstance(offsets, list) and len(offsets) == len(names) + 1:
                durations.extend(
                    (offsets[i + 1] - offsets[i]) / FRAME_RATE_HZ
                    for i in range(len(names))
                )
            # has_melody marks a shard packed with a chroma sidecar. Membership
            # there is identical to the token .bin (missing chroma is zero-filled,
            # never dropped), so every name in the shard is melody-PACKED — though
            # a zero-filled row carries no real contour (see source coverage below).
            if meta.get("has_melody"):
                melody_packed.update(names)
    print(f"packed (trainable) songs: {len(trainable)}")
    n_train = len(trainable)
    if n_train:
        if n_train < FLOOR_NOISE:
            verdict = f"BELOW ~{FLOOR_NOISE//1000}k — expect noise (pipeline-validation only)"
        elif n_train < FLOOR_COHERENT:
            verdict = f"below ~{FLOOR_COHERENT//1000}k recommended floor — usable but thin"
        elif n_train <= CEILING:
            verdict = f"in the recommended ~{FLOOR_COHERENT//1000}k–{CEILING//1000}k band"
        else:
            verdict = f"above the ~{CEILING//1000}k inode ceiling (?)"
        print(f"  scale check:            {verdict}")

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

    # Source chromagrams on nano-melody (the modal_melody.py output). A song with
    # a .mel.npy here has a real contour; whether it reached the dataset also
    # needs it folded into the pack (melody_packed below).
    melody_src = {p.name[: -len(".mel.npy")]
                  for p in Path("/melody").glob("*.mel.npy")}

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
    cov("melody", melody_src)

    # --- melody: source vs packed (the two must both hold for the model to use
    # a contour). source = .mel.npy on nano-melody; packed = folded into a
    # has_melody pack shard. Source-without-packed means the pack predates the
    # chroma and needs a re-pack with --mel-cache-dir before melody trains. ---
    print("\nmelody conditioning:")
    src_in_train = len(melody_src & trainable)
    pk_in_train = len(melody_packed & trainable)
    print(f"  source (.mel.npy):      {src_in_train:>7} / {len(trainable)}  "
          f"({src_in_train / max(len(trainable), 1):.1%})")
    print(f"  packed (.mel.bin):      {pk_in_train:>7} / {len(trainable)}  "
          f"({pk_in_train / max(len(trainable), 1):.1%})")
    src_not_packed = len((melody_src & trainable) - melody_packed)
    if src_not_packed:
        print(f"  source not yet packed:  {src_not_packed}  "
              f"(re-pack with --mel-cache-dir to make these trainable)")
    if pk_in_train and src_in_train < pk_in_train:
        print(f"  packed-but-zero-filled: ~{pk_in_train - src_in_train}  "
              f"(in a melody shard but no source chroma → silent rows)")

    # --- song-duration distribution (from packed offsets) ---
    if durations:
        d = np.array(sorted(durations))
        print("\nsong-duration distribution (packed, seconds):")
        for lo, hi in [(0, 20), (20, 30), (30, 60), (60, 120), (120, 10**9)]:
            n = int(((d >= lo) & (d < hi)).sum())
            label = f"{lo}-{hi}s" if hi < 10**9 else f"{lo}s+"
            print(f"  {label:>9}: {n:>7}  ({n / len(d):.1%})")
        print(f"  mean {d.mean():.0f}s  median {np.median(d):.0f}s  "
              f"min {d.min():.0f}s  max {d.max():.0f}s")
        # Songs shorter than the Modal 60s training segment can't fill a crop
        # (the dataset pads/wraps them) — a large share dilutes the effective set.
        n_short = int((d < 60).sum())
        if n_short:
            print(f"  shorter than 60s crop:  {n_short}  ({n_short / len(d):.1%})")

    # --- model settings vs data scale ---
    # nano is ONE fixed-shape ~2.0B net (the DEFAULTS dict in
    # diskrot/modal_train.py is the source of truth — there is no family of
    # sizes to pick from), so the only training lever that has to match the data
    # scale is the step count. Translate it into effective epochs of unique
    # audio: how many times the configured run sweeps the trainable corpus.
    # Constants mirror DEFAULTS (steps / global batch_size / segment_seconds) —
    # keep in sync if those change.
    TRAIN_STEPS, GLOBAL_BATCH, SEG_SECONDS, MODEL_PARAMS = 400_000, 32, 60, "~2.0B"
    if durations:
        total_audio_s = float(np.array(durations).sum())
        crops_drawn = TRAIN_STEPS * GLOBAL_BATCH          # 60s crops the run draws
        seconds_drawn = crops_drawn * SEG_SECONDS
        epochs = seconds_drawn / max(total_audio_s, 1.0)
        print(f"\nmodel settings vs data scale (DEFAULTS: {MODEL_PARAMS}, "
              f"{TRAIN_STEPS:,} steps, global batch {GLOBAL_BATCH}, {SEG_SECONDS}s crops):")
        print(f"  trainable audio:        {total_audio_s / 3600:,.0f} h  "
              f"({total_audio_s / 1e6:.1f}M s over {len(durations)} songs)")
        print(f"  60s crops over run:     {crops_drawn / 1e6:.1f}M  "
              f"({seconds_drawn / 3600:,.0f} h drawn)")
        print(f"  effective epochs:       {epochs:.1f}x over unique audio")
        # Heuristic band for a fixed-shape run: too few passes leaves the net
        # undertrained, too many invites repetition/memorization. Tune steps (or
        # the corpus) to land in the middle.
        if epochs < 2:
            verdict = "UNDER ~2x — undertraining risk; raise steps or add data"
        elif epochs <= 15:
            verdict = "in the healthy ~2-15x band"
        else:
            verdict = "OVER ~15x — repetition/memorization risk; cut steps or add data"
        print(f"  verdict:                {verdict}")

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

    # NOTE: genre mix is intentionally NOT computed here. A regex over captions
    # in this slim/torch-free container would only reproduce the weak signal
    # eval/genre_gap_eval.py already owns (regex + CLAP zero-shot, per-gap
    # floor/verdict, run-over-run snapshot). The eval-training-data skill runs
    # that tool as a second step for the authoritative genre picture.


@app.local_entrypoint()
def main():
    audit.remote()
