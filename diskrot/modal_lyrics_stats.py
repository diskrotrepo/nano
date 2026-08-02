"""One-command lyrics-stats snapshot / wave-vs-rest diff (read-only).

    modal run diskrot/modal_lyrics_stats.py                 # whole-corpus snapshot
    modal run diskrot/modal_lyrics_stats.py --wave-id base_0  # one wave vs the rest

Prints aggregate lyrics stats: instrumental rate, median/mean word count, vocal-
gender split, top languages, median ``avg_logprob`` (Whisper confidence), and the
``filter_lyrics`` hallucination-flag rate. Useful as a health check on a wave's
transcripts or a baseline corpus snapshot.

Transcribe is Demucs-free now (Whisper on the raw mix) and **vocal gender comes
from the audio-LLM captioner (tags.json)**, not the lyrics entries — so the gender
split is joined from tags.json here (legacy lyrics-entry gender is a fallback).

Read-only: never writes the volume.
"""
from __future__ import annotations

import modal

from diskrot.modal_common import corpus_mount, wave_subdir

app = modal.App("nano-lyrics-stats")

# Needs diskrot on the image to reuse the EXACT is_valid_word / hallucination_reason
# predicates the dataset + filter use (so the stats match training reality).
image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("diskrot")

corpus_vol = corpus_mount()  # R2 audio bucket — only to resolve a wave's stems
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


def _load_tag_gender(tokens_dir: str = "/tokens") -> dict:
    """{stem: 'male'|'female'} from the audio-LLM captioner's per-song ``gender``
    field in tags.json (single file) or tags/ (shard dir). Torch-free so it runs on
    this stage's slim image; mirrors diskrot.dataset._load_tag_gender."""
    import json
    from pathlib import Path

    td = Path(tokens_dir) / "tags"
    tf = Path(tokens_dir) / "tags.json"
    if td.is_dir():
        from diskrot.sharded_store import load_json_shards
        raw = load_json_shards(td, "tags")
    elif tf.exists():
        raw = json.loads(tf.read_text())
    else:
        return {}
    return {k: v["gender"] for k, v in raw.items()
            if isinstance(v, dict) and v.get("gender") in ("male", "female")}


def _group_stats(entries: dict, tag_gender: dict | None = None) -> dict:
    """Aggregate stats for a {stem: entry_or_None} slice of the lyrics dir.

    ``tag_gender`` is the {stem: 'male'|'female'} map from tags.json (the audio-LLM
    captioner's vocal-gender judgment) — the gender split is read from it first,
    with the legacy lyrics-entry ``gender`` field as a fallback for pre-v4 corpora.
    """
    import statistics
    from collections import Counter

    from diskrot.filter_lyrics import hallucination_reason
    from diskrot.transcribe_lyrics import is_valid_word

    tag_gender = tag_gender or {}
    n = len(entries)
    n_none = sum(1 for v in entries.values() if v is None)
    items = [(k, v) for k, v in entries.items() if isinstance(v, dict)]

    word_counts = []
    n_with_words = n_flagged = n_lang = n_logprob = 0
    genders: Counter = Counter()
    langs: Counter = Counter()
    logprobs = []
    for k, v in items:
        nw = sum(1 for w in (v.get("words") or ()) if is_valid_word(w))
        if nw > 0:
            n_with_words += 1
            word_counts.append(nw)
            genders[tag_gender.get(k) or v.get("gender") or "none"] += 1
            lang = v.get("language")
            if lang:
                n_lang += 1
                langs[lang] += 1
            lp = v.get("avg_logprob")
            if isinstance(lp, (int, float)) and not isinstance(lp, bool):
                n_logprob += 1
                logprobs.append(lp)
        if hallucination_reason(v) is not None:
            n_flagged += 1

    def pct(x, d):
        return f"{(100.0 * x / d):.1f}%" if d else "—"

    # "instrumental" at train time = None OR transcribed-but-wordless dict.
    n_wordless = len(items) - n_with_words
    n_instrumental = n_none + n_wordless
    return {
        "present": n,
        "instrumental_rate": pct(n_instrumental, n),
        "with_words_rate": pct(n_with_words, n),
        "median_words": f"{statistics.median(word_counts):.0f}" if word_counts else "—",
        "mean_words": f"{statistics.mean(word_counts):.0f}" if word_counts else "—",
        "halluc_flag_rate": pct(n_flagged, len(items)),
        "gender_male": pct(genders.get("male", 0), n_with_words),
        "gender_female": pct(genders.get("female", 0), n_with_words),
        "gender_none": pct(genders.get("none", 0), n_with_words),
        "median_logprob": f"{statistics.median(logprobs):.3f}" if logprobs else "—",
        "lang_coverage": pct(n_lang, n_with_words),
        "top_langs": ", ".join(f"{k}:{pct(c, n_lang)}" for k, c in langs.most_common(4)) or "—",
    }


_ROWS = [
    ("present", "entries present"),
    ("instrumental_rate", "instrumental %"),
    ("with_words_rate", "with-words %"),
    ("median_words", "median word count"),
    ("mean_words", "mean word count"),
    ("halluc_flag_rate", "halluc-flag % (of dicts)"),
    ("gender_male", "gender male %"),
    ("gender_female", "gender female %"),
    ("gender_none", "gender none %"),
    ("median_logprob", "median avg_logprob"),
    ("lang_coverage", "language coverage %"),
    ("top_langs", "top languages"),
]


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    # Globs the whole corpus + parses every lyrics shard (>1 GB JSON at full
    # coverage) — the 300s default isn't enough.
    timeout=30 * 60,
)
def stats(wave_id: str = "") -> None:
    from diskrot.transcribe_lyrics import load_lyrics_shards

    lyrics = load_lyrics_shards("/tokens/lyrics")
    if not lyrics:
        print("no lyrics/ shards found on /tokens")
        return
    # Vocal gender is sourced from the audio-LLM captions (tags.json), not the
    # lyrics entries. Missing -> lyrics-entry fallback inside _group_stats.
    tag_gender = _load_tag_gender()

    if not wave_id:
        g = _group_stats(lyrics, tag_gender)
        print(f"\nWHOLE CORPUS — {len(lyrics):,} entries\n" + "-" * 40)
        for key, label in _ROWS:
            print(f"  {label:<26} {g[key]}")
        return

    wave_stems = {p.stem for p in (Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3")}
    in_wave = {k: v for k, v in lyrics.items() if k in wave_stems}
    rest = {k: v for k, v in lyrics.items() if k not in wave_stems}
    if not in_wave:
        print(f"wave {wave_id}: 0 of its {len(wave_stems):,} stems are in lyrics/ yet "
              f"— has it been transcribed? (transcribe skips already-done stems)")
        return

    gw, gr = _group_stats(in_wave, tag_gender), _group_stats(rest, tag_gender)
    print(f"\nwave {wave_id} vs REST of corpus")
    print(f"  wave: {len(in_wave):,} entries   rest: {len(rest):,} entries\n")
    print(f"  {'metric':<26} {'wave_' + wave_id:>18} {'rest':>18}")
    print("  " + "-" * 64)
    for key, label in _ROWS:
        print(f"  {label:<26} {str(gw[key]):>18} {str(gr[key]):>18}")
    print("\n  Read: a higher instrumental %, lower median word count, or higher\n"
          "  halluc-flag % in the wave column flags transcript trouble for that wave.")


@app.local_entrypoint()
def main(wave_id: str = ""):
    """--wave-id <id> diffs that wave's lyrics stats vs the rest of the corpus;
    omit it for a whole-corpus snapshot."""
    stats.remote(wave_id=wave_id)
