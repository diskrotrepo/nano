"""One-command lyrics-stats diff — the gate for the skip-Demucs experiment.

Run ONE wave through transcribe with skip-Demucs on::

    modal run --detach diskrot/modal_transcribe.py --wave-id base_0 --skip-demucs

then diff that wave's transcripts against the rest of the corpus (which was
transcribed with full Demucs)::

    modal run diskrot/modal_lyrics_stats.py --wave-id base_0

It prints the wave's aggregate lyrics stats side-by-side with the REST of the
corpus: instrumental rate, median/mean word count, gender split, top languages,
median ``avg_logprob`` (Whisper confidence), and the ``filter_lyrics``
hallucination-flag rate. The decision is a visual match:

* instrumental % jumps in the wave        -> raw-mix LOST vocals (BAD, keep Demucs)
* median word count collapses             -> raw-mix mangled dense mixes  (BAD)
* hallucination-flag rate jumps           -> raw-mix invented words       (BAD)
* avg_logprob drifts a little lower        -> expected & harmless (the floor is ~off)
* gender None-rate jumps                   -> the Demucs-lite clip is missing vocals
* everything matches                       -> skip-Demucs is safe; keep it for all waves

Read-only: never writes the volume. With no --wave-id it just prints the
whole-corpus stats (a baseline snapshot).
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


def _group_stats(entries: dict) -> dict:
    """Aggregate stats for a {stem: entry_or_None} slice of the lyrics dir."""
    import statistics
    from collections import Counter

    from diskrot.filter_lyrics import hallucination_reason
    from diskrot.transcribe_lyrics import is_valid_word

    n = len(entries)
    n_none = sum(1 for v in entries.values() if v is None)
    dicts = [v for v in entries.values() if isinstance(v, dict)]

    word_counts = []
    n_with_words = n_flagged = n_lang = n_logprob = 0
    genders: Counter = Counter()
    langs: Counter = Counter()
    logprobs = []
    for v in dicts:
        nw = sum(1 for w in (v.get("words") or ()) if is_valid_word(w))
        if nw > 0:
            n_with_words += 1
            word_counts.append(nw)
            genders[v.get("gender") or "none"] += 1
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
    n_wordless = len(dicts) - n_with_words
    n_instrumental = n_none + n_wordless
    return {
        "present": n,
        "instrumental_rate": pct(n_instrumental, n),
        "with_words_rate": pct(n_with_words, n),
        "median_words": f"{statistics.median(word_counts):.0f}" if word_counts else "—",
        "mean_words": f"{statistics.mean(word_counts):.0f}" if word_counts else "—",
        "halluc_flag_rate": pct(n_flagged, len(dicts)),
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
    from pathlib import Path

    from diskrot.transcribe_lyrics import load_lyrics_shards

    lyrics = load_lyrics_shards("/tokens/lyrics")
    if not lyrics:
        print("no lyrics/ shards found on /tokens")
        return

    if not wave_id:
        g = _group_stats(lyrics)
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

    gw, gr = _group_stats(in_wave), _group_stats(rest)
    print(f"\nwave {wave_id} (skip-Demucs?) vs REST of corpus (full Demucs)")
    print(f"  wave: {len(in_wave):,} entries   rest: {len(rest):,} entries\n")
    print(f"  {'metric':<26} {'wave_' + wave_id:>18} {'rest':>18}")
    print("  " + "-" * 64)
    for key, label in _ROWS:
        print(f"  {label:<26} {str(gw[key]):>18} {str(gr[key]):>18}")
    print("\n  Read: a higher instrumental %, lower median word count, or higher\n"
          "  halluc-flag % in the wave column is raw-mix degradation. A slightly\n"
          "  lower avg_logprob is expected and harmless (the filter floor is ~off).")


@app.local_entrypoint()
def main(wave_id: str = ""):
    """--wave-id <id> diffs that wave's lyrics stats vs the rest of the corpus;
    omit it for a whole-corpus snapshot."""
    stats.remote(wave_id=wave_id)
