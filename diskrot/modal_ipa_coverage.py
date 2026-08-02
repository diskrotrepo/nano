"""W9 validation spike: does the frozen _IPA inventory cover the corpus's phones?

espeak-ng-phonemizes a sample of real corpus lyrics (each in its DETECTED
language) and reports what fraction of emitted IPA codepoints fall inside
``model.lyric_encoder.PHONEME_VOCAB`` vs map to UNK. Run this BEFORE the
re-phonemize bakes the (checkpoint-incompatible) phoneme vocab in — same
de-risking role the codec spike played for SpectroStream. If coverage is low or a
common codepoint shows up under "top uncovered", add it to _IPA (and bump
PHONEME_VOCAB_SIZE) before committing.

    modal run diskrot/modal_ipa_coverage.py --n-songs 4000
"""
import modal

app = modal.App("nano-ipa-coverage")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng", "libespeak-ng1")
    .pip_install("phonemizer>=3.2", "torch>=2.4", "numpy>=1.26", "tqdm>=4.66")
    .add_local_python_source("diskrot", "model")
)
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=60 * 30)
def coverage(n_songs: int = 4000) -> None:
    import json
    from collections import Counter
    from pathlib import Path

    from diskrot.transcribe_lyrics import is_valid_word, load_lyrics_shards
    from model.lyric_encoder import (
        PHONEME_TO_ID, UNK_PHONEME, _BOUNDARY_PUNCT, _get_espeak, _get_seps,
    )

    raw = load_lyrics_shards(Path("/tokens/lyrics"))
    songs = []
    for name, val in raw.items():
        if not isinstance(val, dict):
            continue
        words = val.get("words")
        clean = [w["word"] for w in words if is_valid_word(w)] if isinstance(words, list) else []
        if clean:
            songs.append((clean, (val.get("language") or "en")))
        if len(songs) >= n_songs:
            break
    print(f"=== IPA coverage over {len(songs)} sampled vocal-ready songs ===", flush=True)

    phone_sep, _ = _get_seps()
    covered = uncovered = 0
    uncov_counter: Counter = Counter()
    lang_total: Counter = Counter()
    lang_uncov: Counter = Counter()
    for i, (words, lang) in enumerate(songs):
        try:
            ipa = _get_espeak(lang).phonemize([" ".join(words)], separator=phone_sep, strip=True)[0]
        except Exception as e:  # noqa: BLE001
            print(f"  phonemize failed (lang={lang}): {type(e).__name__}", flush=True)
            continue
        for ch in ipa:
            if ch.isspace() or ch in _BOUNDARY_PUNCT:
                continue
            lang_total[lang] += 1
            if ch in PHONEME_TO_ID:
                covered += 1
            else:
                uncovered += 1
                uncov_counter[ch] += 1
                lang_uncov[lang] += 1
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(songs)} songs", flush=True)

    total = covered + uncovered
    pct = 100.0 * covered / total if total else 0.0
    print(f"\ncovered {covered:,} / {total:,} IPA codepoints = {pct:.3f}% "
          f"({uncovered:,} -> {UNK_PHONEME})", flush=True)
    print("\ntop-30 UNCOVERED codepoints (codepoint U+XXXX 'char' count):", flush=True)
    for ch, n in uncov_counter.most_common(30):
        print(f"  U+{ord(ch):04X} {ch!r} {n:,}", flush=True)
    print("\nper-language uncovered rate (langs with >=1000 phones):", flush=True)
    for lang, tot in lang_total.most_common():
        if tot >= 1000:
            print(f"  {lang:>5}: {100.0*lang_uncov[lang]/tot:.2f}% uncovered "
                  f"({tot:,} phones)", flush=True)
    print("\nVERDICT: aim for >=99% covered; add any common 'top uncovered' codepoint "
          "to model.lyric_encoder._IPA before the re-phonemize.", flush=True)


@app.local_entrypoint()
def main(n_songs: int = 4000) -> None:
    coverage.remote(n_songs=n_songs)
