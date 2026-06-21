"""Quick read-only audit: do the on-disk lyrics carry a detected `language`?

Determines whether v9 multilingual can reuse the EXISTING transcribe output (no
re-transcribe) or needs a re-transcribe. Reports: % of with-words entries that
have a non-null language field, the language distribution, and avg_logprob
presence (the lyric conf filter's signal).

    modal run diskrot/modal_lyrics_lang_audit.py
"""
import modal

app = modal.App("nano-lyrics-lang-audit")
image = modal.Image.debian_slim(python_version="3.12")
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=60 * 15)
def audit() -> None:
    import json
    from collections import Counter
    from pathlib import Path

    lyrics = Path("/tokens/lyrics")
    shards = sorted(lyrics.glob("lyrics_*.json"))
    if not shards:
        print("no lyrics/ shards found")
        return

    n_entries = n_words = n_lang = n_logprob = n_null = 0
    langs: Counter = Counter()
    for shard in shards:
        for name, val in json.loads(shard.read_text()).items():
            if val is None:
                n_null += 1
                continue
            if not isinstance(val, dict):
                continue
            n_entries += 1
            if val.get("words"):
                n_words += 1
            lang = val.get("language")
            if lang:
                n_lang += 1
                langs[lang] += 1
            if val.get("avg_logprob") is not None:
                n_logprob += 1

    print(f"\n=== lyrics language audit ({len(shards)} shards) ===")
    print(f"non-null entries:            {n_entries:,}")
    print(f"  with words:                {n_words:,}")
    print(f"null (instrumental):         {n_null:,}")
    pct = (lambda n: f"{100*n/n_entries:.1f}%") if n_entries else (lambda n: "-")
    print(f"with `language` field:       {n_lang:,} ({pct(n_lang)})")
    print(f"with `avg_logprob` field:    {n_logprob:,} ({pct(n_logprob)})")
    print("\nlanguage distribution (top 25):")
    for lang, c in langs.most_common(25):
        print(f"  {lang:>6}: {c:,}")
    non_en = sum(c for l, c in langs.items() if l != "en")
    print(f"\nnon-English (with language):  {non_en:,} "
          f"({100*non_en/max(1,n_lang):.1f}% of language-tagged)")
    print("\nVERDICT: if `language` coverage is high -> v9 multilingual reuses the "
          "existing transcribe (NO re-transcribe). If ~0% -> the corpus is the old "
          "forced-English transcribe and needs a re-transcribe for true multilingual.",
          flush=True)


@app.local_entrypoint()
def main() -> None:
    audit.remote()
