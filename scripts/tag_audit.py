"""Tag-format health audit: is the new LONGER audio-LLM tag style healthy?

The chunked-CLAP tag path (model/text_encoder.py:encode_chunked) was built to
condition the decoder on a whole multi-facet description split into <=72-GPT2-
token chunks, instead of one truncated 77-token CLAP vector. That only pays off
if tags.json actually carries the long, rich, audio-LLM (`audio_llm_v1`)
captions and not the legacy ~40-word BART one-liners. This audit measures that.

One CPU container reads /tokens/tags.json and reports:

  - marker mix: audio_llm_v1 (new long) vs bart_v1 (legacy short) vs
    string/none (oldest) — how much of the corpus is on the new format
  - length distribution (chars + words) for the new format vs legacy, the
    share inside the 150-350-word design band, the share degenerately short,
    and the share trimmed at the 3000-char cap (MAX_CHARS — truncated mid-arc)
  - chunk distribution via the SAME chunker as train/inference (CLAP's GPT2
    tokenizer, 72-token windows, 16-chunk cap): chunks-per-caption, the
    resulting n_tag_chunks cross-attn width the corpus would train (capped 16),
    the share at N=1 (no gain over the old single-vector path), and the share
    hitting the 16-chunk cap (tail silently dropped)
  - content health: per-facet coverage (drums/bass/vocals/instruments/
    production/arc — the facets the caption prompt asks for), and degeneracy
    flags (empty, refusal text, repetition loops, markdown/list artifacts)
  - near-duplicate collapse: distinct vs total captions, and the worst
    caption shared across many songs (identical tags can't distinguish songs)

Read-only, safe to run while an auto_tag pass is in flight — a snapshot of disk.

Run:  modal run scripts/tag_audit.py
      modal run scripts/tag_audit.py --max-tokenize 100000   # cap chunk pass
"""
from __future__ import annotations

import modal

app = modal.App("nano-tag-audit")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # transformers' fast GPT2 tokenizer (Rust-backed, no torch) reproduces the
    # train/inference chunker exactly; numpy for the percentiles.
    .pip_install("numpy>=1.26", "transformers>=4.40", "tokenizers>=0.19")
    .add_local_python_source("diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens")

# Mirrors of the format contract in model/text_encoder.py + model/
# audio_llm_captioner.py (duplicated here so the slim image needn't import torch
# via model.text_encoder). Keep in sync if those constants move.
# Any audio-LLM caption (v1, v2, …) is the "new long format" for the deep
# analysis; the marker-mix table still breaks out the exact version so a prompt
# bump (audio_llm_v1 -> v2) shows up as a re-caption rollout, not a regression.
NEW_PREFIX = "audio_llm_"
CHUNK_TOKENS = 72
MAX_CHUNKS = 16
MAX_CHARS = 3000
WORD_BAND = (150, 350)   # the caption prompt's "Aim for 150-350 words"

# Facets the caption prompt asks for (one short sentence each). Coverage of
# these is a proxy for "multi-facet richness" — a rich caption hits most.
FACETS = {
    "drums": ("drum", "percussion", "beat", "kick", "snare", "hi-hat", "hat ",
              "groove", "rhythm", "clap", "shaker", "tom "),
    "bass": ("bass", "808", "sub-bass", "sub bass", "low-end", "low end",
             "bassline", "bass line"),
    "vocals": ("vocal", "voice", "sing", "sung", "rap", "choir", "falsetto",
               "croon", "instrumental", "a cappella", "acappella", "lyric"),
    "instruments": ("guitar", "piano", "synth", "keys", "keyboard", "string",
                    "horn", "sax", "violin", "organ", "pad ", "pluck", "brass",
                    "flute", "cello", "trumpet", "harp", "marimba", "arpeggi"),
    "production": ("production", "mix", "reverb", "delay", "compress", "lo-fi",
                   "lofi", "distort", "saturat", "master", "stereo", "warm",
                   "gritty", "polished", "clean ", "muddy", "crisp", "analog"),
    "arc": ("build", "drop", "intro", "verse", "chorus", "bridge", "outro",
            "section", "transition", "evolv", "climax", "breakdown", "develop",
            "progress", "swell", "crescendo"),
}

REFUSAL = ("i'm sorry", "i am sorry", "i cannot", "i can't", "i can not",
           "as an ai", "i'm unable", "i am unable", "unable to provide",
           "i do not have", "i don't have access", "sorry, i")
ARTIFACTS = ("```", "\n- ", "\n* ", "\n1.", "\n2.", "•", "## ", "**genre",
             "here is", "here's a", "description:")


def _pctiles(xs):
    import numpy as np
    if not xs:
        return None
    a = np.array(xs)
    q = np.percentile(a, [0, 10, 25, 50, 75, 90, 99, 100])
    return {"min": q[0], "p10": q[1], "p25": q[2], "p50": q[3],
            "p75": q[4], "p90": q[5], "p99": q[6], "max": q[7],
            "mean": float(a.mean())}


def _show_pct(label, p, unit=""):
    if p is None:
        print(f"  {label}: (none)")
        return
    print(f"  {label}: min {p['min']:.0f}  p10 {p['p10']:.0f}  p50 "
          f"{p['p50']:.0f}  mean {p['mean']:.0f}  p90 {p['p90']:.0f}  p99 "
          f"{p['p99']:.0f}  max {p['max']:.0f}{unit}")


def _chunk_count(tokenizer, text: str) -> int:
    """Reproduce model/text_encoder.py:chunk_text_ids chunk COUNT (uncapped by
    MAX_CHUNKS so we can see the true tail; we report the cap separately)."""
    text = (text or "").strip()
    if not text:
        return 0
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= CHUNK_TOKENS:
        return 1
    return (len(ids) + CHUNK_TOKENS - 1) // CHUNK_TOKENS


def _repetition_loop(text: str) -> bool:
    """Audio-LLMs sometimes loop a phrase. Flag a 5-gram repeated >=3x OR a
    unique-word ratio < 0.40 on a non-trivial caption."""
    words = text.lower().split()
    if len(words) < 20:
        return False
    if len(set(words)) / len(words) < 0.40:
        return True
    from collections import Counter
    grams = Counter(tuple(words[i:i + 5]) for i in range(len(words) - 4))
    return grams.most_common(1)[0][1] >= 3 if grams else False


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
)
def audit(max_tokenize: int = 200_000):
    import json
    import re
    from collections import Counter
    from pathlib import Path

    tags_path = Path("/tokens/tags.json")
    if not tags_path.exists():
        print("no /tokens/tags.json — has auto_tag run on this corpus?")
        return
    print(f"reading {tags_path} ...", flush=True)
    tags = json.loads(tags_path.read_text())
    n = len(tags)
    print(f"tags.json: {n:,} entries\n")

    # --- normalize entries -> (marker, description) and bucket by marker ---
    markers = Counter()
    new_desc, legacy_desc = [], []
    for v in tags.values():
        if isinstance(v, dict):
            m = v.get("captioner") or "(dict/no-marker)"
            d = v.get("description") or ""
        else:
            m = "(plain-string)"
            d = v or ""
        markers[m] += 1
        (new_desc if m.startswith(NEW_PREFIX) else legacy_desc).append(d)

    print("=== marker mix (which captioner produced each entry) ===")
    for m, c in markers.most_common():
        flag = "  <- new long format" if m.startswith(NEW_PREFIX) else ""
        print(f"  {m:<22} {c:>9,}  ({c / n:.1%}){flag}")
    n_new = len(new_desc)
    print(f"\nnew-format ({NEW_PREFIX}*) coverage: {n_new:,} / {n:,} "
          f"({n_new / n:.1%})")
    if n_new == 0:
        print("\nNO new-format captions present — the corpus is still on the "
              "legacy short tags; re-run auto_tag (audio-LLM) before relying on "
              "the chunked-CLAP path.")
        return

    # --- length distribution: chars + words, new vs legacy ---
    print("\n=== length (the 'longer' claim) ===")
    new_chars = [len(d) for d in new_desc]
    new_words = [len(d.split()) for d in new_desc]
    _show_pct("new  chars", _pctiles(new_chars))
    _show_pct("new  words", _pctiles(new_words))
    if legacy_desc:
        _show_pct("legacy chars", _pctiles([len(d) for d in legacy_desc]))
        _show_pct("legacy words", _pctiles([len(d.split()) for d in legacy_desc]))
    in_band = sum(WORD_BAND[0] <= w <= WORD_BAND[1] for w in new_words)
    short = sum(w < 60 for w in new_words)
    empty = sum(c == 0 for c in new_chars)
    trunc = sum(c >= MAX_CHARS - 5 for c in new_chars)
    print(f"  in 150-350-word design band: {in_band:,} ({in_band / n_new:.1%})")
    print(f"  degenerately short (<60 words): {short:,} ({short / n_new:.1%})")
    print(f"  empty: {empty:,} ({empty / n_new:.1%})")
    print(f"  trimmed at {MAX_CHARS}-char cap (truncated mid-arc): "
          f"{trunc:,} ({trunc / n_new:.1%})")

    # --- chunk distribution (the chunked-CLAP payoff) ---
    print(f"\n=== CLAP chunk distribution (72-token windows, cap {MAX_CHUNKS}) ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    # Tokenizing every caption can be slow at corpus scale. If over the cap,
    # take a deterministic stride sample BUT always include the longest-by-char
    # captions (those drive n_tag_chunks / the 16-cap truncation).
    if n_new <= max_tokenize:
        sample = new_desc
        sampled = False
    else:
        order = sorted(range(n_new), key=lambda i: new_chars[i], reverse=True)
        longest = order[: max_tokenize // 2]
        stride = max(1, n_new // (max_tokenize // 2))
        strided = list(range(0, n_new, stride))
        idx = sorted(set(longest) | set(strided))[:max_tokenize]
        sample = [new_desc[i] for i in idx]
        sampled = True
        print(f"  (sampled {len(sample):,} of {n_new:,} captions for chunking — "
              f"longest-by-char + stride; n_tag_chunks below is exact, rates "
              f"approximate)")
    counts = [_chunk_count(tok, d) for d in sample]
    cc = Counter(min(c, MAX_CHUNKS) for c in counts)
    n_samp = len(counts)
    width = min(max(counts, default=1), MAX_CHUNKS)
    over_cap = sum(c > MAX_CHUNKS for c in counts)
    at_one = sum(c == 1 for c in counts)
    print(f"  n_tag_chunks the corpus would train (max, cap {MAX_CHUNKS}): {width}")
    print(f"  chunks-per-caption: " + "  ".join(
        f"{k}:{cc[k]}" for k in sorted(cc)))
    print(f"  N=1 (no gain over old single-vector): {at_one:,} "
          f"({at_one / n_samp:.1%})")
    multi = n_samp - at_one
    print(f"  N>=2 (chunked-CLAP actually engaged): {multi:,} "
          f"({multi / n_samp:.1%})")
    print(f"  over the {MAX_CHUNKS}-chunk cap (tail dropped): {over_cap:,} "
          f"({over_cap / n_samp:.1%})")

    # --- content health: facet coverage + degeneracy ---
    print("\n=== content health (new-format captions) ===")
    facet_hits = {f: 0 for f in FACETS}
    facet_per_cap = []
    refusal = rep = artifact = 0
    for d in new_desc:
        low = d.lower()
        k = 0
        for f, kws in FACETS.items():
            if any(w in low for w in kws):
                facet_hits[f] += 1
                k += 1
        facet_per_cap.append(k)
        if any(r in low for r in REFUSAL):
            refusal += 1
        if _repetition_loop(d):
            rep += 1
        if any(a in low for a in ARTIFACTS):
            artifact += 1
    print("  per-facet coverage:")
    for f in FACETS:
        h = facet_hits[f]
        print(f"    {f:<12} {h:>9,}  ({h / n_new:.1%})")
    fp = _pctiles(facet_per_cap)
    print(f"  facets-per-caption (of {len(FACETS)}): mean {fp['mean']:.1f}  "
          f"p10 {fp['p10']:.0f}  p50 {fp['p50']:.0f}  p90 {fp['p90']:.0f}")
    rich = sum(k >= 4 for k in facet_per_cap)
    thin = sum(k <= 1 for k in facet_per_cap)
    print(f"  rich (>=4 facets): {rich:,} ({rich / n_new:.1%})   "
          f"thin (<=1 facet): {thin:,} ({thin / n_new:.1%})")
    print(f"  refusal/apology text: {refusal:,} ({refusal / n_new:.1%})")
    print(f"  repetition loops: {rep:,} ({rep / n_new:.1%})")
    print(f"  markdown/list/preamble artifacts: {artifact:,} "
          f"({artifact / n_new:.1%})")

    # --- near-duplicate collapse ---
    print("\n=== duplicate collapse (identical captions across songs) ===")
    norm = Counter(re.sub(r"\s+", " ", d.strip().lower()) for d in new_desc if d)
    distinct = len(norm)
    print(f"  distinct captions: {distinct:,} / {n_new:,} new "
          f"({distinct / n_new:.1%} unique)")
    dup_clusters = [(c, t) for t, c in norm.items() if c > 1]
    songs_in_dups = sum(c for c, _ in dup_clusters)
    print(f"  songs sharing a caption with >=1 other: {songs_in_dups:,} "
          f"({songs_in_dups / n_new:.1%})")
    for c, t in sorted(dup_clusters, reverse=True)[:3]:
        print(f"    x{c:>6,}: {t[:90]!r}")

    # --- one-line verdict ---
    print("\n=== verdict ===")
    cov = n_new / n
    band = in_band / n_new
    bad = (refusal + rep + empty) / n_new
    verdict = "HEALTHY" if (cov >= 0.95 and band >= 0.6 and bad < 0.02
                            and multi / n_samp >= 0.5) else "CHECK"
    print(f"  {verdict}: {cov:.0%} on new format, {band:.0%} in the word band, "
          f"{multi / n_samp:.0%} multi-chunk, n_tag_chunks={width}, "
          f"{bad:.1%} degenerate.")


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol},
    timeout=60 * 10,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
)
def examples(n: int = 20):
    """Return up to *n* audio_llm_v1 entries, evenly strided across all matches
    (not the first n, which would cluster in one wave)."""
    import json
    from pathlib import Path

    tags = json.loads(Path("/tokens/tags.json").read_text())
    matches = [(name, v.get("description", "")) for name, v in tags.items()
               if isinstance(v, dict)
               and (v.get("captioner") or "").startswith(NEW_PREFIX)]
    if not matches:
        return []
    stride = max(1, len(matches) // n)
    picked = matches[::stride][:n]
    return [{"name": name, "description": d} for name, d in picked]


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol},
    timeout=60 * 10,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
)
def compare(n_examples: int = 8):
    """Compare every audio_llm_* marker on the two failure modes the v2 prompt
    targets — the 'electronic, folk, and IDM' genre default and the 'vocals are
    sparse'/'occasional vocals' hedge — plus length, and dump a few v2 examples."""
    import json
    import re
    from pathlib import Path

    tags = json.loads(Path("/tokens/tags.json").read_text())
    by_marker: dict[str, list[tuple[str, str]]] = {}
    for name, v in tags.items():
        if isinstance(v, dict) and (v.get("captioner") or "").startswith(NEW_PREFIX):
            by_marker.setdefault(v["captioner"], []).append(
                (name, v.get("description", "")))

    GENRE_DEFAULT = re.compile(r"electronic,?\s+folk,?\s+and\s+idm", re.I)
    # The full v3 BANNED list + the adjective forms ("sparse vocals") the model
    # also leaks, so the rate isn't undercounted.
    VOCAL_HEDGE = re.compile(
        r"vocals?\s+(are|is)\s+(sparse|minimal)"
        r"|(occasional|minimal|some|sparse)\s+vocals"
        r"|there\s+(may|might)\s+be\s+vocals",
        re.I)
    report = {}
    for m, rows in sorted(by_marker.items()):
        descs = [d for _, d in rows]
        n = len(descs)
        words = [len(d.split()) for d in descs] or [0]
        report[m] = {
            "n": n,
            "mean_words": sum(words) / len(words),
            "genre_default": sum(bool(GENRE_DEFAULT.search(d)) for d in descs),
            "vocal_hedge": sum(bool(VOCAL_HEDGE.search(d)) for d in descs),
        }

    newest = max(by_marker, default=None)  # audio_llm_v3 > v2 > v1 lexicographically
    latest = by_marker.get(newest, [])
    stride = max(1, len(latest) // n_examples)
    examples = [{"name": nm, "description": d}
                for nm, d in latest[::stride][:n_examples]]
    return {"report": report, "examples": examples, "newest": newest}


@app.local_entrypoint()
def main(max_tokenize: int = 200_000):
    audit.remote(max_tokenize=max_tokenize)


@app.local_entrypoint()
def verify(n_examples: int = 8):
    res = compare.remote(n_examples=n_examples)
    print("\n=== audio_llm marker comparison (newest = current prompt) ===")
    print(f"{'marker':<16} {'n':>8} {'mean_w':>7} {'genre-default':>14} "
          f"{'vocal-hedge':>12}")
    for m, r in res["report"].items():
        n = r["n"] or 1
        print(f"{m:<16} {r['n']:>8,} {r['mean_words']:>7.0f} "
              f"{r['genre_default']:>7,} ({r['genre_default']/n:>4.1%}) "
              f"{r['vocal_hedge']:>6,} ({r['vocal_hedge']/n:>4.1%})")
    print(f"\n=== {res.get('newest')} examples ===")
    for i, e in enumerate(res["examples"], 1):
        print(f"\n--- [{i}] {e['name']} ---\n{e['description']}")


@app.local_entrypoint()
def pull(n: int = 20, out: str = "/tmp/audio_llm_v1_examples.json"):
    import json

    rows = examples.remote(n=n)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(rows)} examples -> {out}\n")
    for i, r in enumerate(rows, 1):
        print(f"--- [{i}] {r['name']} ---")
        print(r["description"])
        print()
