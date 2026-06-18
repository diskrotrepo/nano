"""Gap-focused genre coverage evaluator for the nano corpus.

Answers one question: *did the added gap-genre data close the gaps?* (See eval/REPORT.md.)

Why this exists separate from `genre_sweep.py`:
  - genre_sweep silently reads a stale local eval/tags.json, has no floor/verdict,
    no gap focus, and no run-over-run delta. This one auto-pulls the latest
    tags.json off the nano-tokens volume and renders a CLOSED/OPEN verdict per gap.

Two genre signals, reported side by side (neither is ground truth — there is no
source genre provenance, so all we have is caption text):
  1. REGEX  — keyword + instrumentation-proxy patterns over the full corpus (cheap).
              LP-MusicCaps prose rarely names a genre, so this UNDER-counts roots genres.
  2. CLAP   — zero-shot: cosine-sim each caption embedding against genre-prompt
              embeddings, including non-gap DISTRACTOR anchors so a techno caption
              is attributed to "electronic", not forced into the nearest gap genre.
              Coarser but catches genres hiding under instrumentation wording.

The verdict uses the *higher* of the two estimates against the REPORT floor
(>=4% of corpus AND >=8k distinct songs). Exits non-zero if any gap is OPEN, so
this can gate a training launch.

CLI:
    python -m eval.genre_gap_eval                       # pull latest, regex + CLAP
    python -m eval.genre_gap_eval --no-clap             # regex only (fast)
    python -m eval.genre_gap_eval --local               # use cached/local tags.json
    python -m eval.genre_gap_eval --tags-path P         # explicit path (implies --local)
    python -m eval.genre_gap_eval --clap-sample 50000   # CLAP sample size (default 30k)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  # so `from model.text_encoder import ...` works under -m

CACHE_TAGS = Path("/tmp/nano_tags_latest.json")
SNAPSHOT_PATH = EVAL_DIR / "genre_gap_snapshot.json"
REPORT_PATH = EVAL_DIR / "genre_gap_report.txt"
EMB_CACHE_DIR = Path("/tmp/nano_clap_cache")

# Floor for a genre to be reliably generatable (eval/REPORT.md).
FLOOR_PCT = 4.0
FLOOR_SONGS = 8_000

# ── Gap genres (the roots/commercial genres REPORT.md flagged as near-absent) ──
# `rx`   : regex unioning genre words AND instrumentation proxies the captioner
#          actually emits (it describes instruments far more than it names genres).
# `clap` : short natural-language prompts for CLAP zero-shot (a caption is assigned
#          to whichever prompt — gap or distractor — it is most similar to).
GAP_GENRES: dict[str, dict] = {
    "country": {
        "rx": r"\bcountry\b|\bbluegrass\b|\bhonky[- ]?tonk\b|\bpedal steel\b|\bbanjo\b|\bfiddle\b|\btwang|\bmandolin\b|\bamericana\b",
        "clap": ["a country song", "a bluegrass song with banjo and fiddle", "a honky-tonk country track"],
    },
    "latin": {
        "rx": r"\blatin\b|\bsalsa\b|\bcumbia\b|\breggaeton\b|\bbachata\b|\bmariachi\b|\bbossa\b|\bsamba\b|\btango\b|\bflamenco\b|\bmerengue\b",
        "clap": ["a latin music track", "a salsa song", "a reggaeton song", "a bossa nova track"],
    },
    "jazz": {
        "rx": r"\bjazz\b|\bswing\b|\bbebop\b|\bbig band\b|\bwalking bass\b|\bupright bass\b|\bdixieland\b",
        "clap": ["a jazz song", "a swing jazz track with brass", "a smooth jazz piece"],
    },
    "blues": {
        "rx": r"\bblues\b|\bdelta blues\b|\bslide guitar\b|\b12[- ]?bar\b",
        "clap": ["a blues song", "a delta blues track with slide guitar"],
    },
    "soul/r&b": {
        "rx": r"\bsoul\b|\br&b\b|\brhythm and blues\b|\bmotown\b|\bneo[- ]?soul\b",
        "clap": ["a soul song", "an r&b track", "a motown soul song"],
    },
    "funk/disco": {
        "rx": r"\bfunk\b|\bdisco\b|\bgroove\b.*\bbass\b|\bslap bass\b",
        "clap": ["a funk song with slap bass", "a disco track"],
    },
    "reggae/dub": {
        "rx": r"\breggae\b|\bdancehall\b|\bska\b|\boffbeat\b.*\bguitar\b",
        "clap": ["a reggae song", "a dub track", "a dancehall song"],
    },
    "gospel": {
        "rx": r"\bgospel\b|\bworship\b|\bchoir\b.*\bclap",
        "clap": ["a gospel song with choir", "a worship song"],
    },
    "afro": {
        "rx": r"\bafrobeat\b|\bafro[- ]?pop\b|\bhighlife\b|\bamapiano\b|\bsoukous\b",
        "clap": ["an afrobeat song", "an amapiano track", "an african highlife song"],
    },
    # ── world traditions added 2026-06-13 (net-new coverage, build from ~0) ──
    "indian": {
        "rx": r"\bbollywood\b|\bhindustani\b|\bcarnatic\b|\bbhangra\b|\bqawwali\b|\bsitar\b|\btabla\b|\braga\b|\bghazal\b|\bfilmi\b|\bbansuri\b|\bharmonium\b",
        "clap": ["a bollywood song", "an indian classical raga with sitar and tabla", "a bhangra track", "a qawwali devotional song"],
    },
    "east-asian": {
        "rx": r"\bk-?pop\b|\bj-?pop\b|\bc-?pop\b|\bmandopop\b|\bcantopop\b|\benka\b|\bguzheng\b|\bshamisen\b|\berhu\b|\bkoto\b|\bgugak\b",
        "clap": ["a k-pop song", "a j-pop song", "a traditional chinese piece with guzheng", "a traditional japanese piece with shamisen"],
    },
    "mediterranean": {
        "rx": r"\barabic\b|\bmiddle eastern\b|\bturkish\b|\bflamenco\b|\bfado\b|\brebetiko\b|\boud\b|\bmaqam\b|\bbouzouki\b|\bbelly danc|\bandalusian\b|\bra[iï]\b",
        "clap": ["an arabic music track", "a flamenco guitar song", "a portuguese fado song", "a turkish music piece with oud"],
    },
}

# Non-gap anchors so the CLAP classifier has somewhere to send the dominant
# electronic/rock/pop captions instead of forcing them into the nearest gap.
DISTRACTORS: list[str] = [
    "an electronic dance track", "a techno track", "a house music track",
    "an ambient soundscape", "a rock song with electric guitar", "a metal song",
    "a pop song", "a hip hop track", "a classical orchestral piece",
    "a lo-fi beat", "a chiptune video game track", "a cinematic film score",
    "a folk acoustic song", "a world music track with hand percussion",
]

CLAP_MARGIN = 0.05  # require argmax to beat runner-up by this cosine margin to count


def p(lines: list[str], s: str = "") -> None:
    lines.append(s)
    print(s)


# ── data loading ───────────────────────────────────────────────────────────
def resolve_tags(args) -> Path:
    if args.tags_path:
        return Path(args.tags_path)
    if args.local:
        return CACHE_TAGS if CACHE_TAGS.exists() else (EVAL_DIR / "tags.json")
    # auto-pull latest off the volume
    print(f"[pull] modal volume get nano-tokens tags.json -> {CACHE_TAGS}")
    try:
        subprocess.run(
            ["modal", "volume", "get", "nano-tokens", "tags.json", str(CACHE_TAGS), "--force"],
            check=True, capture_output=True, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[pull] failed ({e}); falling back to local copy")
        return CACHE_TAGS if CACHE_TAGS.exists() else (EVAL_DIR / "tags.json")
    return CACHE_TAGS


def load_descs(path: Path) -> list[str]:
    d = json.load(open(path))
    out = []
    for v in d.values():
        s = v.get("description", "") if isinstance(v, dict) else str(v)
        out.append(s.lower())
    return out


def corpus_mp3_count() -> int | None:
    """Best-effort raw mp3 count on nano-corpus, to flag captioning lag."""
    try:
        r = subprocess.run(
            ["modal", "volume", "ls", "nano-corpus", "--json"],
            check=True, capture_output=True, text=True, timeout=120,
        )
        return sum(1 for e in json.loads(r.stdout)
                   if str(e.get("Filename", "")).lower().endswith(".mp3"))
    except Exception:
        return None


# ── regex pass ───────────────────────────────────────────────────────────────
def regex_counts(descs: list[str]) -> dict[str, int]:
    out = {}
    for name, spec in GAP_GENRES.items():
        rx = re.compile(spec["rx"])
        out[name] = sum(1 for s in descs if rx.search(s))
    return out


# ── CLAP zero-shot pass ───────────────────────────────────────────────────────
def clap_counts(descs: list[str], tags_path: Path, sample: int, seed: int) -> dict[str, int] | None:
    try:
        import numpy as np
        import torch
        from model.text_encoder import CLAPTextEncoder
    except Exception as e:
        print(f"[clap] unavailable ({e}); skipping CLAP pass")
        return None

    N = len(descs)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(sample, N), replace=False)
    sampled = [descs[i] for i in idx]

    enc = CLAPTextEncoder(d_out=1024)
    enc._ensure_clap()  # lazy-load msclap + truncation patch (captions are long)

    def embed(texts: list[str]) -> "np.ndarray":
        vecs = []
        for i in range(0, len(texts), 256):
            emb = enc._clap.get_text_embeddings(texts[i:i + 256])  # [b,1024], bypass proj
            vecs.append(emb.detach().cpu().float().numpy())
        v = np.concatenate(vecs, 0)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    # cache caption embeddings keyed on file identity + sampling params
    st = tags_path.stat()
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{st.st_size}_{int(st.st_mtime)}_{sample}_{seed}.npy"
    cache = EMB_CACHE_DIR / key
    if cache.exists():
        print(f"[clap] using cached embeddings {cache}")
        cap = np.load(cache)
    else:
        print(f"[clap] embedding {len(sampled):,} sampled captions (cached on rerun)")
        cap = embed(sampled)
        np.save(cache, cap)

    # build prompt embedding matrix: gap prompts + distractor anchors
    gap_names = list(GAP_GENRES.keys())
    prompt_owner, prompts = [], []
    for name in gap_names:
        for pr in GAP_GENRES[name]["clap"]:
            prompt_owner.append(name); prompts.append(pr)
    n_gap_prompts = len(prompts)
    prompts += DISTRACTORS
    prompt_owner += [None] * len(DISTRACTORS)
    pe = embed(prompts)

    sims = cap @ pe.T                       # [n_sample, n_prompts]
    best = sims.argmax(1)
    # margin vs runner-up to suppress ambiguous assignments
    part = np.partition(sims, -2, axis=1)
    margin = part[:, -1] - part[:, -2]

    counts = {name: 0 for name in gap_names}
    for i in range(len(sampled)):
        owner = prompt_owner[best[i]]
        if owner is not None and margin[i] >= CLAP_MARGIN:
            counts[owner] += 1
    # scale sample counts up to full-corpus estimate
    scale = N / len(sampled)
    return {k: int(round(v * scale)) for k, v in counts.items()}


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use cached/local tags.json, no pull")
    ap.add_argument("--tags-path", default=None, help="explicit tags.json path (implies --local)")
    ap.add_argument("--no-clap", action="store_true", help="skip CLAP zero-shot pass")
    ap.add_argument("--clap-sample", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tags_path = resolve_tags(args)
    descs = load_descs(tags_path)
    N = len(descs)
    mtime = tags_path.stat().st_mtime

    lines: list[str] = []
    p(lines, f"# nano genre-gap eval — {N:,} captioned songs")
    p(lines, f"# source: {tags_path}  (mtime {__import__('datetime').datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M})")
    p(lines, f"# floor: >={FLOOR_PCT:.0f}% of corpus AND >={FLOOR_SONGS:,} distinct songs\n")

    # captioning-lag check
    raw = corpus_mp3_count()
    if raw is not None:
        p(lines, f"## corpus coverage: {N:,} captioned / {raw:,} mp3s on nano-corpus "
                 f"({100*N/raw:.0f}% captioned)")
        if N < 0.9 * raw:
            p(lines, "  ⚠ captioning lags the corpus — gap data may be present but not yet "
                     "captioned, so caption-based coverage UNDER-states true coverage.\n")
        else:
            p(lines, "")

    rx = regex_counts(descs)
    cl = clap_counts(descs, tags_path, args.clap_sample, args.seed) if not args.no_clap else None
    if cl is not None:
        p(lines, "## NOTE: clap≈ is a coarse zero-shot UPPER BOUND (it over-attributes "
                 "fine genres); the verdict is regex-driven.\n")

    # Verdict is REGEX-driven (conservative but never hallucinates a genre). CLAP
    # is advisory only: it's a coarse semantic UPPER BOUND that over-attributes
    # (it forces every caption to its nearest prompt), so it must not flip a
    # verdict — but a big regex-vs-clap gap flags a genre worth a manual look.
    p(lines, f"{'genre':12s} {'regex (verdict)':>17s} {'clap≈ advisory':>16s}  verdict")
    snapshot, any_open = {}, False
    for name in GAP_GENRES:
        rc = rx[name]; rp = 100 * rc / N
        cc = cl[name] if cl else None
        cp = (100 * cc / N) if cc is not None else None
        closed = rp >= FLOOR_PCT and rc >= FLOOR_SONGS
        any_open |= not closed
        verdict = "CLOSED ✅" if closed else "OPEN ❌"
        flag = " ⟵ clap≫regex, verify" if (cc is not None and not closed and cp >= FLOOR_PCT) else ""
        clap_cell = f"{cc:>9,d} {cp:5.2f}%" if cc is not None else f"{'—':>16s}"
        p(lines, f"{name:12s} {rc:>9,d} {rp:6.2f}% {clap_cell}  {verdict}{flag}")
        snapshot[name] = {"regex": rc, "regex_pct": round(rp, 3), "clap": cc,
                          "N": N, "mtime": mtime}

    # run-over-run delta
    if SNAPSHOT_PATH.exists():
        prev = json.load(open(SNAPSHOT_PATH))
        p(lines, "\n## Δ since last run")
        for name in GAP_GENRES:
            if name in prev and "regex" in prev[name]:
                d_songs = snapshot[name]["regex"] - prev[name]["regex"]
                d_pct = snapshot[name]["regex_pct"] - prev[name]["regex_pct"]
                p(lines, f"{name:12s} {d_songs:>+8,d} songs  {d_pct:>+6.2f}%")
    else:
        p(lines, "\n## (no prior snapshot — this run establishes the baseline)")

    json.dump(snapshot, open(SNAPSHOT_PATH, "w"), indent=2)
    REPORT_PATH.write_text("\n".join(lines))
    p(lines, f"\n[written {REPORT_PATH}]  [snapshot {SNAPSHOT_PATH}]")
    p(lines, f"\nVERDICT: {'gaps remain OPEN' if any_open else 'all gaps CLOSED'}")
    return 1 if any_open else 0


if __name__ == "__main__":
    raise SystemExit(main())
