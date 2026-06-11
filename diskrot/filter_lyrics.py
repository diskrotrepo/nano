"""Null out Whisper-hallucinated lyric entries in the sharded ``lyrics/`` dir.

Whisper invents captions over instrumental audio — "Thank you.", "Thanks for
watching!", "We'll be right back." (YouTube-caption artifacts from its training
data). Any entry with usable words trains as ``<vocals>`` with those words
cross-attended, so a hallucinated caption mislabels an instrumental song AND
feeds it garbage lyrics. This pass rewrites such entries to ``null`` — the
existing transcribed-but-wordless convention, which trains as
``<instrumental>`` (see TokenDataset). A 2026-06-11 sweep of the first ~48k
with-words transcripts found ~23% with <=5 words ("Thank you." alone: 2,808
songs) — almost all of it this failure mode.

An entry is nulled when:

- it has fewer than ``MIN_WORDS`` valid words (the dominant case: one invented
  sentence over a whole song), or
- its text contains a known caption-artifact phrase (``JUNK_PHRASES``) and the
  transcript is short (< ``JUNK_MAX_WORDS`` words) — a long real lyric that
  happens to mention e.g. "subscribe" survives, and Whisper's habit of tacking
  "Thank you for watching." onto the END of a real transcript doesn't cost the
  song (only its last few junk words ride along).

Dry-run by default; ``--apply`` rewrites only the changed shards, atomically
(same temp+rename pattern as transcribe_lyrics). Idempotent: nulled entries
are already wordless on a re-run. Ordering: run AFTER transcribe completes —
the transcribe orchestrator holds shard contents in memory and its next flush
would clobber concurrent edits — and BEFORE phonemize, so junk never enters
the phoneme store.

CLI::

    python -m diskrot.filter_lyrics --lyrics-dir /tokens/lyrics [--apply]
"""
from __future__ import annotations

import json
from pathlib import Path

from diskrot.transcribe_lyrics import _atomic_write_json, is_valid_word

# Below this many valid words a "transcript" is a caption artifact, not lyrics.
# Real vocal songs with fewer usable words than this train fine as
# <instrumental> — vocally they're chant-level at most.
MIN_WORDS = 6

# Caption-artifact phrases (lowercase substring match). These come from
# Whisper's subtitle-corpus training data, not from any song.
JUNK_PHRASES = (
    "thank you for watching",
    "thanks for watching",
    "we'll be right back",
    "see you next time",
    "see you in the next video",
    "subscribe",
    "subtitles",
    "copyright",
    "transcript",
    "www.",
    ".com",
)

# A junk phrase only condemns a short transcript; past this many words the
# song is overwhelmingly real lyrics and keeping it costs a few noise words.
JUNK_MAX_WORDS = 30


def hallucination_reason(entry) -> str | None:
    """Why ``entry`` (a per-song lyrics dict) is a hallucination, or None.

    Returns ``"short"`` / ``"junk:<phrase>"`` for entries that should be
    nulled. ``None`` (already instrumental) and non-dict entries are never
    flagged."""
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


def filter_lyrics(
    lyrics_dir: str | Path,
    apply: bool = False,
    verbose: bool = True,
    commit_cb=None,
) -> dict:
    """Sweep every ``lyrics_*.json`` shard and null hallucinated entries.

    Dry-run unless ``apply``; only shards with at least one flagged entry are
    rewritten (atomically). ``commit_cb`` (e.g. a Modal volume commit) runs
    after each shard rewrite. Returns
    ``{"checked", "flagged", "by_reason", "shards_rewritten"}``."""
    lyrics_dir = Path(lyrics_dir)
    n_checked = n_flagged = n_shards = 0
    by_reason: dict[str, int] = {}

    for shard in sorted(lyrics_dir.glob("lyrics_*.json")):
        data = json.loads(shard.read_text())
        flagged = []
        for name, entry in data.items():
            if entry is None:
                continue
            n_checked += 1
            reason = hallucination_reason(entry)
            if reason is None:
                continue
            flagged.append(name)
            family = reason.split(":")[0]
            by_reason[family] = by_reason.get(family, 0) + 1
        n_flagged += len(flagged)
        if flagged and apply:
            for name in flagged:
                data[name] = None
            _atomic_write_json(shard, data)
            n_shards += 1
            if commit_cb is not None:
                commit_cb()
        if verbose and flagged:
            print(f"[filter] {shard.name}: {len(flagged)} hallucinated"
                  f"{' -> nulled' if apply else ' (dry run)'}", flush=True)

    if verbose:
        mode = "applied" if apply else "DRY RUN (pass --apply to write)"
        print(f"[filter] {mode}: {n_flagged}/{n_checked} with-words entries "
              f"flagged {by_reason}, {n_shards} shard(s) rewritten", flush=True)
    return {
        "checked": n_checked,
        "flagged": n_flagged,
        "by_reason": by_reason,
        "shards_rewritten": n_shards,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lyrics-dir", type=str, required=True,
                   help="the sharded lyrics/ dir written by transcribe_lyrics")
    p.add_argument("--apply", action="store_true",
                   help="rewrite shards (default: dry-run report only)")
    args = p.parse_args()
    filter_lyrics(args.lyrics_dir, apply=args.apply)
