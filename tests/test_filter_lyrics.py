"""Tests for diskrot/filter_lyrics.py — Whisper-hallucination nulling.

Two surfaces: (1) ``hallucination_reason`` must flag the observed failure modes
(one invented sentence over a whole song, short caption-artifact phrases)
WITHOUT flagging real lyrics — a false positive silently converts a vocal song
to ``<instrumental>``; and (2) ``filter_lyrics`` must dry-run by default, null
flagged entries in place on ``--apply``, leave everything else byte-identical,
and be idempotent.
"""
from __future__ import annotations

import json

from diskrot.filter_lyrics import filter_lyrics, hallucination_reason


def _entry(text: str, gender: str | None = "female") -> dict:
    words = [
        {"word": w, "start": float(i), "end": float(i) + 0.5}
        for i, w in enumerate(text.split())
    ]
    return {"text": text, "words": words, "gender": gender}


REAL_LYRIC = _entry(
    "written in home burning shame wicked of your shame there's this remain "
    "as in your child you play i lose your days we belong to deserts"
)


def test_flags_short_invented_captions():
    # The dominant observed mode: Whisper inventing one sentence over a song.
    assert hallucination_reason(_entry("Thank you.")) == "short"
    assert hallucination_reason(_entry("We'll be right back.")) == "short"
    assert hallucination_reason(_entry("oh")) == "short"


def test_flags_short_junk_phrases_only():
    junk = _entry("Thank you for watching! Thank you for watching! Thank you so much!")
    assert hallucination_reason(junk).startswith("junk:")
    # The same phrase inside a long real transcript does NOT condemn the song
    # (Whisper tacks it onto the end of genuine lyrics).
    long_real = _entry(REAL_LYRIC["text"] + " thank you for watching")
    assert hallucination_reason(long_real) is None


def test_keeps_real_lyrics_and_instrumentals():
    assert hallucination_reason(REAL_LYRIC) is None
    assert hallucination_reason(None) is None  # already instrumental
    # Words that fail the schema predicate don't count toward MIN_WORDS.
    bogus = {"text": "a b c d e f g", "words": [{"word": "a"}] * 7}
    assert hallucination_reason(bogus) == "short"


def _write_shards(lyrics_dir) -> None:
    lyrics_dir.mkdir()
    (lyrics_dir / "lyrics_000.json").write_text(json.dumps({
        "real_song": REAL_LYRIC,
        "halluc_song": _entry("Thank you."),
        "instrumental": None,
    }))
    (lyrics_dir / "lyrics_001.json").write_text(json.dumps({
        "another_real": REAL_LYRIC,
    }))


def test_filter_lyrics_dry_run_then_apply(tmp_path):
    lyrics_dir = tmp_path / "lyrics"
    _write_shards(lyrics_dir)
    untouched_before = (lyrics_dir / "lyrics_001.json").read_bytes()

    stats = filter_lyrics(lyrics_dir, apply=False, verbose=False)
    assert stats == {"checked": 3, "flagged": 1, "by_reason": {"short": 1},
                     "shards_rewritten": 0, "already_null": 1,
                     "vocal_ready": 2, "total_transcribed": 4}
    data = json.loads((lyrics_dir / "lyrics_000.json").read_text())
    assert data["halluc_song"] is not None  # dry run wrote nothing

    commits = []
    stats = filter_lyrics(lyrics_dir, apply=True, verbose=False,
                          commit_cb=lambda: commits.append(1))
    assert stats["flagged"] == 1 and stats["shards_rewritten"] == 1
    assert commits == [1]  # one commit per rewritten shard
    data = json.loads((lyrics_dir / "lyrics_000.json").read_text())
    assert data["halluc_song"] is None        # nulled = trains <instrumental>
    assert data["real_song"] == REAL_LYRIC    # survivors untouched
    assert data["instrumental"] is None
    # The all-clean shard is never rewritten.
    assert (lyrics_dir / "lyrics_001.json").read_bytes() == untouched_before

    # Idempotent: nulled entries are wordless, nothing left to flag.
    stats = filter_lyrics(lyrics_dir, apply=True, verbose=False)
    assert stats["flagged"] == 0 and stats["shards_rewritten"] == 0
