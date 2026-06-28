"""Tests for diskrot/align_lyrics.py — the pure forced-alignment core.

The torchaudio aligner itself needs model weights (integration-only), but the
normalization, span→word merge with monotonic sanity, idempotent stamp, and the
refine_entry orchestration are pure and fully covered here via a fake aligner."""
from __future__ import annotations

from diskrot.align_lyrics import (
    ALIGN_VERSION,
    frames_to_seconds,
    is_aligned,
    mark_aligned,
    merge_refined_words,
    normalize_for_alignment,
    refine_entry,
)


def test_normalize():
    assert normalize_for_alignment("Hello,") == "hello"
    assert normalize_for_alignment("don't") == "don't"
    assert normalize_for_alignment("123") == ""
    assert normalize_for_alignment("Café") == "café"   # diacritics kept
    assert normalize_for_alignment("  ") == ""


def test_frames_to_seconds():
    s, e = frames_to_seconds(10, 20, n_frames=100, audio_seconds=10.0)
    assert (round(s, 2), round(e, 2)) == (1.0, 2.0)
    assert frames_to_seconds(0, 0, n_frames=0, audio_seconds=10.0) == (0.0, 0.0)
    # end always nudged past start
    s, e = frames_to_seconds(5, 5, n_frames=100, audio_seconds=10.0)
    assert e > s


def test_merge_applies_spans_and_preserves_fields():
    words = [
        {"word": "a", "start": 0.0, "end": 1.0, "prob": 0.9},
        {"word": "b", "start": 1.0, "end": 2.0, "prob": 0.8},
    ]
    refined = [(0.1, 0.5), (1.2, 1.8)]
    out = merge_refined_words(words, refined)
    assert [w["word"] for w in out] == ["a", "b"]
    assert out[0]["prob"] == 0.9 and out[1]["prob"] == 0.8   # other fields preserved
    assert (out[0]["start"], out[0]["end"]) == (0.1, 0.5)
    assert (out[1]["start"], out[1]["end"]) == (1.2, 1.8)


def test_merge_enforces_monotonic_onsets():
    words = [{"word": "a", "start": 0.0, "end": 1.0}, {"word": "b", "start": 1.0, "end": 2.0}]
    # second span starts BEFORE the first ends → clamp to prev_end, don't reorder.
    out = merge_refined_words(words, [(0.1, 0.5), (0.3, 0.9)])
    assert out[1]["start"] == 0.5  # clamped to prev_end
    assert out[1]["end"] >= out[1]["start"]


def test_merge_none_keeps_original_and_advances_cursor():
    words = [
        {"word": "a", "start": 0.0, "end": 3.0},
        {"word": "b", "start": 3.0, "end": 4.0},
    ]
    # a unaligned (None) → keep original; b's refined onset (2.5) is before a's end (3.0)
    # → clamped to 3.0 so it can't precede the kept word.
    out = merge_refined_words(words, [None, (2.5, 3.8)])
    assert (out[0]["start"], out[0]["end"]) == (0.0, 3.0)   # untouched
    assert out[1]["start"] == 3.0


def test_merge_length_mismatch_returns_originals():
    words = [{"word": "a", "start": 0.0, "end": 1.0}]
    out = merge_refined_words(words, [(0.1, 0.5), (0.2, 0.6)])  # wrong length
    assert out == words and out is not words  # copies, untouched


def test_aligned_stamp_roundtrip():
    e = {"words": []}
    assert not is_aligned(e)
    mark_aligned(e)
    assert e["aligned"] == ALIGN_VERSION and is_aligned(e)


class _FakeAligner:
    def __init__(self, spans=None, raises=False):
        self.spans, self.raises = spans, raises

    def align(self, wav, secs, words):
        if self.raises:
            raise RuntimeError("boom")
        return self.spans


def test_refine_entry_applies_alignment():
    entry = {"words": [{"word": "a", "start": 0.0, "end": 1.0}], "language": "en"}
    aligner = _FakeAligner(spans=[(0.2, 0.7)])
    updated, changed = refine_entry(entry, aligner, lambda: (None, 10.0))
    assert changed
    assert (updated["words"][0]["start"], updated["words"][0]["end"]) == (0.2, 0.7)
    assert updated["language"] == "en"        # entry fields preserved
    assert is_aligned(updated)


def test_refine_entry_no_words_is_stamped_unchanged():
    entry = {"words": [], "language": "en"}
    updated, changed = refine_entry(entry, _FakeAligner(), lambda: (None, 10.0))
    assert not changed and is_aligned(updated)


def test_refine_entry_failure_never_regresses():
    entry = {"words": [{"word": "a", "start": 0.0, "end": 1.0}]}
    # Aligner raises → original timestamps preserved, still stamped so it isn't retried.
    updated, changed = refine_entry(entry, _FakeAligner(raises=True), lambda: (None, 10.0))
    assert not changed
    assert (updated["words"][0]["start"], updated["words"][0]["end"]) == (0.0, 1.0)
    assert is_aligned(updated)
