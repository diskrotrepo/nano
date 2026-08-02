"""Forced alignment to sharpen lyric word timestamps (better singing supervision).

Whisper's word timestamps are decoded jointly with the text and drift by tens to
hundreds of ms — fine for ASR, loose for teaching the model WHEN each word is sung.
This re-aligns the EXISTING transcript words to the audio with a CTC forced aligner
(torchaudio ``MMS_FA``, multilingual, wav2vec2-based), optionally on Demucs-isolated
vocals for extra sharpness. Tighter word onsets directly help nano's near-monotonic
sung-alignment learning and the dataset's vocal-crop biasing (which steers crops onto
transcribed words).

Design principle: alignment is a *refinement, never a regression*. Any word/entry the
aligner can't handle (unsupported script, silence, a tokenizer miss) keeps its original
Whisper timestamp. So a partial or failed pass is always safe.

Pure, unit-tested core here (normalization, span→word merge with monotonic sanity, the
idempotent ``aligned`` stamp). The torchaudio aligner (``ForcedAligner``) is lazy and
integration-only — ``modal_align_lyrics.py`` is the GPU stage that drives it over the
sharded lyrics store.
"""
from __future__ import annotations

import re
import unicodedata

# Bump when the alignment method changes so a re-run re-aligns prior entries.
ALIGN_VERSION = 1
TARGET_SR = 16_000  # MMS_FA operates at 16 kHz

# Latin letters + common diacritic ranges + intra-word apostrophe. The MMS_FA
# tokenizer expects a lowercase romanized charset; words that reduce to "" here are
# left with their original timestamp rather than mis-aligned.
_KEEP = re.compile(r"[^a-zÀ-ɏ']")


def normalize_for_alignment(word: str) -> str:
    """Lowercase + strip a word to the aligner's charset. '' if nothing usable."""
    w = unicodedata.normalize("NFKC", str(word)).lower()
    return _KEEP.sub("", w)


def is_aligned(entry: dict) -> bool:
    """True if this lyrics entry was already aligned at the current version
    (re-runs skip it — idempotent/resumable)."""
    return isinstance(entry, dict) and entry.get("aligned") == ALIGN_VERSION


def mark_aligned(entry: dict) -> dict:
    """Stamp an entry as aligned at the current version (in place, returns it)."""
    entry["aligned"] = ALIGN_VERSION
    return entry


def frames_to_seconds(
    start_frame: int, end_frame: int, n_frames: int, audio_seconds: float,
) -> tuple[float, float]:
    """Convert CTC emission frame indices to seconds via the emission frame rate
    (n_frames over audio_seconds). end is nudged past start so spans never collapse."""
    if n_frames <= 0 or audio_seconds <= 0:
        return 0.0, 0.0
    sec_per_frame = audio_seconds / n_frames
    s = max(0.0, start_frame * sec_per_frame)
    e = max(s + 1e-3, end_frame * sec_per_frame)
    return s, e


def merge_refined_words(
    orig_words: list[dict], refined: list[tuple[float, float] | None],
) -> list[dict]:
    """Apply refined ``(start, end)`` spans onto the original words, positionally.

    Preserves each word's text and every other field; only overwrites ``start`` /
    ``end`` where a span is present (None → keep the original timestamp). Enforces
    non-decreasing onsets so a stray refined span can't reorder the stream. If the
    span list length doesn't match (defensive), returns the originals untouched."""
    if len(refined) != len(orig_words):
        return [dict(w) for w in orig_words]
    out: list[dict] = []
    prev_end = 0.0
    for w, span in zip(orig_words, refined):
        nw = dict(w)
        if span is not None:
            s, e = float(span[0]), float(span[1])
            s = max(s, 0.0)
            e = max(e, s + 1e-3)
            # Keep onsets monotonic vs the previous refined word (clamp, don't drop).
            if s < prev_end:
                s = prev_end
                e = max(e, s + 1e-3)
            nw["start"] = round(s, 3)
            nw["end"] = round(e, 3)
            prev_end = e
        else:
            # Unaligned word: advance the monotonic cursor past its original end so a
            # later refined word can't be pulled before it.
            try:
                prev_end = max(prev_end, float(nw.get("end", prev_end)))
            except (TypeError, ValueError):
                pass
        out.append(nw)
    return out


class ForcedAligner:
    """Lazy torchaudio ``MMS_FA`` forced aligner. Construct once per worker (loads the
    wav2vec2 model + tokenizer + aligner), then call ``align`` per song.

    Integration-only (needs the bundle weights), so it isn't unit-tested; the pure
    span→word merge it feeds IS. ``align`` returns a span (or None) per INPUT word,
    positionally — words that normalize away or fail tokenization yield None so the
    caller keeps their original timestamp."""

    def __init__(self, device: str = "cpu"):
        import torch
        from torchaudio.pipelines import MMS_FA as bundle

        self.device = device
        self.torch = torch
        self.model = bundle.get_model().to(device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()

    def align(
        self, waveform_16k, audio_seconds: float, words: list[str],
    ) -> list[tuple[float, float] | None]:
        """waveform_16k: 1-D float tensor at 16 kHz. ``words``: raw word strings.
        Returns one (start_s, end_s) | None per input word (positional)."""
        torch = self.torch
        norm = [normalize_for_alignment(w) for w in words]
        keep_idx = [i for i, w in enumerate(norm) if w]
        result: list[tuple[float, float] | None] = [None] * len(words)
        if not keep_idx:
            return result
        tokens = self.tokenizer([norm[i] for i in keep_idx])
        wav = waveform_16k.to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        with torch.inference_mode():
            emission, _ = self.model(wav)
        n_frames = emission.shape[1]
        try:
            token_spans = self.aligner(emission[0], tokens)
        except Exception:
            return result  # alignment failed wholesale → all original timestamps
        # token_spans: one list of TokenSpan per kept word; first/last token frames
        # bound the word.
        for slot, spans in zip(keep_idx, token_spans):
            if not spans:
                continue
            s_frame = spans[0].start
            e_frame = spans[-1].end
            result[slot] = frames_to_seconds(s_frame, e_frame, n_frames, audio_seconds)
        return result


def refine_entry(
    entry: dict,
    aligner: "ForcedAligner",
    load_waveform,
) -> tuple[dict, bool]:
    """Re-align one lyrics entry. ``load_waveform()`` returns ``(waveform_16k,
    audio_seconds)`` for the song (Demucs vocals or the mix). Returns ``(updated_entry,
    changed)``. Entries with no usable words, or any failure, come back unchanged but
    still stamped aligned (so they aren't retried forever)."""
    words = entry.get("words") if isinstance(entry, dict) else None
    if not words:
        return mark_aligned(dict(entry) if isinstance(entry, dict) else {}), False
    try:
        wav, secs = load_waveform()
        spans = aligner.align(wav, secs, [w.get("word", "") for w in words])
        new_words = merge_refined_words(words, spans)
    except Exception:
        new_words = [dict(w) for w in words]  # never regress on failure
    updated = dict(entry)
    changed = new_words != words
    updated["words"] = new_words
    mark_aligned(updated)
    return updated, changed
