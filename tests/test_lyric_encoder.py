"""Tests for model/lyric_encoder.py — phoneme vocab, g2p, and the encoder.

Two surfaces matter: (1) the frozen phoneme vocab + ``text_to_phoneme_ids`` must
be deterministic and in-range (train and inference both key off the exact id
mapping — a mismatch silently destroys intelligibility), and (2) the encoder must
emit ``[B, L, d_model]`` + a pass-through mask, with padded positions not leaking.
The g2p tests skip when ``g2p_en``/nltk data is unavailable (CI without the
download); the vocab and encoder tests run unconditionally.
"""
from __future__ import annotations

import pytest
import torch

from model.lyric_encoder import (
    BOS_PHONEME_ID,
    GENDER_LABELS,
    GENDER_TOKEN_TO_ID,
    KEY_LABELS,
    KEY_TOKEN_TO_ID,
    NO_SECTION_ID,
    PAD_PHONEME_ID,
    PHONEME_VOCAB,
    PHONEME_VOCAB_SIZE,
    STRUCTURE_LABELS,
    STRUCTURE_TOKEN_TO_ID,
    TEMPO_BPM_EDGES,
    UNKNOWN_GENDER_ID,
    UNKNOWN_KEY_ID,
    UNKNOWN_TEMPO_ID,
    UNKNOWN_VOCALS_ID,
    VOCAL_LABELS,
    VOCAL_TOKEN_TO_ID,
    WORD_BOUNDARY_ID,
    LyricEncoder,
    bpm_to_id,
    gender_label_to_id,
    is_gender_label,
    is_key_label,
    is_tempo_label,
    is_vocal_label,
    key_label_to_id,
    parse_key_label,
    parse_tempo_label,
    structure_label_to_id,
    text_to_phoneme_ids,
    text_to_word_phoneme_groups,
    text_with_markers_to_phoneme_ids,
    vocal_label_to_id,
)
from model.lyric_encoder import N_TEMPO_BUCKETS


def _g2p_available() -> bool:
    # v9: the phonemizer is espeak-ng via the `phonemizer` lib. Probe it directly
    # (text_to_phoneme_ids swallows a missing-backend error and returns [BOS], so
    # it can't be used to detect availability). Tests skip when espeak-ng isn't
    # installed (local dev); they run on the Modal phonemize image.
    try:
        from phonemizer.backend import EspeakBackend

        EspeakBackend("en-us")  # constructs only if the espeak-ng shared lib is present
        return True
    except Exception:
        return False


g2p_required = pytest.mark.skipif(
    not _g2p_available(), reason="phonemizer / espeak-ng not installed",
)


# --- vocab invariants (unconditional) ----------------------------------------
def test_vocab_specials_at_low_ids():
    assert PHONEME_VOCAB[PAD_PHONEME_ID] == "<pad>"
    assert PAD_PHONEME_ID == 0  # padding_idx contract with nn.Embedding
    assert BOS_PHONEME_ID == 1
    assert WORD_BOUNDARY_ID == 2


def test_vocab_size_and_uniqueness():
    assert PHONEME_VOCAB_SIZE == len(PHONEME_VOCAB)
    assert len(set(PHONEME_VOCAB)) == PHONEME_VOCAB_SIZE  # no dup ids
    # v9 multilingual: 4 specials + 9 structure + 3 gender + 15 tempo
    # (1 unknown + 14 buckets) + 25 key (1 unknown + 24 keys) + 3 vocal
    # + 36 lang (1 unknown + 35 langs) + 161 IPA codepoints (incl. the 4 espeak
    # phones added in cec4f1f: ᵻ ä ɫ ᵝ). Bumping this is checkpoint-incompatible.
    assert PHONEME_VOCAB_SIZE == 256


# --- structure markers -------------------------------------------------------
def test_structure_tokens_present_contiguous_below_arpabet():
    assert len(STRUCTURE_LABELS) == 9
    ids = [STRUCTURE_TOKEN_TO_ID[label] for label in STRUCTURE_LABELS]
    # sit right after the 4 specials, contiguous, before gender + ARPABET
    assert ids == list(range(4, 4 + len(STRUCTURE_LABELS)))
    arpabet_start = 4 + len(STRUCTURE_LABELS) + len(GENDER_LABELS)
    assert all(i < arpabet_start for i in ids)
    assert PHONEME_VOCAB[NO_SECTION_ID] == "<no_section>"


# --- gender markers ----------------------------------------------------------
def test_gender_tokens_present_after_structure_below_arpabet():
    assert GENDER_LABELS == ("unknown_gender", "male", "female")
    ids = [GENDER_TOKEN_TO_ID[label] for label in GENDER_LABELS]
    # sit right after the 9 structure markers, contiguous, before ARPABET
    start = 4 + len(STRUCTURE_LABELS)
    assert ids == list(range(start, start + len(GENDER_LABELS)))
    assert PHONEME_VOCAB[UNKNOWN_GENDER_ID] == "<unknown_gender>"


def test_gender_label_to_id_folds_aliases_and_unknown():
    assert gender_label_to_id("male") == GENDER_TOKEN_TO_ID["male"]
    assert gender_label_to_id("FEMALE") == GENDER_TOKEN_TO_ID["female"]
    assert gender_label_to_id(" Man ") == GENDER_TOKEN_TO_ID["male"]
    assert gender_label_to_id("woman") == GENDER_TOKEN_TO_ID["female"]
    assert gender_label_to_id("f") == GENDER_TOKEN_TO_ID["female"]
    # unrecognized / empty -> unknown_gender (never crashes, never bogus id)
    assert gender_label_to_id("robot") == UNKNOWN_GENDER_ID
    assert gender_label_to_id(None) == UNKNOWN_GENDER_ID
    assert gender_label_to_id("") == UNKNOWN_GENDER_ID


def test_is_gender_label_classifies_brackets():
    assert is_gender_label("male") and is_gender_label("WOMAN") and is_gender_label("f")
    # section labels and junk are NOT gender (so the parser routes them to section)
    assert not is_gender_label("chorus")
    assert not is_gender_label("robot")
    assert not is_gender_label(None)


# --- tempo markers -----------------------------------------------------------
def test_tempo_tokens_present_after_gender_below_arpabet():
    from model.lyric_encoder import TEMPO_IDS
    # 1 unknown + N bucket tokens, contiguous, right after the gender markers.
    assert len(TEMPO_IDS) == N_TEMPO_BUCKETS + 1
    start = 4 + len(STRUCTURE_LABELS) + len(GENDER_LABELS)
    assert sorted(TEMPO_IDS) == list(range(start, start + N_TEMPO_BUCKETS + 1))
    assert PHONEME_VOCAB[UNKNOWN_TEMPO_ID] == "<unknown_tempo>"


def test_bpm_to_id_buckets_and_unknown():
    # boundary behavior: edge value goes UP into the next bucket
    assert bpm_to_id(TEMPO_BPM_EDGES[0] - 1) != bpm_to_id(TEMPO_BPM_EDGES[0])
    assert bpm_to_id(TEMPO_BPM_EDGES[0]) == bpm_to_id(TEMPO_BPM_EDGES[0] + 1)
    # below first edge and above last edge are distinct, valid buckets (not unknown)
    assert bpm_to_id(40.0) != UNKNOWN_TEMPO_ID
    assert bpm_to_id(220.0) != UNKNOWN_TEMPO_ID
    assert bpm_to_id(40.0) != bpm_to_id(220.0)
    # missing / nonsensical -> unknown
    assert bpm_to_id(None) == UNKNOWN_TEMPO_ID
    assert bpm_to_id(0) == UNKNOWN_TEMPO_ID
    assert bpm_to_id(-5) == UNKNOWN_TEMPO_ID
    assert bpm_to_id(float("nan")) == UNKNOWN_TEMPO_ID


def test_parse_and_is_tempo_label():
    assert parse_tempo_label("120bpm") == 120.0
    assert parse_tempo_label("120") == 120.0
    assert parse_tempo_label("tempo:128") == 128.0
    assert parse_tempo_label(" Fast ") == parse_tempo_label("fast")
    # not a tempo -> None, so the parser routes it to the section slot
    assert parse_tempo_label("chorus") is None
    assert parse_tempo_label(None) is None
    assert is_tempo_label("120bpm") and is_tempo_label("fast")
    assert not is_tempo_label("chorus") and not is_tempo_label(None)


# --- key markers -------------------------------------------------------------
def test_key_tokens_present_after_tempo_below_arpabet():
    # 1 unknown + 12 majors + 12 minors, contiguous, right after the tempo markers.
    assert len(KEY_LABELS) == 25 and KEY_LABELS[0] == "unknown_key"
    assert KEY_LABELS[1] == "c_major" and KEY_LABELS[13] == "c_minor"  # key_detect contract
    ids = [KEY_TOKEN_TO_ID[label] for label in KEY_LABELS]
    start = 4 + len(STRUCTURE_LABELS) + len(GENDER_LABELS) + (N_TEMPO_BUCKETS + 1)
    assert ids == list(range(start, start + len(KEY_LABELS)))
    assert PHONEME_VOCAB[UNKNOWN_KEY_ID] == "<unknown_key>"


def test_parse_key_label_forms_and_enharmonics():
    assert parse_key_label("Am") == "a_minor"
    assert parse_key_label("a minor") == "a_minor"
    assert parse_key_label("key:Am") == "a_minor"
    assert parse_key_label("F# major") == "f_sharp_major"
    assert parse_key_label("c_sharp_minor") == "c_sharp_minor"  # canonical passes through
    # missing mode -> major; flats and edge enharmonics fold to the sharp canon
    assert parse_key_label("Bb") == "a_sharp_major"
    assert parse_key_label("Db minor") == "c_sharp_minor"
    assert parse_key_label("cb") == "b_major"
    assert parse_key_label("e#") == "f_major"
    # not a key -> None, so the parser routes it elsewhere
    for not_a_key in ("chorus", "break", "inst", "fast", "120bpm", "robot", None, ""):
        assert parse_key_label(not_a_key) is None


def test_key_label_to_id_folds_unknown():
    assert key_label_to_id("a_minor") == KEY_TOKEN_TO_ID["a_minor"]
    assert key_label_to_id("Gb major") == KEY_TOKEN_TO_ID["f_sharp_major"]
    assert key_label_to_id("robot") == UNKNOWN_KEY_ID
    assert key_label_to_id(None) == UNKNOWN_KEY_ID
    assert key_label_to_id("") == UNKNOWN_KEY_ID


def test_is_key_label_classifies_brackets():
    assert is_key_label("a minor") and is_key_label("key:Am") and is_key_label("F#")
    # section labels and junk are NOT keys (so the parser routes them to section)
    assert not is_key_label("chorus") and not is_key_label("robot") and not is_key_label(None)
    # bare f/m ARE parseable keys, but gender is checked first in the parser —
    # documented: use [f major] / [m...] long forms for those keys.
    assert is_key_label("f") and is_gender_label("f")


# --- vocal-presence markers ----------------------------------------------------
def test_vocal_tokens_present_after_key_below_arpabet():
    assert VOCAL_LABELS == ("unknown_vocals", "vocals", "instrumental")
    ids = [VOCAL_TOKEN_TO_ID[label] for label in VOCAL_LABELS]
    start = (4 + len(STRUCTURE_LABELS) + len(GENDER_LABELS)
             + (N_TEMPO_BUCKETS + 1) + len(KEY_LABELS))
    assert ids == list(range(start, start + len(VOCAL_LABELS)))
    assert PHONEME_VOCAB[UNKNOWN_VOCALS_ID] == "<unknown_vocals>"


def test_vocal_label_to_id_folds_aliases_and_unknown():
    assert vocal_label_to_id("vocals") == VOCAL_TOKEN_TO_ID["vocals"]
    assert vocal_label_to_id("Vocal") == VOCAL_TOKEN_TO_ID["vocals"]
    assert vocal_label_to_id("no vocals") == VOCAL_TOKEN_TO_ID["instrumental"]
    assert vocal_label_to_id("INSTRUMENTAL") == VOCAL_TOKEN_TO_ID["instrumental"]
    assert vocal_label_to_id("robot") == UNKNOWN_VOCALS_ID
    assert vocal_label_to_id(None) == UNKNOWN_VOCALS_ID


def test_is_vocal_label_classifies_brackets():
    assert is_vocal_label("instrumental") and is_vocal_label("no vocals") and is_vocal_label("voice")
    # 'inst' is an allin1 SECTION label and must keep routing to the section slot
    # (the parser checks vocal before section, so claiming it here would steal it).
    assert not is_vocal_label("inst")
    assert not is_vocal_label("chorus") and not is_vocal_label(None)


def test_structure_label_to_id_folds_unknown_and_normalizes():
    assert structure_label_to_id("chorus") == STRUCTURE_TOKEN_TO_ID["chorus"]
    assert structure_label_to_id("CHORUS") == STRUCTURE_TOKEN_TO_ID["chorus"]
    assert structure_label_to_id(" Chorus ") == STRUCTURE_TOKEN_TO_ID["chorus"]
    # allin1 never emits prechorus -> folds to no_section, both spellings
    assert structure_label_to_id("prechorus") == NO_SECTION_ID
    assert structure_label_to_id("pre-chorus") == NO_SECTION_ID
    assert structure_label_to_id(None) == NO_SECTION_ID
    assert structure_label_to_id("") == NO_SECTION_ID


@g2p_required
def test_markers_parse_prefix_and_inline():
    ids = text_with_markers_to_phoneme_ids("[verse] hello [chorus] world")
    assert ids[0] == BOS_PHONEME_ID
    assert ids[1] == UNKNOWN_GENDER_ID  # dense gender slot (none given)
    assert ids[2] == UNKNOWN_TEMPO_ID   # dense tempo slot (none given)
    assert ids[3] == UNKNOWN_KEY_ID     # dense key slot (none given)
    assert ids[4] == VOCAL_TOKEN_TO_ID["vocals"]  # words present -> <vocals> default
    assert ids[5] == STRUCTURE_TOKEN_TO_ID["verse"]  # leading section marker = prefix
    assert STRUCTURE_TOKEN_TO_ID["chorus"] in ids[6:]  # inline marker
    # verse appears once (prefix only, not re-emitted)
    assert ids.count(STRUCTURE_TOKEN_TO_ID["verse"]) == 1


@g2p_required
def test_markers_parse_gender_prefix_order_independent():
    # gender given before OR after the section both land in the gender slot;
    # the emitted prefix order is always <gender> <section>.
    a = text_with_markers_to_phoneme_ids("[female] [chorus] hello")
    b = text_with_markers_to_phoneme_ids("[chorus] [female] hello")
    assert a == b
    assert a[0] == BOS_PHONEME_ID
    assert a[1] == GENDER_TOKEN_TO_ID["female"]
    assert a[2] == UNKNOWN_TEMPO_ID  # no tempo given -> unknown_tempo slot
    assert a[3] == UNKNOWN_KEY_ID    # no key given -> unknown_key slot
    assert a[4] == VOCAL_TOKEN_TO_ID["vocals"]  # words present -> <vocals> default
    assert a[5] == STRUCTURE_TOKEN_TO_ID["chorus"]
    # gender appears once (prefix only, never inline)
    assert a.count(GENDER_TOKEN_TO_ID["female"]) == 1


@g2p_required
def test_markers_no_bracket_artifact_reaches_g2p():
    # brackets must be stripped before g2p; only known phoneme/marker ids appear
    ids = text_with_markers_to_phoneme_ids("[chorus] singing the blues")
    assert all(0 <= i < PHONEME_VOCAB_SIZE for i in ids)
    plain = text_to_phoneme_ids("singing the blues", add_bos=False)
    # tail after BOS + <gender> + <tempo> + <key> + <vocals> + <chorus> prefixes +
    # leading WB equals plain g2p (6 header tokens then the word-boundary at index 6)
    assert ids[7:] == plain


@g2p_required
def test_markers_default_prefixes_are_unknown_gender_and_no_section():
    ids = text_with_markers_to_phoneme_ids("hello world")
    assert ids[1] == UNKNOWN_GENDER_ID  # no gender marker -> unknown_gender
    assert ids[2] == UNKNOWN_TEMPO_ID   # no tempo marker -> unknown_tempo
    assert ids[3] == UNKNOWN_KEY_ID     # no key marker -> unknown_key
    assert ids[4] == VOCAL_TOKEN_TO_ID["vocals"]  # words present -> <vocals> default
    assert ids[5] == NO_SECTION_ID      # no section marker -> no_section


@g2p_required
def test_markers_vocal_slot_defaults_and_overrides():
    # No words AND no bracket -> <unknown_vocals> (the train-time never-transcribed case).
    only_markers = text_with_markers_to_phoneme_ids("[verse]")
    assert only_markers[4] == UNKNOWN_VOCALS_ID
    # An explicit [instrumental] survives even with words (contradictory but allowed).
    forced = text_with_markers_to_phoneme_ids("[instrumental] hello")
    assert forced[4] == VOCAL_TOKEN_TO_ID["instrumental"]
    # [vocals] with no words = the vocal-song-instrumental-crop case.
    bare = text_with_markers_to_phoneme_ids("[vocals]")
    assert bare[4] == VOCAL_TOKEN_TO_ID["vocals"]
    # A stray inline vocal marker is dropped (prefix-only).
    inline = text_with_markers_to_phoneme_ids("hello [instrumental] world")
    assert VOCAL_TOKEN_TO_ID["instrumental"] not in inline
    assert inline == text_with_markers_to_phoneme_ids("hello world")


@g2p_required
def test_markers_key_prefix_and_inline_drop():
    ids = text_with_markers_to_phoneme_ids("[a minor] [verse] hello")
    assert ids[3] == KEY_TOKEN_TO_ID["a_minor"]
    assert ids[5] == STRUCTURE_TOKEN_TO_ID["verse"]
    # A stray inline key marker is dropped (prefix-only).
    inline = text_with_markers_to_phoneme_ids("hello [a minor] world")
    assert KEY_TOKEN_TO_ID["a_minor"] not in inline
    assert inline == text_with_markers_to_phoneme_ids("hello world")


@g2p_required
def test_markers_unknown_label_folds_to_no_section():
    ids = text_with_markers_to_phoneme_ids("[prechorus] hello")
    assert ids[1] == UNKNOWN_GENDER_ID  # gender slot still dense
    assert ids[2] == UNKNOWN_TEMPO_ID   # tempo slot still dense
    assert ids[3] == UNKNOWN_KEY_ID     # key slot still dense
    assert ids[4] == VOCAL_TOKEN_TO_ID["vocals"]  # words present -> <vocals> default
    assert ids[5] == NO_SECTION_ID      # unknown label as section prefix


@g2p_required
def test_markers_malformed_brackets_never_reach_g2p():
    # Empty [] and the stray brackets from a nested [[x]] are scrubbed; only valid
    # phoneme/marker ids ever appear (a literal bracket reaching g2p would not).
    for text in ("hello [] world", "[[chorus]] hello", "a [ b ] c"):
        ids = text_with_markers_to_phoneme_ids(text)
        assert all(0 <= i < PHONEME_VOCAB_SIZE for i in ids)
    # [[chorus]] still resolves the chorus label as the section prefix.
    nested = text_with_markers_to_phoneme_ids("[[chorus]] hello")
    assert nested[5] == STRUCTURE_TOKEN_TO_ID["chorus"]
    # A bare [] is a no-op: same stream as the plain text.
    assert text_with_markers_to_phoneme_ids("hello [] world") == \
        text_with_markers_to_phoneme_ids("hello world")


@g2p_required
def test_markers_stray_inline_gender_dropped():
    # A second gender marker is prefix-only; inline it must be dropped, never
    # emitted as a bogus <no_section> the way an unknown inline label would be.
    ids = text_with_markers_to_phoneme_ids("[female] hello [male] world")
    assert ids[1] == GENDER_TOKEN_TO_ID["female"]      # first gender = prefix
    assert GENDER_TOKEN_TO_ID["male"] not in ids       # second gender dropped
    # equals the same line with the stray inline gender simply removed
    assert ids == text_with_markers_to_phoneme_ids("[female] hello world")


@g2p_required
def test_markers_wellformed_unchanged_by_hardening():
    # The hardening must not perturb well-formed input (the equivalence contract).
    ids = text_with_markers_to_phoneme_ids("[female] [verse] hello world [chorus] yeah")
    assert ids[0] == BOS_PHONEME_ID
    assert ids[1] == GENDER_TOKEN_TO_ID["female"]
    assert ids[2] == UNKNOWN_TEMPO_ID  # no tempo given
    assert ids[3] == UNKNOWN_KEY_ID    # no key given
    assert ids[4] == VOCAL_TOKEN_TO_ID["vocals"]  # words present -> <vocals> default
    assert ids[5] == STRUCTURE_TOKEN_TO_ID["verse"]
    assert STRUCTURE_TOKEN_TO_ID["chorus"] in ids[6:]
    assert all(0 <= i < PHONEME_VOCAB_SIZE for i in ids)


@g2p_required
def test_markers_max_len_truncates():
    long_line = "[verse] " + "la " * 50 + "[chorus] " + "na " * 50
    truncated = text_with_markers_to_phoneme_ids(long_line, max_len=9)
    # Whole-unit truncation: never exceeds the cap, and never slices a word/marker
    # mid-unit — so the capped stream is a prefix of the uncapped one.
    assert len(truncated) <= 9
    full = text_with_markers_to_phoneme_ids(long_line)
    assert truncated == full[: len(truncated)]


# --- g2p mapping -------------------------------------------------------------
@g2p_required
def test_text_to_phoneme_ids_deterministic():
    a = text_to_phoneme_ids("singing the blues tonight")
    b = text_to_phoneme_ids("singing the blues tonight")
    assert a == b
    assert a, "non-empty lyric should produce ids"


@g2p_required
def test_text_to_phoneme_ids_in_range_and_bos():
    ids = text_to_phoneme_ids("hello world")
    assert ids[0] == BOS_PHONEME_ID
    assert all(0 <= i < PHONEME_VOCAB_SIZE for i in ids)
    # no padding token ever emitted by g2p
    assert PAD_PHONEME_ID not in ids


@g2p_required
def test_empty_and_whitespace_yield_empty():
    assert text_to_phoneme_ids("") == []
    assert text_to_phoneme_ids("   ") == []


@g2p_required
def test_no_bos_option_and_no_leading_or_trailing_boundary():
    ids = text_to_phoneme_ids("hello world", add_bos=False)
    assert ids[0] != WORD_BOUNDARY_ID
    assert ids[-1] != WORD_BOUNDARY_ID


@g2p_required
def test_max_len_truncates():
    long_line = "singing the blues all night long forever and ever amen"
    assert len(text_to_phoneme_ids(long_line, max_len=7)) == 7


# g2p_en expands digit runs via inflect.number_to_words, which raises
# NumOutOfRangeError past ~10^66 — and Whisper transcripts contain such garbage
# (killed the 2026-06-12 phonemize corpus pass). One bad token must only drop
# itself, never the song (or the whole pass).
_INFLECT_BOMB = "9" * 80


@g2p_required
def test_text_to_phoneme_ids_survives_inflect_bomb():
    ids = text_to_phoneme_ids(f"hello {_INFLECT_BOMB} world")
    assert ids, "surrounding words should still phonemize"
    assert all(0 <= i < PHONEME_VOCAB_SIZE for i in ids)


@g2p_required
def test_word_groups_survive_inflect_bomb_and_keep_alignment():
    words = ["hello", _INFLECT_BOMB, "world"]
    groups = text_to_word_phoneme_groups(words)
    assert len(groups) == len(words), "1:1 word→group alignment must hold"
    assert groups[0] and groups[2], "good words should still phonemize"
    assert groups[1] == [], "the bad word yields an empty group"


# --- encoder -----------------------------------------------------------------
def test_encoder_output_shape_and_mask_passthrough():
    enc = LyricEncoder(d_model=64, n_layers=2, n_heads=8, d_ff=128, max_len=32)
    ids = torch.tensor([[BOS_PHONEME_ID, 10, 11, 12, PAD_PHONEME_ID, PAD_PHONEME_ID]])
    mask = ids != PAD_PHONEME_ID
    emb, out_mask = enc(ids, mask)
    assert emb.shape == (1, 6, 64)
    assert torch.equal(out_mask, mask)


def test_encoder_padding_does_not_affect_real_positions():
    """A real position's encoding must not change when padding is appended —
    proves the key-padding mask actually blocks attention to pad slots."""
    torch.manual_seed(0)
    enc = LyricEncoder(d_model=64, n_layers=2, n_heads=8, d_ff=128, max_len=32)
    enc.eval()
    core = [BOS_PHONEME_ID, 10, 11, 12]
    ids_short = torch.tensor([core])
    ids_pad = torch.tensor([core + [PAD_PHONEME_ID, PAD_PHONEME_ID]])
    with torch.no_grad():
        emb_short, _ = enc(ids_short, ids_short != PAD_PHONEME_ID)
        emb_pad, _ = enc(ids_pad, ids_pad != PAD_PHONEME_ID)
    torch.testing.assert_close(emb_short, emb_pad[:, : len(core)], atol=1e-5, rtol=1e-4)
