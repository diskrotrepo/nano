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
    NO_SECTION_ID,
    PAD_PHONEME_ID,
    PHONEME_VOCAB,
    PHONEME_VOCAB_SIZE,
    STRUCTURE_LABELS,
    STRUCTURE_TOKEN_TO_ID,
    WORD_BOUNDARY_ID,
    LyricEncoder,
    structure_label_to_id,
    text_to_phoneme_ids,
    text_with_markers_to_phoneme_ids,
)


def _g2p_available() -> bool:
    try:
        text_to_phoneme_ids("test")
        return True
    except Exception:
        return False


g2p_required = pytest.mark.skipif(
    not _g2p_available(), reason="g2p_en / nltk data not installed"
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
    assert PHONEME_VOCAB_SIZE == 83  # 4 specials + 9 structure + 70 ARPABET


# --- structure markers -------------------------------------------------------
def test_structure_tokens_present_contiguous_below_arpabet():
    assert len(STRUCTURE_LABELS) == 9
    ids = [STRUCTURE_TOKEN_TO_ID[label] for label in STRUCTURE_LABELS]
    # sit right after the 4 specials, contiguous, before ARPABET (the rest)
    assert ids == list(range(4, 4 + len(STRUCTURE_LABELS)))
    arpabet_start = 4 + len(STRUCTURE_LABELS)
    assert all(i < arpabet_start for i in ids)
    assert PHONEME_VOCAB[NO_SECTION_ID] == "<no_section>"


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
    assert ids[1] == STRUCTURE_TOKEN_TO_ID["verse"]  # leading marker is the prefix
    assert STRUCTURE_TOKEN_TO_ID["chorus"] in ids[2:]  # inline marker
    # verse appears once (prefix only, not re-emitted)
    assert ids.count(STRUCTURE_TOKEN_TO_ID["verse"]) == 1


@g2p_required
def test_markers_no_bracket_artifact_reaches_g2p():
    # brackets must be stripped before g2p; only known phoneme/marker ids appear
    ids = text_with_markers_to_phoneme_ids("[chorus] singing the blues")
    assert all(0 <= i < PHONEME_VOCAB_SIZE for i in ids)
    plain = text_to_phoneme_ids("singing the blues", add_bos=False)
    # the phoneme tail (after BOS + chorus prefix + leading WB) equals plain g2p
    assert ids[3:] == plain


@g2p_required
def test_markers_default_prefix_is_no_section():
    ids = text_with_markers_to_phoneme_ids("hello world")
    assert ids[1] == NO_SECTION_ID  # no leading marker -> no_section prefix


@g2p_required
def test_markers_unknown_label_folds_to_no_section():
    ids = text_with_markers_to_phoneme_ids("[prechorus] hello")
    assert ids[1] == NO_SECTION_ID  # unknown label as prefix


@g2p_required
def test_markers_max_len_truncates():
    long_line = "[verse] " + "la " * 50 + "[chorus] " + "na " * 50
    assert len(text_with_markers_to_phoneme_ids(long_line, max_len=9)) == 9


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
