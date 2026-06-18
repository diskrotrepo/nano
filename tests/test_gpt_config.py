"""Tests for model/nano_audio_gpt.py — GPTConfig vocab/pad invariants.

Trivial but guards the most-likely future regression: the pad id sits at index
``vocab_per_codebook`` (so the embedding/head vocab size is vocab_per_codebook
+ 1). Off-by-one here corrupts every training run.
"""
from __future__ import annotations

from model.nano_audio_gpt import GPTConfig


def test_pad_id_is_vocab_per_codebook():
    cfg = GPTConfig()
    assert cfg.pad_id == cfg.vocab_per_codebook


def test_vocab_with_pad_is_plus_one():
    cfg = GPTConfig()
    assert cfg.vocab_with_pad == cfg.vocab_per_codebook + 1


def test_defaults_match_dac():
    """DAC 44.1kHz uses 9 codebooks of 1024 entries — these defaults are
    load-bearing across the whole pipeline (tokenize.py persists int16 .pt
    assuming max idx < 32767)."""
    cfg = GPTConfig()
    assert cfg.n_codebooks == 9
    assert cfg.vocab_per_codebook == 1024


def test_custom_vocab_size_propagates():
    cfg = GPTConfig(vocab_per_codebook=512)
    assert cfg.pad_id == 512
    assert cfg.vocab_with_pad == 513


def test_fim_off_by_default():
    """FIM is opt-in: default config has only the pad control id, so existing
    checkpoints/tests are unaffected."""
    cfg = GPTConfig()
    assert cfg.use_fim is False
    assert cfg.n_control == 1
    assert cfg.vocab_with_pad == cfg.vocab_per_codebook + 1


def test_fim_adds_two_sentinels():
    """Enabling FIM appends <SUF>/<MID> after pad, growing the vocab by 2."""
    cfg = GPTConfig(use_fim=True)
    assert cfg.n_control == 3
    assert cfg.vocab_with_pad == cfg.vocab_per_codebook + 3
    assert cfg.pad_id == cfg.vocab_per_codebook
    assert cfg.suf_id == cfg.vocab_per_codebook + 1
    assert cfg.mid_id == cfg.vocab_per_codebook + 2
