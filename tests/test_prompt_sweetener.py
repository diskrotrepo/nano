"""Tests for server/prompt_sweetener.py — CPU-only, no model download.

The HF/Qwen calls are mocked. The key invariants under test:
  1. Tags and lyrics now travel as SEPARATE fields (no ". " split), so a
     multi-sentence caption is kept as natural prose — ". " is NOT collapsed.
  2. Only a matched WRAPPING quote pair is stripped; embedded/edge quotes survive.
  3. An already-long / detailed prompt is passed through verbatim (the rewriter
     would only discard the detail the user wrote).
  4. Any failure in the LLM path falls back to the raw prompt unchanged —
     sweetening must never break a generation.
"""
from __future__ import annotations

import torch

from server.prompt_sweetener import PromptSweetener

_MULTI_SENTENCE = (
    "The low quality recording features a mellow lo-fi hip hop beat with a soft "
    "punchy kick, groovy bass and warm keys. It sounds relaxed and emotional."
)


class _FakeBatch(dict):
    def to(self, *args, **kwargs):
        return self


def _install_fake_model(sw: PromptSweetener, caption: str) -> None:
    """Replace the lazy model with fakes that yield ``caption`` verbatim."""

    class _FakeTok:
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

        def __call__(self, text, return_tensors="pt"):
            return _FakeBatch(input_ids=torch.zeros((1, 3), dtype=torch.long))

        def decode(self, tokens, skip_special_tokens=True):
            return caption

    class _FakeModel:
        def generate(self, **kwargs):
            # 3 prompt tokens + 4 "new" tokens; sweeten() slices off the prompt.
            return torch.zeros((1, 7), dtype=torch.long)

    sw._tok = _FakeTok()
    sw._model = _FakeModel()
    sw._ensure_model = lambda: None  # already "loaded"


def test_sanitize_keeps_prose_sentences():
    """Tags/lyrics are separate fields now, so a multi-sentence caption stays as
    natural prose — ". " is preserved, NOT collapsed to "; "."""
    out = PromptSweetener._sanitize(_MULTI_SENTENCE)
    assert ". " in out
    assert "; " not in out
    assert out


def test_sweeten_keeps_prose():
    sw = PromptSweetener(device="cpu")
    _install_fake_model(sw, _MULTI_SENTENCE)
    out = sw.sweeten("lofi beat to study to")
    assert out == _MULTI_SENTENCE  # passed through, prose intact


def test_sanitize_strips_only_wrapping_quotes():
    # a fully-wrapped caption -> wrapping pair removed
    assert PromptSweetener._sanitize('"techno with punchy kick"') == "techno with punchy kick"
    # an embedded/edge quote that is NOT a wrapping pair survives
    kept = PromptSweetener._sanitize('ambient lofi ending in "tennis favorable 2015"')
    assert kept.endswith('"tennis favorable 2015"')


def test_sweeten_clamps_runaway_to_240_words():
    sw = PromptSweetener(device="cpu")
    long_caption = '"' + " ".join(f"w{i}" for i in range(300)) + '"'
    _install_fake_model(sw, long_caption)
    out = sw.sweeten("anything")
    assert not out.startswith('"') and not out.endswith('"')
    assert len(out.split()) <= 240


def test_sweeten_skips_long_prompts_verbatim():
    """A >120-word prompt is already caption-style; pass it through without ever
    calling the model (rewriting would discard the user's detail)."""
    sw = PromptSweetener(device="cpu")

    def _boom():
        raise AssertionError("the model must NOT be loaded for a long prompt")

    sw._ensure_model = _boom
    long_prompt = " ".join(f"word{i}" for i in range(130))
    assert sw.sweeten(long_prompt) == long_prompt


def test_sweeten_falls_back_on_error():
    sw = PromptSweetener(device="cpu")

    def _boom():
        raise RuntimeError("no model here")

    sw._ensure_model = _boom
    raw = "make me some techno"
    assert sw.sweeten(raw) == raw


def test_sweeten_empty_input_returns_empty():
    sw = PromptSweetener(device="cpu")
    assert sw.sweeten("   ") == ""
