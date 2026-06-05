"""Tests for server/prompt_sweetener.py — CPU-only, no model download.

The HF/Qwen calls are mocked. The key invariants under test:
  1. Sweetened output is delimiter-safe — it must contain no ". " sequence,
     because the server splits tags vs lyrics on the FIRST ". " and a
     multi-sentence caption would otherwise mis-split (half the caption lands
     in the lyrics cross-attention slot).
  2. Any failure in the LLM path falls back to the raw prompt unchanged —
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


def test_sanitize_is_delimiter_safe():
    out = PromptSweetener._sanitize(_MULTI_SENTENCE)
    assert ". " not in out
    assert "; " in out  # internal boundary preserved as prose
    assert out


def test_sweeten_output_is_delimiter_safe():
    sw = PromptSweetener(device="cpu")
    _install_fake_model(sw, _MULTI_SENTENCE)
    out = sw.sweeten("lofi beat to study to")
    assert out
    assert ". " not in out


def test_sweeten_strips_quotes_and_clamps_words():
    sw = PromptSweetener(device="cpu")
    long_caption = '"' + " ".join(f"w{i}" for i in range(80)) + '"'
    _install_fake_model(sw, long_caption)
    out = sw.sweeten("anything")
    assert not out.startswith('"') and not out.endswith('"')
    assert len(out.split()) <= 50


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
