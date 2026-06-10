"""Smoke tests for the CUDA weight-only quantization path (NANO_BITS=8|4).

The torchao counterpart to tests/test_mlx_parity.py's quant tests: these only
guard that int8/int4 quantization runs, follows the same layer policy as the MLX
backend (quantize the big Linears; skip embeddings + the lyric encoder), and that
the quantized model still produces in-range tokens. No bit-parity is expected.

Skipped unless running on CUDA with `torchao` installed — int4's tinygemm kernel
needs a GPU (sm80+) and bf16, so there's nothing meaningful to assert on CPU.
"""
from __future__ import annotations

import importlib.util

import pytest
import torch

_HAS_TORCHAO = importlib.util.find_spec("torchao") is not None

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_TORCHAO),
    reason="torch quant path only runs on CUDA with torchao installed",
)

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT  # noqa: E402
from server.inference import _quantize_torch_linears  # noqa: E402


def _tiny_cfg(**overrides) -> GPTConfig:
    # d_model/d_ff divisible by the int4 group_size (64) so no Linear is skipped
    # for divisibility, matching the real model's shape.
    base = dict(
        d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, max_seq_len=128,
        use_lyric_conditioning=True,
    )
    base.update(overrides)
    return GPTConfig(**base)


def _is_quantized(weight: torch.Tensor) -> bool:
    return "AffineQuantized" in type(weight).__name__


@pytest.mark.parametrize("bits", [8, 4])
def test_quantize_layer_policy(bits):
    """Big Linears get quantized; embeddings and the lyric encoder don't."""
    cfg = _tiny_cfg()
    dtype = torch.bfloat16 if bits == 4 else torch.float16
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval().to("cuda").to(dtype)
    _quantize_torch_linears(model, bits)

    # attention / MLP / fused output head quantized
    assert _is_quantized(model.blocks[0].attn.qkv.weight)
    assert _is_quantized(model.blocks[0].mlp.fc1.weight)
    assert _is_quantized(model.head.weight)
    # embeddings + lyric encoder left in full precision
    assert not _is_quantized(model.tok_embeds[0].weight)
    assert not _is_quantized(model.lyric_encoder.layers[0].qkv.weight)


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_generate_in_range(bits):
    """A quantized model still emits valid (in-vocab) tokens."""
    cfg = _tiny_cfg()
    dtype = torch.bfloat16 if bits == 4 else torch.float16
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval().to("cuda").to(dtype)
    _quantize_torch_linears(model, bits)

    prompt = torch.randint(0, cfg.vocab_per_codebook, (cfg.n_codebooks, 4), device="cuda")
    out = model.generate(prompt, num_new_frames=6, temperature=1.0, top_k=10)
    assert out.shape == (cfg.n_codebooks, 10)
    assert out.min() >= 0 and out.max() < cfg.vocab_per_codebook
