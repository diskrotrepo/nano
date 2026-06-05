"""Tests for RoPE correctness in model/nano_audio_gpt.py.

RoPE replaces absolute pos_embed in v6. These tests pin down:
- causal masking still holds (position t can't see t+1, t+2, ...)
- cached decode matches one-shot forward (KV cache writes rotated keys, so
  multi-step decode must equal a single full forward at matching positions)
- position information is actually applied (different start_pos => different
  output for the same tokens)
- max_seq_len cap raises a clear error past the RoPE table
- gradient checkpointing produces the same output as non-checkpointed in eval
"""
from __future__ import annotations

import pytest
import torch

from model.nano_audio_gpt import (
    GPTConfig,
    NanoAudioGPT,
    RotaryEmbedding,
    StaticLayerKVCache,
    apply_rotary,
)


def _tiny_cfg(**overrides) -> GPTConfig:
    base = dict(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        max_seq_len=128,
    )
    base.update(overrides)
    return GPTConfig(**base)


def test_rotary_table_shapes():
    rot = RotaryEmbedding(head_dim=16, max_seq_len=32, base=10000.0)
    assert rot.cos_cached.shape == (32, 16)
    assert rot.sin_cached.shape == (32, 16)
    # Buffers are non-persistent: re-derived from cfg, not saved to ckpt.
    state = rot.state_dict()
    assert "cos_cached" not in state
    assert "sin_cached" not in state


def test_apply_rotary_position_dependent():
    """Same q, k at different positions must yield different rotated tensors."""
    head_dim = 16
    rot = RotaryEmbedding(head_dim=head_dim, max_seq_len=32)
    q = torch.randn(1, 2, 1, head_dim)
    k = torch.randn(1, 2, 1, head_dim)
    cos0, sin0 = rot(0, 1)
    cos5, sin5 = rot(5, 1)
    q0, k0 = apply_rotary(q, k, cos0, sin0)
    q5, k5 = apply_rotary(q, k, cos5, sin5)
    # Position-0 rotation should be (cos=1, sin=0), so q0 == q.
    assert torch.allclose(q0, q, atol=1e-6)
    # Position-5 rotation is non-trivial; q5 must differ from q.
    assert not torch.allclose(q5, q, atol=1e-3)
    assert not torch.allclose(q5, q0, atol=1e-3)


def test_causal_mask_still_holds():
    """Logits at position t must not depend on tokens at positions > t."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    m = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    T = 16
    tokens_a = torch.randint(0, cfg.vocab_per_codebook, (1, K, T))
    tokens_b = tokens_a.clone()
    # Perturb only the last token of each codebook.
    tokens_b[:, :, -1] = (tokens_b[:, :, -1] + 7) % cfg.vocab_per_codebook
    with torch.no_grad():
        out_a = m(tokens_a)
        out_b = m(tokens_b)
    # Logits at positions [0, T-1) must be identical between a and b.
    assert torch.allclose(out_a[:, :, :-1, :], out_b[:, :, :-1, :], atol=1e-5)
    # And the last position must actually differ (sanity that the perturbation
    # was non-trivial).
    assert not torch.allclose(out_a[:, :, -1, :], out_b[:, :, -1, :], atol=1e-3)


def test_cached_decode_matches_oneshot():
    """Prefill + per-step decode with KV cache must match a single forward
    pass over the same tokens at matching positions."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    m = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    T_full = 12
    prefill = 8
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, T_full))

    # One-shot path: feed all T_full at once.
    with torch.no_grad():
        oneshot_logits = m(tokens)  # [B, K, T_full, V]

    # Cached path: prefill `prefill` tokens, then decode the rest one-by-one.
    head_dim = cfg.d_model // cfg.n_heads
    caches = [
        StaticLayerKVCache(
            torch.zeros(1, cfg.n_heads, T_full, head_dim),
            torch.zeros(1, cfg.n_heads, T_full, head_dim),
        )
        for _ in range(cfg.n_layers)
    ]
    with torch.no_grad():
        pre_logits, caches = m(tokens[:, :, :prefill], kv_caches=caches, start_pos=0)
        step_outs = [pre_logits]
        for p in range(prefill, T_full):
            inp = tokens[:, :, p:p + 1]
            step_logits, caches = m(inp, kv_caches=caches, start_pos=p)
            step_outs.append(step_logits)
    cached_logits = torch.cat(step_outs, dim=2)  # [B, K, T_full, V]

    # SDPA + RoPE are deterministic in eval; should be bit-exact modulo float
    # rounding. Allow a tiny tolerance for kernel non-determinism.
    assert torch.allclose(oneshot_logits, cached_logits, atol=1e-4, rtol=1e-4)


def test_max_seq_len_assertion():
    """Asking for positions past max_seq_len must fail with a clear error."""
    cfg = _tiny_cfg(max_seq_len=16)
    m = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, 8))
    with pytest.raises(AssertionError, match="max_seq_len"):
        m(tokens, start_pos=12)  # 12 + 8 = 20 > 16


def test_gradient_checkpointing_matches_plain():
    """Gradient checkpointing should produce the same output as the plain path
    during training (it's a recompute trick, not an algorithmic change)."""
    torch.manual_seed(0)
    cfg_plain = _tiny_cfg(use_gradient_checkpointing=False)
    m_plain = NanoAudioGPT(cfg_plain).train()
    torch.manual_seed(0)
    cfg_gc = _tiny_cfg(use_gradient_checkpointing=True)
    m_gc = NanoAudioGPT(cfg_gc).train()
    # Same init seed => identical weights.
    for p_a, p_b in zip(m_plain.parameters(), m_gc.parameters()):
        assert torch.equal(p_a, p_b)

    K = cfg_plain.n_codebooks
    tokens = torch.randint(0, cfg_plain.vocab_per_codebook, (1, K, 8))
    # Eval-mode for both to disable dropout — the only stochastic element.
    m_plain.eval()
    m_gc.eval()
    with torch.no_grad():
        out_plain = m_plain(tokens)
        out_gc = m_gc(tokens)
    assert torch.allclose(out_plain, out_gc, atol=1e-6)
