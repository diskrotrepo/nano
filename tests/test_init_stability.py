"""Init-time stability hardening guards (the v8 divergence fix).

The 2026-06-12 v8 launch diverged at lr ~4e-5 (cb0-led, val 7.17 -> 8.70 by
step 3000): the fresh lyric/melody conditioning paths injected full-volume
random projections into a 22-layer residual stack from step 0. The fix gates
every conditioning path to an exact no-op at init (zero cross-attn out_proj;
zero ln_final gain in MelodyEncoder) and depth-scales the residual out-projs
(GPT-2 1/sqrt(2*n_layers)). These tests pin both properties: a fresh model's
forward must be bit-identical with and without conditioning.
"""
from __future__ import annotations

import math

import torch

from model.melody_encoder import MelodyEncoder
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _tiny_model() -> NanoAudioGPT:
    cfg = GPTConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=128,
        use_text_conditioning=True, use_lyric_conditioning=True,
        use_melody_conditioning=True, use_gradient_checkpointing=False,
        lyric_enc_layers=1, lyric_enc_heads=2, lyric_enc_d_ff=64,
        max_lyric_len=32, use_qk_norm=True,
    )
    torch.manual_seed(0)
    return NanoAudioGPT(cfg).eval()


def _tokens(cfg: GPTConfig, T: int = 24) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, cfg.vocab_per_codebook, (2, cfg.n_codebooks, T))


def test_conditioning_out_projs_zero_and_resid_scaled_at_init():
    model = _tiny_model()
    resid_std = 0.02 / math.sqrt(2 * model.cfg.n_layers)
    for block in model.blocks:
        assert torch.all(block.cross_attn.out_proj.weight == 0)
        assert torch.all(block.lyric_attn.out_proj.weight == 0)
        # Depth-scaled residual init (loose tolerance — it's a sampled std).
        assert abs(block.attn.proj.weight.std().item() - resid_std) < 0.3 * resid_std
        assert abs(block.mlp.fc2.weight.std().item() - resid_std) < 0.3 * resid_std


def test_text_and_lyric_conditioning_are_noops_at_init():
    model = _tiny_model()
    tokens = _tokens(model.cfg)
    with torch.no_grad():
        base = model(tokens)
        text = model(tokens, text_emb=torch.randn(2, 1, model.cfg.d_model))
        lyric_ids = torch.randint(0, model.cfg.phoneme_vocab_size, (2, 8))
        lyric = model(
            tokens, lyric_ids=lyric_ids, lyric_mask=torch.ones(2, 8, dtype=torch.bool),
        )
    torch.testing.assert_close(text, base)
    torch.testing.assert_close(lyric, base)


def test_melody_content_is_noop_at_init():
    """Two different chroma inputs must produce identical logits at init (the
    encoded melody is zero-gated); the learned null is a separate, deliberate
    constant, so melody=None is NOT compared against melody=<chroma>."""
    model = _tiny_model()
    tokens = _tokens(model.cfg)
    T = tokens.shape[-1]
    torch.manual_seed(2)
    with torch.no_grad():
        a = model(tokens, melody=torch.rand(2, T, model.cfg.melody_n_bins))
        b = model(tokens, melody=torch.rand(2, T, model.cfg.melody_n_bins))
    torch.testing.assert_close(a, b)


def test_melody_encoder_output_zero_at_init():
    enc = MelodyEncoder(d_model=32).eval()
    out = enc.encode_melody(torch.rand(2, 16, 12))
    assert torch.all(out == 0)
    # The null stays a learned non-zero vector (zeros would collide with the
    # valid "silent frame" chroma input).
    assert enc.null.abs().sum() > 0


def test_gates_receive_gradient():
    """The zero gates must not be gradient-dead: one backward pass puts a
    nonzero grad on every gate so the paths can open during training."""
    model = _tiny_model().train()
    tokens = _tokens(model.cfg)
    lyric_ids = torch.randint(0, model.cfg.phoneme_vocab_size, (2, 8))
    logits = model(
        tokens,
        text_emb=torch.randn(2, 1, model.cfg.d_model),
        lyric_ids=lyric_ids,
        lyric_mask=torch.ones(2, 8, dtype=torch.bool),
        melody=torch.rand(2, tokens.shape[-1], model.cfg.melody_n_bins),
    )
    logits.sum().backward()
    for block in model.blocks:
        assert block.cross_attn.out_proj.weight.grad.abs().sum() > 0
        assert block.lyric_attn.out_proj.weight.grad.abs().sum() > 0
    assert model.melody_encoder.ln_final.weight.grad.abs().sum() > 0
