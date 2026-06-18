"""Zeros-uncond CFG equivalence guards.

The training loop drops a conditioning stream by passing a ZEROS embedding
instead of None (diskrot/train.py CFG block) so torch.compile sees stable
argument types (4 graph variants, no eager fallback) — while inference's
unconditional branch still passes None. That is only sound because every
projection on the text/lyric paths is bias-free: attention over zero K/V is
exactly zero, so ``x + cross_attn(..., zeros)`` == skipping the block. These
tests pin that invariant with the zero-init gates deliberately randomized
(otherwise the v8 zero-gated init would make the equality trivially true) —
if someone later adds a bias to those projections, this fails loudly.
"""
from __future__ import annotations

import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _randomized_model() -> NanoAudioGPT:
    """Tiny model with the init-time zero gates overwritten, so conditioning
    paths are live and the None==zeros equality is a real claim."""
    cfg = GPTConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=128,
        use_text_conditioning=True, use_lyric_conditioning=True,
        use_melody_conditioning=True, use_gradient_checkpointing=False,
        lyric_enc_layers=1, lyric_enc_heads=2, lyric_enc_d_ff=64,
        max_lyric_len=32, dropout=0.0, use_qk_norm=True,
    )
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg)
    with torch.no_grad():
        for block in model.blocks:
            block.cross_attn.out_proj.weight.normal_(std=0.02)
            block.lyric_attn.out_proj.weight.normal_(std=0.02)
        model.melody_encoder.ln_final.weight.fill_(1.0)
    return model.eval()


def _tokens(cfg: GPTConfig, B: int = 2, T: int = 24) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, cfg.vocab_per_codebook, (B, cfg.n_codebooks, T))


def test_paths_are_live_after_randomizing_gates():
    """Sanity: with gates randomized, real conditioning DOES change the output
    (otherwise the equality tests below would be vacuous)."""
    model = _randomized_model()
    tokens = _tokens(model.cfg)
    with torch.no_grad():
        base = model(tokens)
        cond = model(tokens, text_emb=torch.randn(2, 1, model.cfg.d_model))
    assert not torch.allclose(cond, base)


def test_zero_text_emb_equals_none():
    model = _randomized_model()
    tokens = _tokens(model.cfg)
    with torch.no_grad():
        none_out = model(tokens)
        zeros_out = model(tokens, text_emb=torch.zeros(2, 1, model.cfg.d_model))
    torch.testing.assert_close(zeros_out, none_out)


def test_zero_lyric_emb_equals_none():
    """The MASKLESS form is what train.py passes (CUDA fused SDPA rejects a
    broadcast size-1 kv_mask — '(*bias): last dimension must be contiguous');
    the masked form is asserted too, but only CPU's math backend can verify it."""
    model = _randomized_model()
    tokens = _tokens(model.cfg)
    zeros_emb = torch.zeros(2, 1, model.cfg.d_model)
    with torch.no_grad():
        none_out = model(tokens)
        maskless = model(tokens, lyric_emb=zeros_emb)
        masked = model(tokens, lyric_emb=zeros_emb,
                       lyric_kv_mask=torch.zeros(2, 1, 1, 1))
    torch.testing.assert_close(maskless, none_out)
    torch.testing.assert_close(masked, none_out)


def test_lyric_keep_zero_equals_none():
    """keep=0 with REAL lyric ids must equal the no-lyrics forward exactly —
    the always-run-encoder CFG gate (what train.py passes on dropped steps)."""
    model = _randomized_model()
    tokens = _tokens(model.cfg)
    ids = torch.randint(0, model.cfg.phoneme_vocab_size, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.bool)
    with torch.no_grad():
        none_out = model(tokens)
        gated = model(tokens, lyric_ids=ids, lyric_mask=mask,
                      lyric_keep=torch.zeros(()))
        kept = model(tokens, lyric_ids=ids, lyric_mask=mask,
                     lyric_keep=torch.ones(()))
        plain = model(tokens, lyric_ids=ids, lyric_mask=mask)
    torch.testing.assert_close(gated, none_out)
    torch.testing.assert_close(kept, plain)


def test_melody_keep_zero_equals_null_path():
    """keep=0 with real chroma must equal melody=None (the null path) exactly,
    and keep=1 must equal the plain conditioned forward."""
    model = _randomized_model()
    tokens = _tokens(model.cfg)
    T = tokens.shape[-1]
    torch.manual_seed(3)
    chroma = torch.rand(2, T, model.cfg.melody_n_bins)
    with torch.no_grad():
        null_out = model(tokens)  # melody=None -> learned null
        gated = model(tokens, melody=chroma, melody_keep=torch.zeros(()))
        kept = model(tokens, melody=chroma, melody_keep=torch.ones(()))
        plain = model(tokens, melody=chroma)
    torch.testing.assert_close(gated, null_out)
    torch.testing.assert_close(kept, plain)


def test_every_param_participates_with_keep_gates():
    """With zeros-text + keep=0 gates (the fully-dropped training step), every
    model parameter must still receive a grad hook — the precondition for DDP
    find_unused_parameters=False. A param with .grad None here would crash
    distributed training with 'Expected to have finished reduction'."""
    model = _randomized_model().train()
    tokens = _tokens(model.cfg)
    ids = torch.randint(0, model.cfg.phoneme_vocab_size, (2, 8))
    logits = model(
        tokens,
        text_emb=torch.zeros(2, 1, model.cfg.d_model),
        lyric_ids=ids, lyric_mask=torch.ones(2, 8, dtype=torch.bool),
        lyric_keep=torch.zeros(()),
        melody=torch.rand(2, tokens.shape[-1], model.cfg.melody_n_bins),
        melody_keep=torch.zeros(()),
    )
    logits.sum().backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad on a fully-dropped step: {missing}"


def test_conditioning_projections_are_bias_free():
    """The structural precondition for the zeros==None equivalence."""
    model = _randomized_model()
    for block in model.blocks:
        for attn in (block.cross_attn, block.lyric_attn):
            assert attn.q_proj.bias is None
            assert attn.kv_proj.bias is None
            assert attn.out_proj.bias is None
