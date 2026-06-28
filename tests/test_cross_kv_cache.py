"""Cross-attention K/V cache byte-identity guard.

`CrossAttention.forward` caches the projected (+ k-normed) K/V of the fixed
conditioning sequence at prefill and reuses it on every decode step, instead of
recomputing ``kv_proj(cond)`` each step (the dominant per-token decode cost when a
long lyric/tag stream is present — see model/nano_audio_gpt.py CrossKVCache).

Because `cond` never changes across decode steps and `k_norm` is
position-independent, the cached value is *exactly* what a recompute would produce.
These tests pin that: generation with the cache ON must be BYTE-IDENTICAL to the
recompute path (cache OFF via the `_use_cross_kv_cache` A/B handle) at temperature 0,
across every CFG-staging shape — single-stage, multi-stage tag CFG, lyrics-only, and
composed lyric guidance (where per-stage cache isolation is load-bearing: each
guidance stage has its own cond and must NOT share a cache).
"""
from __future__ import annotations

import pytest
import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _model() -> NanoAudioGPT:
    """Tiny text+lyric+qk-norm model with cross-attn out-projections randomized so
    the conditioning paths are LIVE (the v8 zero-gated init would otherwise make the
    cached/recomputed equality trivially true on a dead path)."""
    cfg = GPTConfig(
        n_codebooks=4, vocab_per_codebook=64, d_model=128, n_layers=3, n_heads=4,
        d_ff=256, max_seq_len=512, dropout=0.0,
        use_text_conditioning=True, use_lyric_conditioning=True,
        use_qk_norm=True, use_lyric_qk_norm=True,
        lyric_enc_layers=1, lyric_enc_heads=2, lyric_enc_d_ff=64, max_lyric_len=64,
    )
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg)
    with torch.no_grad():
        for block in model.blocks:
            block.cross_attn.out_proj.weight.normal_(std=0.02)
            block.lyric_attn.out_proj.weight.normal_(std=0.02)
    return model.eval()


def _cond(cfg: GPTConfig, B: int = 2):
    torch.manual_seed(7)
    return {
        "text_emb": torch.randn(B, 5, cfg.d_model),
        "text_neg": torch.randn(B, 5, cfg.d_model),
        "lyr": torch.randint(4, cfg.phoneme_vocab_size, (B, 40)),
        "lyr_m": torch.ones(B, 40, dtype=torch.bool),
        "lyr_neg": torch.randint(4, cfg.phoneme_vocab_size, (B, 12)),
        "lyr_neg_m": torch.ones(B, 12, dtype=torch.bool),
    }


def _gen(model: NanoAudioGPT, cache_on: bool, **kw) -> torch.Tensor:
    model._use_cross_kv_cache = cache_on
    torch.manual_seed(123)  # fixed so temp>0 paths are comparable too
    prompt = torch.randint(0, model.cfg.vocab_per_codebook, (2, model.cfg.n_codebooks, 8))
    return model.generate(
        prompt, num_new_frames=24, temperature=0.0, top_k=None, top_p=None, **kw
    )


def _cases(model: NanoAudioGPT):
    c = _cond(model.cfg)
    return {
        "no-cfg tags+lyrics": dict(
            text_emb=c["text_emb"], lyric_ids=c["lyr"], lyric_mask=c["lyr_m"], cfg_scale=1.0),
        "multi-stage tag CFG": dict(
            text_emb=c["text_emb"], text_emb_neg=c["text_neg"],
            lyric_ids=c["lyr"], lyric_mask=c["lyr_m"], cfg_scale=3.0),
        "lyrics-only single-stage": dict(
            lyric_ids=c["lyr"], lyric_mask=c["lyr_m"], cfg_scale=1.0),
        "tags-only CFG": dict(
            text_emb=c["text_emb"], text_emb_neg=c["text_neg"], cfg_scale=2.5),
        "composed lyric CFG": dict(
            text_emb=c["text_emb"], text_emb_neg=c["text_neg"],
            lyric_ids=c["lyr"], lyric_mask=c["lyr_m"],
            lyric_ids_neg=c["lyr_neg"], lyric_mask_neg=c["lyr_neg_m"],
            cfg_scale=2.0, lyric_cfg_scale=4.0),
    }


@pytest.mark.parametrize("case", list(_cases(_model()).keys()))
def test_cross_kv_cache_byte_identical(case: str):
    model = _model()
    kw = _cases(model)[case]
    cached = _gen(model, True, **kw)
    recomputed = _gen(model, False, **kw)
    assert torch.equal(cached, recomputed), f"cache != recompute for: {case}"


def test_conditioning_is_live():
    """Guard against a vacuous test: the cross-attn paths must actually move the
    output (otherwise cached==recomputed proves nothing)."""
    model = _model()
    c = _cond(model.cfg)
    base = _gen(model, True, cfg_scale=1.0)
    cond = _gen(model, True, text_emb=c["text_emb"], lyric_ids=c["lyr"],
                lyric_mask=c["lyr_m"], cfg_scale=1.0)
    assert not torch.equal(base, cond)


def test_cache_off_matches_training_forward_path():
    """cache OFF must drive the SAME code path the (cache-less) training forward
    uses — a single full-sequence forward with no cross_kv_caches arg."""
    model = _model()
    c = _cond(model.cfg)
    tokens = torch.randint(0, model.cfg.vocab_per_codebook, (2, model.cfg.n_codebooks, 16))
    with torch.no_grad():
        # forward() with no cross_kv_caches == the recompute branch (kv_cache=None)
        a = model(tokens, text_emb=c["text_emb"], lyric_ids=c["lyr"], lyric_mask=c["lyr_m"])
        b = model(tokens, text_emb=c["text_emb"], lyric_ids=c["lyr"], lyric_mask=c["lyr_m"],
                  cross_kv_caches=None)
    torch.testing.assert_close(a, b)
