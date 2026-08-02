"""Batched-generation correctness guard (the POST /generate_batch path).

The model decode loop is batch-general; ``generate_audio_batch`` builds a [B,K,1]
seed + batched conditioning and runs ONE forward. The load-bearing contract:
with ``temperature=0`` (argmax, no sampling RNG), every row of a batch must equal
the *independent* B=1 generation of that row's prompt — which pins the batched KV
cache, the per-stage CFG guidance, and the per-frame emission, and proves rows
don't leak into each other through attention or the cache.

These tests need only torch (no DAC/g2p), like ``test_generate_stream.py``.
"""
from __future__ import annotations

import types

import pytest
import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _tiny_cfg(**kw) -> GPTConfig:
    base = dict(
        d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=256,
        use_gradient_checkpointing=False,
    )
    base.update(kw)
    return GPTConfig(**base)


def test_resolve_prompt_batch_shape():
    cfg = _tiny_cfg(
        use_text_conditioning=False, use_lyric_conditioning=False,
        use_melody_conditioning=False,
    )
    model = NanoAudioGPT(cfg).eval()
    p, squeeze = model._resolve_prompt(None, batch_size=4)
    assert p.shape == (4, cfg.n_codebooks, 1)
    assert squeeze is False
    # B=1 default reproduces the original single [K,1] seed (unsqueezed to [1,K,1]).
    p1, _ = model._resolve_prompt(None, batch_size=1)
    assert p1.shape == (1, cfg.n_codebooks, 1)


def test_batched_matches_single_plain():
    """Each row of a batch of identical prompts == the B=1 result (temp=0)."""
    cfg = _tiny_cfg(
        use_text_conditioning=False, use_lyric_conditioning=False,
        use_melody_conditioning=False,
    )
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    prompt = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    ref = model.generate(prompt, num_new_frames=20, temperature=0.0, top_k=50)
    B = 3
    bprompt = prompt.unsqueeze(0).expand(B, K, 3).contiguous()
    out = model.generate(bprompt, num_new_frames=20, temperature=0.0, top_k=50)
    assert out.shape == (B, K, 3 + 20)
    for i in range(B):
        assert torch.equal(out[i], ref)


def test_batched_matches_single_with_cfg():
    """Batched CFG: [B,1,D] text_emb of identical rows == the B=1 cfg run."""
    cfg = _tiny_cfg(use_text_conditioning=True)
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    prompt = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    torch.manual_seed(7)
    emb = torch.randn(1, 1, cfg.d_model)
    ref = model.generate(
        prompt, num_new_frames=18, temperature=0.0, top_k=40,
        text_emb=emb, cfg_scale=3.0,
    )
    B = 3
    bprompt = prompt.unsqueeze(0).expand(B, K, 3).contiguous()
    bemb = emb.expand(B, 1, cfg.d_model).contiguous()
    out = model.generate(
        bprompt, num_new_frames=18, temperature=0.0, top_k=40,
        text_emb=bemb, cfg_scale=3.0,
    )
    assert out.shape == (B, K, 3 + 18)
    for i in range(B):
        assert torch.equal(out[i], ref)


def test_batched_rows_independent():
    """DIFFERENT prompts batched together: each row matches its own B=1 run, so
    rows neither couple through attention nor share KV-cache state."""
    cfg = _tiny_cfg(
        use_text_conditioning=False, use_lyric_conditioning=False,
        use_melody_conditioning=False,
    )
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    p0 = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    p1 = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    ref0 = model.generate(p0, num_new_frames=15, temperature=0.0, top_k=50)
    ref1 = model.generate(p1, num_new_frames=15, temperature=0.0, top_k=50)
    out = model.generate(
        torch.stack([p0, p1], dim=0), num_new_frames=15, temperature=0.0, top_k=50,
    )
    assert torch.equal(out[0], ref0)
    assert torch.equal(out[1], ref1)


# --- _stack_conditioning (server/inference.py) -------------------------------
# Bound onto a stub like test_generate_stream's streaming tests, so no checkpoint
# is needed. The tag path is g2p-free; the lyric-synthesis path needs g2p.


def _cond_stub():
    from server.inference import InferenceEngine

    stub = types.SimpleNamespace(
        device="cpu",
        model=types.SimpleNamespace(cfg=types.SimpleNamespace(max_lyric_len=64)),
    )
    stub._stack_conditioning = InferenceEngine._stack_conditioning.__get__(stub)
    return stub


def test_stack_conditioning_tags_zero_fill():
    stub = _cond_stub()
    D = 8
    e = torch.randn(1, 1, D)  # a single-chunk tag
    # all-None -> the whole axis is None (model skips text conditioning).
    t, tkv, li, lm = stub._stack_conditioning([(None, None, None), (None, None, None)])
    assert t is None and tkv is None and li is None and lm is None
    # some tags -> [B,Nmax,D]; the tag-less row is a single un-masked ZERO chunk
    # (== skip via the bias-free cross-attn), never a fully-masked row.
    t, tkv, li, lm = stub._stack_conditioning([(e, None, None), (None, None, None)])
    assert t.shape == (2, 1, D)
    assert torch.equal(t[0], e[0])
    assert torch.count_nonzero(t[1]) == 0
    assert tkv.shape == (2, 1, 1, 1)
    assert torch.all(tkv == 0)  # both rows attend their single chunk (no padding)
    assert li is None and lm is None


def test_stack_conditioning_chunked_tags_ragged_mask():
    """Different-length chunked tags pad to Nmax with an additive -inf mask on the
    pad of each row, while a tag-less row keeps one un-masked zero chunk."""
    stub = _cond_stub()
    D = 8
    a = torch.randn(1, 3, D)  # 3 chunks
    b = torch.randn(1, 1, D)  # 1 chunk
    t, tkv, _, _ = stub._stack_conditioning([(a, None, None), (b, None, None), (None, None, None)])
    assert t.shape == (3, 3, D)
    assert tkv.shape == (3, 1, 1, 3)
    # row 0: all 3 real (attend); row 1: 1 real + 2 padded (-inf); row 2: 1 zero chunk
    assert torch.all(tkv[0] == 0)
    assert torch.isinf(tkv[1, 0, 0, 1]) and torch.isinf(tkv[1, 0, 0, 2])
    assert tkv[1, 0, 0, 0] == 0
    assert (tkv[2] == 0).any()  # at least one un-masked key -> no fully-masked row


def test_stack_conditioning_lyrics_no_fully_masked_row():
    """Mixed lyric presence: the lyric-less row gets the BOS+header stream so no
    row is fully padded (which would NaN the cross-attn). Needs g2p_en."""
    pytest.importorskip("g2p_en")
    from model.lyric_encoder import PAD_PHONEME_ID

    stub = _cond_stub()
    lids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)  # a row WITH lyrics
    lmask = lids != PAD_PHONEME_ID
    t, tkv, li, lm = stub._stack_conditioning([(None, lids, lmask), (None, None, None)])
    assert li.shape[0] == 2 and lm.shape == li.shape
    # every row keeps at least one valid (non-pad) token -> no fully-masked row.
    assert bool((lm.sum(dim=1) > 0).all())
