"""QK-norm guards (the structural fix for the 2026-06-12 v8 divergences).

Three properties matter: (1) the flag actually adds/omits the norm modules,
(2) a pre-qk-norm checkpoint cfg dict — which lacks the key — must build a
model WITHOUT the norms so old checkpoints keep loading at every
``GPTConfig(**ckpt["cfg"])`` site, and (3) the KV-cached generate path must
agree with the full forward (the norm is applied before the cache write, so
decode steps must see the identical transform).
"""
from __future__ import annotations

import dataclasses

import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _cfg(**over) -> GPTConfig:
    base = dict(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=128,
        use_gradient_checkpointing=False, use_qk_norm=True,
    )
    base.update(over)
    return GPTConfig(**base)


def test_flag_adds_and_omits_norms():
    on = NanoAudioGPT(_cfg(use_text_conditioning=True))
    assert on.blocks[0].attn.q_norm is not None
    assert on.blocks[0].cross_attn.k_norm is not None
    off = NanoAudioGPT(_cfg(use_text_conditioning=True, use_qk_norm=False))
    assert off.blocks[0].attn.q_norm is None
    assert off.blocks[0].cross_attn.q_norm is None


def test_pre_qk_norm_cfg_dict_loads_without_norms():
    """A v7-era cfg dict has no use_qk_norm key — GPTConfig must default it
    False so the rebuilt model matches the old state dict."""
    old_dict = dataclasses.asdict(_cfg(use_qk_norm=False))
    del old_dict["use_qk_norm"]
    model = NanoAudioGPT(GPTConfig(**old_dict))
    assert model.blocks[0].attn.q_norm is None


def test_regional_block_compile_preserves_state_dict_keys():
    """train.py compiles each Block IN PLACE via nn.Module.compile() (regional
    compilation). Unlike `model = torch.compile(model)`, this must not inject
    `_orig_mod.` into state-dict keys — checkpoints stay loadable by the bare
    NanoAudioGPT everywhere (inference/eval/merge)."""
    model = NanoAudioGPT(_cfg())
    keys_before = list(model.state_dict().keys())
    for blk in model.blocks:
        blk.compile()
    assert list(model.state_dict().keys()) == keys_before


def test_generate_path_consistent_with_qk_norm():
    """Prefill+decode through the KV cache must match the cache-free forward
    on the prompt region — proves the norm-before-cache-write ordering."""
    torch.manual_seed(0)
    model = NanoAudioGPT(_cfg()).eval()
    K, T = model.cfg.n_codebooks, 20
    tokens = torch.randint(0, model.cfg.vocab_per_codebook, (1, K, T))
    with torch.no_grad():
        full = model(tokens)
        logits, caches = model(tokens[:, :, :12], kv_caches=[None] * model.cfg.n_layers)
        outs = [logits]
        # The tuple-cache path is single-step by design (is_causal=False once a
        # cache exists) — decode one position at a time, like generate() does.
        for t in range(12, T):
            step, caches = model(tokens[:, :, t:t + 1], kv_caches=caches, start_pos=t)
            outs.append(step)
        stitched = torch.cat(outs, dim=2)
    torch.testing.assert_close(stitched, full, atol=1e-4, rtol=1e-4)
