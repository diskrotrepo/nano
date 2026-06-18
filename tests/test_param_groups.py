"""Guards for the AdamW no-decay split (diskrot/train.py:split_decay_param_groups).

Weight decay on RMSNorm gains actively shrinks the QK-norm gains and the
zero-gated melody ln_final gain — the parameters the 2026-06-12 stability fixes
depend on. The split must put every ndim<2 param plus the melody ``null`` in the
no-decay group, and every matrix weight (attention, MLP, embeddings) in the
decay group, with nothing lost or duplicated.
"""
from __future__ import annotations

import torch

from diskrot.train import split_decay_param_groups
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _model() -> NanoAudioGPT:
    cfg = GPTConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=128,
        use_text_conditioning=True, use_lyric_conditioning=True,
        use_melody_conditioning=True, use_gradient_checkpointing=False,
        lyric_enc_layers=1, lyric_enc_heads=2, lyric_enc_d_ff=64,
        max_lyric_len=32, use_qk_norm=True,
    )
    return NanoAudioGPT(cfg)


def test_split_membership_and_completeness():
    model = _model()
    named = list(model.named_parameters())
    groups = split_decay_param_groups(named, weight_decay=0.02)
    assert groups[0]["weight_decay"] == 0.02 and groups[1]["weight_decay"] == 0.0
    decay = {id(p) for p in groups[0]["params"]}
    no_decay = {id(p) for p in groups[1]["params"]}
    assert not (decay & no_decay), "a param landed in both groups"
    assert len(decay) + len(no_decay) == len(named), "params lost in the split"

    by_name = dict(named)
    # Every norm gain (incl. QK-norm) and the melody null: no decay.
    for name, p in named:
        if p.ndim < 2 or name.endswith(".null"):
            assert id(p) in no_decay, f"{name} should be no-decay"
    assert id(model.blocks[0].attn.q_norm.weight) in no_decay
    assert id(model.melody_encoder.null) in no_decay
    assert id(model.melody_encoder.ln_final.weight) in no_decay
    # Matrix weights: decay.
    assert id(model.blocks[0].attn.qkv.weight) in decay
    assert id(model.blocks[0].mlp.fc1.weight) in decay
    assert id(model.tok_embeds[0].weight) in decay


def test_frozen_params_excluded():
    model = _model()
    model.tok_embeds[0].weight.requires_grad_(False)
    groups = split_decay_param_groups(
        list(model.named_parameters()), weight_decay=0.02)
    all_ids = {id(p) for g in groups for p in g["params"]}
    assert id(model.tok_embeds[0].weight) not in all_ids
