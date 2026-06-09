"""Tests for model/lora.py — LoRA injection, freezing, and merge equivalence.

The load-bearing invariants:
  * adapters start as an exact identity (lora_B init zero), so step 0 of a
    fine-tune reproduces the base model;
  * merging the adapters into plain weights is numerically equivalent to running
    the LoRALinear forward — this is what lets inference stay LoRA-unaware;
  * only the adapters are trainable after mark_only_lora_trainable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.lora import (
    DEFAULT_LORA_TARGETS,
    LoRAConfig,
    LoRALinear,
    apply_lora,
    lora_parameters,
    mark_only_lora_trainable,
    merge_lora_state_dict,
)
from model.delay_pattern import build_train_inputs
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _tiny_cfg(**kw) -> GPTConfig:
    base = dict(
        d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=128,
        dropout=0.0, use_gradient_checkpointing=False,
    )
    base.update(kw)
    return GPTConfig(**base)


# --- LoRALinear unit behaviour ------------------------------------------------

def test_lora_linear_is_identity_at_init():
    """lora_B starts at zero → the adapter contributes nothing until trained."""
    torch.manual_seed(0)
    base = nn.Linear(16, 24, bias=False)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_linear_merged_weight_matches_forward():
    torch.manual_seed(1)
    base = nn.Linear(16, 24, bias=False)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    # Make the adapter non-trivial.
    nn.init.normal_(lora.lora_B, std=0.1)
    x = torch.randn(5, 16)
    merged = nn.Linear(16, 24, bias=False)
    with torch.no_grad():
        merged.weight.copy_(lora.merged_weight())
    assert torch.allclose(lora(x), merged(x), atol=1e-5)


def test_lora_linear_freezes_base():
    base = nn.Linear(8, 8, bias=False)
    lora = LoRALinear(base, rank=2, alpha=2.0)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


# --- injection into the full model --------------------------------------------

def test_apply_lora_wraps_expected_targets():
    cfg = _tiny_cfg(use_text_conditioning=True)
    model = NanoAudioGPT(cfg)
    n = apply_lora(model, LoRAConfig(rank=4, alpha=8.0))
    # per block: qkv, proj (self-attn) + q_proj, kv_proj, out_proj (cross-attn)
    #            + fc1, fc2 (mlp) = 7; ×2 blocks = 14.
    assert n == 14
    wrapped = [m for m in model.modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == n
    # The per-codebook heads (numerically-indexed ModuleList) stay plain.
    assert all(not isinstance(h, LoRALinear) for h in model.heads)


def test_mark_only_lora_trainable():
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg)
    apply_lora(model, LoRAConfig(rank=4, alpha=8.0))
    mark_only_lora_trainable(model)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "expected some trainable params"
    assert all(n.endswith((".lora_A", ".lora_B")) for n in trainable)
    # And the trainable set is exactly the lora_parameters() helper's view.
    assert sum(p.numel() for p in lora_parameters(model)) == sum(
        p.numel() for n, p in model.named_parameters() if p.requires_grad
    )


def test_lora_is_a_small_fraction_of_params():
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg)
    total = model.num_params()
    apply_lora(model, LoRAConfig(rank=2, alpha=4.0))
    mark_only_lora_trainable(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < total * 0.2


# --- end-to-end merge equivalence ---------------------------------------------

def test_merge_state_dict_loads_into_plain_model_and_matches():
    """The headline guarantee: a merged state-dict loads strictly into a vanilla
    NanoAudioGPT and reproduces the LoRA model's outputs bit-for-bit (within fp
    tolerance)."""
    torch.manual_seed(2)
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg).eval()
    lora_cfg = LoRAConfig(rank=4, alpha=8.0)
    apply_lora(model, lora_cfg)
    # Train the adapters a little so they're not the identity.
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.05)

    B, K, T = 2, cfg.n_codebooks, 16
    codes = torch.randint(0, cfg.vocab_per_codebook, (B, K, T))
    inp, _ = build_train_inputs(codes, cfg.pad_id)
    with torch.no_grad():
        ref = model(inp)

    merged_state = merge_lora_state_dict(model.state_dict(), lora_cfg.to_dict())
    plain = NanoAudioGPT(cfg).eval()
    missing, unexpected = plain.load_state_dict(merged_state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    with torch.no_grad():
        got = plain(inp)
    assert torch.allclose(ref, got, atol=1e-5)


def test_merge_passes_through_non_lora_keys():
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg)
    apply_lora(model, LoRAConfig(rank=2, alpha=2.0))
    merged = merge_lora_state_dict(model.state_dict(), LoRAConfig(rank=2, alpha=2.0).to_dict())
    # No adapter keys survive; the embedding/head keys (never wrapped) do.
    assert not any(k.endswith((".lora_A", ".lora_B", ".base.weight")) for k in merged)
    assert "tok_embeds.0.weight" in merged
    assert "heads.0.weight" in merged


def test_lora_config_roundtrips():
    cfg = LoRAConfig(rank=8, alpha=16.0, dropout=0.1, targets=("qkv", "fc1"))
    again = LoRAConfig.from_dict(cfg.to_dict())
    assert again == cfg


def test_default_targets_cover_attn_and_mlp():
    for name in ("qkv", "proj", "fc1", "fc2"):
        assert name in DEFAULT_LORA_TARGETS


# --- checkpoint contract (mirrors diskrot.train + server.inference) ------------

def test_checkpoint_lora_meta_roundtrips_through_inference_merge():
    """A LoRA run's checkpoint carries the adapters under "model" plus a "lora"
    metadata key; the inference path merges them back into a plain model. This
    locks the train.py write side against the server read side."""
    from diskrot.train import _build_ckpt_dict

    torch.manual_seed(3)
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg).eval()
    lora_cfg = LoRAConfig(rank=4, alpha=8.0)
    apply_lora(model, lora_cfg)
    mark_only_lora_trainable(model)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B, std=0.05)

    optim = torch.optim.AdamW(lora_parameters(model), lr=1e-3)
    ckpt = _build_ckpt_dict(
        model=model, optim=optim, text_encoder=None,
        cfg_model_dict=dict(cfg.__dict__), step=10, best_val_loss=1.0,
        best_val_step=10, evals_without_improvement=0, prev_val_loss=None,
        lora_meta=lora_cfg.to_dict(),
    )
    assert ckpt["lora"] == lora_cfg.to_dict()
    assert any(k.endswith(".lora_A") for k in ckpt["model"])

    B, K, T = 1, cfg.n_codebooks, 12
    codes = torch.randint(0, cfg.vocab_per_codebook, (B, K, T))
    inp, _ = build_train_inputs(codes, cfg.pad_id)
    with torch.no_grad():
        ref = model(inp)

    # Replicates server/inference.py's merge-on-load.
    merged = merge_lora_state_dict(ckpt["model"], ckpt["lora"])
    plain = NanoAudioGPT(cfg).eval()
    plain.load_state_dict(merged)
    with torch.no_grad():
        got = plain(inp)
    assert torch.allclose(ref, got, atol=1e-5)
