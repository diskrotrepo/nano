"""LoRA (low-rank adaptation) for parameter-efficient fine-tuning of NanoAudioGPT.

Wraps selected ``nn.Linear`` layers with a *frozen* base weight plus a trainable
low-rank update ``B @ A`` (scaled by ``alpha / rank``). This lets you adapt the
~1.5B base model to a new corpus by training well under 1% of the parameters,
then **merge** the deltas back into plain weights so inference sees a normal
checkpoint with zero runtime overhead and no architecture change.

The wrapper keeps the base ``nn.Linear`` as a child named ``base`` and adds two
parameters ``lora_A`` / ``lora_B``. State-dict keys therefore look like::

    blocks.0.attn.qkv.base.weight   # frozen, == the pretrained weight
    blocks.0.attn.qkv.lora_A        # trainable [rank, in_features]
    blocks.0.attn.qkv.lora_B        # trainable [out_features, rank]

[[merge_lora_state_dict]] folds these three back into a single
``blocks.0.attn.qkv.weight`` so both the PyTorch and MLX inference backends load
the result with no LoRA awareness at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Attribute names of the projections we adapt by default: self-attention
# (qkv/proj), the tag + lyric cross-attention (q_proj/kv_proj/out_proj) and the
# MLP (fc1/fc2), across the main transformer blocks AND the lyric encoder (which
# reuses the same attribute names). The per-codebook output heads and the token
# embeddings live in numerically-indexed ModuleLists, so they are never matched
# and stay frozen — standard LoRA practice.
DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "qkv", "proj", "q_proj", "kv_proj", "out_proj", "fc1", "fc2",
)

_LORA_SUFFIXES = (".lora_A", ".lora_B")


@dataclass
class LoRAConfig:
    """How LoRA adapters are sized and where they attach.

    Round-trips to/from the plain dict stored under the checkpoint's ``"lora"``
    key so a saved fine-tune can be merged or resumed without re-specifying it.
    """
    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = DEFAULT_LORA_TARGETS

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": list(self.targets),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        return cls(
            rank=int(d["rank"]),
            alpha=float(d["alpha"]),
            dropout=float(d.get("dropout", 0.0)),
            targets=tuple(d.get("targets", DEFAULT_LORA_TARGETS)),
        )


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` update.

    ``forward(x) = base(x) + scaling * (dropout(x) @ A^T) @ B^T`` where
    ``scaling = alpha / rank``. ``lora_B`` is initialised to zero so the adapter
    is an exact identity at step 0 — fine-tuning starts from the base model's
    behaviour and departs from it gradually.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert rank > 0, "LoRALinear requires rank > 0"
        self.base = base
        # Freeze the pretrained weight (and bias, though this model is bias-free).
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        w = base.weight  # [out_features, in_features]
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=w.device, dtype=w.dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=w.device, dtype=w.dtype))
        # Kaiming on A (matches the reference LoRA init); B stays zero.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout: nn.Module = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t()
        delta = delta @ self.lora_B.t()
        return out + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """The single effective weight matrix (base + scaled low-rank delta)."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base.weight + delta.to(self.base.weight.dtype)


def apply_lora(module: nn.Module, cfg: LoRAConfig) -> int:
    """Recursively replace target ``nn.Linear`` children with ``LoRALinear``.

    Returns the number of adapters injected. Safe to call on a model already on
    its device — the new LoRA params are created on the base weight's device.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in cfg.targets:
            setattr(module, name, LoRALinear(child, cfg.rank, cfg.alpha, cfg.dropout))
            n += 1
        else:
            # Recurse; a LoRALinear's frozen base is named "base" and won't match
            # a target, so we never double-wrap.
            n += apply_lora(child, cfg)
    return n


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze every parameter except the LoRA adapters (``lora_A`` / ``lora_B``)."""
    for name, p in model.named_parameters():
        p.requires_grad_(name.endswith(_LORA_SUFFIXES))


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The trainable LoRA parameters, for building an optimizer or counting."""
    return [p for n, p in model.named_parameters() if n.endswith(_LORA_SUFFIXES)]


def merge_lora_state_dict(state: dict, lora_meta: dict) -> dict:
    """Fold LoRA adapters in a raw state-dict back into plain ``.weight`` tensors.

    Input keys ``<p>.base.weight`` / ``<p>.lora_A`` / ``<p>.lora_B`` collapse to a
    single ``<p>.weight``; every other key passes through untouched. Pure tensor
    math — no model construction — so both inference backends can call it.
    """
    cfg = LoRAConfig.from_dict(lora_meta)
    scaling = cfg.alpha / cfg.rank
    prefixes = {k[: -len(".lora_A")] for k in state if k.endswith(".lora_A")}

    out: dict = {}
    for k, v in state.items():
        # The two adapter tensors are consumed into the merged weight below.
        if k.endswith(_LORA_SUFFIXES) and k.rsplit(".", 1)[0] in prefixes:
            continue
        if k.endswith(".base.weight"):
            pre = k[: -len(".base.weight")]
            if pre in prefixes:
                A = state[pre + ".lora_A"]
                B = state[pre + ".lora_B"]
                delta = (B.float() @ A.float()) * scaling
                out[pre + ".weight"] = v + delta.to(v.dtype)
                continue
        out[k] = v
    return out
