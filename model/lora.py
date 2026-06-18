"""LoRA (low-rank adaptation) for NanoAudioGPT.

Hand-rolled on purpose: the checkpoint format here is a plain dict of tensors,
and PEFT's PeftModel wrapper + adapter-file format would fight DDP,
torch.compile, and the MLX/quantization inference paths for no gain.

Design: ``LoRALinear`` SUBCLASSES ``nn.Linear``, so the frozen base ``weight``
keeps its original state-dict key (e.g. ``blocks.0.attn.qkv.weight``) and the
adapters are purely additive keys (``...qkv.lora_A`` / ``...qkv.lora_B``).
That makes the whole scheme remap-free:

- adapter extraction is a key-suffix filter (``lora_state_dict``),
- merging is ``weight += (alpha/r) * B @ A`` then dropping the lora keys
  (``merge_state_dicts``), which reproduces the base key layout exactly —
  so a merged checkpoint is indistinguishable from a normally-trained one
  and inference/eval/MLX/quantization need zero changes.

Injection must happen AFTER the base checkpoint is loaded (``from_linear``
reuses the loaded ``weight`` Parameter rather than copying it) and BEFORE the
DDP wrap / torch.compile (DDP's constructor broadcast then syncs the rank-0
adapter init, and its reducer only buckets requires_grad params).

``NanoAudioGPT._init_weights`` runs at model construction — before injection —
so it never clobbers the adapter init. Gradient checkpointing
(use_reentrant=False) and AMP/GradScaler need nothing special: the lora path
is plain tensor ops on Parameters of the wrapped module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suffixes matched against dotted module names under ``model.blocks``: covers
# self-attention (qkv, proj), both cross-attentions (tags + lyrics share the
# CrossAttention projection names), and the MLP. The fused output head, token
# embeddings, norms, and the lyric/melody encoders stay frozen.
DEFAULT_TARGETS = "attn.qkv,attn.proj,q_proj,kv_proj,out_proj,fc1,fc2"


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    targets: str = DEFAULT_TARGETS  # comma-separated module-name suffixes

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        return cls(**d)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r

    @property
    def target_suffixes(self) -> list[str]:
        return [t.strip() for t in self.targets.split(",") if t.strip()]


class LoRALinear(nn.Linear):
    """nn.Linear with an additive low-rank delta: y = Wx + (alpha/r)·B(Ax).

    ``lora_B`` is zero-initialized, so a freshly injected layer computes
    exactly the base layer's output (identity at the start of training).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 r: int, alpha: int, dropout: float = 0.0, **kwargs):
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        assert r > 0, "LoRA rank must be positive"
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        dev_dt = {"device": self.weight.device, "dtype": self.weight.dtype}
        self.lora_A = nn.Parameter(torch.empty(r, in_features, **dev_dt))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, **dev_dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    @classmethod
    def from_linear(cls, base: nn.Linear, r: int, alpha: int,
                    dropout: float = 0.0) -> "LoRALinear":
        """Wrap an existing Linear, REUSING its weight/bias Parameters (no copy)
        so injection after a strict base load keeps the loaded tensors."""
        new = cls(
            base.in_features, base.out_features, bias=base.bias is not None,
            r=r, alpha=alpha, dropout=dropout,
            device=base.weight.device, dtype=base.weight.dtype,
        )
        new.weight = base.weight
        if base.bias is not None:
            new.bias = base.bias
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        # Low-rank order: (x @ A.T) @ B.T keeps the intermediate at rank r.
        return y + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    @torch.no_grad()
    def merge_(self) -> None:
        """Fold the adapter into the base weight in place (fp32 compute)."""
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        self.weight.data += delta.to(self.weight.dtype)
        nn.init.zeros_(self.lora_B)


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Replace every nn.Linear under ``model.blocks`` whose dotted name ends
    with one of ``cfg.targets`` by a LoRALinear sharing its weight.

    Returns the replaced module names (relative to the model root). Raises if
    nothing matched — a silent no-op here would "train" zero parameters.
    """
    suffixes = tuple(cfg.target_suffixes)
    if not suffixes:
        raise ValueError("LoRAConfig.targets is empty")
    replaced: list[str] = []
    for name, module in model.blocks.named_modules():
        if not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not name.endswith(suffixes):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.blocks.get_submodule(parent_name) if parent_name else model.blocks
        setattr(parent, attr, LoRALinear.from_linear(
            module, r=cfg.r, alpha=cfg.alpha, dropout=cfg.dropout))
        replaced.append(f"blocks.{name}")
    if not replaced:
        raise ValueError(
            f"inject_lora matched no modules for targets={cfg.targets!r} — "
            "check the suffix list against the model's block layout"
        )
    return replaced


def mark_only_lora_trainable(model: nn.Module) -> tuple[int, int]:
    """Freeze everything except lora_A/lora_B. Returns (n_trainable, n_total)."""
    n_trainable = 0
    n_total = 0
    for name, p in model.named_parameters():
        is_lora = name.endswith(("lora_A", "lora_B"))
        p.requires_grad_(is_lora)
        n_total += p.numel()
        if is_lora:
            n_trainable += p.numel()
    return n_trainable, n_total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Adapter-only state dict (caller passes the UNWRAPPED module)."""
    return {k: v for k, v in model.state_dict().items()
            if k.endswith((".lora_A", ".lora_B"))}


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load adapter weights into an already-injected model.

    Key sets must match exactly — a mismatch means the LoRAConfig used to
    inject differs from the one the state was saved with.
    """
    own = lora_state_dict(model)
    if set(own) != set(state):
        missing = sorted(set(own) - set(state))[:3]
        unexpected = sorted(set(state) - set(own))[:3]
        raise ValueError(
            f"LoRA adapter keys don't match the injected model "
            f"(missing {missing}..., unexpected {unexpected}...) — was the "
            "checkpoint saved with different lora targets/rank?"
        )
    with torch.no_grad():
        full = model.state_dict()
        for k, v in state.items():
            full[k].copy_(v)


def merge_lora_(model: nn.Module) -> int:
    """In-place merge of every LoRALinear in the model. Returns the count."""
    n = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_()
            n += 1
    return n


def merge_state_dicts(
    base_sd: dict[str, torch.Tensor],
    lora_sd: dict[str, torch.Tensor],
    r: int,
    alpha: int,
) -> dict[str, torch.Tensor]:
    """Pure state-dict-level merge — no model instantiation, CPU-friendly,
    fp16-safe (delta computed in fp32, cast back to the base dtype).

    For every ``{prefix}.lora_A``/``{prefix}.lora_B`` pair the output carries
    ``{prefix}.weight = base + (alpha/r)·B@A``; all other base keys pass
    through untouched. The output key set is EXACTLY the base key set, so the
    merged dict loads strictly into a plain NanoAudioGPT.
    """
    scaling = alpha / r
    out = dict(base_sd)
    consumed = 0
    for key, A in lora_sd.items():
        if not key.endswith(".lora_A"):
            continue
        prefix = key.removesuffix(".lora_A")
        b_key = f"{prefix}.lora_B"
        w_key = f"{prefix}.weight"
        if b_key not in lora_sd:
            raise KeyError(f"lora state has {key} but no {b_key}")
        if w_key not in base_sd:
            raise KeyError(
                f"lora adapter {prefix} has no matching weight in the base "
                f"state dict — wrong base checkpoint?"
            )
        B = lora_sd[b_key]
        base_w = base_sd[w_key]
        delta = (B.float() @ A.float()) * scaling
        out[w_key] = (base_w.float() + delta).to(base_w.dtype)
        consumed += 2
    if consumed != len(lora_sd):
        stray = [k for k in lora_sd if not k.endswith((".lora_A", ".lora_B"))]
        raise ValueError(
            f"merge consumed {consumed}/{len(lora_sd)} lora keys "
            f"(non-adapter keys in lora state: {stray[:3]})"
        )
    return out
