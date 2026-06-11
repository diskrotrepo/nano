"""Merge a LoRA adapter checkpoint into its base — producing a standard,
inference-ready checkpoint.

LoRA training (``diskrot.train --lora``) saves adapter-only checkpoints (no
``model`` key — the frozen base is referenced by path, not copied). Inference,
eval, MLX, and quantization all expect a plain checkpoint, so the adapters
must be folded in first: ``weight += (alpha/r)·B@A`` per adapted layer. The
output here matches the slim-inference schema ``diskrot.modal_export_ckpt``
produces (``model``/``cfg``/``step``/``best_val_loss``/``text_proj``), so the
merged file drops straight into ``server/inference.py``.

Run locally:
    python -m diskrot.merge_lora --base checkpoints/base_best.pt \
        --lora checkpoints/ft_run/best.pt --out checkpoints/merged_inference.pt

On the Modal volume, use ``diskrot/modal_merge_lora.py`` instead.
"""
from __future__ import annotations

from pathlib import Path

import torch

from model.lora import merge_state_dicts
from model.nano_audio_gpt import GPTConfig


def _strip_wrapper_prefixes(state_dict: dict) -> dict:
    return {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state_dict.items()
    }


def _maybe_half(state: dict, half: bool) -> dict:
    if not half:
        return state
    return {
        k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in state.items()
    }


def merge_lora_ckpts(base_ckpt: dict, lora_ckpt: dict, half: bool = True) -> dict:
    """Fold ``lora_ckpt``'s adapters into ``base_ckpt``'s weights.

    Returns the slim-inference checkpoint dict (see module docstring). The
    base may be a full training checkpoint or an ``*_inference`` slim (fp16 is
    fine — the delta is computed in fp32 and cast back). ``step`` and
    ``best_val_loss`` come from the LoRA run (they describe the merged
    weights); ``text_proj`` prefers the LoRA checkpoint's copy (present only
    when the run trained it) and falls back to the base's.
    """
    if "lora" not in lora_ckpt:
        raise ValueError("--lora checkpoint has no 'lora' key — is it actually "
                         "an adapter checkpoint from a --lora run?")
    if "model" not in base_ckpt:
        raise ValueError("--base checkpoint has no 'model' key — pass the base "
                         "(pretrained) checkpoint, not another adapter ckpt")
    # Compare through GPTConfig so an older base whose cfg dict predates newer
    # fields (which train_run round-trips into the lora ckpt as defaults)
    # doesn't false-positive.
    if GPTConfig(**base_ckpt["cfg"]) != GPTConfig(**lora_ckpt["cfg"]):
        raise ValueError(
            "base and lora checkpoints disagree on the model cfg — the "
            "adapters were trained on a different base architecture"
        )

    lcfg = lora_ckpt["lora"]["config"]
    merged = merge_state_dicts(
        _strip_wrapper_prefixes(base_ckpt["model"]),
        lora_ckpt["lora"]["state"],
        r=lcfg["r"], alpha=lcfg["alpha"],
    )

    out: dict = {
        "model": _maybe_half(merged, half),
        "cfg": base_ckpt["cfg"],
        "step": int(lora_ckpt.get("step", -1)),
    }
    if "best_val_loss" in lora_ckpt:
        out["best_val_loss"] = float(lora_ckpt["best_val_loss"])
    text_proj = lora_ckpt.get("text_proj", base_ckpt.get("text_proj"))
    if text_proj is not None:
        out["text_proj"] = _maybe_half(text_proj, half)
    return out


def merge_stats(base_model_sd: dict, lora_state: dict, r: int, alpha: int) -> dict:
    """Sanity numbers for the CLI: how many layers changed and by how much."""
    scaling = alpha / r
    n_layers = 0
    max_delta = 0.0
    for key, A in lora_state.items():
        if not key.endswith(".lora_A"):
            continue
        B = lora_state[key.removesuffix(".lora_A") + ".lora_B"]
        delta = (B.float() @ A.float()) * scaling
        max_delta = max(max_delta, delta.abs().max().item())
        n_layers += 1
    return {"n_layers": n_layers, "max_abs_delta": max_delta}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base", required=True, help="base (pretrained) checkpoint — full or *_inference slim")
    p.add_argument("--lora", required=True, help="adapter checkpoint from a --lora run (best.pt / latest.pt)")
    p.add_argument("--out", required=True, help="where to write the merged inference checkpoint")
    p.add_argument("--no-half", action="store_true", help="keep fp32 weights (default casts to fp16)")
    args = p.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    lora = torch.load(args.lora, map_location="cpu", weights_only=False)

    lcfg = lora["lora"]["config"]
    stats = merge_stats(base["model"], lora["lora"]["state"],
                        r=lcfg["r"], alpha=lcfg["alpha"])
    out = merge_lora_ckpts(base, lora, half=not args.no_half)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"merged {stats['n_layers']} adapted layers "
          f"(r={lcfg['r']} alpha={lcfg['alpha']}, max |Δw| {stats['max_abs_delta']:.4f})")
    print(f"wrote {out_path} (step {out['step']}, "
          f"{'fp16' if not args.no_half else 'fp32'}, "
          f"text_proj: {'text_proj' in out})")
