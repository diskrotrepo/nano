"""Merge a LoRA adapter checkpoint into its base ON the nano-ckpts volume.

Modal wrapper around ``diskrot.merge_lora.merge_lora_ckpts`` — same operation,
but the multi-GB base never leaves the volume. The output is a standard
slim-inference checkpoint (the ``modal_export_ckpt`` schema), ready to
``modal volume get`` and serve.

Run:
    modal run diskrot/modal_merge_lora.py --base v8_sing/best.pt \
        --lora v8_sing_lora/best.pt --out v8_sing_lora/merged_inference.pt
    # then pull it:
    modal volume get nano-ckpts /v8_sing_lora/merged_inference.pt ./checkpoints/latest.pt
"""
from __future__ import annotations

import modal

app = modal.App("nano-merge-lora")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
    .add_local_python_source("model", "diskrot")
)

ckpts_vol = modal.Volume.from_name("nano-ckpts")


@app.function(image=image, volumes={"/ckpts": ckpts_vol}, timeout=1800)
def merge_remote(base: str, lora: str, out: str, half: bool = True) -> dict:
    from pathlib import Path

    import torch

    from diskrot.merge_lora import merge_lora_ckpts, merge_stats

    base_path = Path("/ckpts") / base
    lora_path = Path("/ckpts") / lora
    for p in (base_path, lora_path):
        if not p.exists():
            raise FileNotFoundError(f"no checkpoint at {p}")

    base_ckpt = torch.load(base_path, map_location="cpu", weights_only=False)
    lora_ckpt = torch.load(lora_path, map_location="cpu", weights_only=False)

    lcfg = lora_ckpt["lora"]["config"]
    stats = merge_stats(base_ckpt["model"], lora_ckpt["lora"]["state"],
                        r=lcfg["r"], alpha=lcfg["alpha"])
    merged = merge_lora_ckpts(base_ckpt, lora_ckpt, half=half)

    out_path = Path("/ckpts") / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    ckpts_vol.commit()

    return {
        "base": base,
        "lora": lora,
        "out": out,
        "out_bytes": out_path.stat().st_size,
        "step": merged["step"],
        "half": half,
        "n_layers": stats["n_layers"],
        "max_abs_delta": stats["max_abs_delta"],
        "kept_text_proj": "text_proj" in merged,
    }


@app.local_entrypoint()
def main(base: str, lora: str, out: str = "", half: bool = True):
    from pathlib import Path

    if not out:
        # v8_sing_lora/best.pt -> v8_sing_lora/merged_inference.pt
        out = str(Path(lora).with_name("merged_inference.pt"))
    r = merge_remote.remote(base=base, lora=lora, out=out, half=half)
    gb = 1024 ** 3
    print(f"\nmerged {r['lora']} into {r['base']} -> {r['out']}")
    print(f"  layers: {r['n_layers']}   max |Δw|: {r['max_abs_delta']:.4f}   "
          f"size: {r['out_bytes']/gb:.2f} GB   step: {r['step']}   "
          f"fp16: {r['half']}   text_proj: {r['kept_text_proj']}")
    print(f"\npull it:\n  modal volume get nano-ckpts /{r['out']} ./checkpoints/latest.pt")
