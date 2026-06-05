"""Export a slim, inference-only checkpoint on the nano-ckpts volume.

A training checkpoint (best.pt / latest.pt) carries the AdamW optimizer state,
which for a 1.5B model is ~2x the model size (exp_avg + exp_avg_sq, both fp32).
Inference never reads it — ``server/inference.py`` only loads ``model``, ``cfg``
and ``text_proj``. This entrypoint strips ``optim`` (and anything else inference
ignores) and, by default, casts the weights to fp16 — which inference already
does on GPU/MPS — producing a checkpoint ~6x smaller and far faster to
``modal volume get``.

Run:
    modal run diskrot/modal_export_ckpt.py --src v7_1500m/best.pt
    # then pull the slim file:
    modal volume get nano-ckpts /v7_1500m/best_inference.pt ./checkpoints/latest.pt
"""
from __future__ import annotations

import modal

app = modal.App("nano-export-ckpt")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
)

ckpts_vol = modal.Volume.from_name("nano-ckpts")


@app.function(image=image, volumes={"/ckpts": ckpts_vol}, timeout=1800)
def export_slim(src: str, dst: str = "", half: bool = True) -> dict:
    from pathlib import Path

    import torch

    src_path = Path("/ckpts") / src
    if not src_path.exists():
        raise FileNotFoundError(f"no checkpoint at /ckpts/{src}")

    # weights_only=False so we can read cfg / text_proj dicts.
    ckpt = torch.load(src_path, map_location="cpu", weights_only=False)

    def _maybe_half(state: dict) -> dict:
        if not half:
            return state
        return {
            k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in state.items()
        }

    slim: dict = {
        "model": _maybe_half(ckpt["model"]),
        "cfg": ckpt["cfg"],
        "step": int(ckpt.get("step", -1)),
    }
    # tiny, useful to keep for provenance / health endpoint
    if "best_val_loss" in ckpt:
        slim["best_val_loss"] = float(ckpt["best_val_loss"])
    if "text_proj" in ckpt:
        slim["text_proj"] = _maybe_half(ckpt["text_proj"])

    if not dst:
        # v7_1500m/best.pt -> v7_1500m/best_inference.pt
        dst = str(Path(src).with_name(Path(src).stem + "_inference.pt"))
    dst_path = Path("/ckpts") / dst
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst_path)
    ckpts_vol.commit()

    return {
        "src": src,
        "dst": dst,
        "src_bytes": src_path.stat().st_size,
        "dst_bytes": dst_path.stat().st_size,
        "half": half,
        "step": slim["step"],
        "kept_text_proj": "text_proj" in slim,
    }


@app.local_entrypoint()
def main(src: str = "v7_1500m/best.pt", dst: str = "", half: bool = True):
    r = export_slim.remote(src=src, dst=dst, half=half)
    gb = 1024 ** 3
    print(f"\nexported {r['src']} -> {r['dst']}")
    print(f"  size:  {r['src_bytes']/gb:.2f} GB -> {r['dst_bytes']/gb:.2f} GB "
          f"({r['src_bytes']/max(r['dst_bytes'],1):.1f}x smaller)")
    print(f"  step:  {r['step']}   fp16: {r['half']}   text_proj: {r['kept_text_proj']}")
    print(f"\npull it:\n  modal volume get nano-ckpts /{r['dst']} ./checkpoints/latest.pt")
