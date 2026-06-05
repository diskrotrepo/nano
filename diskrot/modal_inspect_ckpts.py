"""One-off: read every step_*.pt under v2_55m/ on the nano-ckpts volume,
extract training metadata (step, best_val_loss, evals_without_improvement),
return a compact trajectory.

Run:
    modal run diskrot/modal_inspect_ckpts.py
"""
from __future__ import annotations

import modal

app = modal.App("nano-inspect-ckpts")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
)

ckpts_vol = modal.Volume.from_name("nano-ckpts")


@app.function(image=image, volumes={"/ckpts": ckpts_vol}, timeout=600)
def inspect(prefix: str = "v2_55m") -> list[dict]:
    from pathlib import Path

    import torch

    root = Path("/ckpts") / prefix
    files = sorted(p for p in root.glob("*.pt") if p.name.startswith("step_"))
    out: list[dict] = []
    for f in files:
        # weights_only=False so we can read the optim/cfg dicts; we discard
        # the model state_dict immediately to keep memory low.
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        out.append({
            "file": f.name,
            "step": int(ckpt.get("step", -1)),
            "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
            "evals_without_improvement": int(ckpt.get("evals_without_improvement", -1)),
        })
        del ckpt
    # also pull latest.pt + best.pt for reference
    for name in ("latest.pt", "best.pt"):
        p = root / name
        if p.exists():
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            out.append({
                "file": name,
                "step": int(ckpt.get("step", -1)),
                "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
                "evals_without_improvement": int(ckpt.get("evals_without_improvement", -1)),
            })
            del ckpt
    return out


@app.local_entrypoint()
def main(prefix: str = "v2_55m"):
    rows = inspect.remote(prefix=prefix)
    rows.sort(key=lambda r: r["step"])
    print(f"\n{'file':<22} {'step':>7} {'best_val':>9} {'no_improve':>11}")
    print("-" * 55)
    for r in rows:
        print(f"{r['file']:<22} {r['step']:>7} {r['best_val_loss']:>9.4f} {r['evals_without_improvement']:>11}")
