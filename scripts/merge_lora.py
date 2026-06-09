"""Merge a LoRA fine-tune checkpoint into a plain checkpoint.

A checkpoint written by a `lora_rank>0` run stores the frozen base weights plus
the trainable `lora_A`/`lora_B` adapters and a `"lora"` metadata key. The server
already merges these on load, but for distribution / archival it's handy to bake
the deltas into ordinary `.weight` tensors once so the result is byte-identical
to a normally-trained checkpoint (and a hair smaller — the adapters are dropped).

Usage:
    python -m scripts.merge_lora IN.pt OUT.pt

A checkpoint with no `"lora"` key (a from-scratch or full fine-tune run) is
already plain; the script says so and copies it through unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from model.lora import merge_lora_state_dict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path, help="input checkpoint (.pt)")
    ap.add_argument("dst", type=Path, help="output checkpoint (.pt)")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    state = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state.items()
    }

    if "lora" not in ckpt:
        print(f"{args.src} has no LoRA adapters — copying through unchanged.")
        ckpt["model"] = state
    else:
        meta = ckpt["lora"]
        n_before = len(state)
        merged = merge_lora_state_dict(state, meta)
        ckpt["model"] = merged
        # Optimizer state references the LoRA param tensors that no longer exist
        # in the merged model; drop it so the output is a clean inference ckpt.
        ckpt.pop("optim", None)
        del ckpt["lora"]
        print(f"merged LoRA rank={meta['rank']} alpha={meta['alpha']} "
              f"({n_before} → {len(merged)} tensors); dropped optimizer state.")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.dst)
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
