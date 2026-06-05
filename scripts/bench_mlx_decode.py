"""Benchmark the autoregressive decode loop: PyTorch-MPS vs MLX (fp16/int8/int4).

Builds a model at the real default shape (random weights — only timing matters)
and measures wall-clock + frames/sec for a from-scratch text-conditioned generation
at a few durations. Use this to quantify the MLX speedup on your own machine.

    python scripts/bench_mlx_decode.py                 # default shape, 10s + 30s
    python scripts/bench_mlx_decode.py --seconds 5 10 30
    python scripts/bench_mlx_decode.py --backends mlx-int4 torch-mps
    python scripts/bench_mlx_decode.py --n-layers 6    # smaller, faster to bench

The torch-mps baseline is what NANO_MLX=0 forces in the server; the mlx-* rows are
the new path. cfg_scale=3.0 + text conditioning mirror the default /generate call.
"""
from __future__ import annotations

import argparse
import time

import torch

from model.codec import DACodec
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT

FRAME_RATE_HZ = DACodec.FRAME_RATE_HZ  # 86


def _build_cfg(args) -> GPTConfig:
    return GPTConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=0.0,
        max_seq_len=args.max_seq_len,
        use_text_conditioning=True,
    )


def _time_generate(gen_fn, prompt, frames, text_emb, cfg_scale) -> float:
    t0 = time.perf_counter()
    out = gen_fn(prompt, num_new_frames=frames, temperature=0.9, top_k=50, top_p=0.95,
                 text_emb=text_emb, cfg_scale=cfg_scale)
    if hasattr(out, "cpu"):
        out.cpu()  # force materialization
    return time.perf_counter() - t0


def _torch_mps_model(cfg, state):
    m = NanoAudioGPT(cfg)
    m.load_state_dict(state)
    m = m.to("mps").half().eval()
    return m


def _mlx_model(cfg, state, bits):
    import mlx.core as mx

    from model.nano_audio_gpt_mlx import MLXNanoAudioGPT

    return MLXNanoAudioGPT(cfg, state, dtype=mx.float16, bits=bits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, nargs="+", default=[10, 30])
    ap.add_argument("--backends", nargs="+",
                    default=["torch-mps", "mlx-fp16", "mlx-int8", "mlx-int4"])
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--n-layers", type=int, default=22)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    args = ap.parse_args()

    cfg = _build_cfg(args)
    print(f"shape: d_model={cfg.d_model} n_layers={cfg.n_layers} n_heads={cfg.n_heads} "
          f"d_ff={cfg.d_ff}  cfg_scale={args.cfg_scale}")

    torch.manual_seed(0)
    base = NanoAudioGPT(cfg).eval()
    state = base.state_dict()
    print(f"params: {base.num_params()/1e6:.1f}M")
    del base

    text_emb = torch.randn(1, 2, cfg.d_model)

    for backend in args.backends:
        try:
            if backend == "torch-mps":
                if not torch.backends.mps.is_available():
                    print(f"\n[{backend}] skipped — MPS unavailable")
                    continue
                model = _torch_mps_model(cfg, state)
                te = text_emb.to("mps").half()
                gen = model.generate
            elif backend.startswith("mlx-"):
                bits = {"mlx-fp16": None, "mlx-int8": 8, "mlx-int4": 4}[backend]
                model = _mlx_model(cfg, state, bits)
                te = text_emb
                gen = model.generate
            else:
                print(f"\n[{backend}] unknown backend")
                continue
        except Exception as e:  # noqa: BLE001
            print(f"\n[{backend}] build failed: {e}")
            continue

        # warmup (compile/caches/Metal kernels)
        _time_generate(gen, None, 16, te, args.cfg_scale)

        print(f"\n[{backend}]")
        for sec in args.seconds:
            frames = int(sec * FRAME_RATE_HZ)
            dt = _time_generate(gen, None, frames, te, args.cfg_scale)
            print(f"  {sec:>4.0f}s ({frames:>4d} frames): {dt:7.2f}s  "
                  f"{frames/dt:6.1f} frames/s  ({dt/sec:5.2f}x realtime)")

        del model


if __name__ == "__main__":
    main()
