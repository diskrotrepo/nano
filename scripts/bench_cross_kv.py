"""Microbenchmark: cross-attention K/V cache vs recompute (decode latency).

Proves the win from caching projected cross-attn K/V instead of recomputing
``kv_proj(cond)`` every decode step. The speedup is governed by the conditioning
length (lyrics up to 512 phonemes) relative to the per-token work, so it reproduces
on a SMALL model with a LONG lyric stream — a local (MPS/CPU) proxy for the H100
production shape. Also re-asserts byte-identity (temp=0) so a fast-but-wrong cache
can't pass.

Usage:
    python -m scripts.bench_cross_kv                      # local proxy defaults
    python -m scripts.bench_cross_kv --d-model 2048 --n-layers 22 --lyric-len 512 \
        --frames 600 --reps 3                             # ~production shape (needs a big GPU)
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def _build(args, dev: torch.device, dtype: torch.dtype) -> NanoAudioGPT:
    cfg = GPTConfig(
        n_codebooks=args.codebooks, vocab_per_codebook=args.vocab,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        d_ff=args.d_model * 4, max_seq_len=args.frames + 64, dropout=0.0,
        use_text_conditioning=True, use_lyric_conditioning=True,
        use_qk_norm=True, use_lyric_qk_norm=True,
        lyric_enc_layers=2, lyric_enc_heads=args.n_heads,
        lyric_enc_d_ff=args.d_model * 2, max_lyric_len=args.lyric_len + 8,
    )
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg)
    with torch.no_grad():  # make the cross-attn paths live
        for block in model.blocks:
            block.cross_attn.out_proj.weight.normal_(std=0.02)
            block.lyric_attn.out_proj.weight.normal_(std=0.02)
    return model.to(dev, dtype).eval()


def _time_gen(model, kw, dev, frames, reps):
    times = []
    for _ in range(reps):
        _sync(dev)
        t0 = time.perf_counter()
        out = model.generate(**kw)
        _sync(dev)
        times.append((time.perf_counter() - t0) / frames * 1e3)  # ms/frame
    return min(times), statistics.median(times), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=16)
    ap.add_argument("--codebooks", type=int, default=9)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--lyric-len", type=int, default=512)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    dev = _device()
    dtype = torch.bfloat16 if dev.type in ("cuda", "mps") else torch.float32
    model = _build(args, dev, dtype)
    B, K = args.batch, args.codebooks
    n_params = sum(p.numel() for p in model.parameters())

    prompt = torch.randint(0, args.vocab, (B, K, 8), device=dev)
    text_emb = torch.randn(B, 6, args.d_model, device=dev, dtype=dtype)
    text_neg = torch.randn(B, 6, args.d_model, device=dev, dtype=dtype)
    lyr = torch.randint(4, model.cfg.phoneme_vocab_size, (B, args.lyric_len), device=dev)
    lyr_m = torch.ones(B, args.lyric_len, dtype=torch.bool, device=dev)

    base_kw = dict(
        prompt=prompt, num_new_frames=args.frames, temperature=0.0,
        top_k=None, top_p=None, text_emb=text_emb, text_emb_neg=text_neg,
        lyric_ids=lyr, lyric_mask=lyr_m, cfg_scale=args.cfg_scale,
    )

    print(f"device={dev.type} dtype={dtype} params={n_params/1e6:.0f}M "
          f"d_model={args.d_model} layers={args.n_layers} lyric_len={args.lyric_len} "
          f"frames={args.frames} cfg_scale={args.cfg_scale} batch={B}")

    # warmup (triggers any lazy MPS/cuda kernel init for both paths)
    model._use_cross_kv_cache = True
    model.generate(**{**base_kw, "num_new_frames": 8})
    model._use_cross_kv_cache = False
    model.generate(**{**base_kw, "num_new_frames": 8})

    model._use_cross_kv_cache = False
    off_min, off_med, off_out = _time_gen(model, base_kw, dev, args.frames, args.reps)
    model._use_cross_kv_cache = True
    on_min, on_med, on_out = _time_gen(model, base_kw, dev, args.frames, args.reps)

    identical = torch.equal(on_out, off_out)
    print(f"  recompute (cache OFF): {off_med:.3f} ms/frame  (best {off_min:.3f})")
    print(f"  cached    (cache ON ): {on_med:.3f} ms/frame  (best {on_min:.3f})")
    print(f"  SPEEDUP (median): {off_med / on_med:.2f}x   (best-case {off_min / on_min:.2f}x)")
    print(f"  byte-identical (temp=0): {identical}")
    if not identical:
        raise SystemExit("FAIL: cached output diverged from recompute")


if __name__ == "__main__":
    main()
