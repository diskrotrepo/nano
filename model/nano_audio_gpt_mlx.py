"""MLX (Apple Silicon) inference backend for NanoAudioGPT.

Mirrors the autoregressive decode loop of model/nano_audio_gpt.py on Apple's MLX
runtime, which is much faster than PyTorch-MPS for single-token decoding (unified
memory, far less dispatch overhead, real int4/int8 quantized matmul). Only the
transformer runs here — the DAC codec and CLAP text encoder stay on PyTorch/MPS
and exchange tensors at the boundaries.

`MLXNanoAudioGPT` duck-types the parts of NanoAudioGPT that server/inference.py
touches: `.cfg`, `.num_params()`, `.param_dtype`, and a `generate(...)` with the
same signature/return semantics (accepts torch tensors, returns a torch
LongTensor after revert_delay). The math is kept faithful to the PyTorch source so
fp32 logits match within tolerance (see tests/test_mlx_parity.py); fp16 and
quantized weights trade exact parity for speed.

Import is lazy from the engine: this module (and `mlx`) are only loaded on
Apple-Silicon macOS.
"""
from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
import torch

from .delay_pattern import revert_delay
from .nano_audio_gpt import GPTConfig

NEG_INF = float("-inf")


def _t2m(t: torch.Tensor, dtype: mx.Dtype) -> mx.array:
    """torch -> mlx, going through numpy. fp16 numpy isn't always friendly, so
    convert in fp32 then cast to the target mlx dtype."""
    arr = mx.array(t.detach().to(torch.float32).cpu().numpy())
    return arr.astype(dtype)


class _RMSNorm(mnn.Module):
    """Weight-only RMSNorm matching torch.nn.RMSNorm with eps=None.

    torch uses eps = finfo(x.dtype).eps when eps is None; for fp32 that's
    ~1.19e-7 (effectively zero). We upcast to fp32 for the reduction regardless
    of weight dtype — standard for numerical stability and keeps fp16 sane."""

    def __init__(self, dim: int, eps: float = 1.1920929e-07):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        xf = x.astype(mx.float32)
        norm = mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (xf * norm).astype(x.dtype) * self.weight


class _SelfAttention(mnn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = mnn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = mnn.Linear(cfg.d_model, cfg.d_model, bias=False)


class _CrossAttention(mnn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = mnn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_proj = mnn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.out_proj = mnn.Linear(cfg.d_model, cfg.d_model, bias=False)


class _MLP(mnn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = mnn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = mnn.Linear(cfg.d_ff, cfg.d_model, bias=False)


class _Block(mnn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = _RMSNorm(cfg.d_model)
        self.attn = _SelfAttention(cfg)
        self.has_cross_attn = cfg.use_text_conditioning
        if self.has_cross_attn:
            self.ln_cross = _RMSNorm(cfg.d_model)
            self.cross_attn = _CrossAttention(cfg)
        self.ln2 = _RMSNorm(cfg.d_model)
        self.mlp = _MLP(cfg)


class _LayerCache:
    """Pre-allocated KV buffer for one self-attention layer. Mirrors
    StaticLayerKVCache: write rotated K/V at the running offset, read [:end]."""

    __slots__ = ("k", "v", "pos")

    def __init__(self, B: int, n_heads: int, T_max: int, head_dim: int, dtype: mx.Dtype):
        self.k = mx.zeros((B, n_heads, T_max, head_dim), dtype=dtype)
        self.v = mx.zeros((B, n_heads, T_max, head_dim), dtype=dtype)
        self.pos = 0


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    # q/k: [B, H, T, D]; cos/sin: [T, D]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class MLXNanoAudioGPT(mnn.Module):
    """MLX mirror of NanoAudioGPT for inference-time generation."""

    def __init__(
        self,
        cfg: GPTConfig,
        state_dict: dict[str, torch.Tensor],
        dtype: mx.Dtype = mx.float16,
        bits: int | None = None,
        group_size: int = 64,
    ):
        super().__init__()
        self.cfg = cfg
        self.param_dtype = dtype
        self._dtype = dtype
        K = cfg.n_codebooks
        self.n_codebooks = K
        self.head_dim = cfg.d_model // cfg.n_heads

        # Module tree mirrors torch param names exactly so weights load 1:1.
        self.tok_embeds = [mnn.Embedding(cfg.vocab_with_pad, cfg.d_model) for _ in range(K)]
        self.blocks = [_Block(cfg) for _ in range(cfg.n_layers)]
        self.ln_final = _RMSNorm(cfg.d_model)
        self.heads = [mnn.Linear(cfg.d_model, cfg.vocab_with_pad, bias=False) for _ in range(K)]

        # RoPE tables (non-persistent in torch — recomputed here, not loaded).
        inv_freq = 1.0 / (cfg.rope_base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        t = np.arange(cfg.max_seq_len)
        freqs = np.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [max_seq_len, head_dim]
        self._cos = mx.array(np.cos(emb)).astype(dtype)
        self._sin = mx.array(np.sin(emb)).astype(dtype)

        self._load_torch_weights(state_dict)
        self._n_params = sum(v.numel() for k, v in state_dict.items())

        if bits in (4, 8):
            self._quantize(bits, group_size)
            self._bits = bits
        else:
            self._bits = 16

        mx.eval(self.parameters())

    # ---- construction helpers -------------------------------------------------

    def _load_torch_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        weights: list[tuple[str, mx.array]] = []
        for name, tensor in state_dict.items():
            if name.startswith("rotary."):
                continue  # non-persistent RoPE buffers; recomputed
            weights.append((name, _t2m(tensor, self._dtype)))
        # strict=True: every module param must be covered and vice versa.
        self.load_weights(weights, strict=True)

    def _quantize(self, bits: int, group_size: int) -> None:
        # Quantize the big Linear weights; keep token embeddings (small, sensitive)
        # in fp16. RMSNorm/Embedding are skipped by the predicate.
        def predicate(path: str, module: mnn.Module):
            if not isinstance(module, mnn.Linear):
                return False
            if path.startswith("tok_embeds"):
                return False
            # group quant needs in_features divisible by group_size
            if module.weight.shape[-1] % group_size != 0:
                return False
            return True

        mnn.quantize(self, group_size=group_size, bits=bits, class_predicate=predicate)

    # ---- duck-typed surface ---------------------------------------------------

    def num_params(self) -> int:
        return int(self._n_params)

    # ---- forward --------------------------------------------------------------

    def _embed(self, tokens: mx.array) -> mx.array:
        # tokens: [B, K, T] int32
        x = self.tok_embeds[0](tokens[:, 0])
        for k in range(1, self.n_codebooks):
            x = x + self.tok_embeds[k](tokens[:, k])
        return x

    def _self_attn(self, attn: _SelfAttention, x: mx.array, cos, sin, cache: _LayerCache | None):
        B, T, D = x.shape
        qkv = attn.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        q, k = _apply_rotary(q, k, cos, sin)

        if cache is not None:
            old_pos = cache.pos
            end = old_pos + T
            cache.k[:, :, old_pos:end, :] = k
            cache.v[:, :, old_pos:end, :] = v
            k = cache.k[:, :, :end, :]
            v = cache.v[:, :, :end, :]
            cache.pos = end
            mask = "causal" if old_pos == 0 else None
        else:
            mask = "causal"

        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return attn.proj(y)

    def _cross_attn(self, ca: _CrossAttention, x: mx.array, cond: mx.array) -> mx.array:
        B, T, D = x.shape
        T_c = cond.shape[1]
        q = ca.q_proj(x).reshape(B, T, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
        kv = ca.kv_proj(cond)
        k, v = mx.split(kv, 2, axis=-1)
        k = k.reshape(B, T_c, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T_c, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=ca.scale, mask=None)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return ca.out_proj(y)

    def __call__(
        self,
        tokens: mx.array,
        caches: list[_LayerCache] | None = None,
        start_pos: int = 0,
        text_emb: mx.array | None = None,
    ) -> mx.array:
        """tokens: [B, K, T] int32. Returns logits [B, K, T, V]."""
        B, K, T = tokens.shape
        x = self._embed(tokens)
        cos = self._cos[start_pos:start_pos + T]
        sin = self._sin[start_pos:start_pos + T]

        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = x + self._self_attn(block.attn, block.ln1(x), cos, sin, cache)
            if block.has_cross_attn and text_emb is not None:
                x = x + self._cross_attn(block.cross_attn, block.ln_cross(x), text_emb)
            x = x + block.mlp.fc2(mnn.gelu(block.mlp.fc1(block.ln2(x))))

        x = self.ln_final(x)
        logits = mx.stack([head(x) for head in self.heads], axis=1)  # [B, K, T, V]
        return logits

    def logits_oneshot(self, tokens: mx.array, text_emb: mx.array | None = None) -> mx.array:
        """Full causal forward with no KV cache — used by parity tests."""
        return self(tokens, caches=None, start_pos=0, text_emb=text_emb)

    def _new_caches(self, B: int, T_max: int) -> list[_LayerCache]:
        return [
            _LayerCache(B, self.cfg.n_heads, T_max, self.head_dim, self._dtype)
            for _ in range(self.cfg.n_layers)
        ]

    # ---- sampling -------------------------------------------------------------

    def _sample_codebook(self, logits: mx.array, temp, top_k, top_p) -> mx.array:
        """logits: [B, V] -> [B] sampled token ids (int32)."""
        if temp == 0:
            return mx.argmax(logits, axis=-1).astype(mx.int32)
        logits = logits / temp

        if top_p is not None and 0.0 < top_p < 1.0:
            order = mx.argsort(-logits, axis=-1)  # descending
            sorted_logits = mx.take_along_axis(logits, order, axis=-1)
            probs = mx.softmax(sorted_logits, axis=-1)
            cum = mx.cumsum(probs, axis=-1)
            cutoff = cum > top_p
            # keep the first token that crosses the threshold (shift right by 1)
            shifted = mx.concatenate(
                [mx.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], axis=-1
            )
            sorted_logits = mx.where(shifted, NEG_INF, sorted_logits)
            inv = mx.argsort(order, axis=-1)  # permutation back to original order
            logits = mx.take_along_axis(sorted_logits, inv, axis=-1)

        if top_k is not None:
            kth = mx.sort(logits, axis=-1)[..., -top_k]  # kth largest threshold
            logits = mx.where(logits < kth[..., None], NEG_INF, logits)

        return mx.random.categorical(logits, axis=-1).astype(mx.int32)

    # ---- generate -------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor | None,
        num_new_frames: int,
        temperature: float | Sequence[float] = 1.0,
        top_k: int | None | Sequence[int | None] = 250,
        top_p: float | None | Sequence[float | None] = None,
        text_emb: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        text_emb_neg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Drop-in for NanoAudioGPT.generate. Accepts/returns torch tensors."""
        cfg = self.cfg
        K = cfg.n_codebooks
        pad = cfg.pad_id

        if prompt is None:
            seed = np.random.randint(0, cfg.vocab_per_codebook, size=(K, 1))
            prompt_m = mx.array(seed.astype(np.int32))[None]  # [1, K, 1]
            squeeze_batch = True
        else:
            squeeze_batch = prompt.dim() == 2
            p = prompt.unsqueeze(0) if squeeze_batch else prompt
            prompt_m = mx.array(p.detach().to(torch.int64).cpu().numpy().astype(np.int32))

        B, _, T_prompt = prompt_m.shape

        def _per_cb(val, kind):
            if val is None or isinstance(val, (int, float)):
                return [val] * K
            val = list(val)
            assert len(val) == K, f"{kind} must be scalar or length {K}, got {len(val)}"
            return val

        temps = _per_cb(temperature, "temperature")
        top_ks = _per_cb(top_k, "top_k")
        top_ps = _per_cb(top_p, "top_p")

        T_total = T_prompt + num_new_frames
        T_delay = T_total + K - 1
        assert T_delay <= cfg.max_seq_len, (
            f"target delayed length {T_delay} exceeds max_seq_len {cfg.max_seq_len}"
        )

        tokens = mx.full((B, K, T_delay), pad, dtype=mx.int32)
        for k in range(K):
            tokens[:, k, k:k + T_prompt] = prompt_m[:, k]

        cond = None if text_emb is None else _t2m(text_emb, self._dtype)
        neg = None if text_emb_neg is None else _t2m(text_emb_neg, self._dtype)
        use_cfg = cfg_scale != 1.0 and (cond is not None or neg is not None)

        caches_cond = self._new_caches(B, T_delay)
        caches_uncond = self._new_caches(B, T_delay) if use_cfg else None

        prefill_len = max(1, T_prompt)
        logits = self(tokens[:, :, :prefill_len], caches=caches_cond, start_pos=0, text_emb=cond)
        if use_cfg:
            logits_base = self(
                tokens[:, :, :prefill_len], caches=caches_uncond, start_pos=0, text_emb=neg
            )
            logits = logits_base + cfg_scale * (logits - logits_base)
        mx.eval(logits, [c.k for c in caches_cond])

        for pos in range(prefill_len, T_delay):
            if pos > prefill_len:
                inp = tokens[:, :, pos - 1:pos]
                logits = self(inp, caches=caches_cond, start_pos=pos - 1, text_emb=cond)
                if use_cfg:
                    logits_base = self(
                        inp, caches=caches_uncond, start_pos=pos - 1, text_emb=neg
                    )
                    logits = logits_base + cfg_scale * (logits - logits_base)

            step = logits[:, :, -1, :]  # [B, K, V]
            step[:, :, pad] = NEG_INF

            for k in range(K):
                if k + T_prompt <= pos < k + T_total:
                    tok = self._sample_codebook(step[:, k, :], temps[k], top_ks[k], top_ps[k])
                    tokens[:, k, pos] = tok
            mx.eval(tokens[:, :, pos])

        mx.eval(tokens)
        delayed = torch.from_numpy(np.array(tokens, copy=False)).to(torch.long)
        out = revert_delay(delayed, T_total)
        return out.squeeze(0) if squeeze_batch else out
