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


class _LyricEncLayer(mnn.Module):
    """One bidirectional encoder layer; param names mirror torch _EncoderLayer."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = _RMSNorm(d)
        self.qkv = mnn.Linear(d, 3 * d, bias=False)
        self.proj = mnn.Linear(d, d, bias=False)
        self.ln2 = _RMSNorm(d)
        self.fc1 = mnn.Linear(d, cfg.lyric_enc_d_ff, bias=False)
        self.fc2 = mnn.Linear(cfg.lyric_enc_d_ff, d, bias=False)


class _LyricEncoder(mnn.Module):
    """MLX mirror of model.lyric_encoder.LyricEncoder (params only; the
    sinusoidal positional table is non-persistent and recomputed on the parent,
    like the RoPE cos/sin tables)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.embed = mnn.Embedding(cfg.phoneme_vocab_size, cfg.d_model)
        self.layers = [_LyricEncLayer(cfg) for _ in range(cfg.lyric_enc_layers)]
        self.ln_final = _RMSNorm(cfg.d_model)


class _MelodyEncoder(mnn.Module):
    """MLX mirror of model.melody_encoder.MelodyEncoder (params only).

    Note MLX ``Conv1d`` is channels-LAST (input [B, T, C], weight
    [out, kernel, in]) whereas torch ``Conv1d`` is channels-first
    ([B, C, T], weight [out, in, kernel]); the conv weights are transposed at
    load time (see ``_load_torch_weights``) and the math stays channels-last
    here, so no transposes are needed in the forward.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.in_proj = mnn.Linear(cfg.melody_n_bins, d, bias=False)
        self.convs = [
            mnn.Conv1d(d, d, kernel_size=3, padding=1, bias=False)
            for _ in range(cfg.melody_enc_layers)
        ]
        self.norms = [_RMSNorm(d) for _ in range(cfg.melody_enc_layers)]
        self.out_proj = mnn.Linear(d, d, bias=False)
        self.ln_final = _RMSNorm(d)
        self.null = mx.zeros((1, 1, d))


class _Block(mnn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = _RMSNorm(cfg.d_model)
        self.attn = _SelfAttention(cfg)
        self.has_cross_attn = cfg.use_text_conditioning
        if self.has_cross_attn:
            self.ln_cross = _RMSNorm(cfg.d_model)
            self.cross_attn = _CrossAttention(cfg)
        self.has_lyric_attn = cfg.use_lyric_conditioning
        if self.has_lyric_attn:
            self.ln_lyric = _RMSNorm(cfg.d_model)
            self.lyric_attn = _CrossAttention(cfg)
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
        self.has_lyric = cfg.use_lyric_conditioning
        if self.has_lyric:
            self.lyric_encoder = _LyricEncoder(cfg)
            self._lyric_heads = cfg.lyric_enc_heads
            self._lyric_head_dim = cfg.d_model // cfg.lyric_enc_heads
            self._lyric_scale = self._lyric_head_dim ** -0.5
        self.has_melody = cfg.use_melody_conditioning
        if self.has_melody:
            self.melody_encoder = _MelodyEncoder(cfg)
        self.blocks = [_Block(cfg) for _ in range(cfg.n_layers)]
        self.ln_final = _RMSNorm(cfg.d_model)
        # Fused output head (mirrors torch's single `head` Linear; the forward's
        # reshape+transpose recovers [B, K, T, V]).
        self.head = mnn.Linear(cfg.d_model, K * cfg.vocab_with_pad, bias=False)

        # RoPE tables (non-persistent in torch — recomputed here, not loaded).
        inv_freq = 1.0 / (cfg.rope_base ** (np.arange(0, self.head_dim, 2) / self.head_dim))
        t = np.arange(cfg.max_seq_len)
        freqs = np.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)  # [max_seq_len, head_dim]
        self._cos = mx.array(np.cos(emb)).astype(dtype)
        self._sin = mx.array(np.sin(emb)).astype(dtype)

        # Lyric sinusoidal positional table (non-persistent in torch — recomputed,
        # stored as an underscore attr so it stays out of the loaded param tree).
        if self.has_lyric:
            d = cfg.d_model
            pe = np.zeros((cfg.max_lyric_len, d))
            pos = np.arange(cfg.max_lyric_len)[:, None]
            div = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
            pe[:, 0::2] = np.sin(pos * div)
            pe[:, 1::2] = np.cos(pos * div)
            self._lyric_pe = mx.array(pe).astype(dtype)

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
            # MLX Conv1d is channels-last: weight [out, kernel, in] vs torch's
            # [out, in, kernel]. Transpose the melody conv weights so they load
            # 1:1 into mnn.Conv1d (the forward then stays channels-last).
            if (
                "melody_encoder.convs." in name
                and name.endswith(".weight")
                and tensor.dim() == 3
            ):
                tensor = tensor.permute(0, 2, 1).contiguous()
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
            # Keep the lyric encoder full-precision: it's small and phonetically
            # sensitive — quantizing it risks intelligibility for little memory.
            if path.startswith("lyric_encoder"):
                return False
            # Keep the melody encoder full-precision too (small + sensitive).
            if path.startswith("melody_encoder"):
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

    def _cross_attn(
        self, ca: _CrossAttention, x: mx.array, cond: mx.array, mask: mx.array | None = None,
    ) -> mx.array:
        B, T, D = x.shape
        T_c = cond.shape[1]
        q = ca.q_proj(x).reshape(B, T, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
        kv = ca.kv_proj(cond)
        k, v = mx.split(kv, 2, axis=-1)
        k = k.reshape(B, T_c, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T_c, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=ca.scale, mask=mask)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
        return ca.out_proj(y)

    def _encode_lyrics(self, ids: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
        """ids [B, L] int32, mask [B, L] bool. Returns (lyric_emb [B, L, D],
        additive cross-attn mask [B, 1, 1, L]). Mirrors torch encode_lyrics."""
        enc = self.lyric_encoder
        B, L = ids.shape
        D = self.cfg.d_model
        nh, hd = self._lyric_heads, self._lyric_head_dim
        x = enc.embed(ids) + self._lyric_pe[:L][None]
        add = mx.where(mask[:, None, None, :], mx.array(0.0), mx.array(NEG_INF)).astype(x.dtype)
        for layer in enc.layers:
            h = layer.ln1(x)
            q, k, v = mx.split(layer.qkv(h), 3, axis=-1)
            q = q.reshape(B, L, nh, hd).transpose(0, 2, 1, 3)
            k = k.reshape(B, L, nh, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, L, nh, hd).transpose(0, 2, 1, 3)
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self._lyric_scale, mask=add)
            y = y.transpose(0, 2, 1, 3).reshape(B, L, D)
            x = x + layer.proj(y)
            x = x + layer.fc2(mnn.gelu(layer.fc1(layer.ln2(x))))
        x = enc.ln_final(x)
        return x, add

    def _encode_melody(self, chroma: mx.array) -> mx.array:
        """chroma [B, T, n_bins] -> [B, T, D]. Channels-last throughout (MLX
        Conv1d convention); mirrors torch MelodyEncoder.encode_melody."""
        enc = self.melody_encoder
        h = mnn.gelu(enc.in_proj(chroma))  # [B, T, D]
        for conv, norm in zip(enc.convs, enc.norms):
            h = norm(h + conv(h))
        return enc.ln_final(enc.out_proj(h))

    def _encode_melody_delayed(
        self, chroma: mx.array, seq_len: int, offset: int,
    ) -> mx.array:
        """Encode chroma into a full [B, seq_len, D] term: the encoded melody at
        new-frame positions [offset, offset+Tc), the learned null elsewhere.
        Mirrors torch encode_melody_delayed."""
        B = chroma.shape[0]
        D = self.cfg.d_model
        enc = self._encode_melody(chroma)  # [B, Tc, D]
        Tc = enc.shape[1]
        full = mx.tile(self.melody_encoder.null.reshape(1, 1, D), (B, seq_len, 1))
        full = full.astype(enc.dtype)
        end = min(offset + Tc, seq_len)
        if end > offset:
            full[:, offset:end, :] = enc[:, :end - offset, :]
        return full

    def __call__(
        self,
        tokens: mx.array,
        caches: list[_LayerCache] | None = None,
        start_pos: int = 0,
        text_emb: mx.array | None = None,
        lyric_emb: mx.array | None = None,
        lyric_kv_mask: mx.array | None = None,
        melody_emb: mx.array | None = None,
    ) -> mx.array:
        """tokens: [B, K, T] int32. Returns logits [B, K, T, V]."""
        B, K, T = tokens.shape
        x = self._embed(tokens)
        # Time-aligned melody add at the cb0 anchor (slice mirrors the RoPE slice).
        if self.has_melody:
            if melody_emb is not None:
                x = x + melody_emb[:, start_pos:start_pos + T, :]
            else:
                x = x + self.melody_encoder.null  # learned null (dropped/uncond)
        cos = self._cos[start_pos:start_pos + T]
        sin = self._sin[start_pos:start_pos + T]

        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x = x + self._self_attn(block.attn, block.ln1(x), cos, sin, cache)
            if block.has_cross_attn and text_emb is not None:
                x = x + self._cross_attn(block.cross_attn, block.ln_cross(x), text_emb)
            if block.has_lyric_attn and lyric_emb is not None:
                x = x + self._cross_attn(
                    block.lyric_attn, block.ln_lyric(x), lyric_emb, mask=lyric_kv_mask
                )
            x = x + block.mlp.fc2(mnn.gelu(block.mlp.fc1(block.ln2(x))))

        x = self.ln_final(x)
        logits = (
            self.head(x)
            .reshape(B, T, self.n_codebooks, -1)
            .transpose(0, 2, 1, 3)  # [B, K, T, V]
        )
        return logits

    def logits_oneshot(
        self,
        tokens: mx.array,
        text_emb: mx.array | None = None,
        lyric_emb: mx.array | None = None,
        lyric_kv_mask: mx.array | None = None,
        melody_emb: mx.array | None = None,
    ) -> mx.array:
        """Full causal forward with no KV cache — used by parity tests."""
        return self(
            tokens, caches=None, start_pos=0, text_emb=text_emb,
            lyric_emb=lyric_emb, lyric_kv_mask=lyric_kv_mask, melody_emb=melody_emb,
        )

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
        lyric_ids: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        lyric_ids_neg: torch.Tensor | None = None,
        lyric_mask_neg: torch.Tensor | None = None,
        lyric_cfg_scale: float | None = None,
        melody: torch.Tensor | None = None,
        melody_cfg_scale: float | None = None,
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

        def _ids_to_mx(ids):
            return mx.array(ids.detach().to(torch.int64).cpu().numpy().astype(np.int32))

        # Encode lyric streams once (encoder is large; reuse across decode steps).
        lyric_pos = lyric_kv_pos = None
        lyric_neg = lyric_kv_neg = None
        if self.has_lyric and lyric_ids is not None:
            lyric_pos, lyric_kv_pos = self._encode_lyrics(
                _ids_to_mx(lyric_ids), _ids_to_mx(lyric_mask).astype(mx.bool_)
            )
        if self.has_lyric and lyric_ids_neg is not None:
            lyric_neg, lyric_kv_neg = self._encode_lyrics(
                _ids_to_mx(lyric_ids_neg), _ids_to_mx(lyric_mask_neg).astype(mx.bool_)
            )

        # Encode the (fully-known) melody once into a full delayed-length term;
        # the CFG baseline is the learned null. Mirrors torch generate.
        mel_pos = mel_neg = None
        has_melody = self.has_melody and melody is not None
        if self.has_melody:
            D = cfg.d_model
            mel_neg = mx.tile(self.melody_encoder.null.reshape(1, 1, D), (B, T_delay, 1)).astype(self._dtype)
            mel_pos = (
                self._encode_melody_delayed(_t2m(melody, self._dtype), T_delay, T_prompt)
                if melody is not None else mel_neg
            )

        has_cond = any(v is not None for v in (cond, neg, lyric_pos, lyric_neg)) or has_melody
        composed_lyric = (
            lyric_cfg_scale is not None and lyric_cfg_scale != 1.0 and lyric_pos is not None
        )
        composed_melody = (
            melody_cfg_scale is not None and melody_cfg_scale != 1.0 and has_melody
        )
        use_cfg = (cfg_scale != 1.0 and has_cond) or composed_lyric or composed_melody

        # Ordered guidance stages from baseline → full conditioning (mirror the
        # torch generate): logits = stages[0] + Σ scales[i]*(stages[i+1]-stages[i]).
        # Nesting order tags → lyrics → melody; axes without their own scale fold
        # into the cfg (tags) step.
        full = {"text": cond, "lemb": lyric_pos, "lkv": lyric_kv_pos, "mel": mel_pos}
        if not use_cfg:
            stages = [full]
            scales: list[float] = []
        else:
            off = {"text": neg, "lemb": lyric_neg, "lkv": lyric_kv_neg, "mel": mel_neg}
            tags_on = dict(full)
            if composed_lyric:
                tags_on["lemb"], tags_on["lkv"] = lyric_neg, lyric_kv_neg
            if composed_melody:
                tags_on["mel"] = mel_neg
            stages = [off, tags_on]
            scales = [cfg_scale]
            if composed_lyric:
                lyr_on = dict(stages[-1])
                lyr_on["lemb"], lyr_on["lkv"] = lyric_pos, lyric_kv_pos
                stages.append(lyr_on)
                scales.append(lyric_cfg_scale)
            if composed_melody:
                mel_on = dict(stages[-1])
                mel_on["mel"] = mel_pos
                stages.append(mel_on)
                scales.append(melody_cfg_scale)
        for s in stages:
            s["caches"] = self._new_caches(B, T_delay)

        def _run(inp, start):
            for s in stages:
                s["logits"] = self(
                    inp, caches=s["caches"], start_pos=start,
                    text_emb=s["text"], lyric_emb=s["lemb"], lyric_kv_mask=s["lkv"],
                    melody_emb=s["mel"],
                )

        def _combine():
            out = stages[0]["logits"]
            for i, sc in enumerate(scales):
                out = out + sc * (stages[i + 1]["logits"] - stages[i]["logits"])
            return out

        prefill_len = max(1, T_prompt)
        _run(tokens[:, :, :prefill_len], 0)
        logits = _combine()
        mx.eval(logits, [c.k for c in stages[0]["caches"]])

        for pos in range(prefill_len, T_delay):
            if pos > prefill_len:
                inp = tokens[:, :, pos - 1:pos]
                _run(inp, pos - 1)
                logits = _combine()

            step = logits[:, :, -1, :]  # [B, K, V]
            # Mask all control ids (pad, plus FIM <SUF>/<MID> when use_fim) so the
            # generated output only ever contains real DAC tokens. Mirrors the
            # torch path's `step_logits[..., vocab_per_codebook:] = -inf`.
            step[:, :, cfg.vocab_per_codebook:] = NEG_INF

            for k in range(K):
                if k + T_prompt <= pos < k + T_total:
                    tok = self._sample_codebook(step[:, k, :], temps[k], top_ks[k], top_ps[k])
                    tokens[:, k, pos] = tok
            mx.eval(tokens[:, :, pos])

        mx.eval(tokens)
        delayed = torch.from_numpy(np.array(tokens, copy=False)).to(torch.long)
        out = revert_delay(delayed, T_total)
        return out.squeeze(0) if squeeze_batch else out
