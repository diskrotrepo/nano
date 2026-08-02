"""MLX (Apple Silicon) inference backend for NanoAudioGPT.

Mirrors the autoregressive decode loop of model/nano_audio_gpt.py on Apple's MLX
runtime, which is much faster than PyTorch-MPS for single-token decoding (unified
memory, far less dispatch overhead, real int4/int8 quantized matmul). Only the
transformer runs here — the DAC codec and CLAP text encoder stay on PyTorch/MPS
and exchange tensors at the boundaries.

`MLXNanoAudioGPT` duck-types the parts of NanoAudioGPT that server/inference.py
touches: `.cfg`, `.num_params()`, `.param_dtype`, `_resolve_prompt`,
`_generate_stream` (streaming + batched gen), and a `generate(...)` with the same
signature/return semantics (accepts torch tensors, returns a torch LongTensor of
un-delayed frames). The math is kept faithful to the PyTorch source so fp32 logits
match within tolerance (see tests/test_mlx_parity.py); fp16 and quantized weights
trade exact parity for speed.

Import is lazy from the engine: this module (and `mlx`) are only loaded on
Apple-Silicon macOS.
"""
from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import mlx.nn as mnn
import numpy as np
import torch

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
        # Fused kernel; accumulates in fp32 internally — the fused equivalent of
        # the explicit-upcast reduction this used to spell out op by op.
        return mx.fast.rms_norm(x, self.weight, self.eps)


class _SelfAttention(mnn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = mnn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = mnn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # Per-head-dim gains shared across heads — mirrors torch (see
        # GPTConfig.use_qk_norm).
        self.q_norm = _RMSNorm(self.head_dim) if cfg.use_qk_norm else None
        self.k_norm = _RMSNorm(self.head_dim) if cfg.use_qk_norm else None


class _CrossAttention(mnn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = mnn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_proj = mnn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.out_proj = mnn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = _RMSNorm(self.head_dim) if cfg.use_qk_norm else None
        self.k_norm = _RMSNorm(self.head_dim) if cfg.use_qk_norm else None


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
        # Optional QK-norm on the encoder's own attention (GPTConfig.use_lyric_qk_norm,
        # default off but ON in the trained v8 checkpoint). RMSNorm over head_dim.
        head_dim = d // cfg.lyric_enc_heads
        self.q_norm = _RMSNorm(head_dim) if cfg.use_lyric_qk_norm else None
        self.k_norm = _RMSNorm(head_dim) if cfg.use_lyric_qk_norm else None
        self.ln2 = _RMSNorm(d)
        self.fc1 = mnn.Linear(d, cfg.lyric_enc_d_ff, bias=False)
        self.fc2 = mnn.Linear(cfg.lyric_enc_d_ff, d, bias=False)


class _LyricEncoder(mnn.Module):
    """MLX mirror of model.lyric_encoder.LyricEncoder (params only; the
    sinusoidal positional table is non-persistent and recomputed on the
    parent)."""

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


class _CrossKV:
    """Cached projected (+ k-normed) K/V for one cross-attention over a FIXED
    conditioning sequence. Mirrors the torch CrossKVCache: `cond` never changes
    across decode steps, so kv_proj(cond) runs once at prefill and is reused —
    it was the dominant per-token cost with a long lyric stream."""

    __slots__ = ("k", "v")

    def __init__(self):
        self.k: mx.array | None = None
        self.v: mx.array | None = None


class _BlockCrossKV:
    """The two cross-attention caches for one decoder block (tag + lyric)."""

    __slots__ = ("text", "lyric")

    def __init__(self):
        self.text = _CrossKV()
        self.lyric = _CrossKV()


class MLXNanoAudioGPT(mnn.Module):
    """MLX mirror of NanoAudioGPT for inference-time generation."""

    def __init__(
        self,
        cfg: GPTConfig,
        state_dict: dict[str, torch.Tensor],
        dtype: mx.Dtype = mx.float16,
        bits: int | None = None,
        group_size: int = 64,
        fp32_head: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.param_dtype = dtype
        self._dtype = dtype
        # Keep the output head in fp32 (weight + matmul) regardless of the decoder
        # dtype/quant. The head projects straight into the sampling logits with no
        # downstream renorm, so bf16/int8 error there perturbs the sampled tokens
        # and the DAC rollout compounds it into noise (measured on the v10 DAC
        # shape: quantizing the head cost ~4% argmax agreement vs torch fp32).
        self._fp32_head = fp32_head
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

        # RoPE via the fused mx.fast.rope kernel (non-traditional = the same
        # split-half rotation as torch's rotate_half with concatenated freqs);
        # angles are computed in the kernel, so no cos/sin tables are stored.
        self._rope_base = float(cfg.rope_base)

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

        # Force the head to fp32 (weight kept full-precision; the forward casts its
        # input to fp32 too). Done after quant so the predicate has already left it
        # unquantized.
        if self._fp32_head:
            self.head.weight = self.head.weight.astype(mx.float32)

        mx.eval(self.parameters())
        self._prime_kernels()

    # ---- construction helpers -------------------------------------------------

    def _prime_kernels(self) -> None:
        """Run one tiny generation NOW, on the construction (main) thread, so every
        MLX GPU op is initialized here before the first request.

        MLX lazily initializes each *distinct* GPU operation the first time it
        runs, and that first init must happen on the thread that owns the default
        GPU stream (the main / construction thread). The server drives generation
        from ephemeral per-request producer threads (see server/inference.py
        `_stream_mp3`), so any op that was *not* already initialized on the main
        thread raises ``RuntimeError: There is no Stream(gpu, 0) in current
        thread`` the first time a worker thread hits it. The decoder path is
        covered by the server's warmup gen, but the lyric/melody encoders are not
        (warmup passes no lyrics/melody) — so a lyrics or /cover request was the
        first place those kernels ran, on a worker thread, and crashed. Exercising
        every conditioning path here (decoder + 3-stage CFG combine + lyric encoder
        + melody encoder + the sampling kernels) primes them all. Init is
        shape-agnostic, so a tiny fixed-shape pass covers real requests of any
        length. Best-effort: a hiccup here only forfeits priming, it must never
        brick model construction."""
        try:
            cfg = self.cfg
            frames = max(cfg.n_codebooks + 2, 32)  # run the decode loop + one emit
            text = torch.zeros(1, 1, cfg.d_model)
            kw: dict = dict(
                num_new_frames=frames, temperature=0.9, top_k=5, top_p=0.95,
                text_emb=text, text_emb_neg=text, cfg_scale=2.0,
            )
            if self.has_lyric:
                ids = torch.ones(1, 4, dtype=torch.long)
                msk = torch.ones(1, 4, dtype=torch.bool)
                kw.update(lyric_ids=ids, lyric_mask=msk, lyric_ids_neg=ids,
                          lyric_mask_neg=msk, lyric_cfg_scale=2.0)
            if self.has_melody:
                kw.update(melody=torch.zeros(1, frames, 12), melody_cfg_scale=2.0)
            self.generate(prompt=None, **kw)
        except Exception as e:  # noqa: BLE001 — priming is best-effort
            print(f"[mlx] kernel prime skipped: {type(e).__name__}: {e}", flush=True)

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
            # Keep the output head full-precision — it sets the sampling logits
            # directly (no downstream renorm), so quantizing it is what tips the
            # DAC rollout into noise. See self._fp32_head.
            if self._fp32_head and path == "head":
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

    def _self_attn(self, attn: _SelfAttention, x: mx.array, start_pos: int, cache: _LayerCache | None):
        B, T, D = x.shape
        qkv = attn.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        # QK-norm before RoPE (and before the cache write), matching torch.
        if attn.q_norm is not None:
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        q = mx.fast.rope(q, attn.head_dim, traditional=False,
                         base=self._rope_base, scale=1.0, offset=start_pos)
        k = mx.fast.rope(k, attn.head_dim, traditional=False,
                         base=self._rope_base, scale=1.0, offset=start_pos)

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
        self, ca: _CrossAttention, x: mx.array, cond: mx.array,
        mask: mx.array | None = None, cache: _CrossKV | None = None,
    ) -> mx.array:
        B, T, D = x.shape
        q = ca.q_proj(x).reshape(B, T, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
        if ca.q_norm is not None:
            q = ca.q_norm(q)
        if cache is not None and cache.k is not None:
            k, v = cache.k, cache.v
        else:
            T_c = cond.shape[1]
            kv = ca.kv_proj(cond)
            k, v = mx.split(kv, 2, axis=-1)
            k = k.reshape(B, T_c, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, T_c, ca.n_heads, ca.head_dim).transpose(0, 2, 1, 3)
            # k_norm is position-independent, so caching the post-norm K is exactly
            # the value a recompute would produce (mirrors the torch cache).
            if ca.k_norm is not None:
                k = ca.k_norm(k)
            if cache is not None:
                cache.k, cache.v = k, v
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
            if layer.q_norm is not None:
                q = layer.q_norm(q)
                k = layer.k_norm(k)
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
        text_kv_mask: mx.array | None = None,
        lyric_emb: mx.array | None = None,
        lyric_kv_mask: mx.array | None = None,
        melody_emb: mx.array | None = None,
        cross_caches: list[_BlockCrossKV] | None = None,
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

        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            ckv = cross_caches[i] if cross_caches is not None else None
            x = x + self._self_attn(block.attn, block.ln1(x), start_pos, cache)
            if block.has_cross_attn and text_emb is not None:
                x = x + self._cross_attn(
                    block.cross_attn, block.ln_cross(x), text_emb, mask=text_kv_mask,
                    cache=ckv.text if ckv is not None else None,
                )
            if block.has_lyric_attn and lyric_emb is not None:
                x = x + self._cross_attn(
                    block.lyric_attn, block.ln_lyric(x), lyric_emb, mask=lyric_kv_mask,
                    cache=ckv.lyric if ckv is not None else None,
                )
            x = x + block.mlp.fc2(mnn.gelu(block.mlp.fc1(block.ln2(x))))

        x = self.ln_final(x)
        # Head in fp32 (weight is fp32 when _fp32_head) so the logits that feed
        # sampling are computed at full precision, not bf16.
        hx = x.astype(mx.float32) if self._fp32_head else x
        logits = (
            self.head(hx)
            .reshape(B, T, self.n_codebooks, -1)
            .transpose(0, 2, 1, 3)  # [B, K, T, V]
        )
        return logits

    def logits_oneshot(
        self,
        tokens: mx.array,
        text_emb: mx.array | None = None,
        text_kv_mask: mx.array | None = None,
        lyric_emb: mx.array | None = None,
        lyric_kv_mask: mx.array | None = None,
        melody_emb: mx.array | None = None,
    ) -> mx.array:
        """Full causal forward with no KV cache — used by parity tests."""
        return self(
            tokens, caches=None, start_pos=0, text_emb=text_emb, text_kv_mask=text_kv_mask,
            lyric_emb=lyric_emb, lyric_kv_mask=lyric_kv_mask, melody_emb=melody_emb,
        )

    def _new_caches(self, B: int, T_max: int) -> list[_LayerCache]:
        return [
            _LayerCache(B, self.cfg.n_heads, T_max, self.head_dim, self._dtype)
            for _ in range(self.cfg.n_layers)
        ]

    # ---- sampling -------------------------------------------------------------

    def _sample_codebooks(self, logits: mx.array, temps, top_ks, top_ps) -> mx.array:
        """logits: [B, Kv, V] for a contiguous run of codebooks -> [B, Kv] sampled
        ids (int32). temps/top_ks/top_ps are the matching length-Kv slices of the
        per-codebook sampling params.

        One batched kernel sequence instead of a Python loop over codebooks —
        at v9's K=24 the old per-codebook loop was ~150 tiny kernel launches per
        decode step. Filtering semantics are identical per row: temp==0 -> argmax
        of the raw logits; else temp-scale, nucleus (top_p) filter, then top_k
        threshold on the filtered logits. Rows without a top_p/top_k pass through
        (p=2.0 never crosses; k=V keeps everything)."""
        B, Kv, V = logits.shape
        greedy = mx.argmax(logits, axis=-1).astype(mx.int32)
        if all(t == 0 for t in temps):
            return greedy

        safe_t = mx.array([float(t) if t else 1.0 for t in temps]).astype(logits.dtype)
        x = logits / safe_t[None, :, None]

        # Sort descending once; both filters work in the sorted domain, then one
        # inverse permutation restores original order.
        order = mx.argsort(-x, axis=-1)
        xs = mx.take_along_axis(x, order, axis=-1)

        # top_p (nucleus): keep the first token that crosses the threshold
        # (shift right by 1). cum > p is monotone, so the masked set is a suffix
        # and xs stays descending-sorted afterwards.
        p_arr = mx.array(
            [p if (p is not None and 0.0 < p < 1.0) else 2.0 for p in top_ps]
        ).astype(x.dtype)
        probs = mx.softmax(xs, axis=-1)
        cum = mx.cumsum(probs, axis=-1)
        cutoff = cum > p_arr[None, :, None]
        shifted = mx.concatenate(
            [mx.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], axis=-1
        )
        xs = mx.where(shifted, NEG_INF, xs)

        # top_k: threshold = kth largest of the (top_p-filtered) row — index k-1
        # in the descending sort. Same value-comparison (ties kept) as the old
        # mx.sort(...)[..., -top_k] form.
        k_idx = mx.array(
            [min(int(k), V) - 1 if k is not None else V - 1 for k in top_ks],
            dtype=mx.int32,
        ).reshape(1, Kv, 1)
        kth = mx.take_along_axis(xs, mx.broadcast_to(k_idx, (B, Kv, 1)), axis=-1)
        xs = mx.where(xs < kth, NEG_INF, xs)

        inv = mx.argsort(order, axis=-1)  # permutation back to original order
        x = mx.take_along_axis(xs, inv, axis=-1)
        sampled = mx.random.categorical(x, axis=-1).astype(mx.int32)
        if any(t == 0 for t in temps):
            tz = mx.array([t == 0 for t in temps])
            sampled = mx.where(tz[None, :], greedy, sampled)
        return sampled

    # ---- generate -------------------------------------------------------------

    def _resolve_prompt(
        self, prompt: torch.Tensor | None, batch_size: int = 1
    ) -> tuple[torch.Tensor, bool]:
        """torch-tensor counterpart of NanoAudioGPT._resolve_prompt — the server
        calls this and feeds the result back into generate / _generate_stream
        (which convert torch->mx internally). prompt=None -> a random DAC seed
        column ([K,1], or [batch_size,K,1] when batched, each row an independent
        from-scratch clip). A 2D [K,T] prompt unsqueezes to [1,K,T]."""
        if prompt is None:
            shape = (
                (self.n_codebooks, 1) if batch_size == 1
                else (batch_size, self.n_codebooks, 1)
            )
            prompt = torch.randint(0, self.cfg.vocab_per_codebook, shape)
        squeeze_batch = prompt.dim() == 2
        if squeeze_batch:
            prompt = prompt.unsqueeze(0)
        return prompt, squeeze_batch

    @torch.no_grad()
    def _generate_stream(
        self,
        prompt: torch.Tensor,
        num_new_frames: int,
        temperature: float | Sequence[float] = 1.0,
        top_k: int | None | Sequence[int | None] = 250,
        top_p: float | None | Sequence[float | None] = None,
        text_emb: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        text_emb_neg: torch.Tensor | None = None,
        text_kv_mask: torch.Tensor | None = None,
        text_kv_mask_neg: torch.Tensor | None = None,
        lyric_ids: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        lyric_ids_neg: torch.Tensor | None = None,
        lyric_mask_neg: torch.Tensor | None = None,
        lyric_cfg_scale: float | None = None,
        melody: torch.Tensor | None = None,
        melody_cfg_scale: float | None = None,
        emit_every: int = 256,
        first_emit: int | None = None,
    ):
        """Shared decode loop behind generate(); yields the NEW frames as
        un-delayed [B,K,n] torch LongTensor chunks (the cadence the server's
        streaming path consumes). `prompt` must be a resolved [B,K,T] torch tensor
        (see _resolve_prompt). generate() collects these chunks, so streamed and
        one-shot tokens are identical by construction — the MLX analog of the torch
        _generate_stream streaming contract."""
        cfg = self.cfg
        K = cfg.n_codebooks
        pad = cfg.pad_id
        if first_emit is None:
            first_emit = emit_every

        prompt_m = mx.array(prompt.detach().to(torch.int64).cpu().numpy().astype(np.int32))
        B, _, T_prompt = prompt_m.shape

        def _per_cb(val, kind):
            if val is None or isinstance(val, (int, float)):
                return [val] * K
            val = list(val)
            if len(val) == 1:
                return val * K
            if len(val) != K:
                # Mirror the torch _per_cb: resample a foreign-K ladder by
                # linear interpolation instead of rejecting (see
                # nano_audio_gpt._generate_stream).
                L = len(val)
                if any(v is None for v in val):
                    val = [val[round(j * (L - 1) / max(K - 1, 1))] for j in range(K)]
                else:
                    is_int = all(isinstance(v, int) for v in val)
                    fit = []
                    for j in range(K):
                        pos = j * (L - 1) / max(K - 1, 1)
                        lo = int(pos)
                        hi = min(lo + 1, L - 1)
                        frac = pos - lo
                        x = val[lo] * (1 - frac) + val[hi] * frac
                        fit.append(max(1, round(x)) if is_int else x)
                    val = fit
                print(f"[generate-mlx] {kind} ladder length {L} != K={K} — resampled")
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

        def _mask_t2m(m):
            # Convert an incoming torch additive cross-attn mask to mx, replacing
            # any -inf with the finite NEG_INF the mlx SDPA wants (an all-masked
            # flash block with -inf NaNs; cf. the lyric mask). For B=1 inference
            # the incoming tag mask is all-zeros (no padding), so this is a no-op.
            if m is None:
                return None
            a = _t2m(m, self._dtype)
            return mx.where(mx.isinf(a), mx.array(NEG_INF, dtype=a.dtype), a)

        tkv_pos = _mask_t2m(text_kv_mask)
        tkv_neg = _mask_t2m(text_kv_mask_neg)

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
        full = {"text": cond, "tkv": tkv_pos,
                "lemb": lyric_pos, "lkv": lyric_kv_pos, "mel": mel_pos}
        if not use_cfg:
            stages = [full]
            scales: list[float] = []
        else:
            off = {"text": neg, "tkv": tkv_neg,
                   "lemb": lyric_neg, "lkv": lyric_kv_neg, "mel": mel_neg}
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
        # CFG guidance stages, run UNBATCHED — one B-row forward per stage. The
        # batched S*B forward (stacking stages along the batch dim) is NOT
        # logit-identical on the v10 DAC shape: the guided logits come out wrong
        # and the rollout collapses to noise (cfg=1 / single-stage stayed clean;
        # only cfg>1 broke). Per-stage costs len(stages) forwards/step, but the
        # int8 decode is still ~2x faster than torch-MPS here and it is CORRECT.
        S = len(stages)

        # Run each guidance stage as its OWN B-row forward (see note above), each
        # with its own self/cross caches, then combine the per-stage logits. One
        # cross-attn K/V cache per block per stage: the conditioning is FIXED for
        # the whole decode, so kv_proj(cond) runs once at prefill, not per step.
        caches_s = [self._new_caches(B, T_delay) for _ in range(S)]
        cross_s = [
            [_BlockCrossKV() for _ in range(cfg.n_layers)] for _ in range(S)
        ]

        def _run(inp, start):
            return [
                self(
                    inp, caches=caches_s[i], start_pos=start,
                    text_emb=s["text"], text_kv_mask=s["tkv"],
                    lyric_emb=s["lemb"], lyric_kv_mask=s["lkv"], melody_emb=s["mel"],
                    cross_caches=cross_s[i],
                )
                for i, s in enumerate(stages)
            ]

        def _combine(parts):
            if S == 1:
                return parts[0]
            out = parts[0]
            for i, sc in enumerate(scales):
                out = out + sc * (parts[i + 1] - parts[i])
            return out

        def _emit(s, e):
            # Un-delay new frames [s, e): cb k of frame f sits at delayed pos f+k.
            chunk = mx.stack([tokens[:, k, s + k:e + k] for k in range(K)], axis=1)
            mx.eval(chunk)
            return torch.from_numpy(np.array(chunk, copy=False)).to(torch.long)

        prefill_len = max(1, T_prompt)
        logits = _combine(_run(tokens[:, :, :prefill_len], 0))
        mx.eval(logits, [c.k for cs in caches_s for c in cs])

        emitted = 0  # count of new frames already yielded
        threshold = first_emit
        for pos in range(prefill_len, T_delay):
            if pos > prefill_len:
                inp = tokens[:, :, pos - 1:pos]
                logits = _combine(_run(inp, pos - 1))

            step = logits[:, :, -1, :]  # [B, K, V]
            # Mask all control ids (pad, plus FIM <SUF>/<MID> when use_fim) so the
            # generated output only ever contains real DAC tokens. Mirrors the
            # torch path's `step_logits[..., vocab_per_codebook:] = -inf`.
            step[:, :, cfg.vocab_per_codebook:] = NEG_INF

            # The delay pattern makes the codebooks live at this pos a CONTIGUOUS
            # range (cb k samples while k + T_prompt <= pos < k + T_total), so all
            # of them sample in one batched call + one scatter write.
            k0 = max(0, pos - T_total + 1)
            k1 = min(K, pos - T_prompt + 1)
            if k1 > k0:
                toks = self._sample_codebooks(
                    step[:, k0:k1, :], temps[k0:k1], top_ks[k0:k1], top_ps[k0:k1]
                )
                tokens[:, k0:k1, pos] = toks
            # Dispatch without blocking so the GPU works through this step while
            # Python builds the next step's graph; hard-sync every 8 steps to
            # bound the in-flight queue. (A per-step mx.eval serialized the two.)
            if (pos - prefill_len) % 8 == 7:
                mx.eval(tokens)
            else:
                mx.async_eval(tokens)

            # Emit any new frames now fully known: frame f completes at pos=f+K-1.
            n_complete = min(pos - (K - 1), T_total - 1) - T_prompt + 1
            if n_complete - emitted >= threshold:
                yield _emit(T_prompt + emitted, T_prompt + n_complete)
                emitted = n_complete
                threshold = emit_every
        if emitted < num_new_frames:
            yield _emit(T_prompt + emitted, T_total)

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
        text_kv_mask: torch.Tensor | None = None,
        text_kv_mask_neg: torch.Tensor | None = None,
        lyric_ids: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        lyric_ids_neg: torch.Tensor | None = None,
        lyric_mask_neg: torch.Tensor | None = None,
        lyric_cfg_scale: float | None = None,
        melody: torch.Tensor | None = None,
        melody_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Drop-in for NanoAudioGPT.generate. Accepts/returns torch tensors.

        Thin collector over _generate_stream (mirrors the torch generate), so the
        streamed and one-shot token sequences are identical by construction."""
        prompt, squeeze_batch = self._resolve_prompt(prompt)
        chunks = list(self._generate_stream(
            prompt, num_new_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=text_emb, cfg_scale=cfg_scale, text_emb_neg=text_emb_neg,
            text_kv_mask=text_kv_mask, text_kv_mask_neg=text_kv_mask_neg,
            lyric_ids=lyric_ids, lyric_mask=lyric_mask,
            lyric_ids_neg=lyric_ids_neg, lyric_mask_neg=lyric_mask_neg,
            lyric_cfg_scale=lyric_cfg_scale,
            melody=melody, melody_cfg_scale=melody_cfg_scale,
        ))
        new = (
            torch.cat(chunks, dim=-1) if chunks
            else prompt.new_zeros((prompt.shape[0], prompt.shape[1], 0))
        )
        out = torch.cat([prompt, new], dim=-1)
        return out.squeeze(0) if squeeze_batch else out
