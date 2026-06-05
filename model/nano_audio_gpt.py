"""Nano audio GPT.

Decoder-only transformer over delayed multi-codebook audio tokens.
At each step, the K codebook embeddings are summed; K output heads predict
the next token for each codebook. Sized to ~1.5B params with the default config
(1.51B with text-conditioning cross-attention, 1.14B without).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .delay_pattern import revert_delay

KVCache = tuple[torch.Tensor, torch.Tensor]  # (past_k, past_v)


class StaticLayerKVCache:
    """Pre-allocated KV buffer for one attention layer. Avoids per-step allocations."""
    __slots__ = ('k', 'v', 'pos')

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v
        self.pos = 0


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE cos/sin tables, shared across all attention layers.

    Rotates query/key pairs by position-dependent angles so attention scores
    depend on relative positions. Replaces absolute positional embeddings;
    extrapolates better to inference lengths beyond the training segment.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
        # Non-persistent: re-derivable from max_seq_len + base, don't bloat ckpts.
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, start_pos: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        end = start_pos + seq_len
        return self.cos_cached[start_pos:end], self.sin_cached[start_pos:end]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k. q/k: [B, H, T, D]. cos/sin: [T, D]."""
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


@dataclass
class GPTConfig:
    n_codebooks: int = 9
    vocab_per_codebook: int = 1024  # DAC vocab size, excluding pad
    d_model: int = 2048
    n_layers: int = 22
    n_heads: int = 16
    d_ff: int = 8192
    dropout: float = 0.05
    max_seq_len: int = 8192  # delayed sequence length cap (also sizes RoPE cos/sin table)
    use_text_conditioning: bool = False
    rope_base: float = 10000.0
    use_gradient_checkpointing: bool = True

    @property
    def vocab_with_pad(self) -> int:
        return self.vocab_per_codebook + 1

    @property
    def pad_id(self) -> int:
        return self.vocab_per_codebook


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | StaticLayerKVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | StaticLayerKVCache]:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE *before* writing to cache so cached keys carry their
        # original positions; subsequent decode steps don't need to re-rotate.
        q, k = apply_rotary(q, k, cos, sin)

        if isinstance(cache, StaticLayerKVCache):
            old_pos = cache.pos
            end = old_pos + T
            cache.k[:, :, old_pos:end] = k
            cache.v[:, :, old_pos:end] = v
            k = cache.k[:, :, :end]
            v = cache.v[:, :, :end]
            cache.pos = end
            new_cache = cache
            is_causal = old_pos == 0
        elif cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
            new_cache = (k, v)
            is_causal = False
        else:
            new_cache = (k, v)
            is_causal = True

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y), new_cache


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q from audio, K/V from text embeddings."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.kv_proj = nn.Linear(cfg.d_model, 2 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] audio hidden states, cond: [B, T_cond, D] text embeddings."""
        B, T, D = x.shape
        T_c = cond.shape[1]
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(cond).split(D, dim=-1)
        k = k.view(B, T_c, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_c, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.has_cross_attn = cfg.use_text_conditioning
        if self.has_cross_attn:
            self.ln_cross = nn.RMSNorm(cfg.d_model)
            self.cross_attn = CrossAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.use_gradient_checkpointing = cfg.use_gradient_checkpointing

    def _body(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        text_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        # Training-only path (no KV cache). Wrapped by gradient checkpointing.
        attn_out, _ = self.attn(self.ln1(x), cos, sin, cache=None)
        x = x + attn_out
        if self.has_cross_attn and text_emb is not None:
            x = x + self.cross_attn(self.ln_cross(x), text_emb)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        text_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        if cache is None and self.training and self.use_gradient_checkpointing:
            x = checkpoint(self._body, x, cos, sin, text_emb, use_reentrant=False)
            return x, None
        attn_out, new_cache = self.attn(self.ln1(x), cos, sin, cache=cache)
        x = x + attn_out
        if self.has_cross_attn and text_emb is not None:
            x = x + self.cross_attn(self.ln_cross(x), text_emb)
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class NanoAudioGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeds = nn.ModuleList(
            [nn.Embedding(cfg.vocab_with_pad, cfg.d_model) for _ in range(cfg.n_codebooks)]
        )
        head_dim = cfg.d_model // cfg.n_heads
        self.rotary = RotaryEmbedding(head_dim, cfg.max_seq_len, cfg.rope_base)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.RMSNorm(cfg.d_model)
        self.heads = nn.ModuleList(
            [nn.Linear(cfg.d_model, cfg.vocab_with_pad, bias=False) for _ in range(cfg.n_codebooks)]
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        """Forward pass.

        tokens: [B, K, T]
        kv_caches: when None, training path (returns logits only).
                   when a list, returns (logits, new_caches).
        start_pos: positional offset for the tokens (used with KV cache).
        text_emb: [B, T_text, D] optional text conditioning embeddings.
        returns: logits [B, K, T, V] or (logits, new_caches)
        """
        B, K, T = tokens.shape
        assert K == self.cfg.n_codebooks
        assert start_pos + T <= self.cfg.max_seq_len, (
            f"seq pos {start_pos + T} > max_seq_len {self.cfg.max_seq_len} "
            f"(RoPE table size — bump GPTConfig.max_seq_len if you need longer)"
        )

        x = self.tok_embeds[0](tokens[:, 0])
        for k in range(1, K):
            x = x + self.tok_embeds[k](tokens[:, k])
        x = self.drop(x)

        cos, sin = self.rotary(start_pos, T)

        new_caches: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, cos, sin, cache=cache, text_emb=text_emb)
            new_caches.append(new_cache)
        x = self.ln_final(x)

        logits = torch.stack([head(x) for head in self.heads], dim=1)
        if kv_caches is not None:
            return logits, new_caches
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

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
        """Continue a prompt, or generate unconditionally when prompt is None.

        prompt: [K, T_prompt] or [B, K, T_prompt], or None for unconditional
        text_emb: [B, 1, D] optional text conditioning (from CLAPTextEncoder)
        cfg_scale: classifier-free guidance scale. 1.0 = no guidance (single
            forward pass). >1.0 = run an additional baseline forward pass and
            blend logits = base + cfg_scale * (cond - base). Active when either
            text_emb or text_emb_neg is set.
        text_emb_neg: [B, N, D] optional *negative* conditioning. The CFG
            baseline pass is run with this instead of None, so guidance steers
            *away* from it: logits = neg + cfg_scale * (cond - neg). When None
            the baseline is the usual unconditioned pass (steer away from
            "nothing", i.e. plain positive guidance). The model was never
            trained on negative conditioning specifically — this is an
            inference-time use of the same cross-attention, so tune cfg_scale
            empirically.
        temperature / top_k / top_p: scalar (applied to every codebook) or a
            sequence of length n_codebooks (one value per codebook). Later
            codebooks model fine residuals whose true distribution is nearly
            uniform — sampling them at the same temperature as cb0 produces
            audible noise, so a decreasing ladder (e.g. temp/top_k high on cb0,
            low on cb8) usually sounds much better.
        top_p: nucleus sampling cutoff. If set, restrict sampling to the
            smallest set of tokens whose cumulative probability exceeds top_p.
            Applied before top_k (consistent with HF transformers / MusicGen).
        returns: [B, K, T_prompt + num_new_frames]

        When prompt is None a single random seed frame is used internally; the
        InferenceEngine seeds with DAC-encoded silence instead to keep the
        from-scratch path on-distribution.
        Uses KV cache for efficient autoregressive decoding.
        """
        if prompt is None:
            device = next(self.parameters()).device
            prompt = torch.randint(
                0, self.cfg.vocab_per_codebook,
                (self.cfg.n_codebooks, 1),
                device=device,
            )
        squeeze_batch = prompt.dim() == 2
        if squeeze_batch:
            prompt = prompt.unsqueeze(0)
        B, K, T_prompt = prompt.shape
        assert K == self.cfg.n_codebooks

        def _per_cb(val, kind: str) -> list:
            if val is None or isinstance(val, (int, float)):
                return [val] * K
            val = list(val)
            assert len(val) == K, f"{kind} must be scalar or length {K}, got {len(val)}"
            return val

        temps = _per_cb(temperature, "temperature")
        top_ks = _per_cb(top_k, "top_k")
        top_ps = _per_cb(top_p, "top_p")
        device = prompt.device
        pad = self.cfg.pad_id

        T_total = T_prompt + num_new_frames
        T_delay = T_total + K - 1
        assert T_delay <= self.cfg.max_seq_len, (
            f"target delayed length {T_delay} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        tokens = torch.full((B, K, T_delay), pad, dtype=torch.long, device=device)
        for k in range(K):
            tokens[:, k, k:k + T_prompt] = prompt[:, k]

        use_cfg = cfg_scale != 1.0 and (text_emb is not None or text_emb_neg is not None)

        was_training = self.training
        self.eval()
        try:
            head_dim = self.cfg.d_model // self.cfg.n_heads
            dtype = next(self.parameters()).dtype

            def _make_caches() -> list[StaticLayerKVCache]:
                return [
                    StaticLayerKVCache(
                        torch.zeros(B, self.cfg.n_heads, T_delay, head_dim, device=device, dtype=dtype),
                        torch.zeros(B, self.cfg.n_heads, T_delay, head_dim, device=device, dtype=dtype),
                    )
                    for _ in range(self.cfg.n_layers)
                ]

            caches_cond = _make_caches()
            caches_uncond = _make_caches() if use_cfg else None

            # prefill: run full prompt through transformer
            prefill_len = max(1, T_prompt)
            prefill_inp = tokens[:, :, :prefill_len]
            logits, caches_cond = self.forward(
                prefill_inp, kv_caches=caches_cond, start_pos=0, text_emb=text_emb,
            )
            if use_cfg:
                logits_base, caches_uncond = self.forward(
                    prefill_inp, kv_caches=caches_uncond, start_pos=0, text_emb=text_emb_neg,
                )
                logits = logits_base + cfg_scale * (logits - logits_base)

            # decode: one position at a time using cached K/V
            for p in range(prefill_len, T_delay):
                if p > prefill_len:
                    inp = tokens[:, :, p - 1:p]
                    logits, caches_cond = self.forward(
                        inp, kv_caches=caches_cond, start_pos=p - 1, text_emb=text_emb,
                    )
                    if use_cfg:
                        logits_base, caches_uncond = self.forward(
                            inp, kv_caches=caches_uncond, start_pos=p - 1, text_emb=text_emb_neg,
                        )
                        logits = logits_base + cfg_scale * (logits - logits_base)

                step_logits = logits[:, :, -1, :].clone()  # [B, K, V]
                step_logits[..., pad] = float("-inf")

                # Per-codebook sampling: each cb gets its own temperature/top_k/top_p
                # so later codebooks (high-entropy DAC residuals) can be sampled
                # tightly without making cb0 deterministic.
                sampled = torch.empty(B, K, dtype=torch.long, device=device)
                for k in range(K):
                    cb_logits = step_logits[:, k, :]  # [B, V]
                    t_k = temps[k]
                    if t_k == 0:
                        sampled[:, k] = cb_logits.argmax(dim=-1)
                        continue
                    cb_logits = cb_logits / t_k
                    p_k = top_ps[k]
                    if p_k is not None and 0.0 < p_k < 1.0:
                        sorted_logits, sorted_idx = cb_logits.sort(dim=-1, descending=True)
                        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                        cutoff = cum_probs > p_k
                        cutoff[..., 1:] = cutoff[..., :-1].clone()
                        cutoff[..., 0] = False
                        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
                        cb_logits = torch.empty_like(cb_logits).scatter_(
                            -1, sorted_idx, sorted_logits
                        )
                    k_k = top_ks[k]
                    if k_k is not None:
                        vals, _ = cb_logits.topk(k_k, dim=-1)
                        cb_logits = cb_logits.masked_fill(
                            cb_logits < vals[..., -1:], float("-inf")
                        )
                    probs = F.softmax(cb_logits, dim=-1)
                    sampled[:, k] = torch.multinomial(probs, 1).squeeze(-1)

                for k in range(K):
                    if k + T_prompt <= p < k + T_total:
                        tokens[:, k, p] = sampled[:, k]
        finally:
            if was_training:
                self.train()

        out = revert_delay(tokens, T_total)
        return out.squeeze(0) if squeeze_batch else out


if __name__ == "__main__":
    cfg = GPTConfig()
    m = NanoAudioGPT(cfg)
    n = m.num_params()
    print(f"params: {n/1e6:.2f}M  ({n:,})")
    print(f"max_seq_len: {cfg.max_seq_len} (RoPE table); covers "
          f"~{cfg.max_seq_len/86:.0f}s of audio")
    x = torch.randint(0, cfg.vocab_per_codebook, (2, cfg.n_codebooks, 64))
    y = m(x)
    print(f"forward ok: {y.shape}")

    # Cached-decode smoke: prefill 32, then decode 4 steps and verify shape.
    m.eval()
    head_dim = cfg.d_model // cfg.n_heads
    caches = [
        StaticLayerKVCache(
            torch.zeros(1, cfg.n_heads, 64, head_dim),
            torch.zeros(1, cfg.n_heads, 64, head_dim),
        )
        for _ in range(cfg.n_layers)
    ]
    x_prefill = torch.randint(0, cfg.vocab_per_codebook, (1, cfg.n_codebooks, 32))
    _, caches = m(x_prefill, kv_caches=caches, start_pos=0)
    for step in range(4):
        x_step = torch.randint(0, cfg.vocab_per_codebook, (1, cfg.n_codebooks, 1))
        logits, caches = m(x_step, kv_caches=caches, start_pos=32 + step)
    print(f"cached decode ok: {logits.shape}")
