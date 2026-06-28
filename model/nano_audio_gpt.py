"""Nano audio GPT.

Decoder-only transformer over delayed multi-codebook audio tokens.
At each step, the K codebook embeddings are summed; K output heads predict
the next token for each codebook. Sized to ~1.5B params with the default config
(1.51B with text-conditioning cross-attention, 1.14B without).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .lyric_encoder import PHONEME_VOCAB_SIZE, LyricEncoder
from .melody_encoder import MelodyEncoder

KVCache = tuple[torch.Tensor, torch.Tensor]  # (past_k, past_v)


class StaticLayerKVCache:
    """Pre-allocated KV buffer for one attention layer. Avoids per-step allocations."""
    __slots__ = ('k', 'v', 'pos')

    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k
        self.v = v
        self.pos = 0


class CrossKVCache:
    """Cached projected (+ qk-normed) K/V for ONE cross-attention over a FIXED
    conditioning sequence (tags or lyrics).

    Cross-attention K/V depend only on `cond`, which is encoded once at prefill and
    never changes across decode steps — yet `kv_proj(cond)` was being recomputed in
    every block on every step. For a long lyric stream (up to 512 phonemes) that
    recompute dominates per-token decode. Computing it once at prefill and reusing
    it makes the cached value byte-identical to the recompute (k_norm is
    position-independent, so the stored K is post-norm). One cache per (CFG-stage,
    block, stream) — stages must NOT share caches (each has its own cond)."""
    __slots__ = ('k', 'v')

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None


class BlockCrossKV:
    """The two cross-attention caches for one decoder block (tag + lyric streams)."""
    __slots__ = ('text', 'lyric')

    def __init__(self):
        self.text = CrossKVCache()
        self.lyric = CrossKVCache()


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
    # Lyric (phoneme-sequence) conditioning — separate from the pooled-CLAP tag
    # path. When enabled, a LyricEncoder submodule encodes phoneme ids into a
    # sequence the decoder cross-attends to (so the model can sing actual words).
    use_lyric_conditioning: bool = False
    phoneme_vocab_size: int = PHONEME_VOCAB_SIZE
    lyric_enc_layers: int = 3
    lyric_enc_heads: int = 8
    lyric_enc_d_ff: int = 4096
    # 512 (up from 256): a dense 60s crop can carry ~400-600 phoneme tokens, which
    # the 256 cap silently truncated (lyric_encoder.append_unit_capped), dropping
    # the tail words' alignment signal. Only sizes the encoder's non-persistent
    # sinusoidal PE + the dataset truncation cap (no learned-weight reshape).
    max_lyric_len: int = 512
    # Melody (chromagram-sequence) conditioning — a time-aligned, MusicGen-Melody
    # style stream, separate from both the pooled-CLAP tag path and the lyric
    # cross-attention. When enabled, a MelodyEncoder projects a [B,T,12] chroma
    # sequence and the decoder ADDS it to the per-frame token-sum input at the cb0
    # anchor (delayed position p ↔ frame p), so a hummed melody is regenerated in
    # whatever timbre the tags ask for. Dense+time-aligned, so additive rather than
    # cross-attention. Adding it is checkpoint-incompatible (new submodule).
    use_melody_conditioning: bool = False
    melody_n_bins: int = 12
    melody_enc_layers: int = 2
    # Generative stem conditioning (the /addstem path) — a token-domain axis for
    # "add a stem to an existing song". On a stem-add training batch the decoder's
    # TARGET is one isolated stem's tokens; the conditioning is the song's OTHER
    # stems (the accompaniment): a StemEncoder embeds each conditioning stem's full
    # codec tokens [B,K,T] (+ a stem-type id), _stem_add SUMS them, adds a
    # target-type embedding (which stem to generate), and the decoder ADDS the
    # result per-frame at the cb0 anchor — exactly like melody but full-fidelity
    # (token-domain). Lets a user upload a song and generate e.g. a new bassline
    # that fits it. Off by default (checkpoint-incompatible new submodule); enabled
    # in a fresh v9 start with the stem cache (packed_NNN.stem.bin sidecar).
    use_stem_conditioning: bool = False
    stem_enc_layers: int = 2
    n_stem_types: int = 4  # drums/bass/vocals/other (model/stem_encoder.STEM_TYPES)
    # Fill-in-the-middle (infill). When enabled, training reorders a fraction of
    # crops into the canonical FIM layout `prefix <SUF> suffix <MID> middle` (a
    # frame-domain reorder before the delay pattern — attention and the delay
    # logic are untouched) so the decoder learns to bridge a gap it has been
    # shown the suffix of. Adds two per-codebook control ids (<SUF>/<MID>) after
    # pad, so enabling it grows the embedding/head vocab and is
    # checkpoint-incompatible (a fresh start, like adding the melody encoder).
    use_fim: bool = False
    rope_base: float = 10000.0
    use_gradient_checkpointing: bool = True
    # QK-norm: RMSNorm on the per-head queries/keys of self-attention and both
    # cross-attentions (Gemma 2 / OLMo 2 / Qwen style). Bounds attention logits
    # regardless of LR — both 2026-06-12 v8 launches diverged cb0-first with the
    # classic logit-growth signature (onset ~5e-5–1.2e-4, even after the
    # zero-gated conditioning init and with Adam beta2 already 0.95); this is
    # the structural fix. Default False so pre-qk-norm checkpoint cfg dicts
    # (v7) still load everywhere GPTConfig(**ckpt["cfg"]) is called; the train
    # entrypoints (DEFAULTS / TrainConfig) enable it explicitly, and v8+
    # checkpoints carry the key in their saved cfg.
    use_qk_norm: bool = False
    # Same QK-norm, applied inside the LyricEncoder's bidirectional self-attention
    # (model/lyric_encoder.py). Separate flag because the encoder was added after
    # use_qk_norm and the original v8_sing run trained without it: grad forensics
    # traced that run's gradient explosion to lyric_encoder.layers.0 (unbounded
    # encoder attention logits). Default False so every existing checkpoint's cfg
    # dict still loads strictly; the train entrypoints enable it for v8+.
    use_lyric_qk_norm: bool = False

    @property
    def n_control(self) -> int:
        """Reserved ids appended after the DAC vocab: pad, plus the two FIM
        sentinels when use_fim is on."""
        return 1 + (2 if self.use_fim else 0)

    @property
    def vocab_with_pad(self) -> int:
        """Embedding/head size: DAC vocab + control ids (pad [+ FIM sentinels])."""
        return self.vocab_per_codebook + self.n_control

    @property
    def pad_id(self) -> int:
        return self.vocab_per_codebook

    @property
    def suf_id(self) -> int:
        """FIM <SUF> sentinel (separates prefix from suffix). Valid iff use_fim."""
        return self.vocab_per_codebook + 1

    @property
    def mid_id(self) -> int:
        """FIM <MID> sentinel (marks the start of the to-be-generated middle)."""
        return self.vocab_per_codebook + 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        # Per-head-dim gains, shared across heads (Qwen style). See
        # GPTConfig.use_qk_norm for why.
        self.q_norm = nn.RMSNorm(self.head_dim) if cfg.use_qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if cfg.use_qk_norm else None
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | StaticLayerKVCache | None = None,
        input_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | StaticLayerKVCache]:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # QK-norm before RoPE (and before the cache write, so cached keys are
        # already normed — decode steps see the identical transform).
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE *before* writing to cache so cached keys carry their
        # original positions; subsequent decode steps don't need to re-rotate.
        q, k = apply_rotary(q, k, cos, sin)

        if input_pos is not None:
            # CUDA-graph-friendly decode step: write k/v at a TENSOR position and
            # attend over the FULL pre-allocated cache buffer with a static-shape
            # mask (no python-int position, no growing slice) so torch.compile can
            # capture one reusable graph for every step. Correct in eager too.
            assert isinstance(cache, StaticLayerKVCache)
            cache.k[:, :, input_pos] = k
            cache.v[:, :, input_pos] = v
            y = F.scaled_dot_product_attention(
                q, cache.k, cache.v, attn_mask=attn_mask,
            )
            new_cache = cache
        elif isinstance(cache, StaticLayerKVCache):
            old_pos = cache.pos
            end = old_pos + T
            cache.k[:, :, old_pos:end] = k
            cache.v[:, :, old_pos:end] = v
            k = cache.k[:, :, :end]
            v = cache.v[:, :, :end]
            cache.pos = end
            new_cache = cache
            is_causal = old_pos == 0
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )
        elif cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
            new_cache = (k, v)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            new_cache = (k, v)
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
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
        # See GPTConfig.use_qk_norm. RMSNorm(0) == 0, so the zeros-uncond
        # convention (zero cond embedding == skipping the block exactly,
        # tests/test_cfg_uncond.py) survives the norm.
        self.q_norm = nn.RMSNorm(self.head_dim) if cfg.use_qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim) if cfg.use_qk_norm else None
        self.dropout = cfg.dropout

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, kv_mask: torch.Tensor | None = None,
        kv_cache: "CrossKVCache | None" = None,
    ) -> torch.Tensor:
        """x: [B, T, D] audio hidden states, cond: [B, T_cond, D] text embeddings.

        kv_mask: optional additive attention mask broadcastable to
        [B, n_heads, T, T_cond] (0 keep, -inf drop) — used to mask padded
        positions in a variable-length lyric sequence. Callers must guarantee
        each query row has at least one un-masked key (a fully -inf row makes
        SDPA's softmax produce NaN); the lyric path ensures this by always
        keeping the BOS phoneme valid.

        kv_cache: optional CrossKVCache. `cond` is fixed across decode steps, so the
        projected (+ k-normed) K/V are computed once at prefill (when the cache is
        empty) and reused on every subsequent step — skipping kv_proj(cond), the
        dominant per-token decode cost when `cond` is long. None on the training /
        full-sequence path (one forward, already amortized).
        """
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if kv_cache is not None and kv_cache.k is not None:
            k, v = kv_cache.k, kv_cache.v
        else:
            T_c = cond.shape[1]
            k, v = self.kv_proj(cond).split(D, dim=-1)
            k = k.view(B, T_c, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T_c, self.n_heads, self.head_dim).transpose(1, 2)
            # k_norm is position-independent, so caching the post-norm K is exactly
            # the value a recompute would produce.
            if self.k_norm is not None:
                k = self.k_norm(k)
            if kv_cache is not None:
                kv_cache.k = k
                kv_cache.v = v
        # The tag mask is built outside autocast (float32); cast to the (possibly
        # bf16) query dtype so SDPA's flash/mem-efficient kernels accept it. A
        # no-op for the lyric mask, which is already built in the compute dtype.
        if kv_mask is not None and kv_mask.dtype != q.dtype:
            kv_mask = kv_mask.to(q.dtype)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=kv_mask,
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
        # Separate cross-attention for the lyric phoneme sequence (kept distinct
        # from the pooled tag vector so one softmax doesn't pit a global vibe
        # vector against 256 phonemes, and so each stream drops independently).
        self.has_lyric_attn = cfg.use_lyric_conditioning
        if self.has_lyric_attn:
            self.ln_lyric = nn.RMSNorm(cfg.d_model)
            self.lyric_attn = CrossAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)
        self.use_gradient_checkpointing = cfg.use_gradient_checkpointing

    def _body(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        text_emb: torch.Tensor | None,
        text_kv_mask: torch.Tensor | None,
        lyric_emb: torch.Tensor | None,
        lyric_kv_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Training-only path (no KV cache). Wrapped by gradient checkpointing.
        attn_out, _ = self.attn(self.ln1(x), cos, sin, cache=None)
        x = x + attn_out
        if self.has_cross_attn and text_emb is not None:
            x = x + self.cross_attn(self.ln_cross(x), text_emb, kv_mask=text_kv_mask)
        if self.has_lyric_attn and lyric_emb is not None:
            x = x + self.lyric_attn(self.ln_lyric(x), lyric_emb, kv_mask=lyric_kv_mask)
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        text_emb: torch.Tensor | None = None,
        text_kv_mask: torch.Tensor | None = None,
        lyric_emb: torch.Tensor | None = None,
        lyric_kv_mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        cross_kv: "BlockCrossKV | None" = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        if cache is None and self.training and self.use_gradient_checkpointing:
            x = checkpoint(
                self._body, x, cos, sin, text_emb, text_kv_mask,
                lyric_emb, lyric_kv_mask,
                use_reentrant=False,
            )
            return x, None
        attn_out, new_cache = self.attn(
            self.ln1(x), cos, sin, cache=cache,
            input_pos=input_pos, attn_mask=attn_mask,
        )
        x = x + attn_out
        if self.has_cross_attn and text_emb is not None:
            x = x + self.cross_attn(
                self.ln_cross(x), text_emb, kv_mask=text_kv_mask,
                kv_cache=cross_kv.text if cross_kv is not None else None,
            )
        if self.has_lyric_attn and lyric_emb is not None:
            x = x + self.lyric_attn(
                self.ln_lyric(x), lyric_emb, kv_mask=lyric_kv_mask,
                kv_cache=cross_kv.lyric if cross_kv is not None else None,
            )
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class NanoAudioGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # Optional torch.compile'd forward for the per-step decode loop. Set by the
        # inference engine after load (a no-op for training). When present,
        # _generate_stream routes each decode step through it so CUDA graphs apply;
        # the eager self.forward is the fallback (identical results, just slower).
        self._compiled_forward = None
        # Cross-attention K/V caching during generate (see CrossKVCache). On by
        # default; an A/B handle for the benchmark / byte-identity test to fall back
        # to the recompute path. Never affects training (cache is generate-only).
        self._use_cross_kv_cache = True
        self.tok_embeds = nn.ModuleList(
            [nn.Embedding(cfg.vocab_with_pad, cfg.d_model) for _ in range(cfg.n_codebooks)]
        )
        head_dim = cfg.d_model // cfg.n_heads
        self.rotary = RotaryEmbedding(head_dim, cfg.max_seq_len, cfg.rope_base)
        self.drop = nn.Dropout(cfg.dropout)
        # LyricEncoder lives INSIDE the model so DDP syncs its grads and it
        # saves/restores with the model state_dict (no sidecar key like the CLAP
        # projection). It runs once per forward; its output sequence feeds every
        # block's lyric cross-attention.
        if cfg.use_lyric_conditioning:
            self.lyric_encoder = LyricEncoder(
                d_model=cfg.d_model,
                n_layers=cfg.lyric_enc_layers,
                n_heads=cfg.lyric_enc_heads,
                d_ff=cfg.lyric_enc_d_ff,
                max_len=cfg.max_lyric_len,
                vocab_size=cfg.phoneme_vocab_size,
                dropout=cfg.dropout,
                use_qk_norm=cfg.use_lyric_qk_norm,
            )
        # MelodyEncoder also lives INSIDE the model (same rationale as the lyric
        # encoder: DDP grad-sync + saves in the model state_dict). Its output is
        # added to the decoder input per frame rather than cross-attended.
        if cfg.use_melody_conditioning:
            self.melody_encoder = MelodyEncoder(
                d_model=cfg.d_model,
                n_bins=cfg.melody_n_bins,
                n_layers=cfg.melody_enc_layers,
                dropout=cfg.dropout,
            )
        # StemEncoder also lives INSIDE the model (same rationale): token-domain,
        # added per-frame like melody. See _stem_add for the additive injection.
        if cfg.use_stem_conditioning:
            from .stem_encoder import StemEncoder

            self.stem_encoder = StemEncoder(
                d_model=cfg.d_model,
                n_codebooks=cfg.n_codebooks,
                vocab_with_pad=cfg.vocab_with_pad,
                n_layers=cfg.stem_enc_layers,
                n_stem_types=cfg.n_stem_types,
                dropout=cfg.dropout,
            )
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.RMSNorm(cfg.d_model)
        # The K per-codebook output heads, fused into one Linear (one matmul per
        # forward instead of K kernel launches); the forward's view+transpose
        # recovers the per-codebook [B, K, T, V] layout.
        self.head = nn.Linear(
            cfg.d_model, cfg.n_codebooks * cfg.vocab_with_pad, bias=False
        )
        self.apply(self._init_weights)
        # Stability hardening (v8 diverged at lr ~4e-5 without it, cb0-led):
        # (1) GPT-2 depth-scaled init on the residual out-projs — each block
        # adds up to 4 streams to the residual, so shrink every contribution by
        # 1/sqrt(2*n_layers) to keep activation variance flat across the stack.
        # (2) Conditioning cross-attns start as exact no-ops (zero out_proj,
        # ControlNet/Flamingo-style): step-0 dynamics match an unconditioned
        # decoder and the paths open up as their gradients arrive. The melody
        # path is gated the same way inside MelodyEncoder (zero ln_final gain —
        # zeroing a pre-norm Linear would NOT gate, the norm re-amplifies).
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, std=resid_std)
            nn.init.normal_(block.mlp.fc2.weight, std=resid_std)
            if block.has_cross_attn:
                nn.init.zeros_(block.cross_attn.out_proj.weight)
            if block.has_lyric_attn:
                nn.init.zeros_(block.lyric_attn.out_proj.weight)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode_lyrics(
        self, lyric_ids: torch.Tensor, lyric_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the lyric encoder once: (ids [B,L], mask [B,L]) -> (emb [B,L,D],
        additive cross-attn mask [B,1,1,L]).

        The additive mask is 0 at real phonemes and -inf at padding. Callers must
        ensure every row keeps at least one valid phoneme (BOS) so no query row is
        fully masked (that would NaN the cross-attn softmax)."""
        lyric_emb, lyric_mask = self.lyric_encoder(lyric_ids, lyric_mask)
        kv_mask = torch.zeros(
            lyric_mask.shape[0], 1, 1, lyric_mask.shape[1],
            dtype=lyric_emb.dtype, device=lyric_emb.device,
        ).masked_fill(~lyric_mask[:, None, None, :], float("-inf"))
        return lyric_emb, kv_mask

    def _melody_add(
        self,
        melody: torch.Tensor | None,
        melody_emb: torch.Tensor | None,
        B: int,
        T: int,
        start_pos: int,
        keep: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The per-frame melody term added to the decoder input, shape [B, T, D].

        Three sources, in precedence order:
        - ``melody_emb`` (pre-built full delayed-length [B, seq, D], from
          ``encode_melody_delayed``): sliced ``[start_pos:start_pos+T]`` so the
          same tensor serves the prefill and each KV-cached decode step (mirrors
          the RoPE ``self.rotary(start_pos, T)`` slice).
        - ``melody`` (raw chroma [B, Tc, 12], the training path, start_pos=0):
          encoded here so DDP syncs the encoder grads, then null-padded to T (the
          K-2 delay-tail positions) or truncated.
        - neither (melody dropped / unconditional): the learned null, so the model
          always receives a melody signal — train and inference agree that "no
          melody" means the null, not the absence of any add.
        """
        if melody_emb is not None:
            # Decode path indexes by the tensor input_pos (graph-friendly); the
            # prefill / training path uses the contiguous start_pos slice.
            if input_pos is not None:
                return melody_emb[:, input_pos, :]
            return melody_emb[:, start_pos:start_pos + T, :]
        if melody is not None:
            enc = self.melody_encoder.encode_melody(melody)  # [B, Tc, D]
            Tc = enc.shape[1]
            if Tc < T:
                enc = torch.cat([enc, self.melody_encoder.null_emb(B, T - Tc)], dim=1)
            elif Tc > T:
                enc = enc[:, :T, :]
            if keep is not None:
                # CFG gate: keep=0 -> exactly the null path (the dropped /
                # unconditional state), but the encoder ran, so one compiled
                # graph serves both and its params are never DDP-"unused".
                enc = keep * enc + (1 - keep) * self.melody_encoder.null_emb(B, T)
            return enc
        return self.melody_encoder.null_emb(B, T)

    def encode_melody_delayed(
        self, melody: torch.Tensor, seq_len: int, offset: int = 0,
    ) -> torch.Tensor:
        """Encode chroma once into a full delayed-length tensor [B, seq_len, D].

        The encoded melody (one frame per new audio frame) is placed at delayed
        positions ``[offset, offset+Tc)`` — ``offset`` is the prompt length, so the
        melody lines up with the NEW frames at the cb0 anchor — and every other
        position (prompt/seed prefix + the K-1 delay tail) is the learned null.
        Used by ``generate`` to encode the (fully-known) melody ONCE and reuse the
        result across all decode steps via the ``[start_pos:start_pos+T]`` slice.
        """
        B = melody.shape[0]
        enc = self.melody_encoder.encode_melody(melody)  # [B, Tc, D]
        Tc = enc.shape[1]
        full = self.melody_encoder.null_emb(B, seq_len).clone()  # [B, seq_len, D]
        end = min(offset + Tc, seq_len)
        if end > offset:
            full[:, offset:end, :] = enc[:, :end - offset, :]
        return full

    def _encode_stem_accompaniment(
        self,
        stem_tokens: torch.Tensor,
        stem_types: torch.Tensor,
        stem_present: torch.Tensor,
        target_stem_type: torch.Tensor,
    ) -> torch.Tensor:
        """Sum the encoded conditioning stems + the target-type embedding -> [B,Tc,D].

        stem_tokens [B,S,K,T] (S conditioning stems), stem_types [B,S] long,
        stem_present [B,S] float 0/1 (absent slots contribute 0), target_stem_type
        [B] long. ALL S slots are encoded every call (so every embedding table gets
        grad — DDP-safe); an absent slot is masked to a zero contribution by its
        present bit. The target-type embedding is added once, broadcast per-frame.
        """
        B, S = stem_tokens.shape[0], stem_tokens.shape[1]
        se = self.stem_encoder
        acc = None
        for s in range(S):
            enc = se.encode_stem(stem_tokens[:, s], stem_types[:, s])  # [B, Tc, D]
            term = stem_present[:, s].view(B, 1, 1) * enc
            acc = term if acc is None else acc + term
        acc = acc + se.target_type_emb(target_stem_type).unsqueeze(1)  # [B, Tc, D]
        return acc

    def _stem_add(
        self,
        stem_tokens: torch.Tensor | None,
        stem_types: torch.Tensor | None,
        stem_present: torch.Tensor | None,
        target_stem_type: torch.Tensor | None,
        stem_emb: torch.Tensor | None,
        B: int,
        T: int,
        start_pos: int,
        keep: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The per-frame stem term added to the decoder input, shape [B, T, D].

        Mirrors ``_melody_add``. Three sources, in precedence order:
        - ``stem_emb`` (pre-built full delayed-length [B, seq, D], from
          ``encode_stem_delayed``): sliced for the prefill / KV-cached decode step.
        - ``stem_tokens`` (training path, start_pos=0): the S accompaniment stems are
          encoded + summed HERE (so DDP syncs the encoder grads), the target-type is
          added, then null-padded to T (the K-1 delay tail) or truncated.
        - neither (stem axis off for this sample): the learned null, so the model
          always receives a stem signal — train and inference agree that "no stem"
          means the null, not the absence of any add.

        ``keep`` is the CFG-dropout / stem-add-mode gate: keep=0 -> exactly the null
        (the unconditional / non-stem-add state) while the encoder still ran, so one
        compiled graph serves both and no parameter is ever DDP-"unused".
        """
        se = self.stem_encoder
        if stem_emb is not None:
            if input_pos is not None:
                return stem_emb[:, input_pos, :]
            return stem_emb[:, start_pos:start_pos + T, :]
        if stem_tokens is not None:
            acc = self._encode_stem_accompaniment(
                stem_tokens, stem_types, stem_present, target_stem_type
            )
            Tc = acc.shape[1]
            if Tc < T:
                acc = torch.cat([acc, se.null_emb(B, T - Tc)], dim=1)
            elif Tc > T:
                acc = acc[:, :T, :]
            if keep is not None:
                acc = keep * acc + (1 - keep) * se.null_emb(B, T)
            return acc
        return se.null_emb(B, T)

    def encode_stem_delayed(
        self,
        stem_tokens: torch.Tensor,
        stem_types: torch.Tensor,
        stem_present: torch.Tensor,
        target_stem_type: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> torch.Tensor:
        """Encode the accompaniment stems ONCE into a full delayed-length [B,seq,D].

        The summed accompaniment (+ target-type) is placed at delayed positions
        ``[offset, offset+Tc)`` (offset = prompt length, so the stem lines up with
        the NEW frames at the cb0 anchor); every other position is the learned null.
        Used by ``generate`` to encode the fully-known stems once and reuse the
        result across all decode steps via the ``[start_pos:start_pos+T]`` slice
        (the CFG baseline is the plain ``null_emb``)."""
        B = stem_tokens.shape[0]
        acc = self._encode_stem_accompaniment(
            stem_tokens, stem_types, stem_present, target_stem_type
        )  # [B, Tc, D]
        Tc = acc.shape[1]
        full = self.stem_encoder.null_emb(B, seq_len).clone()  # [B, seq_len, D]
        end = min(offset + Tc, seq_len)
        if end > offset:
            full[:, offset:end, :] = acc[:, :end - offset, :]
        return full

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        text_emb: torch.Tensor | None = None,
        text_kv_mask: torch.Tensor | None = None,
        lyric_ids: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        lyric_emb: torch.Tensor | None = None,
        lyric_kv_mask: torch.Tensor | None = None,
        lyric_keep: torch.Tensor | None = None,
        melody: torch.Tensor | None = None,
        melody_emb: torch.Tensor | None = None,
        melody_keep: torch.Tensor | None = None,
        stem_tokens: torch.Tensor | None = None,
        stem_types: torch.Tensor | None = None,
        stem_present: torch.Tensor | None = None,
        target_stem_type: torch.Tensor | None = None,
        stem_emb: torch.Tensor | None = None,
        stem_keep: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        cross_kv_caches: "list[BlockCrossKV] | None" = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        """Forward pass.

        tokens: [B, K, T]
        kv_caches: when None, training path (returns logits only).
                   when a list, returns (logits, new_caches).
        start_pos: positional offset for the tokens (used with KV cache).
        text_emb: [B, T_text, D] optional CLAP tag conditioning. T_text is 1 for a
            single pooled vector or N for a chunked long description (each chunk a
            pooled CLAP vector — see CLAPTextEncoder.encode_chunked).
        text_kv_mask: optional additive cross-attn mask [B, 1, 1, T_text] (0 keep,
            -inf pad) for a ragged chunked tag batch. None = attend to all of
            text_emb (the single-vector and B=1 paths).
        lyric_ids/lyric_mask: [B, L] phoneme ids + bool mask. When given (and
            lyric conditioning is enabled), the lyric encoder runs INSIDE this
            forward — so DDP syncs its grads. Used on the training path.
        lyric_emb/lyric_kv_mask: pre-encoded lyric sequence + additive mask (from
            encode_lyrics). Used on the generate path to encode once and reuse
            across decode steps. Takes precedence over lyric_ids.
        melody: [B, Tc, n_bins] chroma sequence. Encoded INSIDE this forward (DDP
            grad-sync) and added per-frame; the training path. ``melody=None`` with
            melody conditioning enabled adds the learned null (the dropped /
            unconditional state) so train and inference agree.
        melody_emb: pre-built full delayed-length melody term [B, seq, D] (from
            encode_melody_delayed). Used on the generate path to encode once and
            slice per decode step. Takes precedence over ``melody``.
        stem_tokens/stem_types/stem_present/target_stem_type: the stem-add training
            inputs — S accompaniment stems [B,S,K,T] + their stem-type ids [B,S] +
            a present mask [B,S] (0 = absent slot) + the target stem id [B]. Encoded
            + summed INSIDE this forward (DDP grad-sync) and added per-frame. On a
            normal (non-stem-add) batch these are passed zeroed with stem_keep=0 so
            the encoder still runs but contributes the null.
        stem_emb: pre-built full delayed-length stem term [B, seq, D] (from
            encode_stem_delayed). Used on the generate path. Takes precedence over
            ``stem_tokens``.
        lyric_keep / melody_keep / stem_keep: optional 0/1 scalar tensors — the train-time
            CFG-dropout gates. The encoders ALWAYS run and the contribution is
            multiplied by keep (0 -> exactly the unconditional state: zero lyric
            cond / the melody null), so torch.compile sees ONE graph per stream
            and every parameter participates in every step (which is what lets
            DDP run with find_unused_parameters=False). ``None`` = keep=1
            behavior — the inference paths never pass these.
        returns: logits [B, K, T, V] or (logits, new_caches)
        """
        B, K, T = tokens.shape
        assert K == self.cfg.n_codebooks
        assert start_pos + T <= self.cfg.max_seq_len, (
            f"seq pos {start_pos + T} > max_seq_len {self.cfg.max_seq_len} "
            f"(RoPE table size — bump GPTConfig.max_seq_len if you need longer)"
        )

        if (
            lyric_emb is None
            and lyric_ids is not None
            and self.cfg.use_lyric_conditioning
        ):
            lyric_emb, lyric_kv_mask = self.encode_lyrics(lyric_ids, lyric_mask)
        if lyric_emb is not None and lyric_keep is not None:
            # CFG gate: keep=0 zeroes the cond sequence, and zero K/V through
            # the bias-free cross-attn is exactly zero output — identical to
            # skipping the stream, but the encoder stays in the autograd graph.
            lyric_emb = lyric_emb * lyric_keep

        x = self.tok_embeds[0](tokens[:, 0])
        for k in range(1, K):
            x = x + self.tok_embeds[k](tokens[:, k])
        # Melody is time-aligned: add it to the per-frame input at the cb0 anchor
        # (delayed position p ↔ frame p). The slice inside _melody_add keeps the
        # prefill / single-step decode paths aligned, exactly like the RoPE slice.
        if self.cfg.use_melody_conditioning:
            x = x + self._melody_add(melody, melody_emb, B, T, start_pos,
                                     keep=melody_keep, input_pos=input_pos)
        # Stem-add conditioning: the summed accompaniment stems + target-type, added
        # at the same cb0 anchor as melody (delayed position p ↔ frame p). On the
        # training path stem_tokens are encoded here; on decode stem_emb is sliced.
        if self.cfg.use_stem_conditioning:
            x = x + self._stem_add(stem_tokens, stem_types, stem_present,
                                   target_stem_type, stem_emb, B, T, start_pos,
                                   keep=stem_keep, input_pos=input_pos)
        x = self.drop(x)

        # RoPE positions: the CUDA-graph decode path indexes by the TENSOR
        # input_pos (no python-int start_pos guard → one reusable compiled
        # graph); every other path uses the contiguous start_pos slice.
        if input_pos is not None:
            cos = self.rotary.cos_cached[input_pos]
            sin = self.rotary.sin_cached[input_pos]
        else:
            cos, sin = self.rotary(start_pos, T)

        new_caches: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(
                x, cos, sin, cache=cache, text_emb=text_emb, text_kv_mask=text_kv_mask,
                lyric_emb=lyric_emb, lyric_kv_mask=lyric_kv_mask,
                input_pos=input_pos, attn_mask=attn_mask,
                cross_kv=cross_kv_caches[i] if cross_kv_caches is not None else None,
            )
            new_caches.append(new_cache)
        x = self.ln_final(x)

        logits = (
            self.head(x)
            .view(B, T, self.cfg.n_codebooks, self.cfg.vocab_with_pad)
            .transpose(1, 2)  # [B, K, T, V]
        )
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
        text_kv_mask: torch.Tensor | None = None,
        text_kv_mask_neg: torch.Tensor | None = None,
        lyric_ids: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        lyric_ids_neg: torch.Tensor | None = None,
        lyric_mask_neg: torch.Tensor | None = None,
        lyric_cfg_scale: float | None = None,
        melody: torch.Tensor | None = None,
        melody_cfg_scale: float | None = None,
        stem_tokens: torch.Tensor | None = None,
        stem_types: torch.Tensor | None = None,
        stem_present: torch.Tensor | None = None,
        target_stem_type: torch.Tensor | None = None,
        stem_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Continue a prompt, or generate unconditionally when prompt is None.

        prompt: [K, T_prompt] or [B, K, T_prompt], or None for unconditional
        text_emb: [B, T_text, D] optional tag conditioning (from CLAPTextEncoder).
            T_text=1 for a single pooled vector, or N for a chunked long
            description (encode_chunked).
        text_kv_mask / text_kv_mask_neg: additive cross-attn masks [B,1,1,T_text]
            for ragged chunked tags (the positive and CFG-baseline streams). None
            when tags are a single vector or unpadded (B=1).
        melody: [B, num_new_frames, n_bins] chroma for the frames being generated.
            Encoded ONCE into a full delayed-length tensor (placed at the new-frame
            positions, learned-null elsewhere) and reused across decode steps. The
            CFG baseline uses the learned null, so melody is guided like the other
            axes. Length should match num_new_frames (the cover length).
        melody_cfg_scale: when set (and melody is given), guide the melody axis with
            its OWN scale via an extra composed stream — analogous to
            lyric_cfg_scale, nesting after it (tags → lyrics → melody). When None,
            melody is guided jointly with the rest by cfg_scale (one fewer pass).
        stem_tokens/stem_types/stem_present/target_stem_type: the /addstem inputs —
            S accompaniment stems [B,S,K,T] + their stem-type ids [B,S] + a present
            mask [B,S] + the target stem id [B]. Encoded ONCE into a delayed-length
            term (placed at the new-frame positions, learned-null elsewhere) and
            reused across decode steps; the CFG baseline is the learned null.
        stem_cfg_scale: when set (and stems are given), guide the stem axis with its
            OWN scale via an extra composed stream, nesting last (tags → lyrics →
            melody → stem). When None, the stem axis is guided jointly by cfg_scale.
        lyric_ids/lyric_mask: [B, L] phoneme ids + bool mask for the lyric
            conditioning stream. Encoded once and reused across decode steps.
        lyric_ids_neg/lyric_mask_neg: optional *negative* lyric stream for the
            CFG baseline (default baseline drops lyrics entirely).
        lyric_cfg_scale: when set (and lyric conditioning is active), guide the
            lyric axis with its OWN scale via composed guidance — runs a third
            "tags-only, lyrics-dropped" stream and blends
            logits = base + cfg_scale*(tags_only - base)
                          + lyric_cfg_scale*(cond - tags_only).
            This lets a user push lyric intelligibility harder than tag adherence.
            When None, lyrics are guided jointly with tags by cfg_scale (cheaper,
            one fewer forward pass).
        cfg_scale: classifier-free guidance scale. 1.0 = no guidance (single
            forward pass). >1.0 = run an additional baseline forward pass and
            blend logits = base + cfg_scale * (cond - base). Active when any of
            text_emb / text_emb_neg / lyric conditioning is set.
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

        When prompt is None a single random seed frame is used internally — the
        from-scratch path the InferenceEngine relies on (a fresh random seed per
        call). Uses KV cache for efficient autoregressive decoding.
        """
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
            stem_tokens=stem_tokens, stem_types=stem_types,
            stem_present=stem_present, target_stem_type=target_stem_type,
            stem_cfg_scale=stem_cfg_scale,
        ))
        new = (
            torch.cat(chunks, dim=-1) if chunks
            else prompt.new_zeros((prompt.shape[0], prompt.shape[1], 0))
        )
        out = torch.cat([prompt, new], dim=-1)
        return out.squeeze(0) if squeeze_batch else out

    def _resolve_prompt(
        self, prompt: torch.Tensor | None, batch_size: int = 1
    ) -> tuple[torch.Tensor, bool]:
        """Resolve generate()'s prompt arg to a batched [B, K, T_prompt] tensor.

        prompt=None -> a random DAC seed column (the from-scratch path); with
        batch_size>1, B INDEPENDENT random seed columns [B, K, 1] (each row a
        distinct from-scratch clip — the batched-generation path). A 2D [K, T]
        prompt is unsqueezed to [1, K, T] and squeeze_batch=True is returned so the
        caller can squeeze the result back to 2D. batch_size is only consulted when
        prompt is None; an explicit prompt carries its own batch dim. (batch_size=1
        reproduces the original [K, 1] seed exactly, so the B=1 path is unchanged.)
        """
        if prompt is None:
            device = next(self.parameters()).device
            shape = (
                (self.cfg.n_codebooks, 1) if batch_size == 1
                else (batch_size, self.cfg.n_codebooks, 1)
            )
            prompt = torch.randint(
                0, self.cfg.vocab_per_codebook, shape, device=device,
            )
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
        stem_tokens: torch.Tensor | None = None,
        stem_types: torch.Tensor | None = None,
        stem_present: torch.Tensor | None = None,
        target_stem_type: torch.Tensor | None = None,
        stem_cfg_scale: float | None = None,
        emit_every: int = 256,
        first_emit: int | None = None,
    ) -> Iterator[torch.Tensor]:
        """The shared autoregressive decode loop behind generate().

        Yields the NEW frames (never the prompt) as un-delayed [B, K, n]
        LongTensor chunks, in generation order, emitting once >= emit_every new
        frames have become fully known across all K codebooks (the first chunk
        after `first_emit` frames — smaller by default, for a faster
        time-to-first-audio when streaming). generate() concatenates these onto the
        prompt to reproduce its full result bit-identically — this IS the loop, so
        the per-step RNG order is shared and there is no train/inference drift.
        `prompt` must be a resolved, batched [B, K, T_prompt] tensor (see
        _resolve_prompt). emit_every is irrelevant to generate() (it concatenates
        every chunk regardless); it only sets the streaming cadence.

        A new frame f is fully known once the delayed decode step p = f + (K-1)
        completes — codebook k of frame f lands at delayed position f + k.
        """
        B, K, T_prompt = prompt.shape
        assert K == self.cfg.n_codebooks
        if first_emit is None:
            first_emit = emit_every

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

        def _emit(s: int, e: int) -> torch.Tensor:
            # Un-delay new frames [s, e): codebook k's value for frame f sits at
            # delayed position f + k (the inverse of apply_delay; cf. revert_delay).
            return torch.stack([tokens[:, k, s + k:e + k] for k in range(K)], dim=1)

        was_training = self.training
        self.eval()
        try:
            head_dim = self.cfg.d_model // self.cfg.n_heads
            dtype = next(self.parameters()).dtype

            def _make_caches() -> list[StaticLayerKVCache]:
                caches = []
                for _ in range(self.cfg.n_layers):
                    k = torch.zeros(B, self.cfg.n_heads, T_delay, head_dim, device=device, dtype=dtype)
                    v = torch.zeros(B, self.cfg.n_heads, T_delay, head_dim, device=device, dtype=dtype)
                    # These KV buffers persist across decode steps and are mutated
                    # in place — mark them as static-address so torch.compile's
                    # CUDA graphs can own them (without this, the in-place cache
                    # mutation makes cudagraphs fall back to plain compiled kernels).
                    if device.type == "cuda":
                        torch._dynamo.mark_static_address(k)
                        torch._dynamo.mark_static_address(v)
                    caches.append(StaticLayerKVCache(k, v))
                return caches

            # Per-stage cross-attention K/V caches (one BlockCrossKV per layer).
            # Filled lazily on the eager prefill forward, then reused on every decode
            # step so kv_proj(cond) is never recomputed. None when there's no cross
            # conditioning, or when the cache is disabled (A/B benchmark handle).
            has_cross = (
                self.cfg.use_text_conditioning or self.cfg.use_lyric_conditioning
            )

            def _make_cross_kv() -> "list[BlockCrossKV] | None":
                if not (has_cross and self._use_cross_kv_cache):
                    return None
                return [BlockCrossKV() for _ in range(self.cfg.n_layers)]

            # Encode lyric streams ONCE (the encoder is ~100M params — encoding
            # per decode step would dominate cost). Reused across all steps.
            lyric_pos = lyric_kv_pos = None
            lyric_neg = lyric_kv_neg = None
            if self.cfg.use_lyric_conditioning and lyric_ids is not None:
                lyric_pos, lyric_kv_pos = self.encode_lyrics(lyric_ids, lyric_mask)
            if self.cfg.use_lyric_conditioning and lyric_ids_neg is not None:
                lyric_neg, lyric_kv_neg = self.encode_lyrics(lyric_ids_neg, lyric_mask_neg)

            # Encode the (fully-known) melody ONCE into a full delayed-length term,
            # placed at the new-frame positions (offset by the prompt). The CFG
            # baseline is the learned null. Both are reused across decode steps.
            mel_pos = mel_neg = None
            has_melody = self.cfg.use_melody_conditioning and melody is not None
            if self.cfg.use_melody_conditioning:
                mel_neg = self.melody_encoder.null_emb(B, T_delay)  # [B, T_delay, D]
                mel_pos = (
                    self.encode_melody_delayed(melody, T_delay, offset=T_prompt)
                    if melody is not None else mel_neg
                )

            # Encode the accompaniment stems ONCE into a delayed-length term (placed
            # at the new-frame positions), reused across decode steps. The CFG
            # baseline is the learned null (no target-type, like the other axes drop
            # their cond entirely). See encode_stem_delayed / _stem_add.
            stem_pos = stem_neg = None
            has_stem = self.cfg.use_stem_conditioning and stem_tokens is not None
            if self.cfg.use_stem_conditioning:
                stem_neg = self.stem_encoder.null_emb(B, T_delay)  # [B, T_delay, D]
                stem_pos = (
                    self.encode_stem_delayed(
                        stem_tokens, stem_types, stem_present, target_stem_type,
                        T_delay, offset=T_prompt,
                    ) if stem_tokens is not None else stem_neg
                )

            has_cond = any(
                v is not None for v in (text_emb, text_emb_neg, lyric_pos, lyric_neg)
            ) or has_melody or has_stem
            composed_lyric = (
                lyric_cfg_scale is not None and lyric_cfg_scale != 1.0
                and lyric_pos is not None
            )
            composed_melody = (
                melody_cfg_scale is not None and melody_cfg_scale != 1.0
                and has_melody
            )
            composed_stem = (
                stem_cfg_scale is not None and stem_cfg_scale != 1.0
                and has_stem
            )
            use_cfg = (
                (cfg_scale != 1.0 and has_cond)
                or composed_lyric or composed_melody or composed_stem
            )

            # Guidance as an ordered list of stages from the CFG baseline to the
            # fully-conditioned state. Each consecutive pair contributes
            # scale_i * (stage_{i+1} - stage_i); axes WITHOUT their own scale are
            # folded into the cfg_scale (tags) step so they're still guided with no
            # extra forward pass. Nesting order is tags → lyrics → melody → stem.
            #   logits = stages[0] + Σ scales[i] * (stages[i+1] - stages[i])
            # Each stage is a full conditioning state {text, lemb, lkv, mel, stem}
            # with its own KV cache.
            full = {"text": text_emb, "tkv": text_kv_mask,
                    "lemb": lyric_pos, "lkv": lyric_kv_pos, "mel": mel_pos,
                    "stem": stem_pos}
            if not use_cfg:
                stages = [full]
                scales: list[float] = []
            else:
                off = {"text": text_emb_neg, "tkv": text_kv_mask_neg,
                       "lemb": lyric_neg, "lkv": lyric_kv_neg, "mel": mel_neg,
                       "stem": stem_neg}
                # The cfg (tags) step turns on everything that isn't separately
                # composed; composed axes start off and are switched on later.
                tags_on = dict(full)
                if composed_lyric:
                    tags_on["lemb"], tags_on["lkv"] = lyric_neg, lyric_kv_neg
                if composed_melody:
                    tags_on["mel"] = mel_neg
                if composed_stem:
                    tags_on["stem"] = stem_neg
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
                if composed_stem:
                    stem_on = dict(stages[-1])
                    stem_on["stem"] = stem_pos
                    stages.append(stem_on)
                    scales.append(stem_cfg_scale)
            for s in stages:
                s["caches"] = _make_caches()
                s["cross_kv"] = _make_cross_kv()

            # Mutable holder so a runtime compile/cudagraph failure can disable the
            # compiled path mid-generation and fall back to eager (identical
            # results, just slower) instead of failing the whole request.
            # Mutable holder so a runtime compile/cudagraph failure can disable the
            # compiled path mid-generation and fall back to eager (identical
            # results, just slower) instead of failing the whole request.
            compiled_box = [self._compiled_forward]

            def _run(inp, start, input_pos=None, attn_mask=None):
                # Decode steps (input_pos given) go through the compiled forward
                # when available — that's where CUDA graphs remove the per-step
                # launch overhead. Prefill stays eager (it runs once).
                use_compiled = input_pos is not None and compiled_box[0] is not None
                fwd = compiled_box[0] if use_compiled else self.forward
                for s in stages:
                    try:
                        out, s["caches"] = fwd(
                            inp, kv_caches=s["caches"], start_pos=start,
                            text_emb=s["text"], text_kv_mask=s["tkv"],
                            lyric_emb=s["lemb"], lyric_kv_mask=s["lkv"],
                            melody_emb=s["mel"], stem_emb=s["stem"],
                            input_pos=input_pos, attn_mask=attn_mask,
                            cross_kv_caches=s["cross_kv"],
                        )
                    except Exception as e:  # noqa: BLE001 — compile/cudagraph fallback
                        if not use_compiled:
                            raise
                        compiled_box[0] = None
                        fwd = self.forward
                        print(f"[generate] compiled decode failed ({e!r}); "
                              f"falling back to eager for the rest of this run")
                        out, s["caches"] = fwd(
                            inp, kv_caches=s["caches"], start_pos=start,
                            text_emb=s["text"], text_kv_mask=s["tkv"],
                            lyric_emb=s["lemb"], lyric_kv_mask=s["lkv"],
                            melody_emb=s["mel"], stem_emb=s["stem"],
                            input_pos=input_pos, attn_mask=attn_mask,
                            cross_kv_caches=s["cross_kv"],
                        )
                    # Clone: under CUDA graphs the compiled forward's output is a
                    # reused static buffer, so two sequential CFG-stage calls would
                    # otherwise alias — _combine must see each stage's own logits.
                    s["logits"] = out.clone()

            def _combine():
                out = stages[0]["logits"]
                for i, sc in enumerate(scales):
                    out = out + sc * (stages[i + 1]["logits"] - stages[i]["logits"])
                return out

            # prefill: run full prompt through transformer (eager, legacy path)
            prefill_len = max(1, T_prompt)
            prefill_inp = tokens[:, :, :prefill_len]
            _run(prefill_inp, 0)
            logits = _combine()
            # The decode loop may capture a CUDA graph that reads the KV cache the
            # eager prefill just wrote. Sync so those writes are visible before the
            # capture stream reads them (a missing barrier here surfaced as
            # intermittent NaN logits / device-side asserts at capture).
            if device.type == "cuda" and compiled_box[0] is not None:
                torch.cuda.synchronize()

            # decode: one position at a time using cached K/V, yielding new frames
            # as soon as they are fully known across all K codebooks. Each step
            # passes a TENSOR position + a static-shape additive mask over the full
            # cache buffer (no python-int position, no growing slice) so the
            # compiled decode graph is captured once and replayed every step.
            # The mask is a FINITE float (not bool/-inf): a bool mask becomes -inf,
            # and flash-attention tiles keys into blocks — an all-masked early
            # block then has block-max -inf and the online-softmax rescale yields
            # NaN (fp16, CUDA). A large finite negative avoids that and still zeros
            # the masked weights (exp(-1e4) == 0). cf. the cached-decode test.
            # The fixed-shape input_pos path exists to enable torch.compile/CUDA
            # graphs; it's only worth its extra (full-buffer) attention when we're
            # actually compiling. Eager (no compiled forward) uses the original
            # growing-slice legacy path — faster, and the proven fallback.
            use_input_pos = compiled_box[0] is not None
            emitted = 0  # count of new frames already yielded
            threshold = first_emit
            for p in range(prefill_len, T_delay):
                if p > prefill_len:
                    if use_input_pos:
                        pos = p - 1
                        input_pos = torch.tensor([pos], device=device, dtype=torch.long)
                        attn_mask = torch.zeros(T_delay, dtype=dtype, device=device)
                        attn_mask[pos + 1:] = -1e4
                        attn_mask = attn_mask.view(1, 1, 1, T_delay)
                        _run(tokens[:, :, pos:pos + 1], 0,
                             input_pos=input_pos, attn_mask=attn_mask)
                    else:
                        _run(tokens[:, :, p - 1:p], p - 1)
                    logits = _combine()

                step_logits = logits[:, :, -1, :].clone()  # [B, K, V]
                # Mask every control id (pad, plus the FIM <SUF>/<MID> sentinels
                # when use_fim) so generation only ever emits real DAC tokens —
                # the FIM middle must never contain a sentinel.
                step_logits[..., self.cfg.vocab_per_codebook:] = float("-inf")

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

                # Emit any new frames now fully known: frame f completes at p=f+K-1.
                n_complete = min(p - (K - 1), T_total - 1) - T_prompt + 1
                if n_complete - emitted >= threshold:
                    yield _emit(T_prompt + emitted, T_prompt + n_complete)
                    emitted = n_complete
                    threshold = emit_every
            # flush the remaining tail
            if emitted < num_new_frames:
                yield _emit(T_prompt + emitted, T_total)
        finally:
            if was_training:
                self.train()


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
