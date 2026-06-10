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
    max_lyric_len: int = 256
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

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [B, T, D] audio hidden states, cond: [B, T_cond, D] text embeddings.

        kv_mask: optional additive attention mask broadcastable to
        [B, n_heads, T, T_cond] (0 keep, -inf drop) — used to mask padded
        positions in a variable-length lyric sequence. Callers must guarantee
        each query row has at least one un-masked key (a fully -inf row makes
        SDPA's softmax produce NaN); the lyric path ensures this by always
        keeping the BOS phoneme valid.
        """
        B, T, D = x.shape
        T_c = cond.shape[1]
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(cond).split(D, dim=-1)
        k = k.view(B, T_c, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_c, self.n_heads, self.head_dim).transpose(1, 2)
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
        lyric_emb: torch.Tensor | None,
        lyric_kv_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Training-only path (no KV cache). Wrapped by gradient checkpointing.
        attn_out, _ = self.attn(self.ln1(x), cos, sin, cache=None)
        x = x + attn_out
        if self.has_cross_attn and text_emb is not None:
            x = x + self.cross_attn(self.ln_cross(x), text_emb)
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
        lyric_emb: torch.Tensor | None = None,
        lyric_kv_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        if cache is None and self.training and self.use_gradient_checkpointing:
            x = checkpoint(
                self._body, x, cos, sin, text_emb, lyric_emb, lyric_kv_mask,
                use_reentrant=False,
            )
            return x, None
        attn_out, new_cache = self.attn(self.ln1(x), cos, sin, cache=cache)
        x = x + attn_out
        if self.has_cross_attn and text_emb is not None:
            x = x + self.cross_attn(self.ln_cross(x), text_emb)
        if self.has_lyric_attn and lyric_emb is not None:
            x = x + self.lyric_attn(self.ln_lyric(x), lyric_emb, kv_mask=lyric_kv_mask)
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
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.RMSNorm(cfg.d_model)
        # The K per-codebook output heads, fused into one Linear (one matmul per
        # forward instead of K kernel launches); the forward's view+transpose
        # recovers the per-codebook [B, K, T, V] layout.
        self.head = nn.Linear(
            cfg.d_model, cfg.n_codebooks * cfg.vocab_with_pad, bias=False
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
            return melody_emb[:, start_pos:start_pos + T, :]
        if melody is not None:
            enc = self.melody_encoder.encode_melody(melody)  # [B, Tc, D]
            Tc = enc.shape[1]
            if Tc < T:
                enc = torch.cat([enc, self.melody_encoder.null_emb(B, T - Tc)], dim=1)
            elif Tc > T:
                enc = enc[:, :T, :]
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

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: list[KVCache] | None = None,
        start_pos: int = 0,
        text_emb: torch.Tensor | None = None,
        lyric_ids: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        lyric_emb: torch.Tensor | None = None,
        lyric_kv_mask: torch.Tensor | None = None,
        melody: torch.Tensor | None = None,
        melody_emb: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        """Forward pass.

        tokens: [B, K, T]
        kv_caches: when None, training path (returns logits only).
                   when a list, returns (logits, new_caches).
        start_pos: positional offset for the tokens (used with KV cache).
        text_emb: [B, T_text, D] optional (pooled CLAP) tag conditioning.
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

        x = self.tok_embeds[0](tokens[:, 0])
        for k in range(1, K):
            x = x + self.tok_embeds[k](tokens[:, k])
        # Melody is time-aligned: add it to the per-frame input at the cb0 anchor
        # (delayed position p ↔ frame p). The slice inside _melody_add keeps the
        # prefill / single-step decode paths aligned, exactly like the RoPE slice.
        if self.cfg.use_melody_conditioning:
            x = x + self._melody_add(melody, melody_emb, B, T, start_pos)
        x = self.drop(x)

        cos, sin = self.rotary(start_pos, T)

        new_caches: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(
                x, cos, sin, cache=cache, text_emb=text_emb,
                lyric_emb=lyric_emb, lyric_kv_mask=lyric_kv_mask,
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
        lyric_ids: torch.Tensor | None = None,
        lyric_mask: torch.Tensor | None = None,
        lyric_ids_neg: torch.Tensor | None = None,
        lyric_mask_neg: torch.Tensor | None = None,
        lyric_cfg_scale: float | None = None,
        melody: torch.Tensor | None = None,
        melody_cfg_scale: float | None = None,
    ) -> torch.Tensor:
        """Continue a prompt, or generate unconditionally when prompt is None.

        prompt: [K, T_prompt] or [B, K, T_prompt], or None for unconditional
        text_emb: [B, 1, D] optional text conditioning (from CLAPTextEncoder)
        melody: [B, num_new_frames, n_bins] chroma for the frames being generated.
            Encoded ONCE into a full delayed-length tensor (placed at the new-frame
            positions, learned-null elsewhere) and reused across decode steps. The
            CFG baseline uses the learned null, so melody is guided like the other
            axes. Length should match num_new_frames (the cover length).
        melody_cfg_scale: when set (and melody is given), guide the melody axis with
            its OWN scale via an extra composed stream — analogous to
            lyric_cfg_scale, nesting after it (tags → lyrics → melody). When None,
            melody is guided jointly with the rest by cfg_scale (one fewer pass).
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

            has_cond = any(
                v is not None for v in (text_emb, text_emb_neg, lyric_pos, lyric_neg)
            ) or has_melody
            composed_lyric = (
                lyric_cfg_scale is not None and lyric_cfg_scale != 1.0
                and lyric_pos is not None
            )
            composed_melody = (
                melody_cfg_scale is not None and melody_cfg_scale != 1.0
                and has_melody
            )
            use_cfg = (cfg_scale != 1.0 and has_cond) or composed_lyric or composed_melody

            # Guidance as an ordered list of stages from the CFG baseline to the
            # fully-conditioned state. Each consecutive pair contributes
            # scale_i * (stage_{i+1} - stage_i); axes WITHOUT their own scale are
            # folded into the cfg_scale (tags) step so they're still guided with no
            # extra forward pass. Nesting order is tags → lyrics → melody.
            #   logits = stages[0] + Σ scales[i] * (stages[i+1] - stages[i])
            # Each stage is a full conditioning state {text, lemb, lkv, mel} with
            # its own KV cache.
            full = {"text": text_emb, "lemb": lyric_pos, "lkv": lyric_kv_pos, "mel": mel_pos}
            if not use_cfg:
                stages = [full]
                scales: list[float] = []
            else:
                off = {"text": text_emb_neg, "lemb": lyric_neg, "lkv": lyric_kv_neg, "mel": mel_neg}
                # The cfg (tags) step turns on everything that isn't separately
                # composed; composed axes start off and are switched on later.
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
                s["caches"] = _make_caches()

            def _run(inp: torch.Tensor, start: int):
                for s in stages:
                    out, s["caches"] = self.forward(
                        inp, kv_caches=s["caches"], start_pos=start,
                        text_emb=s["text"], lyric_emb=s["lemb"], lyric_kv_mask=s["lkv"],
                        melody_emb=s["mel"],
                    )
                    s["logits"] = out

            def _combine():
                out = stages[0]["logits"]
                for i, sc in enumerate(scales):
                    out = out + sc * (stages[i + 1]["logits"] - stages[i]["logits"])
                return out

            # prefill: run full prompt through transformer
            prefill_len = max(1, T_prompt)
            prefill_inp = tokens[:, :, :prefill_len]
            _run(prefill_inp, 0)
            logits = _combine()

            # decode: one position at a time using cached K/V
            for p in range(prefill_len, T_delay):
                if p > prefill_len:
                    inp = tokens[:, :, p - 1:p]
                    _run(inp, p - 1)
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
