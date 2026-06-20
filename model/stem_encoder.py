"""Token-domain stem encoder for generative stem conditioning (the /addstem path).

The generative inverse of the Demucs ``/stem`` removal: given a SOURCE stem (e.g.
a drum loop) the model generates a full mix built around it. Unlike melody (a
lossy 12-bin chroma) this conditions on the source stem's FULL SpectroStream
tokens — same fidelity the model itself works in — so no audio information is
thrown away (the user picked token-domain over a per-frame feature for exactly
this reason).

Like melody, the source stem is dense + frame-aligned with the target, so it is
conditioned **additively**: this module embeds the source-stem tokens ``[B, K, T]``
(one embedding table per codebook, summed — mirroring the decoder's own input
path), runs a small bidirectional Conv1d stack for temporal context, adds a
learned **stem-type** embedding (which stem is being conditioned on:
drums/bass/vocals/other), and ``NanoAudioGPT`` adds the result to the per-frame
token-sum input at the cb0 anchor (delayed position p <-> frame p), exactly like
``MelodyEncoder``.

A learned **null** stands in for "stem dropped" (classifier-free guidance) and the
unconditional baseline — it must be learned, not zeros (an all-pad source-stem is
a valid input). Lives INSIDE ``NanoAudioGPT`` so DDP syncs its grads and it
saves/restores with the model state_dict.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Demucs' 4 stems; the order is the stem-type id and is checkpoint-baked.
STEM_TYPES: tuple[str, ...] = ("drums", "bass", "vocals", "other")
N_STEM_TYPES = len(STEM_TYPES)
STEM_TYPE_TO_ID: dict[str, int] = {s: i for i, s in enumerate(STEM_TYPES)}


class StemEncoder(nn.Module):
    """Encode source-stem tokens ``[B, K, T]`` (+ a stem-type id) -> ``[B, T, d_model]``.

    Per-codebook embeddings (summed) -> GELU -> ``n_layers`` residual Conv1d(k=3)
    blocks with RMSNorm -> out_proj, plus a learned per-stem-type embedding. The
    convs are bidirectional (the whole source stem is known up front), like
    ``MelodyEncoder``/``LyricEncoder``. ``ln_final`` is zero-gamma-gated so the
    encoded stem is an additive no-op at init and fades in as it learns.
    """

    def __init__(
        self,
        d_model: int,
        n_codebooks: int,
        vocab_with_pad: int,
        n_layers: int = 2,
        n_stem_types: int = N_STEM_TYPES,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        # One embedding table per codebook (mirrors NanoAudioGPT.tok_embeds), so the
        # source-stem tokens are embedded the same way the decoder embeds its own.
        self.tok_embeds = nn.ModuleList(
            [nn.Embedding(vocab_with_pad, d_model) for _ in range(n_codebooks)]
        )
        self.stem_type_emb = nn.Embedding(n_stem_types, d_model)
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
             for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln_final = nn.RMSNorm(d_model)
        # Learned "stem dropped / unconditional" embedding (see module docstring).
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)
        with torch.no_grad():
            self.null.normal_(std=0.02)
        # Zero-gamma gate: encoded stem is exactly zero at init (additive no-op) and
        # fades in as the gain learns — same hardening as MelodyEncoder.
        nn.init.zeros_(self.ln_final.weight)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def encode_stem(self, tokens: torch.Tensor, stem_type: torch.Tensor) -> torch.Tensor:
        """tokens: ``[B, K, T]`` long source-stem codes; stem_type: ``[B]`` long ->
        ``[B, T, d_model]``."""
        x = self.tok_embeds[0](tokens[:, 0])
        for k in range(1, self.n_codebooks):
            x = x + self.tok_embeds[k](tokens[:, k])  # [B, T, D]
        x = x + self.stem_type_emb(stem_type).unsqueeze(1)  # broadcast per-frame
        x = F.gelu(x)
        h = x.transpose(1, 2)  # [B, D, T]
        for conv, norm in zip(self.convs, self.norms):
            h = h + conv(h)
            h = norm(h.transpose(1, 2)).transpose(1, 2)
        x = self.drop(h.transpose(1, 2))  # [B, T, D]
        return self.ln_final(self.out_proj(x))

    def null_emb(self, batch: int, length: int) -> torch.Tensor:
        """The learned null broadcast to ``[batch, length, d_model]`` — the
        stem-dropped / unconditional baseline."""
        return self.null.expand(batch, length, -1)
