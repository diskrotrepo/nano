"""Chromagram melody encoder for time-aligned cover conditioning.

Unlike the lyric stream (a ragged phoneme sequence the decoder cross-attends to,
learning its own alignment), the melody is a **dense, time-aligned** signal: one
12-bin chroma vector per audio frame, already at the DAC frame rate. So instead of
cross-attention it is conditioned **additively** — this module projects the chroma
sequence ``[B, T, 12]`` to ``[B, T, d_model]`` and ``NanoAudioGPT`` adds it to the
per-frame token-sum embedding at the cb0 anchor (delayed position p ↔ frame p). A
small 1-D conv stack gives a little temporal context so the projection isn't purely
pointwise; the house style otherwise mirrors ``LyricEncoder`` (RMSNorm, std=0.02).

A learned **null** embedding stands in for "melody dropped" (classifier-free
guidance) and for the unconditional baseline at inference. It must be learned, not
zeros: an all-zero chroma vector is already a valid input (a silent frame), so
zeros can't mean "no melody conditioning". This is the additive analog of the
lyric path passing ``lyric_emb=None`` to skip its cross-attention entirely.

Lives INSIDE ``NanoAudioGPT`` (like ``LyricEncoder``) so DDP syncs its grads and it
saves/restores with the model state_dict — no sidecar key.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CHROMA = 12


class MelodyEncoder(nn.Module):
    """Encode a chroma sequence ``[B, T, n_bins]`` -> ``[B, T, d_model]``.

    in_proj (n_bins->d_model) -> GELU -> ``n_conv`` depthwise-ish Conv1d(k=3) blocks
    with residual + RMSNorm -> out_proj. The conv blocks are causal-agnostic
    (bidirectional, ``padding=1``) — the whole melody is known up front, exactly
    like the bidirectional ``LyricEncoder``. ``forward`` returns the per-frame
    embedding indexed by absolute frame; the decoder slices it by ``start_pos`` so
    the same tensor serves the full-length training pass and the KV-cached
    single-step decode.
    """

    def __init__(
        self,
        d_model: int,
        n_bins: int = N_CHROMA,
        n_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.in_proj = nn.Linear(n_bins, d_model, bias=False)
        self.convs = nn.ModuleList(
            [nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
             for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln_final = nn.RMSNorm(d_model)
        # Learned "melody dropped / unconditional" embedding (see module docstring).
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.apply(self._init_weights)
        # Init the null AFTER apply (apply doesn't touch bare Parameters) so it
        # starts as a small learnable vector rather than exactly zero.
        with torch.no_grad():
            self.null.normal_(std=0.02)
        # Zero-gamma gate: with ln_final's gain at zero the encoded melody is
        # exactly zero at init — an additive no-op for the decoder — and fades
        # in smoothly as the gain learns. Zeroing out_proj would NOT gate this:
        # RMSNorm rescales any nonzero input back to unit RMS. (Same stability
        # hardening as the zeroed cross-attn out-projs in NanoAudioGPT.)
        nn.init.zeros_(self.ln_final.weight)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode_melody(self, chroma: torch.Tensor) -> torch.Tensor:
        """chroma: ``[B, T, n_bins]`` float -> ``[B, T, d_model]``."""
        x = self.in_proj(chroma)  # [B, T, D]
        x = F.gelu(x)
        # Conv1d wants [B, D, T]; transpose around the residual conv stack.
        h = x.transpose(1, 2)  # [B, D, T]
        for conv, norm in zip(self.convs, self.norms):
            y = conv(h)
            h = h + y
            # RMSNorm over the channel dim — transpose back, norm, transpose.
            h = norm(h.transpose(1, 2)).transpose(1, 2)
        x = h.transpose(1, 2)  # [B, T, D]
        x = self.drop(x)
        x = self.out_proj(x)
        return self.ln_final(x)

    def null_emb(self, batch: int, length: int) -> torch.Tensor:
        """The learned null broadcast to ``[batch, length, d_model]`` — the
        melody-dropped / unconditional baseline."""
        return self.null.expand(batch, length, -1)
