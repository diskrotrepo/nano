"""Tests for fill-in-the-middle (infill) support.

Guards the two contracts that, if broken, silently corrupt a v8 FIM run:
- ``fim_reorder_batch`` produces the canonical ``prefix <SUF> suffix <MID>
  middle`` layout (and co-reorders melody with zeroed sentinel frames), keeping
  the [B,K,T] shape so it slots straight into the existing delay/train path.
- The training reorder and the inference-side ``build_fim_prompt`` build a
  byte-identical ``prefix <SUF> suffix <MID>`` prefix (mirrors
  test_structure_markers' train==inference guard for the lyric stream).
- ``generate`` never emits a control id (pad / sentinels) into the output.
"""
from __future__ import annotations

import random

import torch

from model.fim import build_fim_prompt, fim_reorder_batch
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _ramp_codes(B: int, K: int, T: int) -> torch.Tensor:
    """codes[b,k,t] = t, so any column reveals its original frame index."""
    return torch.arange(T)[None, None, :].expand(B, K, T).contiguous()


def _locate_sentinels(row: torch.Tensor, suf_id: int, mid_id: int) -> tuple[int, int]:
    """Columns (frames) where every codebook holds the sentinel id."""
    suf_cols = [t for t in range(row.shape[1]) if bool((row[:, t] == suf_id).all())]
    mid_cols = [t for t in range(row.shape[1]) if bool((row[:, t] == mid_id).all())]
    assert len(suf_cols) == 1, f"expected exactly one <SUF> frame, got {suf_cols}"
    assert len(mid_cols) == 1, f"expected exactly one <MID> frame, got {mid_cols}"
    return suf_cols[0], mid_cols[0]


def test_reorder_preserves_shape_and_layout():
    B, K, T = 4, 9, 200
    cfg = GPTConfig(use_fim=True)
    codes = _ramp_codes(B, K, T)
    melody = torch.arange(T, dtype=torch.float32)[None, :, None].expand(B, T, 12).contiguous()
    out, mel = fim_reorder_batch(codes, melody, cfg.suf_id, cfg.mid_id, random.Random(0))

    assert out.shape == (B, K, T)
    assert mel.shape == (B, T, 12)

    for b in range(B):
        row = out[b]
        suf_pos, mid_pos = _locate_sentinels(row, cfg.suf_id, cfg.mid_id)
        assert suf_pos < mid_pos, "<SUF> must precede <MID>"

        a = suf_pos  # prefix length
        prefix = row[0, :suf_pos]
        suffix = row[0, suf_pos + 1:mid_pos]
        middle = row[0, mid_pos + 1:]

        # prefix is the original head [0, a); suffix is the original tail ending
        # at T-1; middle is the original [a, a+len) — i.e. P, S, M in order with
        # at most the last 2 middle frames dropped to hold length T.
        assert torch.equal(prefix, torch.arange(a))
        assert int(suffix[-1]) == T - 1
        assert torch.equal(suffix, torch.arange(int(suffix[0]), T))
        assert torch.equal(middle, torch.arange(a, a + middle.shape[0]))

        # Melody co-reordered identically, with zeros at the two sentinel frames.
        assert bool((mel[b, suf_pos] == 0).all())
        assert bool((mel[b, mid_pos] == 0).all())
        assert torch.equal(mel[b, :suf_pos, 0], prefix.float())
        assert torch.equal(mel[b, suf_pos + 1:mid_pos, 0], suffix.float())


def test_reorder_without_melody():
    cfg = GPTConfig(use_fim=True)
    codes = _ramp_codes(2, 9, 128)
    out, mel = fim_reorder_batch(codes, None, cfg.suf_id, cfg.mid_id, random.Random(1))
    assert out.shape == codes.shape
    assert mel is None


def test_train_inference_layout_identical():
    """The reorder's `prefix <SUF> suffix <MID>` prefix must equal what
    build_fim_prompt produces from the same prefix/suffix — so an infill request
    feeds the model the exact byte layout it trained on."""
    cfg = GPTConfig(use_fim=True)
    B, K, T = 3, 9, 150
    codes = _ramp_codes(B, K, T)
    out, _ = fim_reorder_batch(codes, None, cfg.suf_id, cfg.mid_id, random.Random(7))
    for b in range(B):
        row = out[b]
        suf_pos, mid_pos = _locate_sentinels(row, cfg.suf_id, cfg.mid_id)
        # Recover P and S from the reordered row and rebuild the prompt.
        P = row[:, :suf_pos]
        S = row[:, suf_pos + 1:mid_pos]
        prompt = build_fim_prompt(P, S, cfg.suf_id, cfg.mid_id)
        # The training row up to and including <MID> IS the inference prompt.
        assert torch.equal(prompt, row[:, :mid_pos + 1])


def _tiny_fim_model() -> NanoAudioGPT:
    cfg = GPTConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=256,
        use_fim=True, use_gradient_checkpointing=False,
    )
    return NanoAudioGPT(cfg).eval()


def test_forward_accepts_reordered_batch():
    model = _tiny_fim_model()
    cfg = model.cfg
    codes = torch.randint(0, cfg.vocab_per_codebook, (2, cfg.n_codebooks, 64))
    out, _ = fim_reorder_batch(codes, None, cfg.suf_id, cfg.mid_id, random.Random(0))
    logits = model(out)
    assert logits.shape == (2, cfg.n_codebooks, 64, cfg.vocab_with_pad)


def test_generate_never_emits_control_ids():
    """Infill-style generate: feed `P <SUF> S <MID>` and check the produced
    middle contains only real DAC tokens (no pad / <SUF> / <MID>)."""
    model = _tiny_fim_model()
    cfg = model.cfg
    K = cfg.n_codebooks
    P = torch.randint(0, cfg.vocab_per_codebook, (K, 20))
    S = torch.randint(0, cfg.vocab_per_codebook, (K, 20))
    prompt = build_fim_prompt(P, S, cfg.suf_id, cfg.mid_id)
    out = model.generate(prompt, num_new_frames=16, temperature=1.0, top_k=50)
    middle = out[:, prompt.shape[1]:]
    assert middle.shape == (K, 16)
    assert int(middle.max()) < cfg.vocab_per_codebook
    assert int(middle.min()) >= 0
