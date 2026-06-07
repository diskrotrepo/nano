"""Tests for melody (chromagram) conditioning.

Two load-bearing contracts:

1. The chroma the dataset stores at train time (``diskrot.melody.extract_chroma``
   via the packer) and the chroma inference computes from the uploaded hum MUST be
   byte-identical — otherwise the model is conditioned on a different signal than
   it trained on. ``test_train_inference_chroma_equivalence`` is that guard (the
   melody analog of ``test_structure_markers``'s stream-equivalence test).
2. The melody is added to the decoder at the cb0 anchor (delayed position p ↔
   frame p) with null on the prompt prefix + delay tail — the alignment math in
   ``encode_melody_delayed`` and the per-frame add.

The chroma/audio tests are gated on librosa being importable; the model-shape and
delay-alignment tests need only torch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from model.delay_pattern import build_train_inputs
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _librosa_available() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


librosa_required = pytest.mark.skipif(
    not _librosa_available(), reason="librosa not installed"
)


def _tiny_cfg(**kw) -> GPTConfig:
    base = dict(
        d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=256,
        use_melody_conditioning=True, use_gradient_checkpointing=False,
    )
    base.update(kw)
    return GPTConfig(**base)


# --- chroma extraction contract -----------------------------------------------

@librosa_required
def test_extract_chroma_frame_count():
    import librosa

    from diskrot.melody import N_CHROMA, _dac_frame_count, extract_chroma

    y = librosa.chirp(fmin=110, fmax=880, sr=44100, duration=1.5)
    # Default frame count follows the DAC convention ceil(samples/512).
    c = extract_chroma(y)
    assert c.shape == (N_CHROMA, _dac_frame_count(len(y)))
    # Forced frame count is honored exactly (the packer passes the token frames).
    for n in (50, 173, 400):
        assert extract_chroma(y, n_frames=n).shape == (N_CHROMA, n)


@librosa_required
def test_extract_chroma_frames_unit_normalized():
    import librosa

    from diskrot.melody import extract_chroma

    y = librosa.chirp(fmin=110, fmax=880, sr=44100, duration=1.0)
    c = extract_chroma(y, n_frames=100)
    norms = np.linalg.norm(c, axis=0)
    # Every frame is unit-norm (or zero) — loudness doesn't leak into the signal.
    assert np.all((np.abs(norms - 1.0) < 1e-3) | (norms < 1e-3))


@librosa_required
def test_dac_frame_count_matches_ceil_hop():
    from diskrot.melody import HOP_LENGTH, _dac_frame_count

    for n in (511, 512, 513, 44100, 44101):
        assert _dac_frame_count(n) == (n + HOP_LENGTH - 1) // HOP_LENGTH


# --- train==inference equivalence + crop alignment (the key guard) ------------

@librosa_required
def test_train_inference_chroma_equivalence(tmp_path, monkeypatch):
    """The dataset's stored/cropped chroma must equal what inference recomputes
    from the same audio — byte-for-byte at the stored fp16 precision."""
    import librosa

    from diskrot import dataset as dataset_mod
    from diskrot.melody import extract_chroma
    from diskrot.pack_cache import pack
    from diskrot.dataset import TokenDataset, load_mmap_bundle

    cache = tmp_path / "tokens"
    cache.mkdir()
    # Two synth songs sharing identical chroma so whichever lands in the train
    # split (the 1-song-min val split sends one song to val) matches `stored`.
    # We don't need real DAC tokens, only the right frame count, so derive it.
    y = librosa.chirp(fmin=110, fmax=1760, sr=44100, duration=4.0).astype(np.float32)
    from diskrot.melody import _dac_frame_count
    token_T = _dac_frame_count(len(y))
    stored = extract_chroma(y, n_frames=token_T).astype(np.float16)
    for name in ("song0", "song1"):
        torch.save(torch.randint(0, 1024, (9, token_T), dtype=torch.int16), cache / f"{name}.pt")
        np.save(cache / f"{name}.mel.npy", stored)

    pack(cache, mel_cache_dir=cache, shard_target_songs=5, verbose=False)

    seg = token_T - 4
    bundle = load_mmap_bundle(cache / "packed", segment_frames=seg, val_ratio=0.0)
    assert bundle["has_melody"]
    ds = TokenDataset.from_mmap(bundle, "train", seg)

    # Full per-song chroma in the pack == the stored fp16 array, byte-for-byte.
    full = np.asarray(ds._get_mel(0))
    assert np.array_equal(full, stored)

    # Inference path recomputes the SAME chroma (natural frame count == token_T).
    infer = extract_chroma(y).astype(np.float16)
    assert np.array_equal(infer, stored)

    # The crop window applied to the melody matches the one applied to the tokens.
    monkeypatch.setattr(dataset_mod.random, "randint", lambda lo, hi: 2)
    tokens, _tags, _lids, melody = ds[0]
    assert melody.shape == (seg, 12)
    expect = torch.from_numpy(stored[:, 2:2 + seg]).to(torch.float32).T
    assert torch.allclose(melody, expect)


# --- delay-pattern alignment math ---------------------------------------------

def test_encode_melody_delayed_placement():
    """Encoded melody sits at [offset, offset+Tc); prompt prefix + delay tail are
    the learned null. This is the cb0-anchor alignment the decoder relies on."""
    torch.manual_seed(0)
    model = NanoAudioGPT(_tiny_cfg()).eval()  # eval -> dropout off, deterministic
    Tc, offset = 6, 1  # T_prompt=1 (prompt prefix), K=9 -> 8-frame delay tail
    seq_len = offset + Tc + (9 - 1)
    melody = torch.randn(1, Tc, 12)
    full = model.encode_melody_delayed(melody, seq_len, offset=offset)
    enc = model.melody_encoder.encode_melody(melody)
    null = model.melody_encoder.null  # [1,1,D]
    # body matches the encoded melody
    assert torch.allclose(full[:, offset:offset + Tc, :], enc)
    # prompt prefix (positions before offset) is null
    assert torch.allclose(full[:, :offset, :], null.expand(1, offset, -1))
    # delay tail (positions after the melody) is null
    assert torch.allclose(full[:, offset + Tc:, :], null.expand(1, seq_len - offset - Tc, -1))


# --- forward / CFG-drop / generate threading ----------------------------------

def test_forward_melody_and_null_shapes():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg)
    B, K, Tseg = 2, cfg.n_codebooks, 16
    codes = torch.randint(0, cfg.vocab_per_codebook, (B, K, Tseg))
    inp, _ = build_train_inputs(codes, cfg.pad_id)
    chroma = torch.randn(B, Tseg, 12)
    V = cfg.vocab_with_pad
    # melody present and melody=None (dropped → learned null) both run, no NaN.
    y1 = model(inp, melody=chroma)
    y2 = model(inp, melody=None)
    assert y1.shape == (B, K, inp.shape[2], V)
    assert y2.shape == y1.shape
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()


def test_melody_cfg_drop_trains():
    """A fully-dropped batch still produces finite gradients (the null path)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg)
    B, K, Tseg = 2, cfg.n_codebooks, 16
    codes = torch.randint(0, cfg.vocab_per_codebook, (B, K, Tseg))
    inp, tgt = build_train_inputs(codes, cfg.pad_id)
    logits = model(inp, melody=None)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1)
    )
    loss.backward()
    assert torch.isfinite(model.melody_encoder.null.grad).all()
    assert model.melody_encoder.null.grad.abs().sum() > 0


def test_generate_encodes_melody_once():
    """The fully-known melody is encoded ONCE, not per decode step."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg).eval()
    mel = torch.randn(1, 12, 12)  # [B=1, num_new_frames=12, bins]

    calls = {"n": 0}
    orig = model.melody_encoder.encode_melody

    def _counting(x):
        calls["n"] += 1
        return orig(x)

    model.melody_encoder.encode_melody = _counting
    out = model.generate(prompt=None, num_new_frames=12, melody=mel,
                         cfg_scale=3.0, temperature=1.0, top_k=20)
    assert out.shape == (cfg.n_codebooks, 13)  # seed + 12
    assert calls["n"] == 1


def test_generate_composed_melody_runs():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = NanoAudioGPT(cfg).eval()
    mel = torch.randn(1, 10, 12)
    out = model.generate(prompt=None, num_new_frames=10, melody=mel,
                         cfg_scale=3.0, melody_cfg_scale=2.0, temperature=1.0, top_k=20)
    assert out.shape == (cfg.n_codebooks, 11)


def test_melody_disabled_ignores_chroma():
    """A model without melody conditioning ignores a melody arg (no crash)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(use_melody_conditioning=False)
    model = NanoAudioGPT(cfg)
    assert not hasattr(model, "melody_encoder")
    B, K, Tseg = 1, cfg.n_codebooks, 12
    codes = torch.randint(0, cfg.vocab_per_codebook, (B, K, Tseg))
    inp, _ = build_train_inputs(codes, cfg.pad_id)
    y = model(inp, melody=torch.randn(B, Tseg, 12))  # melody arg simply unused
    assert y.shape[0] == B
