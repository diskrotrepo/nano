"""Parity tests for the MLX inference backend (model/nano_audio_gpt_mlx.py).

Skipped unless running on Apple Silicon with `mlx` installed. The fp32 tests pin
the math port against the PyTorch reference; the quant tests only guard that the
int8/int4 paths run and stay in range (no bit-parity is expected there).

Mirrors tests/test_rope_equivalence.py: tiny configs, CPU torch reference.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from server.inference import _mlx_available

pytestmark = pytest.mark.skipif(
    not _mlx_available(), reason="MLX backend only runs on Apple Silicon with mlx installed"
)

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT  # noqa: E402


def _tiny_cfg(**overrides) -> GPTConfig:
    base = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, max_seq_len=128)
    base.update(overrides)
    return GPTConfig(**base)


def _build_pair(cfg, seed=0):
    import mlx.core as mx

    from model.nano_audio_gpt_mlx import MLXNanoAudioGPT

    torch.manual_seed(seed)
    m = NanoAudioGPT(cfg).eval()
    mlx_m = MLXNanoAudioGPT(cfg, m.state_dict(), dtype=mx.float32, bits=16)
    return m, mlx_m


@pytest.mark.parametrize("use_text", [False, True])
@pytest.mark.parametrize("use_qk_norm", [False, True])
def test_oneshot_logits_match_torch(use_text, use_qk_norm):
    """MLX full forward must match PyTorch logits in fp32 (within float noise)."""
    import mlx.core as mx

    cfg = _tiny_cfg(use_text_conditioning=use_text, use_qk_norm=use_qk_norm)
    m, mlx_m = _build_pair(cfg)
    K, T = cfg.n_codebooks, 12
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, T))
    text_emb = torch.randn(1, 2, cfg.d_model) if use_text else None

    with torch.no_grad():
        ref = m(tokens, text_emb=text_emb).numpy()
    mt = mx.array(tokens.numpy().astype(np.int32))
    met = mx.array(text_emb.numpy()) if use_text else None
    got = np.array(mlx_m.logits_oneshot(mt, text_emb=met))

    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-3


@pytest.mark.parametrize("use_qk_norm", [False, True])
def test_cached_decode_matches_oneshot(use_qk_norm):
    """The KV-cached generate path (greedy) must reproduce a one-shot forward's
    argmax trajectory — and match PyTorch's generate exactly in fp32. qk-norm is
    covered in both states (it must be applied before the cache write)."""
    cfg = _tiny_cfg(use_text_conditioning=True, use_qk_norm=use_qk_norm)
    m, mlx_m = _build_pair(cfg, seed=3)
    K, T = cfg.n_codebooks, 10
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, T))
    text_emb = torch.randn(1, 2, cfg.d_model)

    out_t = m.generate(tokens[0], num_new_frames=6, temperature=0.0, top_k=None,
                       top_p=None, text_emb=text_emb, cfg_scale=1.0)
    out_m = mlx_m.generate(tokens[0], num_new_frames=6, temperature=0.0, top_k=None,
                           top_p=None, text_emb=text_emb, cfg_scale=1.0)
    assert out_t.shape == out_m.shape
    assert torch.equal(out_t, out_m)


def test_cfg_generate_matches_torch():
    """CFG-guided (cfg_scale>1) greedy generate must match PyTorch token-for-token.

    Regression guard for the MLX CFG path: it runs each guidance stage as its own
    forward and combines the logits. A prior batched-forward implementation (all
    stages stacked along the batch dim) was NOT logit-identical on the DAC shape
    and collapsed the rollout to noise — but the old test only checked shape/range
    (temperature=0.9, no torch reference), so it passed anyway. Greedy + exact
    equality vs torch is what actually pins the guidance math.

    NOTE: use_qk_norm=True is load-bearing — the batched-forward bug only surfaced
    with qk-norm on (the v10 checkpoint's setting); with it off the old code
    matched torch, which is exactly why the bug shipped. Don't drop it."""
    cfg = _tiny_cfg(use_text_conditioning=True, use_qk_norm=True)
    m, mlx_m = _build_pair(cfg, seed=4)
    K = cfg.n_codebooks
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, 8))
    text_emb = torch.randn(1, 2, cfg.d_model)
    kw = dict(num_new_frames=5, temperature=0.0, top_k=None, top_p=None,
              text_emb=text_emb, cfg_scale=3.0)
    out_t = m.generate(tokens[0], **kw)
    out_m = mlx_m.generate(tokens[0], **kw)
    assert out_t.shape == (K, 8 + 5)
    assert torch.equal(out_t, out_m)


def test_unconditional_generate_from_none():
    cfg = _tiny_cfg(use_text_conditioning=False)
    _, mlx_m = _build_pair(cfg, seed=5)
    out = mlx_m.generate(None, num_new_frames=7, temperature=0.9, top_k=50, top_p=0.95)
    assert out.shape == (cfg.n_codebooks, 1 + 7)  # internal 1-frame seed + new
    assert int(out.min()) >= 0 and int(out.max()) < cfg.vocab_per_codebook


def _lyric_cfg(**overrides) -> GPTConfig:
    base = dict(
        use_lyric_conditioning=True, lyric_enc_layers=2, lyric_enc_heads=4,
        lyric_enc_d_ff=128, max_lyric_len=16,
    )
    base.update(overrides)
    return _tiny_cfg(**base)


def _lyric_inputs():
    from model.lyric_encoder import BOS_PHONEME_ID, PAD_PHONEME_ID

    # BOS + a few phones + one pad slot; mask keeps BOS valid (no fully-masked row).
    ids = torch.tensor([[BOS_PHONEME_ID, 10, 11, 12, 13, PAD_PHONEME_ID]])
    mask = ids != PAD_PHONEME_ID
    return ids, mask


def test_oneshot_logits_match_torch_with_lyrics():
    """The lyric encoder + lyric cross-attn port must match torch end-to-end."""
    import mlx.core as mx

    cfg = _lyric_cfg(use_text_conditioning=True)
    m, mlx_m = _build_pair(cfg)
    K, T = cfg.n_codebooks, 12
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, T))
    text_emb = torch.randn(1, 1, cfg.d_model)
    ids, mask = _lyric_inputs()

    with torch.no_grad():
        ref = m(tokens, text_emb=text_emb, lyric_ids=ids, lyric_mask=mask).numpy()
    lemb, lkv = mlx_m._encode_lyrics(
        mx.array(ids.numpy().astype(np.int32)), mx.array(mask.numpy())
    )
    got = np.array(
        mlx_m.logits_oneshot(
            mx.array(tokens.numpy().astype(np.int32)),
            text_emb=mx.array(text_emb.numpy()),
            lyric_emb=lemb, lyric_kv_mask=lkv,
        )
    )
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-3


def test_cached_decode_matches_with_lyrics():
    """Greedy KV-cached generate with lyric conditioning must match torch exactly."""
    cfg = _lyric_cfg(use_text_conditioning=True)
    m, mlx_m = _build_pair(cfg, seed=7)
    K = cfg.n_codebooks
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, 10))
    text_emb = torch.randn(1, 1, cfg.d_model)
    ids, mask = _lyric_inputs()
    kw = dict(num_new_frames=6, temperature=0.0, top_k=None, top_p=None,
              text_emb=text_emb, cfg_scale=1.0, lyric_ids=ids, lyric_mask=mask)
    out_t = m.generate(tokens[0], **kw)
    out_m = mlx_m.generate(tokens[0], **kw)
    assert torch.equal(out_t, out_m)


def test_lyric_composed_cfg_matches_torch():
    """Composed dual-axis guidance (tags cfg + separate lyric_cfg_scale, S=3 stages)
    greedy-generates identically to PyTorch — the multi-stage combine + per-stage
    cross/lyric caches must stay isolated (see test_cfg_generate_matches_torch).

    qk-norm on (the trigger for the batched-forward bug — see that test)."""
    cfg = _lyric_cfg(use_text_conditioning=True, use_qk_norm=True)
    m, mlx_m = _build_pair(cfg, seed=8)
    K = cfg.n_codebooks
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, 8))
    text_emb = torch.randn(1, 1, cfg.d_model)
    ids, mask = _lyric_inputs()
    kw = dict(num_new_frames=5, temperature=0.0, top_k=None, top_p=None,
              text_emb=text_emb, cfg_scale=2.0, lyric_cfg_scale=4.0,
              lyric_ids=ids, lyric_mask=mask)
    out_t = m.generate(tokens[0], **kw)
    out_m = mlx_m.generate(tokens[0], **kw)
    assert out_t.shape == (K, 8 + 5)
    assert torch.equal(out_t, out_m)


def _melody_cfg(**overrides) -> GPTConfig:
    base = dict(use_melody_conditioning=True, melody_enc_layers=2)
    base.update(overrides)
    return _tiny_cfg(**base)


def test_oneshot_logits_match_torch_with_melody():
    """The melody encoder (incl. the channels-last Conv1d weight transpose) +
    additive melody term must match torch end-to-end."""
    import mlx.core as mx

    cfg = _melody_cfg(use_text_conditioning=True)
    m, mlx_m = _build_pair(cfg)
    K, T = cfg.n_codebooks, 12
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, T))
    text_emb = torch.randn(1, 1, cfg.d_model)
    chroma = torch.randn(1, T, cfg.melody_n_bins)  # length == seq so no null-pad

    with torch.no_grad():
        ref = m(tokens, text_emb=text_emb, melody=chroma).numpy()
    mel_emb = mlx_m._encode_melody(mx.array(chroma.numpy()))
    got = np.array(
        mlx_m.logits_oneshot(
            mx.array(tokens.numpy().astype(np.int32)),
            text_emb=mx.array(text_emb.numpy()),
            melody_emb=mel_emb,
        )
    )
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-3


def test_cached_decode_matches_with_melody():
    """Greedy KV-cached generate with melody conditioning must match torch exactly
    (validates encode_melody_delayed placement + the generate threading)."""
    cfg = _melody_cfg(use_text_conditioning=True)
    m, mlx_m = _build_pair(cfg, seed=11)
    K = cfg.n_codebooks
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, 10))
    text_emb = torch.randn(1, 1, cfg.d_model)
    melody = torch.randn(1, 6, cfg.melody_n_bins)  # one per new frame
    kw = dict(num_new_frames=6, temperature=0.0, top_k=None, top_p=None,
              text_emb=text_emb, cfg_scale=1.0, melody=melody)
    out_t = m.generate(tokens[0], **kw)
    out_m = mlx_m.generate(tokens[0], **kw)
    assert torch.equal(out_t, out_m)


def test_melody_composed_cfg_runs_and_in_range():
    """Composed melody guidance (separate melody_cfg_scale) runs on MLX."""
    cfg = _melody_cfg(use_text_conditioning=True)
    _, mlx_m = _build_pair(cfg, seed=12)
    K = cfg.n_codebooks
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, 8))
    text_emb = torch.randn(1, 1, cfg.d_model)
    melody = torch.randn(1, 5, cfg.melody_n_bins)
    out = mlx_m.generate(tokens[0], num_new_frames=5, temperature=0.9, top_k=50,
                         top_p=0.95, text_emb=text_emb, cfg_scale=2.0,
                         melody=melody, melody_cfg_scale=4.0)
    assert out.shape == (K, 8 + 5)
    assert int(out.min()) >= 0 and int(out.max()) < cfg.vocab_per_codebook


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_runs_in_range(bits):
    """int8/int4 just need to run and produce valid tokens — no bit-parity."""
    import mlx.core as mx

    from model.nano_audio_gpt_mlx import MLXNanoAudioGPT

    cfg = _tiny_cfg(use_text_conditioning=True)
    torch.manual_seed(6)
    m = NanoAudioGPT(cfg).eval()
    qm = MLXNanoAudioGPT(cfg, m.state_dict(), dtype=mx.float16, bits=bits)
    assert qm._bits == bits

    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, cfg.n_codebooks, 8))
    text_emb = torch.randn(1, 2, cfg.d_model)
    out = qm.generate(tokens[0], num_new_frames=5, temperature=0.9, top_k=50,
                      top_p=0.95, text_emb=text_emb, cfg_scale=3.0)
    assert int(out.min()) >= 0 and int(out.max()) < cfg.vocab_per_codebook


# --- streaming + batched entry points (server /generate_stream, /generate_batch) ---
# These guard the MLX _generate_stream / _resolve_prompt added so the streaming and
# batched server paths work on Apple Silicon (they crash without them). Mirrors
# tests/test_generate_stream.py and tests/test_generate_batch.py.


def _stream_equiv(cfg, *, num_new_frames, emit_every, seed=0, **gen_kw):
    """MLX generate() (collector) vs _generate_stream() chunks — identical tokens.
    temperature=0 (argmax) so both paths are deterministic regardless of RNG state."""
    _, mlx_m = _build_pair(cfg, seed=seed)
    K = cfg.n_codebooks
    prompt = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    T_prompt = prompt.shape[1]
    ref = mlx_m.generate(prompt, num_new_frames=num_new_frames, temperature=0.0, **gen_kw)
    chunks = list(mlx_m._generate_stream(
        prompt.unsqueeze(0), num_new_frames=num_new_frames, temperature=0.0,
        emit_every=emit_every, **gen_kw,
    ))
    streamed = torch.cat(chunks, dim=-1).squeeze(0)
    assert ref.shape == (K, T_prompt + num_new_frames)
    assert torch.equal(ref[:, T_prompt:], streamed)
    return chunks, T_prompt


def test_mlx_stream_matches_generate_plain():
    _stream_equiv(_tiny_cfg(use_text_conditioning=False),
                  num_new_frames=20, emit_every=7, top_k=50)


def test_mlx_stream_matches_generate_with_cfg():
    cfg = _tiny_cfg(use_text_conditioning=True)
    torch.manual_seed(7)
    text_emb = torch.randn(1, 1, cfg.d_model)
    _stream_equiv(cfg, num_new_frames=18, emit_every=8, top_k=40,
                  text_emb=text_emb, cfg_scale=3.0)


def test_mlx_stream_matches_generate_with_lyrics():
    cfg = _lyric_cfg(use_text_conditioning=True)
    torch.manual_seed(7)
    text_emb = torch.randn(1, 1, cfg.d_model)
    ids, mask = _lyric_inputs()
    _stream_equiv(cfg, num_new_frames=16, emit_every=6, top_k=40,
                  text_emb=text_emb, cfg_scale=2.0, lyric_ids=ids, lyric_mask=mask)


def test_mlx_stream_cadence():
    """first_emit makes the first chunk smaller, then steady emit_every; sizes
    sum to num_new_frames (mirrors the torch streaming cadence)."""
    cfg = _tiny_cfg(use_text_conditioning=False)
    _, mlx_m = _build_pair(cfg, seed=0)
    K = cfg.n_codebooks
    prompt = torch.randint(0, cfg.vocab_per_codebook, (K, 1))
    chunks = list(mlx_m._generate_stream(
        prompt.unsqueeze(0), num_new_frames=30, emit_every=10, first_emit=4,
        temperature=1.0, top_k=50, top_p=0.95,
    ))
    sizes = [c.shape[-1] for c in chunks]
    assert sum(sizes) == 30
    assert sizes[0] == 4 and sizes[1] == 10


def test_mlx_resolve_prompt_batch_shape():
    cfg = _tiny_cfg(use_text_conditioning=False)
    _, mlx_m = _build_pair(cfg, seed=0)
    p, sq = mlx_m._resolve_prompt(None, batch_size=4)
    assert tuple(p.shape) == (4, cfg.n_codebooks, 1) and sq is False
    p1, _ = mlx_m._resolve_prompt(None, batch_size=1)
    assert tuple(p1.shape) == (1, cfg.n_codebooks, 1)


def test_mlx_batched_matches_single():
    """Batched generation (the /generate_batch path): each row of a batch of
    identical prompts equals the B=1 result (temp=0, deterministic)."""
    cfg = _tiny_cfg(use_text_conditioning=False)
    _, mlx_m = _build_pair(cfg, seed=0)
    K = cfg.n_codebooks
    prompt = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    ref = mlx_m.generate(prompt, num_new_frames=15, temperature=0.0, top_k=50)
    B = 3
    bprompt = prompt.unsqueeze(0).expand(B, K, 3).contiguous()
    out = mlx_m.generate(bprompt, num_new_frames=15, temperature=0.0, top_k=50)
    assert out.shape == (B, K, 3 + 15)
    for i in range(B):
        assert torch.equal(out[i], ref)


def test_mlx_batched_rows_independent():
    """Different prompts batched together: each row matches its own B=1 run, so
    rows don't leak through attention or the KV cache."""
    cfg = _tiny_cfg(use_text_conditioning=False)
    _, mlx_m = _build_pair(cfg, seed=1)
    K = cfg.n_codebooks
    p0 = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    p1 = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    ref0 = mlx_m.generate(p0, num_new_frames=12, temperature=0.0, top_k=50)
    ref1 = mlx_m.generate(p1, num_new_frames=12, temperature=0.0, top_k=50)
    out = mlx_m.generate(torch.stack([p0, p1], 0), num_new_frames=12,
                         temperature=0.0, top_k=50)
    assert torch.equal(out[0], ref0)
    assert torch.equal(out[1], ref1)


@pytest.mark.parametrize("lyric_qk", [False, True])
def test_lyric_qk_norm_generate_matches_torch(lyric_qk):
    """Lyric-encoder QK-norm (GPTConfig.use_lyric_qk_norm, ON in the trained v8
    checkpoint) must load and match torch — guards the MLX _LyricEncLayer
    q_norm/k_norm port (without it, the real checkpoint won't even load on MLX)."""
    cfg = _lyric_cfg(use_text_conditioning=True, use_lyric_qk_norm=lyric_qk)
    m, mlx_m = _build_pair(cfg, seed=9)
    K = cfg.n_codebooks
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, 10))
    text_emb = torch.randn(1, 1, cfg.d_model)
    ids, mask = _lyric_inputs()
    kw = dict(num_new_frames=6, temperature=0.0, top_k=None, top_p=None,
              text_emb=text_emb, cfg_scale=1.0, lyric_ids=ids, lyric_mask=mask)
    out_t = m.generate(tokens[0], **kw)
    out_m = mlx_m.generate(tokens[0], **kw)
    assert torch.equal(out_t, out_m)
