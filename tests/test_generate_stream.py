"""Streaming-generation drift guard.

The server's progressive-audio path consumes ``NanoAudioGPT._generate_stream``
(it yields the new frames in chunks as they become known), while ``generate()``
is a thin collector over the *same* generator. The load-bearing contract is that
the two produce byte-identical tokens — otherwise the streamed audio would drift
from what a normal ``generate()`` produces. This is the streaming analog of the
melody / structure-marker equivalence guards.

These tests need only torch (no DAC, no g2p), so they run everywhere.
"""
from __future__ import annotations

import shutil
import types

import pytest
import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT, StaticLayerKVCache

ffmpeg_required = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not installed"
)


def _tiny_cfg(**kw) -> GPTConfig:
    base = dict(
        d_model=64, n_layers=2, n_heads=4, d_ff=128, max_seq_len=256,
        use_gradient_checkpointing=False,
    )
    base.update(kw)
    return GPTConfig(**base)


def _run_equiv(cfg: GPTConfig, *, num_new_frames: int, emit_every: int, **gen_kw):
    """generate() vs _generate_stream must yield identical new-frame tokens."""
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    # Explicit prompt (NOT None) so neither path draws a random seed frame — the
    # only divergence we are NOT testing.
    prompt = torch.randint(0, cfg.vocab_per_codebook, (K, 3))
    T_prompt = prompt.shape[1]

    torch.manual_seed(123)
    ref = model.generate(prompt, num_new_frames=num_new_frames, **gen_kw)

    torch.manual_seed(123)
    chunks = list(model._generate_stream(
        prompt.unsqueeze(0), num_new_frames=num_new_frames,
        emit_every=emit_every, **gen_kw,
    ))
    streamed = torch.cat(chunks, dim=-1).squeeze(0)

    # generate() returns prompt + new frames; the stream yields only new frames.
    assert ref.shape == (K, T_prompt + num_new_frames)
    assert torch.equal(ref[:, T_prompt:], streamed)
    return chunks, T_prompt


def test_stream_matches_generate_plain():
    cfg = _tiny_cfg(
        use_text_conditioning=False, use_lyric_conditioning=False,
        use_melody_conditioning=False,
    )
    _run_equiv(cfg, num_new_frames=40, emit_every=7,
               temperature=1.0, top_k=50, top_p=0.95)


def test_stream_matches_generate_with_cfg():
    """cfg_scale > 1 exercises the multi-stage _run/_combine guidance path."""
    cfg = _tiny_cfg(use_text_conditioning=True)
    torch.manual_seed(7)
    text_emb = torch.randn(1, 1, cfg.d_model)
    _run_equiv(cfg, num_new_frames=37, emit_every=8,
               temperature=0.9, top_k=40, top_p=None,
               text_emb=text_emb, cfg_scale=3.0)


def test_stream_chunk_cadence():
    """First chunk after first_emit, then emit_every; final flush covers the tail.

    With first_emit defaulting to emit_every, num_new_frames=40, emit_every=7:
    chunks of 7,7,7,7,7 then a 5-frame flush → 6 chunks summing to 40.
    """
    cfg = _tiny_cfg(
        use_text_conditioning=False, use_lyric_conditioning=False,
        use_melody_conditioning=False,
    )
    chunks, _ = _run_equiv(cfg, num_new_frames=40, emit_every=7,
                           temperature=1.0, top_k=50, top_p=0.95)
    sizes = [c.shape[-1] for c in chunks]
    assert sum(sizes) == 40
    assert all(s <= 7 for s in sizes)
    assert sizes[:-1] == [7, 7, 7, 7, 7] and sizes[-1] == 5


def test_stream_first_emit_smaller_then_steady():
    """first_emit makes the first chunk smaller (fast time-to-first-audio)."""
    cfg = _tiny_cfg(
        use_text_conditioning=False, use_lyric_conditioning=False,
        use_melody_conditioning=False,
    )
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval()
    K = cfg.n_codebooks
    prompt = torch.randint(0, cfg.vocab_per_codebook, (K, 1))
    chunks = list(model._generate_stream(
        prompt.unsqueeze(0), num_new_frames=30,
        emit_every=10, first_emit=4,
        temperature=1.0, top_k=50, top_p=0.95,
    ))
    sizes = [c.shape[-1] for c in chunks]
    assert sum(sizes) == 30
    assert sizes[0] == 4              # smaller first chunk
    assert sizes[1] == 10            # steady cadence after


# --- CUDA-graph decode correctness (KV cache + masked attention) -------------
# The decode loop uses a TENSOR position + a static-shape mask over the full KV
# buffer (the CUDA-graph-friendly path). It must produce logits identical to a
# plain uncached full forward, or the speedup would change the output. Eager
# here (no GPU/compile needed); compile only changes speed, not these results.

def _cached_decode_matches_full(cfg: GPTConfig, **fwd_kw):
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval()
    K, T = cfg.n_codebooks, 12
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, T))
    head_dim = cfg.d_model // cfg.n_heads
    with torch.no_grad():
        ref = model(tokens, **fwd_kw)                      # [1, K, T, V], no cache
        caches = [
            StaticLayerKVCache(
                torch.zeros(1, cfg.n_heads, T, head_dim),
                torch.zeros(1, cfg.n_heads, T, head_dim),
            )
            for _ in range(cfg.n_layers)
        ]
        got = torch.empty_like(ref)
        # prefill position 0 (legacy path), then decode positions 1..T-1 via the
        # input_pos + masked-full-buffer path.
        out0, caches = model(tokens[:, :, 0:1], kv_caches=caches, start_pos=0, **fwd_kw)
        got[:, :, 0] = out0[:, :, 0]
        for t in range(1, T):
            input_pos = torch.tensor([t], dtype=torch.long)
            # Finite additive mask (matches _generate_stream): 0 for keys 0..t,
            # large-negative beyond — keeps flash-attention from NaNing in fp16.
            attn_mask = torch.zeros(T)
            attn_mask[t + 1:] = -1e4
            attn_mask = attn_mask.view(1, 1, 1, T)
            out_t, caches = model(
                tokens[:, :, t:t + 1], kv_caches=caches, start_pos=0,
                input_pos=input_pos, attn_mask=attn_mask, **fwd_kw,
            )
            got[:, :, t] = out_t[:, :, 0]
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max().item()


def test_cached_decode_matches_full_plain():
    _cached_decode_matches_full(_tiny_cfg(
        use_text_conditioning=False, use_lyric_conditioning=False,
        use_melody_conditioning=False,
    ))


def test_cached_decode_matches_full_with_text():
    cfg = _tiny_cfg(use_text_conditioning=True)
    torch.manual_seed(3)
    text_emb = torch.randn(1, 1, cfg.d_model)
    _cached_decode_matches_full(cfg, text_emb=text_emb)


def test_cached_decode_matches_full_with_melody():
    # Exercises the melody input_pos slice in _melody_add (the /cover path):
    # per-step melody[input_pos] must equal the contiguous full-forward slice.
    cfg = _tiny_cfg(use_melody_conditioning=True)
    torch.manual_seed(5)
    melody_emb = torch.randn(1, 12, cfg.d_model)  # full-length [1, T, D]
    _cached_decode_matches_full(cfg, melody_emb=melody_emb)


# --- streaming orchestration (server/inference.py) ---------------------------
# These drive the REAL generate_audio_stream/_decode_chunk (threading + the
# persistent ffmpeg PCM->MP3 pipe + on_complete-only-on-completion) with a stub
# standing in for the model/codec, so they need ffmpeg but no checkpoint/GPU/DAC.

HOP = 512


def _make_stub_engine():
    from server.inference import InferenceEngine

    K = 9

    def _fake_resolve_prompt(_prompt):
        return torch.zeros(1, K, 1, dtype=torch.long), False

    def _fake_generate_stream(prompt, num_new_frames, emit_every=256,
                              first_emit=None, **_kw):
        if first_emit is None:
            first_emit = emit_every
        produced, threshold = 0, first_emit
        while produced < num_new_frames:
            n = min(threshold, num_new_frames - produced)
            yield torch.arange(produced, produced + n).view(1, 1, n).expand(1, K, n).contiguous()
            produced += n
            threshold = emit_every

    def _fake_decode(window):  # [K, w] long -> [w*HOP] float (DAC: 512/frame)
        return torch.zeros(window.shape[-1] * HOP)

    model = types.SimpleNamespace(
        cfg=types.SimpleNamespace(n_codebooks=K, max_seq_len=8192),
        _resolve_prompt=_fake_resolve_prompt,
        _generate_stream=_fake_generate_stream,
    )
    codec = types.SimpleNamespace(SAMPLE_RATE=44100, FRAME_RATE_HZ=86, decode=_fake_decode)

    stub = types.SimpleNamespace(
        model=model,
        codec=codec,
        _build_conditioning=lambda *a, **k: (None, None, None),
        _gen_metadata=lambda *a, **k: {},
        # No silence seed in the stub → the (None, 1) path, so the stream falls
        # back to _resolve_prompt (like the real engine without a seed).
        _bootstrap=lambda: (None, 1),
    )
    # Bind the real methods under test onto the stub.
    stub._decode_chunk = InferenceEngine._decode_chunk.__get__(stub)
    stub._decoded_chunks = InferenceEngine._decoded_chunks.__get__(stub)
    stub._stream_mp3 = InferenceEngine._stream_mp3.__get__(stub)
    stub.generate_audio_stream = InferenceEngine.generate_audio_stream.__get__(stub)
    return stub


@ffmpeg_required
def test_stream_yields_mp3_and_saves_on_completion():
    stub = _make_stub_engine()
    saved = {}
    blocks = list(stub.generate_audio_stream(
        seconds=20 / 86,  # 20 frames
        emit_every=8, first_emit=4, decode_ctx=4,
        on_complete=lambda body, mime: saved.update(body=body, mime=mime),
    ))
    out = b"".join(blocks)
    assert blocks and out  # bytes flowed
    # Valid MPEG audio: ID3 header or an MPEG frame sync (0xFFEx).
    assert out[:3] == b"ID3" or out[0] == 0xFF
    # on_complete fired once with the canonical (re-encoded) clip.
    assert saved.get("mime") == "audio/mpeg"
    assert saved.get("body")


@ffmpeg_required
def test_stream_disconnect_does_not_save():
    """Closing the response generator early (client disconnect) must cancel the
    producer and skip on_complete — no partial clip is persisted."""
    stub = _make_stub_engine()
    saved = {}
    gen = stub.generate_audio_stream(
        seconds=200 / 86,  # long enough to not finish in one block
        emit_every=8, first_emit=4, decode_ctx=4,
        on_complete=lambda body, mime: saved.update(body=body),
    )
    next(gen)        # pull the first block, then "disconnect"
    gen.close()      # -> GeneratorExit -> finally cancels, no on_complete
    assert "body" not in saved


@ffmpeg_required
def test_decode_chunk_left_context_drop():
    """_decode_chunk decodes [f0-ctx, f1) and drops the ctx*HOP warmup samples."""
    stub = _make_stub_engine()
    K = 9
    all_new = torch.arange(40).view(1, 40).expand(K, 40).contiguous()
    # f0=8, f1=16, ctx=4 -> decode [4,16)=12 frames, drop 4*HOP, keep 8 frames.
    pcm = stub._decode_chunk(all_new, f0=8, f1=16, ctx=4, hop=HOP)
    assert pcm.dtype.name == "float32"
    assert len(pcm) == 8 * HOP
    # f0=0 (first chunk): start clamps to 0, nothing dropped.
    pcm0 = stub._decode_chunk(all_new, f0=0, f1=4, ctx=4, hop=HOP)
    assert len(pcm0) == 4 * HOP
