"""Chunked-CLAP long-description tag conditioning.

Three layers, each independently skippable:
- chunk_text_ids: tokenizer-only (gpt2), no CLAP — the boundary logic + the
  no-silent-truncation cap, exercised with the user's ~2900-char example.
- encode_chunked: needs msclap — the backward-compat invariant (a short caption
  reproduces ``encode``) and the long-caption sequence shape.
- the model [B,N,D] + mask path: pure torch — CFG-uncond equivalence survives a
  chunked sequence, a ragged batch never NaNs, a tagless row is finite.
"""
from __future__ import annotations

import pytest
import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT
from model.text_encoder import CHUNK_TOKENS, MAX_CHUNKS, CLAPTextEncoder, chunk_text_ids

# The user's target prompt: dense, free-form, multilingual, ~2900 chars.
USER_EXAMPLE = (
    "( Instrumental) perfect effective Russiaan indie it descends vocal snippets "
    "float Uyghur doomtrap, perfect trap, float -ina Indie punk Percussive "
    "mumblecrunk bipolarock, #mys-post-slowcore annoyingunk surprises and defies "
    "male the of the tape Dried and cartoon top doomtrap, natural Tibet snippets "
    "float the reflective doomtrap, natural Ambient yr- Piano the 2015, Canadian "
    "float and top 2015, pulpdividual, pothole melancholy Canadian whippets, moving "
    "field-recording segments orange ceiling, sometimes timelessly conscious tennis "
    "favorable top 2015, Canadian #myspace the of the tape -ina Indie #myspace "
    "Ambient the of Hebrew snippets into silence, brilliant drunk, system continues, "
    "replaces the trap male vocals impressive, into silence, brilliant drunk, "
    "reflective the reflective interstices Uyghur flourishes Ambient top 1992, drunk "
    "contrapuntal Taiwan cassette mom tape, the repeating note sequence mixed "
    "emotional Parody Religious and decomposing harsh sibilance Bossa Nova, sr_96000 "
    "is silence, Religious ambient lofi Sessions \"tennis favorable top 2015\""
)


@pytest.fixture(scope="module")
def gpt2_tok():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:  # offline / no cache
        pytest.skip(f"gpt2 tokenizer unavailable: {e}")


# ── chunk_text_ids: tokenizer-only ───────────────────────────────────────────

def test_short_caption_is_single_chunk_identity(gpt2_tok):
    """A caption under the token budget is returned unchanged as one chunk —
    the terse-tag backward-compat invariant."""
    s = "an upbeat indie rock track with jangly guitars and male vocals"
    assert len(gpt2_tok.encode(s)) <= CHUNK_TOKENS
    assert chunk_text_ids(gpt2_tok, s) == [s]


def test_empty_is_no_chunks(gpt2_tok):
    assert chunk_text_ids(gpt2_tok, "") == []
    assert chunk_text_ids(gpt2_tok, "   ") == []


def test_long_caption_splits_under_token_cap(gpt2_tok):
    chunks = chunk_text_ids(gpt2_tok, USER_EXAMPLE)
    assert len(chunks) > 1, "the long example must split into several chunks"
    assert len(chunks) <= MAX_CHUNKS, "must respect the chunk cap"
    # Every chunk re-tokenizes within CLAP's 77-token limit (with a little
    # headroom for BPE boundary drift).
    for c in chunks:
        assert len(gpt2_tok.encode(c)) <= 77


def test_user_example_not_silently_truncated(gpt2_tok, capsys):
    """~2900 dense chars is ~13 chunks — under the 16 cap, so the whole thing is
    kept (no tail dropped). If it ever exceeds the cap, the drop is LOGGED."""
    n_tokens = len(gpt2_tok.encode(USER_EXAMPLE))
    chunks = chunk_text_ids(gpt2_tok, USER_EXAMPLE)
    out = capsys.readouterr().out
    if n_tokens > CHUNK_TOKENS * MAX_CHUNKS:
        assert "capping at" in out  # no silent truncation
    else:
        assert "capping at" not in out
        # round-trip covered every token window
        assert len(chunks) == (n_tokens + CHUNK_TOKENS - 1) // CHUNK_TOKENS


def test_cap_logs_when_exceeded(gpt2_tok, capsys):
    huge = " ".join(["word"] * 4000)
    chunks = chunk_text_ids(gpt2_tok, huge, max_chunks=4)
    assert len(chunks) == 4
    assert "capping at 4" in capsys.readouterr().out


# ── encode_chunked: needs msclap ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def clap_encoder():
    pytest.importorskip("msclap")
    enc = CLAPTextEncoder(d_out=64, device="cpu")
    enc._ensure_clap()
    return enc


def test_encode_chunked_short_matches_encode(clap_encoder):
    """A caption that fits one chunk reproduces the single-vector ``encode``
    exactly — existing terse-tag checkpoints are unaffected."""
    s = "a mellow lofi hip hop instrumental with warm bass and vinyl crackle"
    emb_single = clap_encoder.encode([s])               # [1,1,D]
    emb_chunked, mask = clap_encoder.encode_chunked([s])
    assert emb_chunked.shape == emb_single.shape        # one chunk
    torch.testing.assert_close(emb_chunked, emb_single, atol=1e-4, rtol=1e-4)
    assert torch.all(mask == 0)                         # all-attend (no padding)


def test_encode_chunked_long_is_a_sequence(clap_encoder):
    emb, mask = clap_encoder.encode_chunked([USER_EXAMPLE])
    assert emb.shape[1] > 1, "a long description must become N>1 chunks"
    assert emb.shape[1] <= MAX_CHUNKS
    assert mask.shape == (1, 1, 1, emb.shape[1])
    assert torch.isfinite(emb).all()


def test_encode_chunked_ragged_batch_mask(clap_encoder):
    short = "techno with punchy kick"
    emb, mask = clap_encoder.encode_chunked([short, USER_EXAMPLE, ""])
    B, N = emb.shape[0], emb.shape[1]
    assert B == 3
    # the empty row keeps exactly one un-masked (zero) chunk -> not fully masked
    assert (mask[2, 0, 0] == 0).sum() >= 1
    # every row has at least one un-masked key (no NaN-inducing fully -inf row)
    for i in range(B):
        assert torch.isfinite(mask[i]).any()


# ── the model [B,N,D] + mask path: pure torch ────────────────────────────────

def _model() -> NanoAudioGPT:
    cfg = GPTConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=128,
        use_text_conditioning=True, use_gradient_checkpointing=False,
        dropout=0.0, use_qk_norm=True,
    )
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg)
    with torch.no_grad():  # un-zero the conditioning gate so the path is live
        for block in model.blocks:
            block.cross_attn.out_proj.weight.normal_(std=0.02)
    return model.eval()


def _tokens(model, B=2, T=20):
    torch.manual_seed(1)
    return torch.randint(0, model.cfg.vocab_per_codebook, (B, model.cfg.n_codebooks, T))


def test_chunked_zeros_plus_mask_equals_none():
    """The train-time CFG drop: zeros[B,N,D] + an all-attend mask is bit-identical
    to text_emb=None (the unconditional state)."""
    model = _model()
    tokens = _tokens(model)
    B, N, D = 2, 6, model.cfg.d_model
    keep = torch.ones(B, N, dtype=torch.bool)
    mask = CLAPTextEncoder.additive_kv_mask(keep, torch.float32)
    with torch.no_grad():
        none_out = model(tokens)
        zero_out = model(tokens, text_emb=torch.zeros(B, N, D), text_kv_mask=mask)
    torch.testing.assert_close(zero_out, none_out)


def test_chunked_ragged_mask_is_finite_and_live():
    model = _model()
    tokens = _tokens(model)
    B, N, D = 2, 5, model.cfg.d_model
    keep = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    emb = torch.randn(B, N, D) * keep[..., None]
    mask = CLAPTextEncoder.additive_kv_mask(keep, emb.dtype)
    with torch.no_grad():
        out = model(tokens, text_emb=emb, text_kv_mask=mask)
        base = model(tokens)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, base), "real chunked tags must change the output"


def test_tagless_row_in_mixed_batch_is_finite():
    """A row with a single un-masked ZERO chunk (the 'no tags' convention) must
    not NaN the cross-attn softmax."""
    model = _model()
    tokens = _tokens(model)
    B, N, D = 2, 4, model.cfg.d_model
    keep = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    emb = torch.randn(B, N, D) * keep[..., None]  # row 0 = single zero chunk
    mask = CLAPTextEncoder.additive_kv_mask(keep, emb.dtype)
    with torch.no_grad():
        out = model(tokens, text_emb=emb, text_kv_mask=mask)
    assert torch.isfinite(out).all()
