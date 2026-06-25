"""Tests for generative stem conditioning (the /addstem path).

Guards the contracts that, if broken, silently corrupt a stem-trained run or the
/addstem inference path:
- The packer's per-song stem sidecar co-crops with the SAME window as the token
  stream (the melody co-crop contract), and a missing song is flagged absent
  (present=False) rather than treated as a real all-zero stem.
- The train-time stem conditioning (``_stem_add`` from raw stem tokens) and the
  inference-time term (``encode_stem_delayed``) are byte-identical at the aligned
  new-frame positions (the FIM/melody train==inference guard).
- ``generate`` with stem conditioning never emits a control id.
- The StemEncoder's params all receive grad on a dropped (stem_keep=0) batch — the
  DDP find_unused_parameters=False invariant the train loop relies on.
"""
from __future__ import annotations

import numpy as np
import torch

from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _tiny_stem_model(K: int = 4) -> NanoAudioGPT:
    cfg = GPTConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, n_codebooks=K,
        vocab_per_codebook=64, max_seq_len=256,
        use_stem_conditioning=True, stem_enc_layers=1, n_stem_types=4,
        use_melody_conditioning=False, use_lyric_conditioning=False,
        use_gradient_checkpointing=False,
    )
    return NanoAudioGPT(cfg)


def _pack_with_stems(tmp_path, ramp: bool = True, drop_one: bool = False):
    """Pack a tiny corpus with a parallel stem sidecar. With ``ramp`` every value
    equals its frame index so a crop reveals which frames it covers (alignment).
    Returns the cache dir."""
    from diskrot import pack_cache as pc

    tok_dir = tmp_path / "tok"; tok_dir.mkdir()
    stem_dir = tmp_path / "stems"; stem_dir.mkdir()
    K, NS, vocab = 4, 4, 256
    names = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    for i, nm in enumerate(names):
        T = 40 + i
        if ramp:
            tok = np.tile(np.arange(T, dtype=np.int16), (K, 1))      # tok[k,t]=t
            st = np.tile(np.arange(T, dtype=np.int16), (NS, K, 1))   # st[s,k,t]=t
        else:
            tok = rng.integers(0, vocab, (K, T)).astype(np.int16)
            st = rng.integers(0, vocab, (NS, K, T)).astype(np.int16)
        torch.save(torch.from_numpy(tok), tok_dir / f"{nm}.pt")
        if not (drop_one and nm == "b"):  # 'b' optionally has no stems -> absent
            np.save(stem_dir / f"{nm}.stems.npy", st)
    pc.pack(str(tok_dir), out_dir=str(tmp_path / "packed"), shard_target_songs=2,
            stem_cache_dir=str(stem_dir), verbose=False)
    return tmp_path


def test_pack_marks_missing_stems_absent(tmp_path):
    """A song without a .stems.npy is zero-filled AND flagged present=False, so the
    dataset never trains on its zeros as if they were a real stem."""
    from diskrot import pack_cache as pc

    cache = _pack_with_stems(tmp_path, ramp=False, drop_one=True)
    out = cache / "packed"
    idx = pc.load_shard_index(out)
    assert idx["has_stems"] and idx["n_stems"] == 4
    present_by_name = {}
    for shard_id, local, name, _ in pc.iter_all_names(out):
        meta = pc.load_shard_meta(out, shard_id)
        present_by_name[name] = meta["stem_present"][local]
    assert present_by_name["a"] is True
    assert present_by_name["b"] is False  # dropped -> absent
    assert present_by_name["c"] is True


def test_dataset_stem_cocrop_aligned(tmp_path):
    """The dataset's stem crop covers the EXACT same frame window as the token crop
    (both ramps value==frame), so the conditioning lines up with the target."""
    from diskrot.dataset import TokenDataset

    cache = _pack_with_stems(tmp_path, ramp=True)
    seg = 16
    ds = TokenDataset(str(cache), segment_frames=seg, split="train", val_ratio=0.0,
                      n_codebooks=4, pad_short=False)
    assert ds._has_stems
    for _ in range(8):  # several random crops
        tokens, _tags, _lids, _mel, stem = ds[0]
        assert stem is not None
        st, present = stem[0], stem[1]
        assert present == 1.0
        # tokens cb0 ramp == the frames cropped; every stem's cb0 must match it.
        token_frames = tokens[0]              # [seg] = the cropped frame indices
        for s in range(st.shape[0]):
            assert torch.equal(st[s, 0], token_frames), "stem crop window != token window"


def test_train_inference_stem_conditioning_identical():
    """``_stem_add`` (train, from raw stem tokens) and ``encode_stem_delayed``
    (inference) must produce the SAME per-frame conditioning at the aligned
    new-frame positions — the train==inference contract for the stem axis."""
    torch.manual_seed(0)
    model = _tiny_stem_model().eval()
    cfg = model.cfg
    K, T, S = cfg.n_codebooks, 24, 3
    B = 1
    stem_tokens = torch.randint(0, cfg.vocab_per_codebook, (B, S, K, T))
    stem_types = torch.tensor([[0, 2, 3]])  # drums, vocals, other (target=bass)
    stem_present = torch.tensor([[1.0, 1.0, 1.0]])
    target_stem_type = torch.tensor([1])  # bass

    with torch.no_grad():
        train_term = model._stem_add(
            stem_tokens, stem_types, stem_present, target_stem_type,
            None, B, T, start_pos=0, keep=torch.ones(B, 1, 1))
        offset = 5
        seq_len = T + offset + 4
        delayed = model.encode_stem_delayed(
            stem_tokens, stem_types, stem_present, target_stem_type,
            seq_len=seq_len, offset=offset)
    # The delayed term at the new-frame positions == the training term's frames.
    assert torch.allclose(delayed[:, offset:offset + T, :], train_term[:, :T, :], atol=1e-6)
    # And positions outside the placed window are the learned null (not the acc).
    null = model.stem_encoder.null_emb(B, offset)
    assert torch.allclose(delayed[:, :offset, :], null, atol=1e-6)


def test_stem_encoder_grad_on_dropped_batch():
    """Every StemEncoder param must receive grad even when the stem axis is dropped
    (stem_keep=0) — the DDP find_unused_parameters=False invariant. Mirrors how the
    train loop passes zeroed stem args + keep=0 on normal (non-stem-add) batches."""
    model = _tiny_stem_model().train()
    cfg = model.cfg
    B, K, T, S = 2, cfg.n_codebooks, 16, 3
    tokens = torch.randint(0, cfg.vocab_per_codebook, (B, K, T))
    logits = model(
        tokens,
        stem_tokens=torch.zeros(B, S, K, T, dtype=torch.long),
        stem_types=torch.zeros(B, S, dtype=torch.long),
        stem_present=torch.zeros(B, S),
        target_stem_type=torch.zeros(B, dtype=torch.long),
        stem_keep=torch.zeros(B, 1, 1),
    )
    logits.float().pow(2).mean().backward()
    missing = [n for n, p in model.stem_encoder.named_parameters() if p.grad is None]
    assert not missing, f"StemEncoder params without grad on a dropped batch: {missing}"


def test_dropped_stem_equals_no_stem():
    """stem_keep=0 must reproduce the exact logits of a forward with no stem args
    (the learned-null / unconditional state) — so train and inference agree that
    'no stem' is the null, and CFG's baseline is well-defined."""
    torch.manual_seed(1)
    model = _tiny_stem_model().eval()
    cfg = model.cfg
    B, K, T, S = 2, cfg.n_codebooks, 16, 3
    tokens = torch.randint(0, cfg.vocab_per_codebook, (B, K, T))
    with torch.no_grad():
        plain = model(tokens)  # _stem_add(None) -> null
        dropped = model(
            tokens,
            stem_tokens=torch.randint(0, cfg.vocab_per_codebook, (B, S, K, T)),
            stem_types=torch.zeros(B, S, dtype=torch.long),
            stem_present=torch.ones(B, S),
            target_stem_type=torch.ones(B, dtype=torch.long),
            stem_keep=torch.zeros(B, 1, 1),
        )
    assert torch.allclose(plain, dropped, atol=1e-5)


def test_stem_and_lyrics_coexist():
    """A vocals-target /addstem keeps the lyric stream, so the model must accept
    lyric conditioning AND stem conditioning together (forward + generate)."""
    torch.manual_seed(3)
    cfg = GPTConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64, n_codebooks=4,
        vocab_per_codebook=64, max_seq_len=256,
        use_stem_conditioning=True, stem_enc_layers=1, n_stem_types=4,
        use_lyric_conditioning=True, use_melody_conditioning=False,
        use_gradient_checkpointing=False,
    )
    model = NanoAudioGPT(cfg).eval()
    K, T, S, L = cfg.n_codebooks, 16, 3, 12
    tokens = torch.randint(0, cfg.vocab_per_codebook, (1, K, T))
    lyric_ids = torch.randint(1, cfg.phoneme_vocab_size, (1, L))
    lyric_mask = torch.ones(1, L, dtype=torch.bool)
    with torch.no_grad():
        logits = model(
            tokens, lyric_ids=lyric_ids, lyric_mask=lyric_mask,
            stem_tokens=torch.randint(0, cfg.vocab_per_codebook, (1, S, K, T)),
            stem_types=torch.tensor([[0, 1, 3]]), stem_present=torch.ones(1, S),
            target_stem_type=torch.tensor([2]),  # vocals
        )
    assert logits.shape == (1, K, T, cfg.vocab_with_pad)
    # generate with both lyric and stem conditioning (the /addstem vocals path).
    out = model.generate(
        prompt=None, num_new_frames=6, temperature=1.0, top_k=20,
        lyric_ids=lyric_ids, lyric_mask=lyric_mask,
        stem_tokens=torch.randint(0, cfg.vocab_per_codebook, (1, S, K, 6)),
        stem_types=torch.tensor([[0, 1, 3]]), stem_present=torch.ones(1, S),
        target_stem_type=torch.tensor([2]), cfg_scale=2.0,
    )
    assert int(out[:, 1:].max()) < cfg.vocab_per_codebook


def test_generate_stem_never_emits_control_ids():
    """Stem-conditioned generate must produce only real DAC tokens (no pad)."""
    torch.manual_seed(2)
    model = _tiny_stem_model().eval()
    cfg = model.cfg
    K, T, S = cfg.n_codebooks, 12, 3
    stem_tokens = torch.randint(0, cfg.vocab_per_codebook, (1, S, K, T))
    out = model.generate(
        prompt=None, num_new_frames=8, temperature=1.0, top_k=20,
        stem_tokens=stem_tokens,
        stem_types=torch.tensor([[0, 2, 3]]),
        stem_present=torch.ones(1, S),
        target_stem_type=torch.tensor([1]),
        cfg_scale=2.0, stem_cfg_scale=3.0,
    )
    new = out[:, 1:]  # strip the seed frame
    assert int(new.max()) < cfg.vocab_per_codebook
    assert int(new.min()) >= 0
