"""Regression tests for the v9 changes (SpectroStream stereo codec, K=24 + stored
depth, full-song short-song padding, frame-rate decoupling, EMA, the data-win
filters). The SpectroStream codec ROUND-TRIP itself needs the magenta-rt stack
(Linux+CUDA), so it's exercised on Modal (diskrot/modal_spectrostream_spike.py),
not here; these cover everything that runs locally."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from diskrot import pack_cache
from diskrot.dataset import TokenDataset, load_mmap_bundle


# ---------------------------------------------------------------- codec abstraction
def test_codec_constants_switch():
    from model.codec import DACodec, SpectroStreamCodec, codec_constants

    assert DACodec.N_CHANNELS == 1 and DACodec.FRAME_RATE_HZ == 86
    assert SpectroStreamCodec.N_CHANNELS == 2 and SpectroStreamCodec.FRAME_RATE_HZ == 25
    assert SpectroStreamCodec.SAMPLE_RATE == 48000 and SpectroStreamCodec.VOCAB_SIZE == 1024

    dac = codec_constants("dac")
    assert (dac["sample_rate"], dac["frame_rate_hz"], dac["hop"]) == (44100, 86, 512)
    ss = codec_constants("spectrostream")
    assert (ss["sample_rate"], ss["frame_rate_hz"], ss["hop"]) == (48000, 25, 1920)


def test_get_codec_unknown_raises():
    from model.codec import get_codec

    with pytest.raises(ValueError):
        get_codec(codec="encodec")


def test_codec_module_imports_without_dac_or_magenta():
    # Importing model.codec must NOT require dac or magenta_rt (lazy in __init__).
    import importlib

    importlib.import_module("model.codec")  # no error == pass


# --------------------------------------------------------------- frame-rate decouple
def test_melody_frame_rate_tracks_codec(monkeypatch):
    import importlib

    import diskrot.melody as melody

    monkeypatch.setenv("NANO_CODEC", "spectrostream")
    importlib.reload(melody)
    try:
        assert (melody.SAMPLE_RATE, melody.HOP_LENGTH) == (48000, 1920)
        assert melody._dac_frame_count(48000 * 8) == 200  # 8s @ 25Hz
    finally:
        monkeypatch.delenv("NANO_CODEC", raising=False)
        importlib.reload(melody)  # restore DAC default for other tests
    assert (melody.SAMPLE_RATE, melody.HOP_LENGTH) == (44100, 512)


# ------------------------------------------------------ dataset slice + short padding
def _pack(tmp_path, lengths: dict[str, int], stored_k: int = 32):
    toks = tmp_path / "tokens"
    toks.mkdir()
    g = torch.Generator().manual_seed(0)
    for name, L in lengths.items():
        torch.save(torch.randint(0, 1024, (stored_k, L), generator=g, dtype=torch.int16),
                   toks / f"song_{name}.pt")
    return pack_cache.pack(toks, out_dir=tmp_path / "packed", shard_target_songs=10,
                           verbose=False)


def test_dataset_slices_stored_to_model_k(tmp_path):
    packed = _pack(tmp_path, {"A": 300, "B": 300}, stored_k=32)  # 2 songs: 1 train, 1 val
    shapes = []
    for split in ("train", "val"):
        ds = TokenDataset.from_mmap(
            load_mmap_bundle(packed, segment_frames=100, val_ratio=0.34),
            split, 100, n_codebooks=24)
        shapes += [tuple(ds[i][0].shape) for i in range(len(ds))]
    assert shapes and all(s == (24, 100) for s in shapes)  # all sliced 32 -> 24


def test_short_song_padding_keeps_and_masks(tmp_path):
    SEG, K, PAD = 100, 24, 1024
    packed = _pack(tmp_path, {"A": 250, "B": 40, "C": 150}, stored_k=32)

    def kept(pad_short):
        names = set()
        rows = []
        for split in ("train", "val"):
            ds = TokenDataset.from_mmap(
                load_mmap_bundle(packed, segment_frames=SEG, val_ratio=0.34, pad_short=pad_short),
                split, SEG, n_codebooks=K, pad_short=pad_short, pad_id=PAD)
            names |= set(ds.names)
            rows += [(ds.names[i], ds[i][0]) for i in range(len(ds))]
        return names, dict(rows)

    names, rows = kept(True)
    assert names == {"song_A", "song_B", "song_C"}  # short song kept
    assert all(t.shape == (K, SEG) for t in rows.values())
    b = rows["song_B"]
    assert (b[:, 40:] == PAD).all() and (b[:, :40] != PAD).any()  # tail padded, head intact

    names_off, _ = kept(False)
    assert "song_B" not in names_off  # legacy drops the short song


# --------------------------------------------------------------------------- EMA
def test_model_ema_update_and_swap():
    import torch.nn as nn

    from diskrot.train import ModelEMA

    torch.manual_seed(0)
    m = nn.Linear(8, 8)
    ema = ModelEMA(m, decay=0.9)
    w0 = ema.shadow["weight"].clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    assert torch.allclose(ema.shadow["weight"], 0.9 * w0 + 0.1 * (w0 + 1.0), atol=1e-5)

    live = m.weight.detach().clone()
    ema.copy_to(m)
    assert torch.allclose(m.weight, ema.shadow["weight"].to(m.weight.dtype), atol=1e-5)
    ema.restore(m)
    assert torch.allclose(m.weight, live, atol=1e-6)  # live weights restored

    ss = ema.served_state(m)
    assert set(ss) == set(m.state_dict()) and torch.allclose(ss["weight"], ema.shadow["weight"], atol=1e-5)

    ema2 = ModelEMA(nn.Linear(8, 8), 0.9)
    ema2.load_state_dict(ema.state_dict())
    assert torch.allclose(ema2.shadow["weight"], ema.shadow["weight"], atol=1e-6)


# --------------------------------------------------------- tokenize loudness + gate
def test_loudness_uniform_gain_and_clip_guard():
    import diskrot.tokenize as tk

    t = np.linspace(0, 1, 48000, endpoint=False).astype(np.float32)
    st = np.stack([0.03 * np.sin(2 * np.pi * 220 * t), 0.05 * np.sin(2 * np.pi * 330 * t)])
    out = tk._normalize_loudness(st.copy(), 48000)
    nz = np.abs(st) > 1e-3
    c = out[nz] / st[nz]
    assert np.allclose(c, c.mean(), rtol=1e-4)  # uniform scalar gain (stereo image preserved)
    assert np.abs(out).max() <= 1.0
    assert np.abs(tk._normalize_loudness(np.ones((2, 100), np.float32) * 5.0, 48000)).max() <= 1.0


def test_quality_gate_conservative():
    import diskrot.tokenize as tk

    t = np.linspace(0, 1, 48000, endpoint=False).astype(np.float32)
    assert tk._audio_quality_reason(0.12 * np.sin(2 * np.pi * 220 * t), 48000) is None
    assert tk._audio_quality_reason(0.98 * np.sin(2 * np.pi * 220 * t), 48000) is None  # loud master kept
    assert tk._audio_quality_reason(np.zeros(48000, np.float32), 48000) == "silent"
    assert tk._audio_quality_reason(0.001 * np.sin(2 * np.pi * 220 * t), 48000) == "silent"
    sq = np.sign(np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert tk._audio_quality_reason(sq, 48000) == "clipped"


# ----------------------------------------------------- lyric filter (multilingual-safe)
def test_filter_keeps_non_english_drops_hallucinations():
    from diskrot.filter_lyrics import hallucination_reason

    def words(n):
        return [{"word": "x", "start": 0.0, "end": 0.1} for _ in range(n)]

    assert hallucination_reason({"words": words(50), "avg_logprob": -0.3}) is None
    assert hallucination_reason({"words": words(40), "text": "我爱你"}) is None  # non-EN kept
    assert hallucination_reason({"words": words(40), "avg_logprob": -1.0}) is None  # moderate lp kept
    assert hallucination_reason({"words": words(2)}) == "short"
    assert hallucination_reason({"words": words(40), "avg_logprob": -3.0}) == "low_confidence"


# ---------------------------------------------------------------------- GPTConfig K
def test_gptconfig_k24_shapes():
    from model.nano_audio_gpt import GPTConfig

    c = GPTConfig(n_codebooks=24)
    assert c.n_codebooks == 24 and c.vocab_per_codebook == 1024 and c.pad_id == 1024


# ------------------------------------------------------------- multilingual (W9)
def test_phoneme_vocab_unique_and_sized():
    import model.lyric_encoder as le

    assert len(set(le.PHONEME_VOCAB)) == len(le.PHONEME_VOCAB) == le.PHONEME_VOCAB_SIZE == 256
    assert le.PAD_PHONEME_ID == 0  # PAD stays at id 0
    # IPA phones + language markers are present
    assert all(c in le.PHONEME_TO_ID for c in ["ɛ", "ʃ", "ŋ", "θ", "ː"])
    assert le.PHONEME_TO_ID["<lang_fr>"] and le.PHONEME_TO_ID["<lang_zh>"]


def test_lang_mapper_aliases_and_unknown():
    import model.lyric_encoder as le

    assert le.lang_label_to_id("en") == le.PHONEME_TO_ID["<lang_en>"]
    assert le.lang_label_to_id("french") == le.PHONEME_TO_ID["<lang_fr>"]
    assert le.lang_label_to_id("mandarin") == le.lang_label_to_id("lang:zh") == le.PHONEME_TO_ID["<lang_zh>"]
    assert le.lang_label_to_id("klingon") == le.UNKNOWN_LANG_ID
    assert le.is_lang_label("japanese") and le.is_lang_label("ko") and not le.is_lang_label("chorus")
    assert le.parse_lang_label("spanish") == "es" and le.parse_lang_label("klingon") is None


def test_inference_header_has_lang_slot():
    """The inference parser's 6-marker header (with <lang>) matches the train order
    BOS <gender> <tempo> <key> <vocals> <lang> <section>. Markers-only path so
    espeak isn't required."""
    import model.lyric_encoder as le

    ids = le.text_with_markers_to_phoneme_ids("[male] [spanish] [verse]")
    expected = [
        le.BOS_PHONEME_ID,
        le.gender_label_to_id("male"), le.bpm_to_id(None), le.key_label_to_id(None),
        le.UNKNOWN_VOCALS_ID, le.lang_label_to_id("es"), le.structure_label_to_id("verse"),
    ]
    assert ids == expected
