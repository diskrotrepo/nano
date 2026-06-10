"""Tests for diskrot/key_detect.py — Krumhansl key estimation over packed chroma.

Two surfaces: (1) ``estimate_key`` must label obviously-keyed chroma correctly
and refuse degenerate input (the packer zero-fills missing melody — guessing a
key there would poison the marker), and (2) ``detect_keys`` must sweep a packed
layout, emit the dataset's keys.json schema, and resume by skip.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from diskrot.key_detect import detect_keys, estimate_key


def _triad(*bins, weights=(1.0, 0.8, 0.9)) -> np.ndarray:
    c = np.zeros(12)
    for b, w in zip(bins, weights):
        c[b] = w
    return c


def test_estimate_key_major_minor_and_rotation():
    assert estimate_key(_triad(0, 4, 7)) == "c_major"     # C E G
    assert estimate_key(_triad(9, 0, 4)) == "a_minor"     # A C E
    # The estimator is rotation-equivariant: shift C major up 6 semitones -> F#.
    assert estimate_key(np.roll(_triad(0, 4, 7), 6)) == "f_sharp_major"


def test_estimate_key_rejects_degenerate():
    assert estimate_key(np.zeros(12)) is None        # packer zero-fill
    assert estimate_key(np.ones(12)) is None         # flat — no key information
    assert estimate_key(np.full(12, np.nan)) is None
    assert estimate_key(np.zeros(13)) is None        # wrong shape


def test_detect_keys_sweeps_pack_and_resumes(tmp_path):
    from diskrot.pack_cache import pack

    cache = tmp_path / "tokens"
    cache.mkdir()
    T = 600
    # song0: C major chroma. song1: all-zero chroma (missing melody -> skipped).
    chroma = np.tile(_triad(0, 4, 7)[:, None], (1, T)).astype(np.float16)
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True)
    for name, mel in (("song0", chroma), ("song1", np.zeros((12, T), np.float16))):
        torch.save(torch.randint(0, 1024, (9, T), dtype=torch.int16), cache / f"{name}.pt")
        np.save(cache / f"{name}.mel.npy", mel)
    pack(cache, mel_cache_dir=cache, shard_target_songs=5, verbose=False)

    out = detect_keys(cache, verbose=False)
    payload = json.loads(out.read_text())
    assert payload == {"song0": {"key": "c_major"}}  # zero-chroma song omitted

    # Resume-by-skip: a second run leaves the existing estimate untouched.
    out.write_text(json.dumps({"song0": {"key": "b_minor"}}))  # poison to prove skip
    detect_keys(cache, verbose=False)
    assert json.loads(out.read_text())["song0"]["key"] == "b_minor"

    # The dataset loader consumes the schema as the <key_*> marker source.
    out.write_text(json.dumps(payload))
    from diskrot.dataset import _load_keys
    assert _load_keys(out, verbose=False) == {"song0": "c_major"}
