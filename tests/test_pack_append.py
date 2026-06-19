"""Guards for the append-capable packer (``diskrot.pack_cache.pack_append``) and
the stable, corpus-growth-invariant train/val split (``diskrot.dataset``).

Core invariant: appending a wave adds NEW shards (ids ``max+1…``) and merges the
index, leaving every prior shard **byte-identical** — so unbounded wave ingestion
never repacks the existing corpus. These tests need no GPU/Modal/librosa.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from diskrot import pack_cache
from diskrot.dataset import _build_mmap_split_index


def _make_wave(
    root: Path, name: str, n: int, T: int = 300, seed: int = 0, melody: bool = True,
) -> Path:
    """Write a wave dir of ``n`` synthetic int16 ``[9, T]`` ``.pt`` files (+ an
    optional ``[12, T]`` float16 ``.mel.npy`` each), uniquely named ``<name>_song_NNN``
    so multiple waves never collide. Returns the dir."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(seed)
    for i in range(n):
        codes = torch.randint(0, 1024, (9, T), generator=gen, dtype=torch.int16)
        song = f"{name}_song_{i:03d}"
        torch.save(codes, d / f"{song}.pt")
        if melody:
            rng = np.random.default_rng(seed * 1000 + i)
            chroma = rng.random((12, T), dtype=np.float32).astype(np.float16)
            np.save(d / f"{song}.mel.npy", chroma)
    return d


def _shard_file_hashes(out: Path) -> dict[str, str]:
    """SHA-256 of every per-shard file (``.bin``/``.json``/``.mel.bin``),
    EXCLUDING the global index (which is expected to change on append)."""
    h: dict[str, str] = {}
    for p in sorted(out.glob("packed_*")):
        if p.name == pack_cache.SHARD_INDEX_NAME:
            continue
        h[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return h


def test_append_grows_pack_prior_shards_byte_identical(tmp_path):
    out = tmp_path / "packed"
    waveA = _make_wave(tmp_path, "waveA", n=7, seed=1)
    pack_cache.pack_append(waveA, out, shard_target_songs=3,
                           mel_cache_dir=waveA, verbose=False)
    idxA = pack_cache.load_shard_index(out)
    assert idxA["n_songs_total"] == 7
    assert idxA["n_shards"] == 3          # 7 / 3 -> 3 + 3 + 1
    assert idxA["has_melody"] is True
    hashesA = _shard_file_hashes(out)

    waveB = _make_wave(tmp_path, "waveB", n=5, seed=2)
    pack_cache.pack_append(waveB, out, shard_target_songs=3,
                           mel_cache_dir=waveB, verbose=False)
    idxB = pack_cache.load_shard_index(out)

    # Pack strictly grew; new shard ids continue from max+1 (no gap / collision).
    assert idxB["n_songs_total"] == 12
    assert idxB["n_shards"] == 5          # +2 (5 / 3 -> 2)
    assert [s["shard_id"] for s in idxB["shards"]] == list(range(5))
    # All wave-A entries still present, before wave-B's (index merged, not rebuilt).
    assert idxB["shards"][:3] == idxA["shards"]
    # CORE INVARIANT: every wave-A shard file is byte-identical after the append.
    hashesB = _shard_file_hashes(out)
    for fname, digest in hashesA.items():
        assert hashesB[fname] == digest, f"{fname} changed on append — not append-safe!"


def test_append_cold_start_no_existing_index(tmp_path):
    out = tmp_path / "packed"
    waveA = _make_wave(tmp_path, "waveA", n=4, seed=1)
    pack_cache.pack_append(waveA, out, shard_target_songs=2,
                           mel_cache_dir=waveA, verbose=False)
    idx = pack_cache.load_shard_index(out)
    assert idx["n_shards"] == 2 and idx["n_songs_total"] == 4
    assert [s["shard_id"] for s in idx["shards"]] == [0, 1]


def test_append_roundtrip_byte_exact(tmp_path):
    """Every appended song reads back byte-identical via the mmap."""
    out = tmp_path / "packed"
    waveA = _make_wave(tmp_path, "waveA", n=3, T=250, seed=1)
    waveB = _make_wave(tmp_path, "waveB", n=3, T=270, seed=2)
    pack_cache.pack_append(waveA, out, shard_target_songs=2, mel_cache_dir=waveA, verbose=False)
    pack_cache.pack_append(waveB, out, shard_target_songs=2, mel_cache_dir=waveB, verbose=False)
    wave_dirs = {"waveA": waveA, "waveB": waveB}
    seen = 0
    for shard_id, local_idx, name, n_frames in pack_cache.iter_all_names(out):
        mm, meta = pack_cache.open_shard_mmap(out, shard_id)
        try:
            s, e = meta["offsets"][local_idx], meta["offsets"][local_idx + 1]
            got = np.array(mm[:, s:e])
        finally:
            del mm
        wave = name.split("_song_")[0]
        original = torch.load(wave_dirs[wave] / f"{name}.pt", weights_only=True)
        assert np.array_equal(got, original.numpy()), f"{name} mismatch"
        seen += 1
    assert seen == 6


def test_append_orphan_recovery_idempotent_merge(tmp_path):
    """Re-running an append whose shards committed but whose index merge was
    interrupted reconstructs the SAME merged index and re-adopts the orphan
    shards (byte-identical), rather than duplicating or rebuilding them."""
    out = tmp_path / "packed"
    waveA = _make_wave(tmp_path, "waveA", n=4, seed=1)
    waveB = _make_wave(tmp_path, "waveB", n=4, seed=2)
    pack_cache.pack_append(waveA, out, shard_target_songs=2, mel_cache_dir=waveA, verbose=False)
    idxA_text = (out / pack_cache.SHARD_INDEX_NAME).read_text()  # A-only index

    pack_cache.pack_append(waveB, out, shard_target_songs=2, mel_cache_dir=waveB, verbose=False)
    idxAB = pack_cache.load_shard_index(out)
    hashesAB = _shard_file_hashes(out)

    # Simulate interruption AFTER B's shards committed but BEFORE the index merge
    # landed: roll the on-disk index back to A-only (B shards are now orphans).
    (out / pack_cache.SHARD_INDEX_NAME).write_text(idxA_text)

    # Re-run the same wave append: base_shard_id derives from the (A-only) index,
    # so it targets B's ids again, finds them valid, re-adopts, and re-merges.
    pack_cache.pack_append(waveB, out, shard_target_songs=2, mel_cache_dir=waveB, verbose=False)
    idx_recovered = pack_cache.load_shard_index(out)
    assert idx_recovered["shards"] == idxAB["shards"]
    assert idx_recovered["n_songs_total"] == idxAB["n_songs_total"] == 8
    # Orphan shards re-adopted untouched (not rewritten).
    assert _shard_file_hashes(out) == hashesAB


def test_append_melody_consistency_gate(tmp_path):
    # has_melody=True pack: appending WITHOUT mel_cache_dir must fail fast.
    out = tmp_path / "packed"
    waveA = _make_wave(tmp_path, "waveA", n=3, seed=1, melody=True)
    pack_cache.pack_append(waveA, out, shard_target_songs=5, mel_cache_dir=waveA, verbose=False)
    waveB = _make_wave(tmp_path, "waveB", n=3, seed=2, melody=False)
    with pytest.raises(ValueError, match="has_melody=True"):
        pack_cache.pack_append(waveB, out, shard_target_songs=5, verbose=False)

    # has_melody=False pack: appending WITH melody must fail fast.
    out2 = tmp_path / "packed2"
    waveC = _make_wave(tmp_path, "waveC", n=3, seed=3, melody=False)
    pack_cache.pack_append(waveC, out2, shard_target_songs=5, verbose=False)
    waveD = _make_wave(tmp_path, "waveD", n=3, seed=4, melody=True)
    with pytest.raises(ValueError, match="has_melody=False"):
        pack_cache.pack_append(waveD, out2, shard_target_songs=5, mel_cache_dir=waveD, verbose=False)


def test_pack_survives_wave_cleanup(tmp_path):
    """Deleting a wave's source .pt/.mel.npy (the per-wave inode reclaim) leaves
    the pack fully readable — shards are the source of truth."""
    out = tmp_path / "packed"
    waveA = _make_wave(tmp_path, "waveA", n=5, seed=1)
    pack_cache.pack_append(waveA, out, shard_target_songs=3, mel_cache_dir=waveA, verbose=False)
    for p in list(waveA.glob("*")):   # simulate per-wave cleanup
        p.unlink()
    names = [n for _, _, n, _ in pack_cache.iter_all_names(out)]
    assert len(names) == 5
    train, val, _ = _build_mmap_split_index(out, segment_frames=100, val_ratio=0.2, seed=42)
    assert len(train) + len(val) == 5


def test_stable_split_invariant_to_corpus_growth(tmp_path):
    """A song's train/val assignment must not change when the corpus grows."""
    out = tmp_path / "packed"
    waveA = _make_wave(tmp_path, "waveA", n=40, T=300, seed=1)
    pack_cache.pack_append(waveA, out, shard_target_songs=20, mel_cache_dir=waveA, verbose=False)
    trainA, valA, _ = _build_mmap_split_index(out, segment_frames=100, val_ratio=0.2, seed=42)
    valA_names = {e[2] for e in valA}
    namesA = {e[2] for e in trainA} | valA_names
    assert len(namesA) == 40
    assert 0 < len(valA_names) < 40           # split is non-trivial

    waveB = _make_wave(tmp_path, "waveB", n=40, T=300, seed=2)
    pack_cache.pack_append(waveB, out, shard_target_songs=20, mel_cache_dir=waveB, verbose=False)
    _, valB, _ = _build_mmap_split_index(out, segment_frames=100, val_ratio=0.2, seed=42)
    valB_names = {e[2] for e in valB}

    for name in namesA:                       # STABILITY across growth
        assert (name in valA_names) == (name in valB_names), \
            f"{name} moved train<->val when the corpus grew"
