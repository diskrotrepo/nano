"""Shared test fixtures.

The repo uses flat top-level packages (``model``, ``diskrot``, ``server``)
declared via setuptools' ``packages.find``. When tests are run from the repo
root, pytest's rootdir-on-syspath behavior already makes those importable, but
we add the parent dir explicitly so the suite also works when invoked from
inside ``tests/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def synth_tokens_dir(tmp_path: Path):
    """Factory fixture: writes N int16 .pt files of shape [n_codebooks, T] into
    a fresh tmp_path subdir and returns the dir. The token values are random
    within [0, vocab) — enough for shape/dtype/split tests; no semantic content.

    Returned callable signature:
        make(n_files: int = 5, T: int = 1000, n_codebooks: int = 9,
             vocab: int = 1024, seed: int = 0) -> Path

    ``T`` may be an int (all files same length) or a list[int] of length
    ``n_files`` (per-file length, useful for testing the too-short skip path).
    """
    def _make(
        n_files: int = 5,
        T: int | list[int] = 1000,
        n_codebooks: int = 9,
        vocab: int = 1024,
        seed: int = 0,
    ) -> Path:
        gen = torch.Generator().manual_seed(seed)
        out = tmp_path / "tokens"
        out.mkdir(exist_ok=True)
        if isinstance(T, int):
            lengths = [T] * n_files
        else:
            assert len(T) == n_files, "len(T) must equal n_files"
            lengths = T
        for i, t in enumerate(lengths):
            codes = torch.randint(0, vocab, (n_codebooks, t), generator=gen, dtype=torch.int16)
            torch.save(codes, out / f"song_{i:03d}.pt")
        return out

    return _make
