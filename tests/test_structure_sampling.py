"""Tests for diskrot/structure._sample_keep — deterministic subset sampling.

The structure (allin1) pass is the most expensive optional data-prep step, so it
now runs on only a stem-hash-keyed fraction of the corpus (the rest fall back to
``<no_section>``). The predicate must be (1) deterministic across processes — so a
re-run with the same pct never re-decides membership and resume-by-skip stays
correct — (2) hit the requested fraction, and (3) be INDEPENDENT of the shard-bucket
grid so the kept set isn't correlated with shard id.
"""
from __future__ import annotations

import hashlib

from diskrot.structure import _sample_keep, _struct_bucket


def _stems(n: int) -> list[str]:
    return [f"wave_{i:07d}" for i in range(n)]


def test_boundaries_keep_all_or_none():
    for stem in _stems(50):
        assert _sample_keep(stem, 100) is True
        assert _sample_keep(stem, 1000) is True   # >=100 keeps all
        assert _sample_keep(stem, 0) is False
        assert _sample_keep(stem, -5) is False


def test_deterministic_and_uses_sha1_not_builtin_hash():
    # Stable across calls, and pinned to the exact sha1 formula — a switch to the
    # process-salted builtin hash() (or any other grid) would diverge here and
    # break resume across containers/restarts.
    for stem in _stems(200):
        expected = int(hashlib.sha1(("sample:" + stem).encode()).hexdigest()[:8], 16) % 100 < 37
        assert _sample_keep(stem, 37) == expected
        assert _sample_keep(stem, 37) == expected  # idempotent


def test_kept_fraction_matches_pct():
    stems = _stems(20000)
    for pct in (25, 30, 50, 75):
        kept = sum(_sample_keep(s, pct) for s in stems)
        frac = kept / len(stems)
        assert abs(frac - pct / 100) < 0.02, f"pct={pct}: got {frac:.3f}"


def test_monotone_in_pct():
    # A higher pct keeps a superset (each stem crosses its own fixed threshold once).
    stems = _stems(5000)
    kept30 = {s for s in stems if _sample_keep(s, 30)}
    kept60 = {s for s in stems if _sample_keep(s, 60)}
    assert kept30 <= kept60


def test_independent_of_shard_bucket_grid():
    # The 'sample:' salt decorrelates sampling from _struct_bucket: the kept-rate
    # among stems in even shard buckets must match the kept-rate among odd ones
    # (≈ pct), i.e. sampling doesn't favor any shard.
    stems = _stems(20000)
    even = [s for s in stems if _struct_bucket(s) % 2 == 0]
    odd = [s for s in stems if _struct_bucket(s) % 2 == 1]
    rate_even = sum(_sample_keep(s, 50) for s in even) / len(even)
    rate_odd = sum(_sample_keep(s, 50) for s in odd) / len(odd)
    assert abs(rate_even - 0.5) < 0.03 and abs(rate_odd - 0.5) < 0.03
    assert abs(rate_even - rate_odd) < 0.04
