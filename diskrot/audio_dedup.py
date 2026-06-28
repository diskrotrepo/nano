"""Near-duplicate AUDIO dedup (catch re-uploads SHA-256 misses).

``prepare`` already drops byte-identical files (SHA-256), but the same song
re-encoded at a different bitrate, with different tags/padding, or from a different
source is byte-DIFFERENT yet acoustically identical. At corpus scale those
near-duplicates over-represent a handful of tracks, waste model capacity, and invite
memorization. This finds them acoustically.

The approach (all scale-safe — NO all-pairs over 1.3M):
  1. ``fpcalc`` (chromaprint) → a raw acoustic fingerprint per song (a list of 32-bit
     subfingerprints over the first ~2 min). Re-encodes of the same audio produce
     nearly identical fingerprint sequences.
  2. ``simhash64`` collapses that variable-length fingerprint into one 64-bit
     signature where acoustically-similar songs land at small Hamming distance.
  3. ``group_near_dups`` buckets signatures by LSH bands (songs only compared if they
     share a band → near-linear, not O(n^2)), Hamming-checks within buckets, and
     union-finds the survivors into duplicate groups.
  4. ``choose_keeper`` keeps the best copy of each group (highest bitrate, then the
     lexicographically-first name — same tie-break as prepare's SHA dedup).

This module is the pure, unit-tested core; ``modal_audio_dedup.py`` is the Modal stage
that runs fpcalc across CPU containers and applies the verdict (dry-run by default).
"""
from __future__ import annotations

import re
from collections import defaultdict

_MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    """A 64-bit avalanche hash (SplitMix64). Spreads a 32-bit subfingerprint's bits
    across all 64 so SimHash votes aren't biased toward chromaprint's low bits."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def simhash64(fingerprint: list[int]) -> int:
    """Collapse a chromaprint fingerprint (list of 32-bit ints) to a 64-bit SimHash.

    Acoustically-similar songs share most subfingerprints, so a few differing ones
    move only a handful of bit-votes → the signatures stay at small Hamming distance.
    Empty fingerprint → 0."""
    votes = [0] * 64
    for sub in fingerprint:
        h = _splitmix64(int(sub) & 0xFFFFFFFF)
        for b in range(64):
            votes[b] += 1 if (h >> b) & 1 else -1
    sig = 0
    for b in range(64):
        if votes[b] > 0:
            sig |= 1 << b
    return sig


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit signatures."""
    return bin((a ^ b) & _MASK64).count("1")


def parse_fpcalc_raw(output: str) -> list[int]:
    """Parse ``fpcalc -raw`` stdout (``FINGERPRINT=12,34,...``) into a list of ints.

    Accepts the whole stdout (with or without the ``DURATION=`` line). Returns []
    when no fingerprint line is present (caller treats that as un-fingerprintable)."""
    m = re.search(r"FINGERPRINT=([0-9,\-]+)", output)
    if not m:
        return []
    return [int(tok) for tok in m.group(1).split(",") if tok]


def band_keys(sig: int, n_bands: int = 4) -> list[tuple[int, int]]:
    """LSH band keys for a signature: split 64 bits into ``n_bands`` equal bands;
    two signatures are only compared if they share a (band_index, band_value) key.
    More bands = more recall (catches more distant dups) at more candidate pairs."""
    band_bits = 64 // n_bands
    mask = (1 << band_bits) - 1
    return [(b, (sig >> (b * band_bits)) & mask) for b in range(n_bands)]


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def group_near_dups(
    sigs: dict[str, int],
    max_hamming: int = 3,
    n_bands: int = 4,
    max_bucket: int = 2_000,
) -> list[list[str]]:
    """Group near-duplicate songs from ``{name: simhash}``.

    Two songs are near-dups if their signatures are within ``max_hamming`` bits AND
    share at least one LSH band (the band filter keeps this near-linear). Returns a
    list of groups (each a sorted list of >= 2 names); singletons are omitted.

    ``max_bucket`` skips pathologically-large LSH buckets (e.g. thousands of true-
    silent files colliding) to avoid an O(n^2) blow-up; such buckets are reported by
    the caller, not silently compared. Raising ``n_bands`` / ``max_hamming`` trades
    recall for cost."""
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    for name, sig in sigs.items():
        for key in band_keys(sig, n_bands):
            buckets[key].append(name)

    uf = _UnionFind(sigs.keys())
    for members in buckets.values():
        if len(members) < 2 or len(members) > max_bucket:
            continue
        for i in range(len(members)):
            si = sigs[members[i]]
            for j in range(i + 1, len(members)):
                if hamming(si, sigs[members[j]]) <= max_hamming:
                    uf.union(members[i], members[j])

    groups: dict[str, list[str]] = defaultdict(list)
    for name in sigs:
        groups[uf.find(name)].append(name)
    return sorted(
        (sorted(g) for g in groups.values() if len(g) > 1),
        key=lambda g: g[0],
    )


def oversized_buckets(
    sigs: dict[str, int], n_bands: int = 4, max_bucket: int = 2_000,
) -> int:
    """Count LSH buckets skipped by ``group_near_dups`` for being too large — so the
    Modal stage can log how many candidates went uncompared (no silent truncation)."""
    buckets: dict[tuple[int, int], int] = defaultdict(int)
    for sig in sigs.values():
        for key in band_keys(sig, n_bands):
            buckets[key] += 1
    return sum(1 for c in buckets.values() if c > max_bucket)


def choose_keeper(group: list[str], bit_rates: dict[str, int | None]) -> str:
    """Pick the copy to KEEP from a duplicate group: highest bitrate wins, ties broken
    by the lexicographically-first name (same discipline as prepare's SHA dedup, so
    the two dedup passes agree on which copy survives)."""
    return min(group, key=lambda n: (-(bit_rates.get(n) or 0), n))


def dedup_verdict(
    sigs: dict[str, int],
    bit_rates: dict[str, int | None] | None = None,
    max_hamming: int = 3,
    n_bands: int = 4,
) -> tuple[list[list[str]], dict[str, str]]:
    """Full verdict: ``(groups, drop_to_keeper)`` where ``drop_to_keeper`` maps each
    droppable near-dup name to the keeper it duplicates. ``groups`` is every
    near-dup cluster (for the report); ``drop_to_keeper`` excludes the keepers."""
    bit_rates = bit_rates or {}
    groups = group_near_dups(sigs, max_hamming=max_hamming, n_bands=n_bands)
    drop_to_keeper: dict[str, str] = {}
    for g in groups:
        keeper = choose_keeper(g, bit_rates)
        for name in g:
            if name != keeper:
                drop_to_keeper[name] = keeper
    return groups, drop_to_keeper
