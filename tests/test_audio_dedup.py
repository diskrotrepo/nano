"""Tests for diskrot/audio_dedup.py — near-duplicate audio detection.

The SimHash + LSH-banding core is pure, so explicit signatures exercise the grouping
logic deterministically (no fpcalc / audio files needed)."""
from __future__ import annotations

import random

from diskrot.audio_dedup import (
    band_keys,
    choose_keeper,
    dedup_verdict,
    group_near_dups,
    hamming,
    oversized_buckets,
    parse_fpcalc_raw,
    simhash64,
)


def test_simhash_deterministic_and_identical():
    fp = [1, 2, 3, 4, 5, 999, 123456]
    assert simhash64(fp) == simhash64(fp)
    assert simhash64([]) == 0


def test_simhash_small_change_stays_close_far_change_is_far():
    rng = random.Random(0)
    base = [rng.getrandbits(32) for _ in range(400)]
    near = list(base)
    for i in (10, 120, 250):  # replace 3 of 400 subfingerprints (a light re-encode)
        near[i] = rng.getrandbits(32)
    far = [rng.getrandbits(32) for _ in range(400)]
    a, b, c = simhash64(base), simhash64(near), simhash64(far)
    assert hamming(a, b) <= 4              # a light perturbation barely moves the sig
    assert hamming(a, c) > 8               # an independent fingerprint is far
    assert hamming(a, b) < hamming(a, c)


def test_hamming():
    assert hamming(0, 0) == 0
    assert hamming(0b1011, 0b0001) == 2


def test_parse_fpcalc_raw():
    assert parse_fpcalc_raw("DURATION=123\nFINGERPRINT=10,20,30\n") == [10, 20, 30]
    assert parse_fpcalc_raw("FINGERPRINT=-5,7") == [-5, 7]
    assert parse_fpcalc_raw("DURATION=99\n") == []  # no fingerprint line


def test_band_keys_partition():
    keys = band_keys(0xFFFFFFFFFFFFFFFF, n_bands=4)
    assert len(keys) == 4
    assert all(val == 0xFFFF for _, val in keys)
    assert [b for b, _ in keys] == [0, 1, 2, 3]


# Explicit signatures: two near pairs + one outlier.
A = 0
B = 0b111                          # hamming 3 from A
C = 0x00000000FFFFFFFF             # 32 bits — far from A/B
D = C ^ 0b11                       # hamming 2 from C
E = 0x0F0F0F0F0F0F0F0F             # shares no band with any of the above


def test_group_near_dups_clusters_correctly():
    sigs = {"a": A, "b": B, "c": C, "d": D, "e": E}
    groups = group_near_dups(sigs, max_hamming=3, n_bands=4)
    assert groups == [["a", "b"], ["c", "d"]]  # 'e' is a singleton → omitted


def test_within_threshold_always_shares_a_band():
    # Invariant: with n_bands=4, any two sigs <=3 bits apart share a band (pigeonhole),
    # so the LSH filter never misses a true near-dup at max_hamming=3.
    assert hamming(A, B) <= 3
    a_bands = set(band_keys(A, 4))
    b_bands = set(band_keys(B, 4))
    assert a_bands & b_bands


def test_far_pair_in_same_bucket_not_unioned():
    # A and C share band 2 and band 3 (both zero) so they're CANDIDATES, but the
    # Hamming check (32 > 3) correctly keeps them apart.
    assert set(band_keys(A, 4)) & set(band_keys(C, 4))
    groups = group_near_dups({"a": A, "c": C}, max_hamming=3, n_bands=4)
    assert groups == []


def test_choose_keeper_prefers_bitrate_then_name():
    # Higher bitrate wins.
    assert choose_keeper(["x", "y"], {"x": 128_000, "y": 320_000}) == "y"
    # Tie on bitrate → lexicographically-first name (matches prepare's SHA dedup).
    assert choose_keeper(["zebra", "apple"], {"zebra": 320_000, "apple": 320_000}) == "apple"
    # Missing bitrate counts as 0.
    assert choose_keeper(["x", "y"], {"y": 256_000}) == "y"


def test_dedup_verdict_maps_drops_to_keepers():
    sigs = {"a": A, "b": B, "c": C, "d": D, "e": E}
    bit_rates = {"a": 320_000, "b": 128_000, "c": 256_000, "d": 320_000, "e": 320_000}
    groups, drops = dedup_verdict(sigs, bit_rates, max_hamming=3, n_bands=4)
    assert groups == [["a", "b"], ["c", "d"]]
    # a kept (higher bitrate) → b drops to a; d kept (higher bitrate) → c drops to d.
    assert drops == {"b": "a", "c": "d"}
    assert "e" not in drops


def test_oversized_buckets_skipped():
    # 5 identical signatures with max_bucket=3 → the shared bucket is skipped, so no
    # group forms, and oversized_buckets reports the skipped bands.
    sigs = {f"s{i}": 0 for i in range(5)}
    assert group_near_dups(sigs, max_hamming=3, n_bands=4, max_bucket=3) == []
    assert oversized_buckets(sigs, n_bands=4, max_bucket=3) == 4  # all 4 bands oversized
