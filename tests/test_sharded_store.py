"""Guards for the hash-sharded metadata store (diskrot.sharded_store) and the
dataset loaders' single-file-or-sharded-dir branch."""
from __future__ import annotations

import json

from diskrot import sharded_store as ss
from diskrot.dataset import _load_keys, _load_tags


def test_round_trip_write_load(tmp_path):
    mapping = {f"song_{i:04d}": {"description": f"tags {i}"} for i in range(500)}
    d = tmp_path / "tags"
    ss.write_json_shards(d, "tags", mapping)
    shards = list(d.glob("tags_*.json"))
    assert 1 < len(shards) <= ss.N_SHARDS         # sharded, not one big file
    assert ss.load_json_shards(d, "tags") == mapping


def test_shard_index_stable_and_bounded():
    for name in ("a", "song_123", "weird name", "ünïçødé"):
        s = ss.shard_index(name)
        assert 0 <= s < ss.N_SHARDS
        assert ss.shard_index(name) == s          # deterministic


def test_only_shards_touched_flush(tmp_path):
    """A flush with only_shards rewrites just the changed shard — the O(touched)
    flush that keeps metadata writes cheap as the corpus grows."""
    mapping = {f"s{i}": {"key": "a_minor"} for i in range(300)}
    d = tmp_path / "keys"
    ss.write_json_shards(d, "keys", mapping)
    before = {p.name: p.read_bytes() for p in d.glob("keys_*.json")}

    new = "newsong"
    mapping[new] = {"key": "c_major"}
    ss.write_json_shards(d, "keys", mapping, only_shards={ss.shard_index(new)})
    after = {p.name: p.read_bytes() for p in d.glob("keys_*.json")}

    changed = [n for n in before if before[n] != after.get(n)]
    assert len(changed) <= 1                      # at most the touched shard
    assert ss.load_json_shards(d, "keys")[new] == {"key": "c_major"}


def test_convert_file_to_shards(tmp_path):
    single = tmp_path / "tags.json"
    mapping = {f"song_{i}": {"description": f"d{i}"} for i in range(120)}
    single.write_text(json.dumps(mapping))
    out = tmp_path / "tags"
    assert ss.convert_file_to_shards(single, out, "tags") == 120
    assert ss.load_json_shards(out, "tags") == mapping


def test_dataset_loaders_accept_single_or_sharded(tmp_path):
    tags = {f"song_{i}": {"description": f"vibe {i}"} for i in range(50)}
    keys = {f"song_{i}": {"key": "a_minor"} for i in range(50)}
    (tmp_path / "tags.json").write_text(json.dumps(tags))
    (tmp_path / "keys.json").write_text(json.dumps(keys))
    t1 = _load_tags(tmp_path / "tags.json", verbose=False)
    k1 = _load_keys(tmp_path / "keys.json", verbose=False)

    ss.write_json_shards(tmp_path / "tags", "tags", tags)
    ss.write_json_shards(tmp_path / "keys", "keys", keys)
    t2 = _load_tags(tmp_path / "tags", verbose=False)
    k2 = _load_keys(tmp_path / "keys", verbose=False)

    assert t1 == t2 == {f"song_{i}": f"vibe {i}" for i in range(50)}
    assert k1 == k2 == {f"song_{i}": "a_minor" for i in range(50)}
