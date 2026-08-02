"""Endpoint wiring for POST /generate_batch and POST /generate_stream.

Exercises the real request parsing / manifest building / base64 encoding / stream
wrapping with a STUB engine bound onto the module globals (mirrors how
test_generate_stream stubs the engine), so no checkpoint / GPU / DAC is needed.
"""
from __future__ import annotations

import base64
import types

import pytest

import server.main as M


def _install_stub(monkeypatch, *, batch_fn=None, stream_fn=None):
    stub = types.SimpleNamespace()
    if batch_fn is not None:
        stub.generate_audio_batch = batch_fn
    if stream_fn is not None:
        stub.generate_audio_stream = stream_fn
    stub.sweeten_prompt = lambda p: p
    monkeypatch.setattr(M, "MODELS", {"default": None})
    monkeypatch.setattr(M, "DEFAULT_MODEL", "default")
    monkeypatch.setattr(M, "ENGINES", {"default": stub})
    monkeypatch.setattr(M, "engine", stub)
    monkeypatch.setattr(M, "OUTPUT_DIR", "")  # don't touch disk
    return stub


def test_generate_batch_count(monkeypatch):
    cap = {}

    def gen_batch(requests, *, seconds, temperature, top_k, top_p, cfg_scale,
                  lyric_cfg_scale=None, on_item=None):
        cap["requests"] = requests
        cap["seconds"] = seconds
        out = []
        for i in range(len(requests)):
            body = f"MP3-{i}".encode()
            if on_item is not None:
                on_item(i, body, "audio/mpeg")
            out.append((body, "audio/mpeg"))
        return out

    _install_stub(monkeypatch, batch_fn=gen_batch)
    resp = M.generate_batch_endpoint(
        M.BatchRequest(count=3, prompt="techno", sweeten=False, seconds=5.0)
    )
    assert resp["count"] == 3 and len(resp["items"]) == 3
    assert cap["seconds"] == 5.0 and len(cap["requests"]) == 3
    for i, it in enumerate(resp["items"]):
        assert it["req_id"]
        assert it["mime"] == "audio/mpeg"
        assert base64.b64decode(it["audio_b64"]) == f"MP3-{i}".encode()
        assert "_prompt" not in it  # internal scratch field stripped


def test_generate_batch_items_inherit_shared(monkeypatch):
    cap = {}

    def gen_batch(requests, **_kw):
        cap["requests"] = requests
        return [(b"x", "audio/mpeg") for _ in requests]

    _install_stub(monkeypatch, batch_fn=gen_batch)
    M.generate_batch_endpoint(M.BatchRequest(
        items=[M.BatchItem(prompt="alpha"), M.BatchItem(lyrics="la la la")],
        prompt="shared", sweeten=False,
    ))
    reqs = cap["requests"]
    assert reqs[0]["text"].startswith("alpha")          # per-item prompt wins
    # tags and lyrics are SEPARATE fields now (no ". " join): the inherited
    # shared prompt lands in text, the item's own lyrics in lyrics.
    assert reqs[1]["text"] == "shared" and "la la" in reqs[1]["lyrics"]


def test_generate_batch_too_large(monkeypatch):
    _install_stub(monkeypatch, batch_fn=lambda *a, **k: [])
    monkeypatch.setattr(M, "MAX_BATCH", 4)
    with pytest.raises(M.HTTPException) as e:
        M.generate_batch_endpoint(M.BatchRequest(count=5, sweeten=False))
    assert e.value.status_code == 400


def test_generate_stream_response_streams_bytes(monkeypatch):
    """The shared body behind GET+POST /generate_stream — bytes flow through and
    long lyrics reach the engine as their OWN field (not joined to the tags)."""
    cap = {}
    chunks = [b"ID3header", b"frame1", b"frame2"]

    def gen_stream(**kw):  # eager (records args at call time) + returns an iterable
        cap.update(kw)
        return chunks

    _install_stub(monkeypatch, stream_fn=gen_stream)
    resp = M._generate_stream_response(
        seconds=5.0, temperature=0.9, top_k=50, top_p=0.95,
        per_cb_temperature="", per_cb_top_k="", per_cb_top_p="",
        cfg_scale=7.0, prompt="techno", lyrics="word " * 400,  # long lyrics
        gender="", bpm=0.0, negative_prompt="", sweeten=False,
        lyric_cfg_scale=0.0, req_id="abc", model="",
    )
    assert resp.media_type == "audio/mpeg"
    # tags and long lyrics reach the engine as SEPARATE args (no ". " join).
    assert (cap.get("text") or "") == "techno"
    assert "word" in (cap.get("lyrics") or "")
