"""Tests for the server's multi-model registry (server/main.py).

Covers NANO_MODELS parsing and the lazy `_get_engine` selector — the switch
between checkpoints (e.g. the 2.0B teacher vs the distilled student) at serve
time. The real InferenceEngine is stubbed so these stay fast and GPU-free.
"""
from __future__ import annotations

import pytest

import server.main as sm


class _StubEngine:
    """Minimal stand-in for InferenceEngine — records which ckpt it 'loaded'."""
    instances = 0

    def __init__(self, ckpt_path=None, device=None):
        type(self).instances += 1
        self.ckpt_path = ckpt_path
        self.ckpt_step = 0


@pytest.fixture
def stub_engine(monkeypatch):
    _StubEngine.instances = 0
    monkeypatch.setattr(sm, "InferenceEngine", _StubEngine)
    # Fresh registry per test.
    monkeypatch.setattr(sm, "ENGINES", {})
    monkeypatch.setattr(sm, "engine", None)
    monkeypatch.setattr(sm, "ACTIVE_MODEL", None)
    return _StubEngine


# ----- _parse_models_env -----

def test_parse_models_env_unset_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("NANO_MODELS", raising=False)
    monkeypatch.delenv("NANO_CKPT", raising=False)
    models, default = sm._parse_models_env()
    assert default == "default"
    assert models == {"default": None}


def test_parse_models_env_unset_uses_nano_ckpt(monkeypatch):
    monkeypatch.delenv("NANO_MODELS", raising=False)
    monkeypatch.setenv("NANO_CKPT", "/ckpts/v8_sing4/best.pt")
    models, default = sm._parse_models_env()
    assert models == {"default": "/ckpts/v8_sing4/best.pt"}


def test_parse_models_env_multi(monkeypatch):
    monkeypatch.setenv(
        "NANO_MODELS", "fast=/ckpts/v8_distill/best_inference.pt,full=/ckpts/v8_sing4/best_inference.pt")
    monkeypatch.delenv("NANO_DEFAULT_MODEL", raising=False)
    models, default = sm._parse_models_env()
    assert list(models) == ["fast", "full"]
    assert models["full"] == "/ckpts/v8_sing4/best_inference.pt"
    assert default == "fast"  # first listed when NANO_DEFAULT_MODEL unset


def test_parse_models_env_explicit_default(monkeypatch):
    monkeypatch.setenv("NANO_MODELS", "fast=/a.pt,full=/b.pt")
    monkeypatch.setenv("NANO_DEFAULT_MODEL", "full")
    _, default = sm._parse_models_env()
    assert default == "full"


def test_parse_models_env_bad_default_raises(monkeypatch):
    monkeypatch.setenv("NANO_MODELS", "fast=/a.pt")
    monkeypatch.setenv("NANO_DEFAULT_MODEL", "nope")
    with pytest.raises(ValueError):
        sm._parse_models_env()


def test_parse_models_env_malformed_entry_raises(monkeypatch):
    monkeypatch.setenv("NANO_MODELS", "fast")  # no '=path'
    with pytest.raises(ValueError):
        sm._parse_models_env()


# ----- _get_engine (lazy selection / switching) -----

def test_get_engine_lazy_loads_and_caches(stub_engine, monkeypatch):
    monkeypatch.setattr(sm, "MODELS", {"fast": "/fast.pt", "full": "/full.pt"})
    monkeypatch.setattr(sm, "DEFAULT_MODEL", "fast")

    e1 = sm._get_engine("fast")
    assert e1.ckpt_path == "/fast.pt"
    assert sm.ACTIVE_MODEL == "fast"
    assert stub_engine.instances == 1
    # Second call to the same id reuses the cached engine (no new load).
    e1b = sm._get_engine("fast")
    assert e1b is e1
    assert stub_engine.instances == 1
    # Switching loads the other engine lazily and makes it active.
    e2 = sm._get_engine("full")
    assert e2.ckpt_path == "/full.pt"
    assert sm.ACTIVE_MODEL == "full"
    assert stub_engine.instances == 2
    # Switching back reuses the first (still cached).
    assert sm._get_engine("fast") is e1
    assert stub_engine.instances == 2


def test_get_engine_empty_selects_default(stub_engine, monkeypatch):
    monkeypatch.setattr(sm, "MODELS", {"fast": "/fast.pt", "full": "/full.pt"})
    monkeypatch.setattr(sm, "DEFAULT_MODEL", "full")
    e = sm._get_engine("")
    assert e.ckpt_path == "/full.pt"
    assert sm.ACTIVE_MODEL == "full"


def test_get_engine_unknown_raises_400(stub_engine, monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(sm, "MODELS", {"fast": "/fast.pt"})
    monkeypatch.setattr(sm, "DEFAULT_MODEL", "fast")
    with pytest.raises(HTTPException) as ei:
        sm._get_engine("bogus")
    assert ei.value.status_code == 400
