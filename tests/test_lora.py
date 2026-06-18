"""Tests for model/lora.py — LoRA injection, adapter extraction, and merging.

The contract pinned down here:
- injection is remap-free: base state-dict keys are unchanged (the weight
  Parameter is REUSED, not copied), adapters are purely additive keys
- a freshly injected model computes exactly the base model's output (lora_B
  zero-init), so fine-tuning starts from the base behavior
- merge_state_dicts folds adapters into the base key layout exactly, so a
  merged checkpoint loads strictly into a plain NanoAudioGPT and matches the
  lora model's forward
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from model.lora import (
    DEFAULT_TARGETS,
    LoRAConfig,
    LoRALinear,
    inject_lora,
    load_lora_state,
    lora_state_dict,
    mark_only_lora_trainable,
    merge_state_dicts,
)
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT


def _tiny_cfg(**overrides) -> GPTConfig:
    base = dict(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        max_seq_len=128,
        use_gradient_checkpointing=False,
    )
    base.update(overrides)
    return GPTConfig(**base)


def _tokens(cfg: GPTConfig, B: int = 2, T: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, cfg.vocab_per_codebook, (B, cfg.n_codebooks, T), generator=g)


@pytest.fixture(params=[False, True], ids=["plain", "text_cond"])
def model_and_cfg(request):
    cfg = _tiny_cfg(use_text_conditioning=request.param)
    torch.manual_seed(0)
    return NanoAudioGPT(cfg).eval(), cfg


def test_inject_preserves_base_keys_and_freezes(model_and_cfg):
    model, cfg = model_and_cfg
    base_sd = model.state_dict()
    base_keys = set(base_sd)
    # Hold a reference to a base weight tensor to prove it's reused, not copied.
    qkv_weight_before = model.blocks[0].attn.qkv.weight

    lcfg = LoRAConfig(r=4, alpha=8)
    replaced = inject_lora(model, lcfg)
    assert all(n.startswith("blocks.") for n in replaced)
    expected_per_block = 4 if not cfg.use_text_conditioning else 7
    assert len(replaced) == expected_per_block * cfg.n_layers

    new_sd = model.state_dict()
    extra = set(new_sd) - base_keys
    assert base_keys <= set(new_sd)
    assert all(k.endswith((".lora_A", ".lora_B")) for k in extra)
    assert len(extra) == 2 * len(replaced)
    # weight Parameter reused — same tensor object, not a copy
    assert model.blocks[0].attn.qkv.weight is qkv_weight_before

    n_trainable, n_total = mark_only_lora_trainable(model)
    assert 0 < n_trainable < n_total
    for name, p in model.named_parameters():
        assert p.requires_grad == name.endswith(("lora_A", "lora_B")), name
    # head / embeddings / norms untouched and frozen
    assert not isinstance(model.head, LoRALinear)
    assert not model.head.weight.requires_grad
    assert not model.tok_embeds[0].weight.requires_grad


def test_forward_identity_at_init(model_and_cfg):
    model, cfg = model_and_cfg
    tokens = _tokens(cfg)
    text_emb = torch.randn(2, 1, cfg.d_model) if cfg.use_text_conditioning else None
    with torch.no_grad():
        before = model(tokens, text_emb=text_emb)
    inject_lora(model, LoRAConfig(r=4, alpha=8))
    model.eval()
    with torch.no_grad():
        after = model(tokens, text_emb=text_emb)
    # lora_B is zero-initialized, so the delta is exactly zero.
    assert torch.equal(before, after)


def _randomize_lora_b(model: NanoAudioGPT, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("lora_B"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)


def test_merge_matches_lora_forward(model_and_cfg):
    model, cfg = model_and_cfg
    lcfg = LoRAConfig(r=4, alpha=8)
    inject_lora(model, lcfg)
    _randomize_lora_b(model)
    tokens = _tokens(cfg)
    text_emb = torch.randn(2, 1, cfg.d_model) if cfg.use_text_conditioning else None
    with torch.no_grad():
        lora_out = model(tokens, text_emb=text_emb)

    base_sd = {k: v for k, v in model.state_dict().items()
               if not k.endswith((".lora_A", ".lora_B"))}
    merged_sd = merge_state_dicts(base_sd, lora_state_dict(model),
                                  r=lcfg.r, alpha=lcfg.alpha)
    assert set(merged_sd) == set(base_sd)

    plain = NanoAudioGPT(cfg).eval()
    plain.load_state_dict(merged_sd)  # strict — proves key fidelity
    with torch.no_grad():
        merged_out = plain(tokens, text_emb=text_emb)
    assert torch.allclose(lora_out, merged_out, atol=1e-5)

    # Per-layer check: merged weight == base + (alpha/r)·B@A
    w_key = "0.attn.qkv.weight"
    A = model.blocks[0].attn.qkv.lora_A
    B = model.blocks[0].attn.qkv.lora_B
    expected = base_sd[f"blocks.{w_key}"] + (lcfg.alpha / lcfg.r) * (B @ A)
    assert torch.allclose(merged_sd[f"blocks.{w_key}"], expected, atol=1e-6)


def test_merge_state_dicts_key_fidelity():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg)
    lcfg = LoRAConfig(r=4, alpha=8)
    inject_lora(model, lcfg)
    _randomize_lora_b(model)
    lora_sd = lora_state_dict(model)
    base_sd = {k: v for k, v in model.state_dict().items() if k not in lora_sd}

    merged = merge_state_dicts(base_sd, lora_sd, r=lcfg.r, alpha=lcfg.alpha)
    assert set(merged) == set(base_sd)
    assert not any(k.endswith((".lora_A", ".lora_B")) for k in merged)

    # fp16 base: delta computed fp32, output dtype matches base
    base_fp16 = {k: v.half() for k, v in base_sd.items()}
    merged_fp16 = merge_state_dicts(base_fp16, lora_sd, r=lcfg.r, alpha=lcfg.alpha)
    assert all(v.dtype == torch.float16 for v in merged_fp16.values())

    # an adapter whose prefix isn't in the base raises
    bad = dict(lora_sd)
    bad["blocks.99.attn.qkv.lora_A"] = torch.zeros(4, cfg.d_model)
    bad["blocks.99.attn.qkv.lora_B"] = torch.zeros(3 * cfg.d_model, 4)
    with pytest.raises(KeyError):
        merge_state_dicts(base_sd, bad, r=lcfg.r, alpha=lcfg.alpha)

    # a non-adapter key smuggled into the lora state raises
    bad2 = dict(lora_sd)
    bad2["blocks.0.attn.qkv.weight"] = torch.zeros(3 * cfg.d_model, cfg.d_model)
    with pytest.raises(ValueError):
        merge_state_dicts(base_sd, bad2, r=lcfg.r, alpha=lcfg.alpha)


def test_adapter_state_roundtrip():
    cfg = _tiny_cfg()
    lcfg = LoRAConfig(r=4, alpha=8)

    torch.manual_seed(0)
    src = NanoAudioGPT(cfg).eval()
    base_sd = {k: v.clone() for k, v in src.state_dict().items()}
    inject_lora(src, lcfg)
    _randomize_lora_b(src)
    state = lora_state_dict(src)
    tokens = _tokens(cfg)
    with torch.no_grad():
        want = src(tokens)

    # Fresh model from the same base weights, freshly injected (different
    # random lora_A), then adapter state loaded over it.
    torch.manual_seed(123)
    dst = NanoAudioGPT(cfg).eval()
    dst.load_state_dict(base_sd)
    inject_lora(dst, lcfg)
    load_lora_state(dst, state)
    with torch.no_grad():
        got = dst(tokens)
    assert torch.equal(want, got)

    # Mismatched key sets (different targets) raise
    other = NanoAudioGPT(cfg)
    other.load_state_dict(base_sd)
    inject_lora(other, LoRAConfig(r=4, alpha=8, targets="fc1,fc2"))
    with pytest.raises(ValueError):
        load_lora_state(other, state)


def test_inject_no_match_raises():
    model = NanoAudioGPT(_tiny_cfg())
    with pytest.raises(ValueError):
        inject_lora(model, LoRAConfig(targets="does_not_exist"))


def test_lora_config_roundtrip():
    lcfg = LoRAConfig(r=8, alpha=16, dropout=0.1, targets="fc1,fc2")
    assert LoRAConfig.from_dict(lcfg.to_dict()) == lcfg
    assert LoRAConfig().targets == DEFAULT_TARGETS


# ---------------------------------------------------------------------------
# train.py plumbing: startup precedence (resume > init-from > scratch) and the
# adapter-only LoRA checkpoint schema.
# ---------------------------------------------------------------------------

from diskrot.train import (  # noqa: E402
    TrainConfig,
    _build_ckpt_dict,
    _resolve_startup,
    _restore_train_state,
)


def _save_base_ckpt(path, cfg: GPTConfig, step: int = 1000) -> NanoAudioGPT:
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg)
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "step": step,
        "best_val_loss": 1.23,
    }, path)
    return model


def _train_cfg(tmp_path, **overrides) -> TrainConfig:
    base = dict(ckpt_dir=str(tmp_path / "run"), model=_tiny_cfg())
    base.update(overrides)
    return TrainConfig(**base)


def test_resolve_startup_scratch(tmp_path):
    plan = _resolve_startup(_train_cfg(tmp_path), verbose=False)
    assert plan.mode == "scratch"
    assert plan.base_state is None and plan.lora_cfg is None


def test_resolve_startup_lora_without_init_raises(tmp_path):
    with pytest.raises(ValueError, match="init"):
        _resolve_startup(_train_cfg(tmp_path, lora=True), verbose=False)


def test_resolve_startup_init_mode(tmp_path):
    base_path = tmp_path / "base.pt"
    ckpt_cfg = _tiny_cfg(d_ff=256)  # differs from the flag-built cfg
    _save_base_ckpt(base_path, ckpt_cfg)
    plan = _resolve_startup(
        _train_cfg(tmp_path, init_from=str(base_path)), verbose=False)
    assert plan.mode == "init"
    assert plan.ckpt is None  # nothing to restore — fresh optimizer/step
    assert plan.model_cfg == ckpt_cfg  # architecture comes FROM the checkpoint
    assert plan.base_state is not None
    assert plan.base_ckpt_path == str(base_path)
    assert plan.lora_cfg is None

    lora_plan = _resolve_startup(
        _train_cfg(tmp_path, init_from=str(base_path), lora=True, lora_r=8),
        verbose=False)
    assert lora_plan.lora_cfg == LoRAConfig(r=8)
    assert lora_plan.lora_state is None  # fresh adapters


def test_resolve_startup_init_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_startup(
            _train_cfg(tmp_path, init_from=str(tmp_path / "nope.pt")),
            verbose=False)


def test_resolve_startup_resume_beats_init(tmp_path):
    cfg = _train_cfg(tmp_path)
    run_dir = Path(cfg.ckpt_dir)
    run_dir.mkdir(parents=True)
    _save_base_ckpt(run_dir / "latest.pt", cfg.model, step=500)
    base_path = tmp_path / "base.pt"
    _save_base_ckpt(base_path, cfg.model)

    cfg.init_from = str(base_path)
    plan = _resolve_startup(cfg, verbose=False)
    assert plan.mode == "resume"
    assert plan.ckpt is not None and plan.ckpt["step"] == 500
    assert _restore_train_state(plan.ckpt)["step"] == 500


def test_resolve_startup_lora_flag_on_full_latest_raises(tmp_path):
    cfg = _train_cfg(tmp_path, lora=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _save_base_ckpt(run_dir / "latest.pt", cfg.model)
    cfg.init_from = str(run_dir / "latest.pt")
    with pytest.raises(RuntimeError, match="non-LoRA"):
        _resolve_startup(cfg, verbose=False)


def _save_lora_ckpt(run_dir, model_cfg: GPTConfig, lcfg: LoRAConfig,
                    base_path: str, step: int = 700) -> dict:
    torch.manual_seed(0)
    model = NanoAudioGPT(model_cfg)
    inject_lora(model, lcfg)
    _randomize_lora_b(model)
    state = lora_state_dict(model)
    ckpt = {
        "lora": {"config": lcfg.to_dict(), "state": state, "base_ckpt": base_path},
        "optim": {},
        "step": step,
        "cfg": model_cfg.__dict__,
        "best_val_loss": 2.0,
    }
    torch.save(ckpt, run_dir / "latest.pt")
    return state


def test_resolve_startup_lora_resume(tmp_path):
    cfg = _train_cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    base_path = tmp_path / "base.pt"
    base_model = _save_base_ckpt(base_path, cfg.model)
    # The ckpt's lora config wins over (different) CLI flags.
    saved_lcfg = LoRAConfig(r=4, alpha=8)
    saved_state = _save_lora_ckpt(run_dir, cfg.model, saved_lcfg, str(base_path))
    cfg.lora, cfg.lora_r = True, 64

    plan = _resolve_startup(cfg, verbose=False)
    assert plan.mode == "resume"
    assert plan.lora_cfg == saved_lcfg
    assert set(plan.lora_state) == set(saved_state)
    assert plan.model_cfg == cfg.model
    assert plan.base_ckpt_path == str(base_path)
    # base weights resolved from the recorded path
    assert torch.equal(plan.base_state["head.weight"],
                       base_model.state_dict()["head.weight"])
    # ckpt carries no text_proj -> the proj stays frozen on resume
    assert cfg.lora_train_text_proj is False

    # Rebuild exactly as train_run would: load base, inject, restore adapters.
    rebuilt = NanoAudioGPT(plan.model_cfg)
    rebuilt.load_state_dict(plan.base_state)
    inject_lora(rebuilt, plan.lora_cfg)
    load_lora_state(rebuilt, plan.lora_state)
    got = lora_state_dict(rebuilt)
    assert all(torch.equal(got[k], saved_state[k]) for k in saved_state)


def test_resolve_startup_lora_resume_missing_base_raises(tmp_path):
    cfg = _train_cfg(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    _save_lora_ckpt(run_dir, cfg.model, LoRAConfig(r=4), str(tmp_path / "gone.pt"))
    with pytest.raises(FileNotFoundError):
        _resolve_startup(cfg, verbose=False)


def test_lora_ckpt_schema(tmp_path):
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg)
    lcfg = LoRAConfig(r=4, alpha=8)
    inject_lora(model, lcfg)
    mark_only_lora_trainable(model)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])

    payload = {"config": lcfg.to_dict(), "state": lora_state_dict(model),
               "base_ckpt": "/ckpts/base.pt"}
    ckpt = _build_ckpt_dict(
        model=model, optim=optim, text_encoder=None,
        cfg_model_dict=cfg.__dict__, step=10, best_val_loss=2.5,
        best_val_step=10, evals_without_improvement=0, prev_val_loss=2.6,
        lora_payload=payload, init_from="/ckpts/base.pt",
    )
    assert "model" not in ckpt  # adapter-only: the frozen base is NOT copied
    assert ckpt["lora"]["base_ckpt"] == "/ckpts/base.pt"
    assert all(k.endswith((".lora_A", ".lora_B")) for k in ckpt["lora"]["state"])
    # "cfg" stays a pure GPTConfig dict — the purity guard for inference/eval
    assert GPTConfig(**ckpt["cfg"]) == cfg
    restored = _restore_train_state(ckpt)
    assert restored["step"] == 10 and restored["prev_val_loss"] == 2.6

    # Full (non-LoRA) checkpoints keep today's schema, plus inert provenance.
    full = _build_ckpt_dict(
        model=model, optim=optim, text_encoder=None,
        cfg_model_dict=cfg.__dict__, step=10, best_val_loss=2.5,
        best_val_step=10, evals_without_improvement=0, prev_val_loss=None,
        init_from="/ckpts/base.pt",
    )
    assert "model" in full and "lora" not in full
    assert full["init_from"] == "/ckpts/base.pt"


def test_merge_lora_ckpts_cli_function():
    from diskrot.merge_lora import merge_lora_ckpts

    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = NanoAudioGPT(cfg).eval()
    base_ckpt = {
        "model": {f"_orig_mod.{k}": v for k, v in model.state_dict().items()},
        "cfg": cfg.__dict__,
        "step": 1000,
        "text_proj": {"weight": torch.randn(4, 4)},
    }
    lcfg = LoRAConfig(r=4, alpha=8)
    inject_lora(model, lcfg)
    _randomize_lora_b(model)
    lora_ckpt = {
        "lora": {"config": lcfg.to_dict(), "state": lora_state_dict(model),
                 "base_ckpt": "base.pt"},
        "cfg": cfg.__dict__,
        "step": 1500,
        "best_val_loss": 2.0,
    }

    out = merge_lora_ckpts(base_ckpt, lora_ckpt, half=True)
    # slim-inference schema: same keys export_slim produces
    assert set(out) == {"model", "cfg", "step", "best_val_loss", "text_proj"}
    assert out["step"] == 1500  # from the LoRA run
    # _orig_mod. prefixes stripped, no lora keys, half-cast floats only
    assert all(not k.startswith("_orig_mod.") for k in out["model"])
    assert not any(k.endswith((".lora_A", ".lora_B")) for k in out["model"])
    assert all(v.dtype == torch.float16 for v in out["model"].values()
               if v.is_floating_point())
    # lora run didn't train text_proj -> base's copy survives
    assert torch.allclose(out["text_proj"]["weight"].float(),
                          base_ckpt["text_proj"]["weight"], atol=1e-3)

    # the merged weights load strictly into a plain model and match the
    # lora model's forward (fp16 tolerance)
    plain = NanoAudioGPT(cfg).eval()
    plain.load_state_dict({k: v.float() for k, v in out["model"].items()})
    tokens = _tokens(cfg)
    with torch.no_grad():
        want = model(tokens)
        got = plain(tokens)
    assert torch.allclose(want, got, atol=2e-2)

    # cfg mismatch is a hard error
    other_cfg = _tiny_cfg(d_ff=256)
    with pytest.raises(ValueError, match="cfg"):
        merge_lora_ckpts({**base_ckpt, "cfg": other_cfg.__dict__}, lora_ckpt)
    # wrong-direction arguments are hard errors
    with pytest.raises(ValueError, match="lora"):
        merge_lora_ckpts(base_ckpt, base_ckpt)
    with pytest.raises(ValueError, match="model"):
        merge_lora_ckpts(lora_ckpt, lora_ckpt)
