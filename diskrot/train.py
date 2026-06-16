"""Device-agnostic training loop (mps on M4, cuda on A100).

Loss is per-codebook cross entropy over the MusicGen-style delayed sequence,
with the PAD token ignored.

Multi-GPU (DDP) is opt-in by setting ``world_size > 1`` on TrainConfig and
providing the per-rank ``local_rank``. The Modal entrypoint wires this up via
``torch.multiprocessing.spawn`` — see ``modal_train.py``.
"""
from __future__ import annotations

import math
import os
import random as _rng
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.codec import DACodec
from model.delay_pattern import build_train_inputs
from model.fim import fim_reorder_batch
from model.lora import (
    DEFAULT_TARGETS,
    LoRAConfig,
    inject_lora,
    load_lora_state,
    lora_state_dict,
    mark_only_lora_trainable,
)
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT
from model.text_encoder import CLAPTextEncoder
from diskrot.dataset import TokenDataset, collate_lyrics


@dataclass
class TrainConfig:
    cache_dir: str = "./token_cache"
    ckpt_dir: str = "./checkpoints"
    device: str = "mps"

    steps: int = 125_000
    batch_size: int = 8
    lr: float = 2.5e-4
    weight_decay: float = 0.02
    warmup_steps: int = 1_500
    grad_clip: float = 1.0

    log_every: int = 25
    eval_every: int = 1000
    # Bumped from 2000 → 5000: Modal volume commits are non-trivial and we
    # don't need step-level durability on H100 runs. The async commit wrapper
    # below also prevents commits from blocking the training loop.
    ckpt_every: int = 5000
    eval_batches: int = 26

    segment_seconds: float = 30.0
    val_ratio: float = 0.12
    patience: int = 20  # evals without val loss improvement before stopping (0 = disabled)
    tags_path: str | None = None  # path to tags.json for text conditioning
    lyrics_path: str | None = None  # path to lyrics (sharded dir or legacy lyrics.json) for lyric conditioning
    structure_path: str | None = None  # path to structure (sharded dir or JSON) for section-marker conditioning
    keys_path: str | None = None  # path to keys.json (diskrot.key_detect) for the <key_*> header marker
    phonemes_path: str | None = None  # path to the pre-phonemized phonemes/ dir (diskrot.phonemize)
    cfg_dropout: float = 0.1  # probability of dropping text conditioning (classifier-free guidance)
    # Fraction of training batches reordered into the FIM (infill) layout. Only
    # active when model.use_fim is True. A FIM batch drops lyric conditioning
    # (its sung alignment can't survive a frame reorder) but keeps tags and the
    # co-reordered melody, so this trades directly against lyric/melody training
    # — keep it modest since singing is the headline objective.
    fim_prob: float = 0.0

    seed: int = 42

    # Fine-tuning: load model weights (and the GPTConfig, which the checkpoint
    # carries) from this checkpoint, then start a FRESH run — new optimizer,
    # step 0, fresh LR schedule and early-stop counters. Ignored when
    # {ckpt_dir}/latest.pt exists: resume always wins, so an interrupted
    # fine-tune relaunched with the same command picks up where it left off.
    init_from: str | None = None
    # LoRA: freeze the base model and train low-rank adapters only (requires
    # init_from — adapters train on top of a pretrained base). Checkpoints are
    # adapter-only (no "model" key, ~100 MB instead of multi-GB); fold them
    # into a standard checkpoint with diskrot.merge_lora before serving.
    lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_targets: str = DEFAULT_TARGETS
    # The CLAP tag projection is frozen in LoRA mode by default. Set True to
    # train it with the adapters — it's then saved in the LoRA checkpoint and
    # preferred over the base's copy by the merge tool.
    lora_train_text_proj: bool = False

    # Distributed (multi-GPU) — DDP is active when world_size > 1.
    # batch_size is interpreted as the *per-rank* batch; global batch is
    # batch_size * world_size. lr at TrainConfig creation should already be the
    # sqrt-scaled target for the global batch.
    world_size: int = 1
    local_rank: int = 0
    dist_backend: str = "nccl"

    # Optional WandB instrumentation. wandb_project="" or None disables it.
    # The init is rank-0 only and silently no-ops if WANDB_API_KEY isn't set
    # or the wandb package isn't installed.
    wandb_project: str | None = None
    wandb_run_name: str | None = None

    # use_qk_norm / use_lyric_qk_norm = True is the bespoke model's shape (the
    # GPTConfig defaults stay False only so pre-qk-norm checkpoint cfg dicts keep
    # loading). use_lyric_qk_norm extends QK-norm into the LyricEncoder (the fix
    # for the v8_sing gradient runaway traced to lyric_encoder.layers.0).
    model: GPTConfig = field(
        default_factory=lambda: GPTConfig(use_qk_norm=True, use_lyric_qk_norm=True))


def _setup_dist(cfg: TrainConfig) -> bool:
    """Initialize torch.distributed when world_size > 1.

    Returns True if DDP is active. Caller is responsible for calling
    ``_teardown_dist`` at the end of the run. After this call, every
    ``.to("cuda")`` and ``.cuda()`` in the process targets this rank's GPU,
    because ``torch.cuda.set_device(local_rank)`` is set."""
    if cfg.world_size <= 1:
        return False
    import torch.distributed as dist

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend=cfg.dist_backend,
            rank=cfg.local_rank,
            world_size=cfg.world_size,
        )
    torch.cuda.set_device(cfg.local_rank)
    return True


def _teardown_dist(use_ddp: bool) -> None:
    if not use_ddp:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main(cfg: TrainConfig) -> bool:
    return cfg.local_rank == 0


def _init_wandb(cfg: TrainConfig):
    """Rank-0-only WandB init. Returns the wandb module if initialized, else
    None. Silently no-ops if wandb isn't installed, ``wandb_project`` is unset,
    or ``WANDB_API_KEY`` isn't present in the environment — so launches
    that don't pass a project keep working unchanged.
    """
    if cfg.local_rank != 0:
        return None
    if not cfg.wandb_project:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] WANDB_API_KEY not set; skipping wandb init", flush=True)
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[wandb] package not installed; skipping wandb init", flush=True)
        return None
    # Flatten the nested GPTConfig so the W&B UI shows model fields directly.
    flat_cfg = {k: v for k, v in cfg.__dict__.items() if k != "model"}
    for k, v in cfg.model.__dict__.items():
        flat_cfg[f"model.{k}"] = v
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=flat_cfg,
        resume="allow",
    )
    print(f"[wandb] initialized project={cfg.wandb_project} "
          f"run={wandb.run.name}", flush=True)
    return wandb


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all ranks in-place. No-op if DDP isn't active."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


class _AsyncCommit:
    """Background-thread wrapper around a checkpoint-volume commit callback so
    the training loop never blocks waiting for the volume RPC to land.

    Pending-flag semantics (queue depth of 1): if a commit is already in
    flight when ``submit()`` is called, we set a ``pending`` flag instead of
    dropping the request. When the in-flight commit finishes, a completion
    hook checks the flag and immediately fires one more commit if requested.
    ``close()`` repeatedly awaits in-flight + pending until both are clear.

    Why this matters (B5): previously the submit-while-busy path silently
    dropped the request. For mid-training that was fine — the NEXT step's
    submit covered it. But the FINAL submit before ``close()`` could be the
    one dropped, leaving the volume one checkpoint stale at run end. With
    H100 runs costing $30+, losing the last several thousand steps' worth
    of progress is a real regression."""

    def __init__(self, callback):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock

        self._cb = callback
        self._pool = ThreadPoolExecutor(max_workers=1) if callback is not None else None
        self._future = None
        self._pending = False
        self._lock = Lock()

    def _run(self) -> None:
        try:
            self._cb()
        except Exception as e:
            print(f"[ckpt-commit] commit raised: {e}", flush=True)

    def _on_done(self, _fut) -> None:
        # Fires on the pool thread when a commit finishes. If submit() was
        # called while we were running, kick off one more commit to honor it.
        with self._lock:
            if not self._pending or self._pool is None:
                self._future = None
                return
            self._pending = False
            self._future = self._pool.submit(self._run)
            self._future.add_done_callback(self._on_done)

    def submit(self) -> None:
        if self._cb is None or self._pool is None:
            return
        with self._lock:
            if self._future is not None and not self._future.done():
                # In flight — remember that another commit was requested.
                self._pending = True
                return
            self._future = self._pool.submit(self._run)
            self._future.add_done_callback(self._on_done)

    def close(self) -> None:
        if self._pool is None:
            return
        # Drain in-flight + any pending-then-rescheduled work. The done-callback
        # races us to schedule the pending commit; loop until both flags clear.
        while True:
            with self._lock:
                fut = self._future
            if fut is None:
                break
            try:
                fut.result()
            except Exception as e:
                print(f"[ckpt-commit] final commit raised: {e}", flush=True)
            with self._lock:
                # If on_done scheduled a follow-up, self._future is the new one;
                # otherwise it's been cleared and we're done.
                if self._future is fut:
                    self._future = None
                    break
        self._pool.shutdown(wait=True)


def _build_ckpt_dict(
    *,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    text_encoder: "CLAPTextEncoder | None",
    cfg_model_dict: dict,
    step: int,
    best_val_loss: float,
    best_val_step: int,
    evals_without_improvement: int,
    prev_val_loss: float | None,
    lora_payload: dict | None = None,
    init_from: str | None = None,
) -> dict:
    """Build the checkpoint dict used by both the best-checkpoint and
    step-checkpoint save sites. Single source of truth for the on-disk
    schema; if a new field is added (like ``prev_val_loss``), this is the
    only place to update — see [[_restore_train_state]] for the read side.

    Why: the two save sites previously duplicated this dict literally,
    drifting easily and dropping fields (B2: ``prev_val_loss`` was missing
    from BOTH writers AND the reader, so resume always lost the comparison
    baseline).

    ``lora_payload`` switches the schema to an adapter-only LoRA checkpoint:
    the (frozen, multi-GB) base weights are NOT saved — ``"model"`` is omitted
    and ``"lora" = {config, state, base_ckpt}`` records the adapter weights
    plus where the base lives. ``"cfg"`` stays a pure GPTConfig dict either
    way, so ``GPTConfig(**ckpt["cfg"])`` always works. ``init_from`` is a
    provenance-only key on fine-tune runs (no reader depends on it)."""
    ckpt = {
        "optim": optim.state_dict(),
        "step": step,
        "cfg": cfg_model_dict,
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
        "evals_without_improvement": evals_without_improvement,
        "prev_val_loss": prev_val_loss,
    }
    if lora_payload is not None:
        ckpt["lora"] = lora_payload
    else:
        ckpt["model"] = _unwrapped_state_dict(model)
    if init_from:
        ckpt["init_from"] = init_from
    if text_encoder is not None:
        ckpt["text_proj"] = text_encoder.proj.state_dict()
    return ckpt


def _restore_train_state(ckpt: dict) -> dict:
    """Pull just the training-loop scalar state out of a checkpoint dict
    (everything except model/optim/text_proj weights, which the caller loads
    via state_dict). Returns sensible defaults for missing keys.

    Why: pairs with [[_build_ckpt_dict]] so the read side knows about every
    field the write side stores. The ``best_val_step`` default is **0**
    (sentinel: never recorded), not the current step — using the current
    step would silently claim "the best was just achieved" on resume from
    older checkpoints that predated the field (B3)."""
    return {
        "step": ckpt.get("step", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "best_val_step": ckpt.get("best_val_step", 0),
        "evals_without_improvement": ckpt.get("evals_without_improvement", 0),
        "prev_val_loss": ckpt.get("prev_val_loss", None),
    }


def _strip_wrapper_prefixes(state_dict: dict) -> dict:
    """Normalize legacy torch.compile (``_orig_mod.``) and DDP (``module.``)
    prefixes; checkpoints written by this script are already saved bare."""
    return {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state_dict.items()
    }


@dataclass
class StartupPlan:
    """How train_run should start, resolved by [[_resolve_startup]].

    - mode="resume": continue an interrupted run from {ckpt_dir}/latest.pt
      (``ckpt`` carries the optimizer state + loop scalars to restore).
    - mode="init": fine-tune — ``base_state`` are pretrained weights to load,
      but the optimizer/step/schedule start fresh (``ckpt`` is None).
    - mode="scratch": today's from-zero path (everything None).

    ``lora_cfg`` non-None means LoRA is active (inject adapters, freeze base);
    ``lora_state`` carries saved adapter weights on a LoRA resume.
    ``model_cfg`` non-None means the architecture comes from a checkpoint and
    overrides the flag-built GPTConfig.
    """
    mode: str
    ckpt: dict | None = None
    base_state: dict | None = None
    model_cfg: GPTConfig | None = None
    lora_cfg: LoRAConfig | None = None
    lora_state: dict | None = None
    text_proj_state: dict | None = None
    base_ckpt_path: str | None = None


def _flag_lora_cfg(cfg: TrainConfig) -> LoRAConfig:
    return LoRAConfig(r=cfg.lora_r, alpha=cfg.lora_alpha,
                      dropout=cfg.lora_dropout, targets=cfg.lora_targets)


def _load_base_ckpt(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"base checkpoint {path} not found — pass --init-from with the "
            "path to the pretrained checkpoint (full or *_inference slim)"
        )
    return torch.load(p, map_location="cpu", weights_only=False)


def _print_cfg_diff(flag_cfg: GPTConfig, ckpt_cfg: GPTConfig) -> None:
    diffs = [
        f"  {k}: {flag_cfg.__dict__[k]} (flags) -> {v} (checkpoint)"
        for k, v in ckpt_cfg.__dict__.items()
        if flag_cfg.__dict__.get(k) != v
    ]
    if diffs:
        print("GPTConfig taken from the checkpoint; differs from flags/defaults on:")
        for d in diffs:
            print(d)


def _resolve_startup(cfg: TrainConfig, verbose: bool = True) -> StartupPlan:
    """Decide how this run starts. Precedence: an existing
    ``{ckpt_dir}/latest.pt`` ALWAYS wins (so re-running the launch command
    resumes an interrupted run, fine-tune or not); else ``cfg.init_from``
    starts a fresh fine-tune from those weights; else train from scratch.

    Checkpoints are loaded to CPU — the model lives on cfg.device and
    ``load_state_dict`` copies across; optimizer state is moved to the param
    device by ``Optimizer.load_state_dict``.
    """
    latest = Path(cfg.ckpt_dir) / "latest.pt"
    if latest.exists():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        if "lora" in ckpt:
            # Resume an interrupted LoRA run: rebuild the frozen base from the
            # recorded path (or --init-from), re-inject, restore adapters.
            lora_cfg = LoRAConfig.from_dict(ckpt["lora"]["config"])
            if cfg.lora and _flag_lora_cfg(cfg) != lora_cfg:
                if verbose:
                    print(f"WARNING: lora flags differ from the resumed checkpoint; "
                          f"the checkpoint wins ({lora_cfg})")
            base_path = cfg.init_from or ckpt["lora"].get("base_ckpt")
            if not base_path:
                raise FileNotFoundError(
                    f"{latest} is a LoRA checkpoint with no recorded base — "
                    "pass --init-from with the base checkpoint path"
                )
            base = _load_base_ckpt(base_path)
            # A LoRA ckpt carries text_proj iff the run trained it — the ckpt
            # wins over the CLI flag here too, so a bare relaunch keeps
            # training (and saving) the projection it was training before.
            cfg.lora_train_text_proj = "text_proj" in ckpt
            return StartupPlan(
                mode="resume", ckpt=ckpt,
                base_state=_strip_wrapper_prefixes(base["model"]),
                model_cfg=GPTConfig(**ckpt["cfg"]),
                lora_cfg=lora_cfg,
                lora_state=ckpt["lora"]["state"],
                text_proj_state=ckpt.get("text_proj", base.get("text_proj")),
                base_ckpt_path=base_path,
            )
        if cfg.lora:
            raise RuntimeError(
                f"--lora was passed but {latest} is a full (non-LoRA) checkpoint — "
                "use a fresh --ckpt-dir for the LoRA run"
            )
        if cfg.init_from and verbose:
            print(f"{latest} exists — resuming it and ignoring --init-from")
        return StartupPlan(
            mode="resume", ckpt=ckpt,
            base_state=_strip_wrapper_prefixes(ckpt["model"]),
            text_proj_state=ckpt.get("text_proj"),
            base_ckpt_path=ckpt.get("init_from"),
        )

    if cfg.init_from:
        ckpt = _load_base_ckpt(cfg.init_from)
        model_cfg = GPTConfig(**ckpt["cfg"])
        if verbose:
            _print_cfg_diff(cfg.model, model_cfg)
        return StartupPlan(
            mode="init",
            base_state=_strip_wrapper_prefixes(ckpt["model"]),
            model_cfg=model_cfg,
            lora_cfg=_flag_lora_cfg(cfg) if cfg.lora else None,
            text_proj_state=ckpt.get("text_proj"),
            base_ckpt_path=cfg.init_from,
        )

    if cfg.lora:
        raise ValueError(
            "--lora requires --init-from: adapters train on top of a "
            "pretrained base, not from scratch"
        )
    return StartupPlan(mode="scratch")


def _broadcast_flag(flag: bool, src: int = 0, device: str = "cuda") -> bool:
    """Broadcast a boolean from src to all ranks. No-op if DDP isn't active."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return flag
    t = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
    dist.broadcast(t, src=src)
    return bool(t.item())


def _broadcast_module_params(module: torch.nn.Module, src: int = 0) -> None:
    """Make every rank's copy of ``module`` identical to ``src``'s by
    broadcasting each parameter. Use for modules that aren't DDP-wrapped (DDP
    handles initial-weight sync internally for wrapped modules). No-op if
    DDP isn't active."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)


def _all_reduce_module_grads(module: torch.nn.Module) -> None:
    """Average gradients across ranks for a module that's NOT inside the DDP
    wrapper. Call between ``loss.backward()`` and ``optim.step()``. No-op if
    DDP isn't active. DDP itself handles this for wrapped modules via its
    reducer hook.

    CRITICAL: ranks must call all_reduce in lockstep — skipping the collective
    on any single rank deadlocks the whole group. So when this rank has no
    grad (e.g. CFG dropout fired and the module's forward path was skipped),
    we materialize a zero grad before the collective. The optimizer then does
    a zero-grad step for that param (momentum/variance decay slightly toward
    zero on skipped steps), which is the correct behavior for ranks that
    contributed no gradient signal."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return
    for p in module.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


def split_decay_param_groups(
    named_params, weight_decay: float,
) -> list[dict]:
    """AdamW param groups: weight decay only on matrix-shaped weights.

    ndim < 2 catches every RMSNorm gain (incl. QK-norm) — decaying those
    actively shrinks the gains the stability fixes rely on; the melody
    encoder's learned ``null`` ([1,1,D]) is excluded by name for the same
    reason. Embeddings and Linear/Conv weights keep cfg.weight_decay
    (nanoGPT-style). Guarded by tests/test_param_groups.py."""
    decay, no_decay = [], []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".null"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _trend_label(history) -> str:
    """Coarse trend tag from the slope of recent val losses. Threshold of 0.01
    per checkup — over a 5-checkup window that's a total move of ~0.05, which
    matches the eyeball threshold for "is this meaningfully changing"."""
    n = len(history)
    if n < 3:
        return ""
    mean_x = (n - 1) / 2.0
    mean_y = sum(history) / n
    num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(history))
    den = sum((i - mean_x) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    if slope < -0.01:
        return f"🙂 getting better over last {n} checkups"
    if slope > 0.01:
        return f"😟 getting worse over last {n} checkups"
    return f"😐 about the same over last {n} checkups"


def _loss_fn(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: [B, K, T, V], targets: [B, K, T] -> (total_loss, per_cb_loss [K]).

    Computes per-token CE once with ``reduction='none'`` (which honors
    ignore_index by zeroing pad positions) and derives both the global mean and
    per-codebook means from that single tensor. This replaces a Python loop of
    1 + K=9 ``F.cross_entropy`` calls with 1 call + a couple of reductions —
    fewer kernel launches and less Python overhead per training step."""
    B, K, T, V = logits.shape
    per_pos = F.cross_entropy(
        logits.reshape(B * K * T, V),
        targets.reshape(B * K * T),
        ignore_index=pad_id,
        reduction="none",
    ).view(B, K, T)
    mask = (targets != pad_id).to(per_pos.dtype)  # [B, K, T]

    cb_sum = (per_pos * mask).sum(dim=(0, 2))  # [K]
    cb_count = mask.sum(dim=(0, 2)).clamp(min=1.0)  # [K]
    per_cb = cb_sum / cb_count

    total = (per_pos * mask).sum() / mask.sum().clamp(min=1.0)
    return total, per_cb


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip torch.compile (``_orig_mod``) and DDP (``module``) wrappers."""
    inner = getattr(model, "_orig_mod", model)
    inner = getattr(inner, "module", inner)
    return inner


def _unwrapped_state_dict(model: torch.nn.Module) -> dict:
    """Return state_dict without any wrapper prefixes (compile/DDP)."""
    return _unwrap(model).state_dict()


def _build_cond(
    text_encoder: CLAPTextEncoder, tags: list[str], device: str,
    tag_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Encode tags as a pooled CLAP embedding -> [B, 1, D], or None if no tags.

    Lyrics no longer flow through CLAP — they are conditioned via the phoneme
    LyricEncoder + lyric cross-attention (see model/lyric_encoder.py). This builds
    only the pooled TAG vector. If tag_cache is provided, looks up pre-computed
    CLAP embeddings and applies the projection layer (avoids running CLAP).
    """
    if not any(t != "" for t in tags):
        return None
    if tag_cache is not None:
        zero = torch.zeros(text_encoder.CLAP_DIM, device=device)
        raw = torch.stack([tag_cache.get(t, zero) for t in tags])  # [B, 1024]
        return text_encoder.proj(raw).unsqueeze(1)                 # [B, 1, d_out]
    return text_encoder.encode(tags).to(device)                    # [B, 1, D]


@torch.no_grad()
def _evaluate(
    model: NanoAudioGPT, loader: DataLoader, cfg: TrainConfig,
    n_batches: int, text_encoder: CLAPTextEncoder | None = None,
    tag_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[float, list[float]]:
    # Read pad_id from cfg (not model.cfg) so this works whether `model` is
    # the bare NanoAudioGPT, a DDP wrapper, or a torch.compile wrapper —
    # those wrappers don't passthrough attribute access for .cfg.
    pad_id = cfg.model.pad_id
    amp_enabled = cfg.device == "cuda"
    # try/finally so the model is returned to train mode even if a forward
    # pass raises mid-eval (NCCL hiccup, OOM, assertion). Without this the
    # model stays in eval — dropout disabled, training silently degraded.
    model.eval()
    try:
        use_lyrics = cfg.model.use_lyric_conditioning
        use_melody = cfg.model.use_melody_conditioning
        losses, per_cb_sums = [], None
        for i, (batch, tags, lyric_ids, lyric_mask, melody) in enumerate(loader):
            if i >= n_batches:
                break
            # int16 on host (P3 — saves ~24 GB shared RAM at the production
            # corpus size). Cast to int64 on the GPU because nn.Embedding's
            # index_select kernel only accepts int64.
            batch = batch.to(cfg.device, non_blocking=True).long()
            inputs, targets = build_train_inputs(batch, pad_id)

            text_emb = None
            if text_encoder is not None:
                text_emb = _build_cond(text_encoder, list(tags), cfg.device, tag_cache)
            l_ids = l_mask = None
            if use_lyrics:
                l_ids = lyric_ids.to(cfg.device, non_blocking=True)
                l_mask = lyric_mask.to(cfg.device, non_blocking=True)
            mel = None
            if use_melody and melody is not None:
                mel = melody.to(cfg.device, non_blocking=True)

            # bf16 to match the train loop (see train_run's scaler comment).
            with torch.amp.autocast(cfg.device, dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(inputs, text_emb=text_emb, lyric_ids=l_ids,
                               lyric_mask=l_mask, melody=mel)
                _, per_cb = _loss_fn(logits, targets, pad_id)
            if per_cb_sums is None:
                per_cb_sums = per_cb.clone()
            else:
                per_cb_sums += per_cb
            losses.append(per_cb.mean().item())
    finally:
        model.train()
    if not losses:
        return float("nan"), []
    return sum(losses) / len(losses), (per_cb_sums / len(losses)).tolist()


def train_run(
    cfg: TrainConfig,
    ckpt_callback=None,
    shared_bundle: dict | None = None,
    precomputed_tag_cache: dict[str, torch.Tensor] | None = None,
) -> None:
    """Train per cfg. Optional preloaded inputs (used by the DDP path so each
    rank doesn't redundantly re-read off the volume / re-encode CLAP):

    - shared_bundle: result of ``dataset.load_mmap_bundle()`` — mmap-backed
      token index + tags + lyrics. When given, skips per-rank disk I/O.
    - precomputed_tag_cache: dict[tag_str -> CPU [1024] tensor] of CLAP
      embeddings encoded once in the parent. When given, skips the serial
      CLAP-per-tag loop on every rank."""
    use_ddp = _setup_dist(cfg)
    main = _is_main(cfg)
    wandb = _init_wandb(cfg)
    # Only rank 0 ever commits, but every rank instantiates the wrapper so the
    # call sites can use it unconditionally without ``if main`` guards.
    async_commit = _AsyncCommit(ckpt_callback if main else None)
    # Diverge per-rank RNG so that CFG dropout and any other host-side random
    # decisions vary across ranks (data shuffling itself goes through
    # DistributedSampler, which uses cfg.seed independently).
    torch.manual_seed(cfg.seed + cfg.local_rank)
    _rng.seed(cfg.seed + cfg.local_rank)
    if cfg.device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # Resolve resume / fine-tune / scratch BEFORE anything reads cfg.model:
    # when starting from a checkpoint, the architecture comes from the ckpt's
    # cfg dict (flags are advisory) and the datasets below read
    # cfg.model.max_lyric_len.
    plan = _resolve_startup(cfg, verbose=main)
    if plan.model_cfg is not None:
        cfg.model = plan.model_cfg
        # A conditioning axis that's on in the ckpt but has no data wired up
        # would silently train its cross-attention against nothing.
        if main:
            if cfg.model.use_text_conditioning and cfg.tags_path is None:
                print("WARNING: checkpoint enables text conditioning but "
                      "--tags-path is not set — tags train without signal")
            if cfg.model.use_lyric_conditioning and cfg.lyrics_path is None:
                print("WARNING: checkpoint enables lyric conditioning but "
                      "--lyrics-path is not set — lyrics train without signal")
    lora_active = plan.lora_cfg is not None

    segment_frames = int(cfg.segment_seconds * DACodec.FRAME_RATE_HZ)
    if main:
        print(f"segment_frames={segment_frames} (delayed seq len = {segment_frames + cfg.model.n_codebooks - 2})")
        if use_ddp:
            print(f"DDP active: world_size={cfg.world_size}, "
                  f"per-rank batch={cfg.batch_size}, global batch={cfg.batch_size * cfg.world_size}")

    if shared_bundle is not None:
        if main:
            print("using preloaded mmap bundle (skipping per-rank disk load)",
                  flush=True)
        train_ds = TokenDataset.from_mmap(
            shared_bundle, "train", segment_frames, cfg.model.max_lyric_len)
        val_ds = TokenDataset.from_mmap(
            shared_bundle, "val", segment_frames, cfg.model.max_lyric_len)
    else:
        train_ds = TokenDataset(cfg.cache_dir, segment_frames=segment_frames, split="train",
                                val_ratio=cfg.val_ratio, seed=cfg.seed, tags_path=cfg.tags_path,
                                lyrics_path=cfg.lyrics_path, structure_path=cfg.structure_path,
                                keys_path=cfg.keys_path, phonemes_path=cfg.phonemes_path,
                                max_lyric_len=cfg.model.max_lyric_len)
        val_ds = TokenDataset(cfg.cache_dir, segment_frames=segment_frames, split="val",
                              val_ratio=cfg.val_ratio, seed=cfg.seed, tags_path=cfg.tags_path,
                              lyrics_path=cfg.lyrics_path, structure_path=cfg.structure_path,
                              keys_path=cfg.keys_path, phonemes_path=cfg.phonemes_path,
                              max_lyric_len=cfg.model.max_lyric_len)

    # Steer training crops toward sung regions so most <vocals> crops actually
    # carry phonemes (uniform crops often land on a vocal song's instrumental
    # intro/solo/outro). Train split only — val stays a clean whole-distribution
    # metric; singing is measured separately via WER (scripts/eval_lyric_wer.py).
    train_ds.bias_vocal_crops = True

    pin = cfg.device == "cuda"
    if use_ddp:
        train_sampler: DistributedSampler | None = DistributedSampler(
            train_ds, num_replicas=cfg.world_size, rank=cfg.local_rank,
            shuffle=True, seed=cfg.seed, drop_last=True,
        )
        val_sampler: DistributedSampler | None = DistributedSampler(
            val_ds, num_replicas=cfg.world_size, rank=cfg.local_rank,
            shuffle=True, seed=cfg.seed, drop_last=True,
        )
        # 8 workers per rank: lyric phonemization (g2p) on a cache miss costs
        # ~20-200 ms/song for OOV-heavy Whisper transcripts, and at corpus scale
        # nearly every item misses the LRU — 2 workers starve the H100s. The
        # pre-phonemized store (cfg.phonemes_path) removes most of that cost;
        # the extra workers cover the fallback + mmap paging.
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=train_sampler,
                                  num_workers=8, pin_memory=pin, drop_last=True,
                                  persistent_workers=True, prefetch_factor=4,
                                  collate_fn=collate_lyrics)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, sampler=val_sampler,
                                num_workers=1, pin_memory=pin, drop_last=True,
                                persistent_workers=True, collate_fn=collate_lyrics)
    else:
        train_sampler = None
        val_sampler = None
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=8, pin_memory=pin, drop_last=True, persistent_workers=True,
                                  prefetch_factor=4, collate_fn=collate_lyrics)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=1, pin_memory=pin, drop_last=True, persistent_workers=True,
                                collate_fn=collate_lyrics)

    model = NanoAudioGPT(cfg.model).to(cfg.device)
    n_params = model.num_params()
    if main:
        print(f"model: {n_params/1e6:.2f}M params on {cfg.device}")

    # Pretrained weights (fine-tune init or LoRA base) load BEFORE the LoRA
    # injection (LoRALinear reuses the loaded weight Parameters) and before the
    # DDP wrap / torch.compile below.
    if plan.base_state is not None:
        model.load_state_dict(plan.base_state)
        plan.base_state = None  # free the CPU copy (~6 GB at 1.5B fp32)
        if main and plan.mode == "init":
            print(f"initialized weights from {plan.base_ckpt_path} (fresh "
                  f"optimizer/step — fine-tune run)")

    if lora_active:
        replaced = inject_lora(model, plan.lora_cfg)
        if plan.lora_state is not None:
            load_lora_state(model, plan.lora_state)
        n_trainable, n_total = mark_only_lora_trainable(model)
        if main:
            print(f"LoRA active: r={plan.lora_cfg.r} alpha={plan.lora_cfg.alpha} "
                  f"targets={plan.lora_cfg.targets} — {len(replaced)} layers, "
                  f"trainable {n_trainable/1e6:.2f}M / {n_total/1e6:.2f}M "
                  f"({100 * n_trainable / n_total:.2f}%)")

    # text conditioning
    text_encoder: CLAPTextEncoder | None = None
    tag_cache: dict[str, torch.Tensor] = {}
    use_text = cfg.model.use_text_conditioning and train_ds.has_tags
    if use_text:
        text_encoder = CLAPTextEncoder(d_out=cfg.model.d_model, device=cfg.device)
        text_encoder.to(cfg.device)
        text_encoder.eval()
        # text_encoder.proj is the only trainable part of text_encoder (CLAP
        # itself is frozen). It lives OUTSIDE the NanoAudioGPT module that DDP
        # wraps, so DDP won't sync its initial weights or its per-step grads
        # for us. Without these two calls, each rank silently trains its own
        # divergent copy of proj — see "Known DDP gotchas" in README.modal.md.
        # Broadcast happens regardless of resume: if a checkpoint is loaded
        # later, all ranks load the same on-disk weights, so they stay in sync.
        _broadcast_module_params(text_encoder.proj, src=0)
        # Pretrained projection (resume or fine-tune init). Loading after the
        # broadcast is fine: every rank loads the same on-disk weights.
        if plan.text_proj_state is not None:
            text_encoder.proj.load_state_dict(plan.text_proj_state)
        if main:
            print(f"text conditioning enabled (cfg_dropout={cfg.cfg_dropout})")
        if precomputed_tag_cache is not None:
            # DDP fast path: parent ran CLAP once and handed us CPU tensors.
            # Move them to this rank's GPU so the in-step lookup is local.
            for tag, emb in precomputed_tag_cache.items():
                tag_cache[tag] = emb.to(cfg.device)
            if main:
                print(f"loaded {len(tag_cache)} precomputed CLAP tag embeddings", flush=True)
        else:
            # pre-compute CLAP embeddings for all tags (they're fixed per song).
            # Every rank computes them so the in-step embedding lookup is local.
            unique_tags = sorted(set(train_ds._tags.values()) | set(val_ds._tags.values()))
            if unique_tags:
                if main:
                    print(f"pre-computing CLAP embeddings for {len(unique_tags)} unique tags...", flush=True)
                text_encoder._ensure_clap()
                t0_clap = time.time()
                with torch.no_grad():
                    for i, tag in enumerate(unique_tags):
                        emb = text_encoder._clap.get_text_embeddings([tag])  # [1, 1024]
                        tag_cache[tag] = emb.squeeze(0).to(cfg.device)       # [1024]
                        if main and (i + 1) % 500 == 0:
                            elapsed = time.time() - t0_clap
                            rate = (i + 1) / max(elapsed, 1e-6)
                            eta = (len(unique_tags) - i - 1) / max(rate, 1e-6)
                            print(f"  CLAP precompute: {i+1}/{len(unique_tags)} "
                                  f"({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)
                if main:
                    print(f"cached {len(tag_cache)} tag embeddings", flush=True)
    elif cfg.model.use_text_conditioning and main:
        print("WARNING: use_text_conditioning=True but no tags found — training without text")

    if cfg.model.use_lyric_conditioning and main:
        n_lyrics = len(train_ds._lyrics) + len(val_ds._lyrics)
        print(f"lyric (phoneme) conditioning enabled: {n_lyrics} songs with lyrics "
              f"(max_lyric_len={cfg.model.max_lyric_len}, cfg_dropout={cfg.cfg_dropout})")

    if cfg.model.use_melody_conditioning:
        # Fail fast on every rank: without the chroma sidecar the melody encoder
        # only ever sees its learned null, so a full run would silently train no
        # melody signal at all. Don't burn the GPU hours — abort and tell the user.
        if not getattr(train_ds, "_has_melody", False):
            raise RuntimeError(
                "use_melody_conditioning=True but the pack has NO chroma sidecar "
                "(packed_NNN.mel.bin). Repack with pack_cache --mel-cache-dir (run "
                "diskrot.modal_melody first), or disable melody conditioning."
            )
        if main:
            print(f"melody (chroma) conditioning enabled (n_bins={cfg.model.melody_n_bins}, "
                  f"cfg_dropout={cfg.cfg_dropout})")

    # REGIONAL compilation, before the DDP wrap: compile each decoder Block
    # in place (nn.Module.compile keeps state-dict keys clean — no _orig_mod
    # interior prefixes) instead of one monolithic 22-layer graph. The blocks
    # are identical code, so dynamo compiles ONE block program and reuses it
    # 22x — cold compile drops from tens of minutes to single digits. The thin
    # eager remainder (embedding sum, melody add, head) is a few percent of
    # step time at most; DDP's comm hooks live at the autograd level, so
    # gradient-overlap behavior is unchanged by compiling under the wrapper.
    if cfg.device == "cuda":
        for blk in model.blocks:
            blk.compile()
        if main:
            print(f"torch.compile enabled (regional: {len(model.blocks)} blocks)")

    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # find_unused_parameters=False is safe (and ~5%/step cheaper — no extra
        # autograd-graph traversal) because every param now participates in
        # every step: dropped-text steps pass a zeros embedding through the
        # cross-attn, and dropped-lyric/melody steps run the encoders with a
        # keep=0 gate (see the conditioning block in the loop). If DDP ever
        # crashes with "Expected to have finished reduction", a stream
        # regressed to a skip-style None path — fix the caller, don't flip
        # this back to True.
        model = DDP(model, device_ids=[cfg.local_rank], find_unused_parameters=False)

    # Only train params that require grad (everything in full/scratch mode;
    # just the adapters in LoRA mode) + the text_encoder projection (CLAP
    # itself is always frozen). The proj is frozen too in LoRA mode unless
    # opted in — its grad all-reduce in the step loop is skipped when frozen.
    proj_trainable = text_encoder is not None and (
        not lora_active or cfg.lora_train_text_proj
    )
    if text_encoder is not None and not proj_trainable:
        text_encoder.proj.requires_grad_(False)
    named = list(model.named_parameters())
    if proj_trainable:
        named += [(f"text_proj.{n}", p)
                  for n, p in text_encoder.proj.named_parameters()]
    trainable = [p for _, p in named if p.requires_grad]
    # Two groups: no weight decay on norm gains / the melody null (see
    # split_decay_param_groups). NOTE: changes the optimizer state-dict group
    # layout — resumes of pre-split checkpoints' optim state are incompatible
    # (v8 starts fresh; no live run predates this).
    optim = torch.optim.AdamW(
        split_decay_param_groups(named, cfg.weight_decay),
        lr=cfg.lr, betas=(0.9, 0.95), fused=(cfg.device == "cuda"),
    )

    amp_enabled = cfg.device == "cuda"
    # bf16, NOT the fp16 autocast-on-CUDA default: fp16's narrow exponent range
    # on a 2B-param × 5167-token model was the root-cause candidate for the
    # 2026-06-12 warmup divergences (cb0-led — the deepest gradient path
    # saturates first; v7 at half the depth×seq survived the same code). H100
    # bf16 removes the overflow class and the GradScaler with it — the disabled
    # scaler object is kept so the scale/unscale_/step call sites stay uniform.
    scaler = torch.amp.GradScaler(cfg.device, enabled=False)

    def _lora_payload() -> dict | None:
        """Adapter-only checkpoint payload (None in full/scratch mode). The
        frozen base is referenced by path, not copied — see _build_ckpt_dict."""
        if not lora_active:
            return None
        return {
            "config": plan.lora_cfg.to_dict(),
            "state": lora_state_dict(_unwrap(model)),
            "base_ckpt": plan.base_ckpt_path,
        }

    # In LoRA mode with a frozen proj, the checkpoint must NOT carry text_proj
    # (the merge tool would prefer it over the base's; the base's copy is the
    # trained one). _build_ckpt_dict skips it when handed None.
    ckpt_text_encoder = text_encoder if proj_trainable else None

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    pad_id = cfg.model.pad_id

    step = 0
    best_val_loss = float("inf")
    best_val_step = 0
    evals_without_improvement = 0
    prev_val_loss: float | None = None
    val_history: deque[float] = deque(maxlen=5)
    if plan.mode == "resume":
        # Model (and LoRA adapter) weights were already loaded before the DDP
        # wrap; here we restore the optimizer + loop scalars. Optimizer state
        # was loaded to CPU — Optimizer.load_state_dict moves it to the param
        # device. init mode restores nothing: fresh optimizer, step 0, fresh
        # warmup/cosine schedule and early-stop counters.
        optim.load_state_dict(plan.ckpt["optim"])
        restored = _restore_train_state(plan.ckpt)
        step = restored["step"]
        best_val_loss = restored["best_val_loss"]
        best_val_step = restored["best_val_step"]
        evals_without_improvement = restored["evals_without_improvement"]
        prev_val_loss = restored["prev_val_loss"]
        plan.ckpt = None  # free the CPU copy
        if main:
            print(f"resumed from {Path(cfg.ckpt_dir) / 'latest.pt'} at step {step}")
    t0 = time.time()
    running: torch.Tensor | None = None
    running_count = 0
    max_gnorm: torch.Tensor | None = None  # window-max pre-clip grad norm
    # Pre-built CFG keep gates (see the conditioning block in the loop) — one
    # H2D copy each at startup instead of a tiny blocking copy per step.
    keep_one = torch.ones((), device=cfg.device)
    keep_zero = torch.zeros((), device=cfg.device)
    # Rolling sum of time spent in next(train_iter); reported per log_every and
    # logged to wandb as ``dataloader_wait_ms``. Catches I/O regressions on the
    # mmap dataset path (if it climbs to ~100ms+, mmap throughput isn't keeping
    # up with the GPU and we need to warm the page cache or shard differently).
    dataloader_wait_s = 0.0
    epoch = 0
    if use_ddp and train_sampler is not None:
        train_sampler.set_epoch(epoch)
    train_iter = iter(train_loader)
    while step < cfg.steps:
        t_io = time.time()
        try:
            batch, tags, lyric_ids, lyric_mask, melody = next(train_iter)
        except StopIteration:
            epoch += 1
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_iter = iter(train_loader)
            batch, tags, lyric_ids, lyric_mask, melody = next(train_iter)
        dataloader_wait_s += time.time() - t_io

        # int16 on host (P3 — saves ~24 GB shared RAM). Cast to int64 on
        # GPU because nn.Embedding's index_select kernel requires int64.
        batch = batch.to(cfg.device, non_blocking=True).long()
        mel_dev = melody.to(cfg.device, non_blocking=True) if melody is not None else None

        # Fill-in-the-middle: with prob fim_prob reorder this batch into the
        # infill layout `prefix <SUF> suffix <MID> middle` (frame-domain reorder
        # before the delay pattern). FIM scrambles frame order, so the lyric
        # stream (its near-monotonic sung alignment) is dropped for the batch;
        # tags stay (order-invariant) and melody is co-reordered to match.
        do_fim = cfg.model.use_fim and _rng.random() < cfg.fim_prob
        if do_fim:
            batch, mel_dev = fim_reorder_batch(
                batch, mel_dev, cfg.model.suf_id, cfg.model.mid_id, _rng
            )
        inputs, targets = build_train_inputs(batch, pad_id)

        # Tag + lyric + melody conditioning, each dropped INDEPENDENTLY for
        # classifier-free guidance (teaches every on/off combination so
        # inference can guide each axis). "Drop" never passes None: text gets a
        # ZEROS embedding (bias-free projections make attending over zero K/V
        # exactly zero — bit-identical to skipping, guarded by
        # tests/test_cfg_uncond.py), lyrics/melody get keep=0 gates below. The
        # forward's argument TYPES are therefore identical on every step — one
        # compiled graph total, no dynamo variant explosion, no eager fallback.
        # Inference's uncond branch still passes None — same math, no drift.
        B_in = inputs.shape[0]
        d_model = cfg.model.d_model
        text_emb = None
        if text_encoder is not None:
            if _rng.random() >= cfg.cfg_dropout:
                text_emb = _build_cond(text_encoder, list(tags), cfg.device, tag_cache)
            else:
                text_emb = torch.zeros((B_in, 1, d_model), device=cfg.device)
        # Lyrics and melody: the encoders ALWAYS run; a 0/1 keep tensor zeroes
        # the contribution on dropped steps (keep=0 is exactly the uncond
        # state: zero lyric cond / the melody null). One compiled graph per
        # stream instead of a branch pair, and every param participates every
        # step — which is what lets DDP run find_unused_parameters=False.
        # (Tensor VALUES don't create dynamo guards; None-vs-tensor does.)
        # On a FIM batch the lyric stream is dropped (alignment can't survive
        # the reorder); mel_dev is the co-reordered chroma.
        l_ids = l_mask = lyric_keep = None
        if cfg.model.use_lyric_conditioning:
            l_ids = lyric_ids.to(cfg.device, non_blocking=True)
            l_mask = lyric_mask.to(cfg.device, non_blocking=True)
            lyric_drop = do_fim or _rng.random() < cfg.cfg_dropout
            lyric_keep = keep_zero if lyric_drop else keep_one
        mel = melody_keep = None
        if cfg.model.use_melody_conditioning and mel_dev is not None:
            mel = mel_dev
            melody_keep = (keep_zero if _rng.random() < cfg.cfg_dropout
                           else keep_one)

        for g in optim.param_groups:
            g["lr"] = _cosine_lr(step, cfg)

        with torch.amp.autocast(cfg.device, dtype=torch.bfloat16, enabled=amp_enabled):
            logits = model(inputs, text_emb=text_emb, lyric_ids=l_ids,
                           lyric_mask=l_mask, lyric_keep=lyric_keep,
                           melody=mel, melody_keep=melody_keep)
            loss, per_cb = _loss_fn(logits, targets, pad_id)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        # DDP wraps `model` and auto-averages its grads via .backward()'s comm
        # hook. text_encoder.proj is OUTSIDE that wrapper, so its grads are
        # per-rank — we have to all-reduce them ourselves before the optim step
        # to keep ranks in sync. Skipped when proj is frozen (LoRA mode): no
        # grads exist and no rank enters the collective, so no deadlock.
        if proj_trainable:
            _all_reduce_module_grads(text_encoder.proj)
        # Clip the same combined param list the optimizer trains
        # (model + text_encoder.proj if present). Clipping only model.parameters()
        # leaves proj's grad uncapped, which can produce unstable updates when a
        # rare-tag batch yields a huge CLAP-projection gradient.
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        # Track the window max pre-clip norm — a ramp here is the divergence
        # precursor (the 2026-06-12 failures were diagnosed blind without it).
        max_gnorm = gnorm if max_gnorm is None else torch.maximum(max_gnorm, gnorm)
        scaler.step(optim)
        scaler.update()

        running = loss.detach() if running is None else running + loss.detach()
        running_count += 1
        step += 1

        if step % cfg.log_every == 0 and main:
            elapsed = time.time() - t0
            # Effective batch (across all ranks) drives reported tok/s so that
            # multi-GPU runs report aggregate throughput.
            global_bs = cfg.batch_size * cfg.world_size
            tok_per_step = global_bs * cfg.model.n_codebooks * inputs.shape[2]
            tps = tok_per_step * cfg.log_every / max(elapsed, 1e-6)
            avg_loss = (running / running_count).item()
            cb_str = " ".join(f"{x:.2f}" for x in per_cb.tolist())
            avg_io_ms = (dataloader_wait_s / cfg.log_every) * 1000
            cur_lr = optim.param_groups[0]["lr"]
            gnorm_val = max_gnorm.item() if max_gnorm is not None else float("nan")
            print(f"step {step:>6}/{cfg.steps}  loss {avg_loss:.4f}  "
                  f"lr {cur_lr:.2e}  grad {gnorm_val:.2f}  tok/s {tps/1e3:.1f}k  "
                  f"cb[{cb_str}]",
                  flush=True)
            if wandb is not None:
                log_payload = {
                    "train/loss": avg_loss,
                    "train/lr": cur_lr,
                    "train/grad_norm_max": gnorm_val,
                    "train/tokens_per_sec": tps,
                    "train/dataloader_wait_ms": avg_io_ms,
                    "train/epoch": epoch,
                }
                for cb_i, cb_loss in enumerate(per_cb.tolist()):
                    log_payload[f"train/cb_{cb_i}"] = cb_loss
                if cfg.device == "cuda":
                    log_payload["train/gpu_mem_alloc_gb"] = (
                        torch.cuda.memory_allocated() / 1e9
                    )
                wandb.log(log_payload, step=step)
            running, running_count, t0 = None, 0, time.time()
            dataloader_wait_s = 0.0
            max_gnorm = None

        if step % cfg.eval_every == 0:
            # All ranks evaluate their shard; we average loss + per-codebook
            # losses across ranks so every rank takes the same early-stop
            # decision (otherwise DDP deadlocks at the next collective).
            val_loss, val_per_cb = _evaluate(model, val_loader, cfg, cfg.eval_batches,
                                            text_encoder=text_encoder, tag_cache=tag_cache)
            if use_ddp:
                vl = torch.tensor([val_loss], device=cfg.device, dtype=torch.float32)
                _all_reduce_mean(vl)
                val_loss = vl.item()
                if val_per_cb:
                    pcb = torch.tensor(val_per_cb, device=cfg.device, dtype=torch.float32)
                    _all_reduce_mean(pcb)
                    val_per_cb = pcb.tolist()
            cb_str = " ".join(f"{x:.2f}" for x in val_per_cb)
            val_history.append(val_loss)

            if wandb is not None:
                val_payload = {
                    "val/loss": val_loss,
                    "val/best_loss": min(best_val_loss, val_loss),
                    "val/evals_without_improvement": evals_without_improvement,
                }
                for cb_i, cb_loss in enumerate(val_per_cb):
                    val_payload[f"val/cb_{cb_i}"] = cb_loss
                wandb.log(val_payload, step=step)

            if prev_val_loss is None:
                change_phrase = "first checkup"
            else:
                diff = val_loss - prev_val_loss
                if abs(diff) < 5e-5:
                    change_phrase = "same as last checkup"
                elif diff < 0:
                    change_phrase = f"{-diff:.4f} better than last checkup"
                else:
                    change_phrase = f"{diff:.4f} worse than last checkup"

            trend = _trend_label(val_history)
            trend_str = f"   {trend}" if trend else ""

            improved = val_loss < best_val_loss
            if improved:
                if best_val_loss == float("inf"):
                    best_phrase = "🎉 first checkup — this is our new best"
                else:
                    gain = best_val_loss - val_loss
                    best_phrase = (
                        f"🎉 NEW BEST! beat previous best ({best_val_loss:.4f}) by {gain:.4f}"
                    )
                best_val_loss = val_loss
                best_val_step = step
                evals_without_improvement = 0
                if main:
                    best_path = Path(cfg.ckpt_dir) / "best.pt"
                    ckpt_data = _build_ckpt_dict(
                        model=model, optim=optim, text_encoder=ckpt_text_encoder,
                        cfg_model_dict=cfg.model.__dict__,
                        step=step, best_val_loss=best_val_loss,
                        best_val_step=best_val_step,
                        evals_without_improvement=evals_without_improvement,
                        prev_val_loss=prev_val_loss,
                        lora_payload=_lora_payload(),
                        init_from=cfg.init_from or plan.base_ckpt_path,
                    )
                    torch.save(ckpt_data, best_path)
                    print(f"  checkup at step {step:>6}  score {val_loss:.4f}  "
                          f"({change_phrase})   {best_phrase}{trend_str}  cb[{cb_str}]",
                          flush=True)
            else:
                evals_without_improvement += 1
                if main:
                    gap = val_loss - best_val_loss
                    print(f"  checkup at step {step:>6}  score {val_loss:.4f}  "
                          f"({change_phrase})   😕 {gap:.4f} worse than best "
                          f"({best_val_loss:.4f} from step {best_val_step}) "
                          f"— strike {evals_without_improvement} of {cfg.patience}{trend_str}  cb[{cb_str}]",
                          flush=True)
            prev_val_loss = val_loss
            stop = cfg.patience > 0 and evals_without_improvement >= cfg.patience
            stop = _broadcast_flag(stop, src=0, device=cfg.device) if use_ddp else stop
            if stop:
                if main:
                    print(f"early stopping at step {step} — val loss has not improved "
                          f"for {cfg.patience} evals", flush=True)
                break

        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            if main:
                ckpt = _build_ckpt_dict(
                    model=model, optim=optim, text_encoder=ckpt_text_encoder,
                    cfg_model_dict=cfg.model.__dict__,
                    step=step, best_val_loss=best_val_loss,
                    best_val_step=best_val_step,
                    evals_without_improvement=evals_without_improvement,
                    prev_val_loss=prev_val_loss,
                    lora_payload=_lora_payload(),
                    init_from=cfg.init_from or plan.base_ckpt_path,
                )
                p = Path(cfg.ckpt_dir) / f"step_{step:07d}.pt"
                torch.save(ckpt, p)
                latest = Path(cfg.ckpt_dir) / "latest.pt"
                torch.save(ckpt, latest)
                print(f"  saved ckpt → {p}", flush=True)
                async_commit.submit()

    async_commit.close()
    if wandb is not None:
        wandb.finish()
    _teardown_dist(use_ddp)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=125_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--cache-dir", default="./token_cache")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--tags-path", type=str, default=None, help="path to tags.json for text conditioning")
    p.add_argument("--lyrics-path", type=str, default=None, help="path to lyrics (sharded dir or legacy lyrics.json) for lyric conditioning")
    p.add_argument("--structure-path", type=str, default=None, help="path to structure (sharded dir or JSON) for section-marker conditioning")
    p.add_argument("--keys-path", type=str, default=None, help="path to keys.json (diskrot.key_detect) for the <key_*> header marker")
    p.add_argument("--phonemes-path", type=str, default=None, help="path to the pre-phonemized phonemes/ dir (diskrot.phonemize)")
    p.add_argument("--melody", action="store_true",
                   help="enable melody (chroma) conditioning — requires the pack to "
                        "have been built with --mel-cache-dir (parallel .mel.bin)")
    p.add_argument("--init-from", type=str, default=None,
                   help="fine-tune: load model weights + GPTConfig from this checkpoint "
                        "and start a fresh run (new optimizer/step/schedule). Ignored "
                        "when {ckpt-dir}/latest.pt exists — resume always wins")
    p.add_argument("--lora", action="store_true",
                   help="freeze the base and train LoRA adapters only (requires "
                        "--init-from). Saves small adapter-only checkpoints; merge "
                        "with `python -m diskrot.merge_lora` before serving")
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling = alpha/r)")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="dropout on the LoRA path")
    p.add_argument("--lora-targets", type=str, default=DEFAULT_TARGETS,
                   help="comma-separated Linear-name suffixes under model.blocks to adapt")
    p.add_argument("--lora-train-text-proj", action="store_true",
                   help="also train the CLAP tag projection in LoRA mode (frozen by default)")
    args = p.parse_args()

    # Tags drive the pooled-CLAP path (use_text_conditioning); lyrics drive the
    # phoneme LyricEncoder path (use_lyric_conditioning) — independent flags now.
    # Melody drives the additive chroma path (use_melody_conditioning).
    # With --init-from, this flag-built GPTConfig is advisory only: train_run
    # replaces it with the checkpoint's cfg (so a fine-tune/LoRA run always
    # matches the base architecture and conditioning axes).
    model_cfg = GPTConfig(
        use_text_conditioning=args.tags_path is not None,
        use_lyric_conditioning=args.lyrics_path is not None,
        use_melody_conditioning=args.melody,
    )
    cfg = TrainConfig(
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        cache_dir=args.cache_dir,
        ckpt_dir=args.ckpt_dir,
        tags_path=args.tags_path,
        lyrics_path=args.lyrics_path,
        structure_path=args.structure_path,
        keys_path=args.keys_path,
        phonemes_path=args.phonemes_path,
        init_from=args.init_from,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        lora_train_text_proj=args.lora_train_text_proj,
        model=model_cfg,
    )
    train_run(cfg)
