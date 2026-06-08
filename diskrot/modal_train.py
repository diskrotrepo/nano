"""Modal entrypoint for training on an H100.

Setup (one-time):
    pip install modal
    modal token new
    # tokenize locally first, then upload the cache to a Modal volume:
    modal volume create nano-tokens
    modal volume put nano-tokens token_cache/ /

Run the default ~1.5B model (single H100):
    modal run --detach diskrot/modal_train.py

Override architecture / hyperparams via CLI:
    modal run --detach diskrot/modal_train.py \\
      --d-model 2048 --n-layers 22 --n-heads 16 --d-ff 8192 \\
      --batch-size 64 --lr 3.0e-4 --warmup-steps 5000 \\
      --eval-batches 50 --ckpt-subdir v7_1500m

Multi-GPU DDP on 8×H100 (per-rank batch_size; global = n_gpus × that):
    modal run --detach diskrot/modal_train.py --n-gpus 8 --batch-size 8

Pull checkpoints back when training is done (substitute the subdir):
    modal volume get nano-ckpts /v7_1500m/best.pt ./checkpoints/best.pt
"""
from __future__ import annotations

import modal

app = modal.App("nano-train")


def _prefetch_clap() -> None:
    """Download CLAP weights into the image so runtime never blocks on HF.
    msclap's CLAP() ctor has no timeout — a CLOSE_WAIT on HF's CDN hangs forever."""
    from msclap import CLAP

    CLAP(version="2023", use_cuda=False)


def _prefetch_g2p() -> None:
    """Bake g2p_en's nltk data + model into the image so dataloader workers never
    block on a download. The phoneme LyricEncoder phonemizes lyrics with g2p_en;
    its first G2p() call needs the CMUdict + POS tagger nltk corpora."""
    import nltk

    for res in ("averaged_perceptron_tagger_eng", "cmudict", "averaged_perceptron_tagger"):
        nltk.download(res, quiet=True)
    from g2p_en import G2p

    G2p()("warm up the cache")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4",
        "torchaudio>=2.4",
        "librosa>=0.10",
        "descript-audio-codec>=1.0.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "msclap",
        "wandb>=0.16",
        "g2p_en==2.1.0",
    )
    # Override descript-audiotools' protobuf<3.20 pin, but stay under 7 so
    # wandb's `protobuf<7` constraint is satisfied. Intersection of the three
    # constraints lands on protobuf 4.x; 4.25 picks a recent stable.
    .run_commands("pip install 'protobuf>=4.25,<5'")
    .run_function(_prefetch_clap)
    .run_function(_prefetch_g2p)
    .add_local_python_source("model", "diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
ckpts_vol = modal.Volume.from_name("nano-ckpts", create_if_missing=True)

# WandB secret: mounted into training containers so train.py's _init_wandb()
# sees WANDB_API_KEY and can call wandb.init(). The secret must already exist;
# create once locally with:
#   modal secret create wandb WANDB_API_KEY=<your-key>
# If you don't want wandb on a particular run, leave --wandb-project empty and
# the env var is ignored. If the secret doesn't exist yet, function calls
# below will fail at launch with a clear NotFoundError -- create it and retry.
wandb_secret = modal.Secret.from_name("wandb")


# ~1.5B model on 8×H100 DDP, served via the sharded mmap dataset path,
# 30s segments with RoPE so inference can extrapolate to 90s single-shot
# generation. At the recommended corpus scale the token count is roughly
# Chinchilla-matched to 1.5B, so this is the right-sized model for the data —
# the prior 287M was heavily over-data'd. Gradient checkpointing is on because 1.5B × 30s
# segments × bf16 is tight on 80 GB H100s without it; the ~25% recompute cost
# is well worth the headroom. Batch is conservative (global 64, per-rank 8 on
# 8 ranks) to fit 1.5B at 30s; raise toward 96 if memory allows, drop to 48
# (per-rank 6) on OOM.
DEFAULTS = {
    "d_model": 2048,
    "n_layers": 22,
    "n_heads": 16,                 # head_dim = 128 (even, RoPE-safe; 2048 % 16 == 0)
    "d_ff": 8192,
    "dropout": 0.05,
    "max_seq_len": 8192,           # RoPE table covers ~95s at 86 Hz
    "use_gradient_checkpointing": True,
    "segment_seconds": 30.0,
    "batch_size": 64,              # global; per-rank = 8 on 8 ranks
    "lr": 3.0e-4,
    "warmup_steps": 5000,
    "steps": 400_000,
    "patience": 20,
    "eval_batches": 50,
    # v8_sing: phoneme LyricEncoder cross-attention (intelligible singing) + the
    # additive chroma (melody) conditioning stream that lets a hummed melody be
    # re-rendered in the prompt's timbre (the /cover path). Both land in the same
    # fresh start — v8 is still pre-ship, so melody folds into its train rather
    # than minting a new version. Incompatible with v7 checkpoints (new modules +
    # GPTConfig fields). Melody requires the pack to carry the parallel chroma
    # sidecar (modal_melody.py → --mel-cache-dir); without it the model trains the
    # null path only.
    "ckpt_subdir": "v8_sing",
    # Lyric (phoneme) conditioning — a ~100M bidirectional encoder feeding a
    # per-block lyric cross-attention. Enabled together with tag conditioning.
    "lyric_enc_layers": 3,
    "lyric_enc_heads": 8,
    "lyric_enc_d_ff": 4096,
    "max_lyric_len": 256,
    # Melody (chroma) conditioning — a small additive encoder over the 12-bin
    # chromagram (~13M params). Enabled together with tags + lyrics.
    "melody_n_bins": 12,
    "melody_enc_layers": 2,
    # Fill-in-the-middle (infill, the /infill path). Adds two per-codebook control
    # ids (<SUF>/<MID>) and reorders fim_prob of batches into the FIM layout so the
    # model learns to bridge a gap given prefix+suffix. FIM batches drop lyrics
    # (frame reorder scrambles sung alignment) but keep tags + co-reordered melody,
    # so keep fim_prob modest — singing is the headline objective. Enabling use_fim
    # grows the vocab and is checkpoint-incompatible (fresh v8 start).
    "use_fim": True,
    "fim_prob": 0.15,
}
DDP_PER_RANK_BATCH = DEFAULTS["batch_size"] // 8   # = 8 (global 64 on 8 ranks)


def _build_cfg_kwargs(
    steps: int, batch_size: int, lr: float, warmup_steps: int,
    patience: int, eval_batches: int, ckpt_subdir: str, text_conditioned: bool,
    segment_seconds: float = 10.0,
    fim_prob: float = DEFAULTS["fim_prob"],
    wandb_project: str | None = None, wandb_run_name: str | None = None,
) -> dict:
    """Shared TrainConfig builder for both single- and multi-GPU paths.
    Returns a plain dict so it survives mp.spawn pickling."""
    tags_path = "/tokens/tags.json" if text_conditioned else None
    lyrics_path = "/tokens/lyrics" if text_conditioned else None
    structure_path = "/tokens/structure" if text_conditioned else None
    return dict(
        fim_prob=fim_prob,
        cache_dir="/tokens",
        ckpt_dir=f"/ckpts/{ckpt_subdir}",
        device="cuda",
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        patience=patience,
        eval_batches=eval_batches,
        tags_path=tags_path,
        lyrics_path=lyrics_path,
        structure_path=structure_path,
        text_conditioned=text_conditioned,
        segment_seconds=segment_seconds,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )


def _build_model_cfg(
    d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float,
    text_conditioned: bool,
    max_seq_len: int = DEFAULTS["max_seq_len"],
    use_gradient_checkpointing: bool = True,
    lyric_enc_layers: int = DEFAULTS["lyric_enc_layers"],
    lyric_enc_heads: int = DEFAULTS["lyric_enc_heads"],
    lyric_enc_d_ff: int = DEFAULTS["lyric_enc_d_ff"],
    max_lyric_len: int = DEFAULTS["max_lyric_len"],
    melody_n_bins: int = DEFAULTS["melody_n_bins"],
    melody_enc_layers: int = DEFAULTS["melody_enc_layers"],
    use_fim: bool = DEFAULTS["use_fim"],
):
    from model.nano_audio_gpt import GPTConfig

    # text_conditioned drives the pooled-CLAP tag path, the phoneme lyric path,
    # AND the additive melody path — this bespoke model ships all three together.
    # FIM (infill) is independent of conditioning: it adds the <SUF>/<MID> control
    # ids and is trained via fim_prob batch reordering in the loop.
    return GPTConfig(
        use_text_conditioning=text_conditioned,
        use_lyric_conditioning=text_conditioned,
        use_melody_conditioning=text_conditioned,
        use_fim=use_fim,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, dropout=dropout,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
        lyric_enc_layers=lyric_enc_layers,
        lyric_enc_heads=lyric_enc_heads,
        lyric_enc_d_ff=lyric_enc_d_ff,
        max_lyric_len=max_lyric_len,
        melody_n_bins=melody_n_bins,
        melody_enc_layers=melody_enc_layers,
    )


def _ddp_worker(
    local_rank: int, world_size: int, cfg_kwargs: dict, model_kwargs: dict,
    shared_bundle: dict | None = None,
    precomputed_tag_cache: dict | None = None,
) -> None:
    """One DDP rank. Runs in a subprocess launched by mp.spawn.

    rank 0 re-acquires the checkpoints volume by name so it can commit
    incrementally; non-zero ranks don't touch the volume.

    shared_bundle and precomputed_tag_cache (if provided) are produced once
    by the parent process — workers reuse them instead of redundantly
    re-loading 50 GB off the volume / re-encoding CLAP per rank."""
    from diskrot.train import TrainConfig, train_run

    text_conditioned = cfg_kwargs.pop("text_conditioned")
    model_cfg = _build_model_cfg(text_conditioned=text_conditioned, **model_kwargs)
    cfg = TrainConfig(
        **cfg_kwargs,
        model=model_cfg,
        world_size=world_size,
        local_rank=local_rank,
    )
    callback = None
    if local_rank == 0:
        import modal as _modal

        callback = _modal.Volume.from_name("nano-ckpts").commit
    train_run(
        cfg,
        ckpt_callback=callback,
        shared_bundle=shared_bundle,
        precomputed_tag_cache=precomputed_tag_cache,
    )


def _encode_clap_shard(rank: int, shard: list[str], d_model: int, batch: int) -> dict:
    """Worker body shared by single- and multi-GPU paths. Sets the active
    device, loads CLAP onto it, encodes the shard, returns a CPU dict.
    Caller is responsible for tearing down the CUDA context (process exit
    in multi-GPU case, explicit del/empty_cache in single-GPU case)."""
    import time as _t

    import torch

    torch.cuda.set_device(rank)
    from model.text_encoder import CLAPTextEncoder

    enc = CLAPTextEncoder(d_out=d_model, device="cuda")
    enc.to(f"cuda:{rank}")
    enc.eval()
    enc._ensure_clap()
    n = len(shard)
    print(f"[clap-gpu{rank}] encoding {n} tags (batch={batch}, bf16)...",
          flush=True)
    t0 = _t.time()
    cache: dict[str, torch.Tensor] = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, n, batch):
            chunk = shard[start:start + batch]
            embs = enc._clap.get_text_embeddings(chunk)
            embs_cpu = embs.float().cpu()
            for tag, e in zip(chunk, embs_cpu):
                cache[tag] = e
            done = start + len(chunk)
            if done % (batch * 8) == 0 or done == n:
                elapsed = _t.time() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"[clap-gpu{rank}] {done}/{n} "
                      f"({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)
    return cache


def _clap_shard_worker(rank: int, shards: list[list[str]], d_model: int,
                       batch: int, return_dict) -> None:
    """mp.spawn entrypoint for multi-GPU sharded encoding."""
    return_dict[rank] = _encode_clap_shard(rank, shards[rank], d_model, batch)


def _precompute_clap_cache(unique_tags: list[str], d_model: int,
                            n_gpus: int = 1) -> dict:
    """Encode every unique tag with CLAP once, return a dict of CPU [1024]
    tensors that survive mp.spawn pickling. Workers move them to their own
    GPU. This replaces the previous 4× per-rank serial encoding loop.

    When n_gpus > 1 and that many devices are visible, shards the tag list
    across GPUs via mp.spawn and merges the results — near-linear speedup
    since encoding is GPU-bound at BATCH=512+bf16."""
    import time as _t

    import torch

    if not unique_tags:
        return {}

    BATCH = 512
    visible = torch.cuda.device_count()
    if n_gpus > 1 and visible >= n_gpus:
        # P4: shard across GPUs. mp.spawn uses the "spawn" start method, so
        # each worker boots a fresh CUDA context — no conflict with whatever
        # the parent has touched. Round-robin sharding (i::n_gpus) interleaves
        # short/long tags so per-GPU padding waste is even.
        import torch.multiprocessing as mp

        shards = [unique_tags[i::n_gpus] for i in range(n_gpus)]
        print(f"[clap-parent] sharding {len(unique_tags)} tags across "
              f"{n_gpus} GPUs (batch={BATCH}, bf16)...", flush=True)
        t0 = _t.time()
        manager = mp.get_context("spawn").Manager()
        return_dict = manager.dict()
        mp.spawn(
            _clap_shard_worker,
            args=(shards, d_model, BATCH, return_dict),
            nprocs=n_gpus, join=True,
        )
        cache: dict[str, torch.Tensor] = {}
        for rank in range(n_gpus):
            cache.update(return_dict[rank])
        print(f"[clap-parent] done: {len(cache)} embeddings in "
              f"{_t.time()-t0:.0f}s ({n_gpus}-GPU)", flush=True)
        return cache

    # P1: batched CLAP encoding. msclap pads variable-length tokenizations
    # internally — verified by tests/test_pipeline_audit.py::
    # test_clap_batched_matches_single_tag (batched matches per-tag to <1e-4
    # atol across 36 mixed-length tags). At the production ~34k-tag corpus
    # this drops CLAP precompute from ~11 min to ~1 min. BATCH=512 + bf16
    # autocast on H100 takes it further; reduce BATCH if a future CLAP
    # variant OOMs. bf16 is safe — the learned projection runs in fp32 on
    # the embeddings after they've been cast back here.
    t0 = _t.time()
    cache = _encode_clap_shard(0, unique_tags, d_model, BATCH)
    # Free CLAP GPU memory before mp.spawn workers initialize their own
    # CUDA contexts on rank 0.
    torch.cuda.empty_cache()
    print(f"[clap-parent] done: {len(cache)} embeddings in {_t.time()-t0:.0f}s",
          flush=True)
    return cache


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 12,  # 12 hours
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol},
    secrets=[wandb_secret],
    # Self-heal on worker preemption. train_run() auto-resumes from latest.pt
    # (and wandb resume="allow"), so a restart picks up at the last checkpoint
    # instead of stranding the run. A deterministic failure still surfaces
    # after the retries are exhausted.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_remote(
    steps: int = DEFAULTS["steps"],
    batch_size: int = DEFAULTS["batch_size"],
    lr: float = DEFAULTS["lr"],
    warmup_steps: int = DEFAULTS["warmup_steps"],
    patience: int = DEFAULTS["patience"],
    eval_batches: int = DEFAULTS["eval_batches"],
    ckpt_subdir: str = DEFAULTS["ckpt_subdir"],
    d_model: int = DEFAULTS["d_model"],
    n_layers: int = DEFAULTS["n_layers"],
    n_heads: int = DEFAULTS["n_heads"],
    d_ff: int = DEFAULTS["d_ff"],
    dropout: float = DEFAULTS["dropout"],
    text_conditioned: bool = True,
    segment_seconds: float = DEFAULTS["segment_seconds"],
    max_seq_len: int = DEFAULTS["max_seq_len"],
    # Default ON: 300M params × 30s segments × bf16 is tight on 80 GB H100s
    # without it. The ~25% step-time cost is negligible vs the cost of an OOM
    # mid-run. Pass --no-use-gradient-checkpointing to disable for small-config
    # benchmarks.
    use_gradient_checkpointing: bool = True,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    from diskrot.train import TrainConfig, train_run

    tags_path = "/tokens/tags.json" if text_conditioned else None
    lyrics_path = "/tokens/lyrics" if text_conditioned else None
    model_cfg = _build_model_cfg(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
        dropout=dropout, text_conditioned=text_conditioned,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    cfg = TrainConfig(
        cache_dir="/tokens",
        ckpt_dir=f"/ckpts/{ckpt_subdir}",
        device="cuda",
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        patience=patience,
        eval_batches=eval_batches,
        tags_path=tags_path,
        lyrics_path=lyrics_path,
        segment_seconds=segment_seconds,
        fim_prob=DEFAULTS["fim_prob"],
        wandb_project=wandb_project or None,
        wandb_run_name=wandb_run_name or None,
        model=model_cfg,
    )
    train_run(cfg, ckpt_callback=ckpts_vol.commit)


# Multi-GPU DDP — 8×H100 in a single container, ranks coordinated via mp.spawn.
# batch_size on this path is *per-rank*; the global batch is n_gpus× that. lr
# should be sqrt-scaled against the global batch the same way as the single-GPU
# path (global stays 64 with per-rank 8 on 8 ranks, so the tuned lr holds).
@app.function(
    image=image,
    gpu="H100:8",
    timeout=60 * 60 * 24,  # 24 hours
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol},
    secrets=[wandb_secret],
    # Self-heal on worker preemption — see train_remote(). Auto-resume from
    # latest.pt makes a restart cheap; the DDP ranks re-spawn fresh each time.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_remote_multi(
    steps: int = DEFAULTS["steps"],
    batch_size: int = DDP_PER_RANK_BATCH,  # per-rank; global = 8 × per-rank = 64
    lr: float = DEFAULTS["lr"],
    warmup_steps: int = DEFAULTS["warmup_steps"],
    patience: int = DEFAULTS["patience"],
    eval_batches: int = DEFAULTS["eval_batches"],
    ckpt_subdir: str = DEFAULTS["ckpt_subdir"],
    d_model: int = DEFAULTS["d_model"],
    n_layers: int = DEFAULTS["n_layers"],
    n_heads: int = DEFAULTS["n_heads"],
    d_ff: int = DEFAULTS["d_ff"],
    dropout: float = DEFAULTS["dropout"],
    text_conditioned: bool = True,
    n_gpus: int = 8,
    segment_seconds: float = DEFAULTS["segment_seconds"],
    max_seq_len: int = DEFAULTS["max_seq_len"],
    # On by default; see train_remote() for the OOM rationale.
    use_gradient_checkpointing: bool = True,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    import torch
    import torch.multiprocessing as mp
    from pathlib import Path as _Path

    from diskrot.dataset import load_mmap_bundle
    from diskrot.pack_cache import PACKED_DIR, SHARD_INDEX_NAME
    from diskrot.train import TrainConfig
    from model.codec import DACodec

    assert torch.cuda.device_count() >= n_gpus, (
        f"requested n_gpus={n_gpus} but only {torch.cuda.device_count()} CUDA devices visible"
    )
    cfg_kwargs = _build_cfg_kwargs(
        steps, batch_size, lr, warmup_steps, patience, eval_batches,
        ckpt_subdir, text_conditioned,
        segment_seconds=segment_seconds,
        wandb_project=wandb_project or None,
        wandb_run_name=wandb_run_name or None,
    )
    model_kwargs = dict(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, dropout=dropout,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    # ----- Preload once in the parent so the 4 ranks share one copy ----- #
    _defaults = TrainConfig()
    # cfg_kwargs already carries the resolved segment_seconds; use it (not the
    # TrainConfig default) so the bundle covers the right crop window.
    segment_frames = int(cfg_kwargs["segment_seconds"] * DACodec.FRAME_RATE_HZ)
    packed_dir = _Path(cfg_kwargs["cache_dir"]) / PACKED_DIR
    if not (packed_dir / SHARD_INDEX_NAME).exists():
        raise FileNotFoundError(
            f"No sharded packed layout at {packed_dir} — run diskrot.pack_cache first"
        )
    print(f"[parent] sharded layout detected at {packed_dir} — using mmap bundle "
          f"(segment_frames={segment_frames})", flush=True)
    shared_bundle = load_mmap_bundle(
        packed_dir=packed_dir,
        segment_frames=segment_frames,
        val_ratio=_defaults.val_ratio,
        seed=_defaults.seed,
        tags_path=cfg_kwargs["tags_path"],
        lyrics_path=cfg_kwargs["lyrics_path"],
        structure_path=cfg_kwargs["structure_path"],
    )
    tag_cache: dict = {}
    if text_conditioned and shared_bundle["tags"]:
        unique_tags = sorted(set(shared_bundle["tags"].values()))
        tag_cache = _precompute_clap_cache(unique_tags, d_model, n_gpus=n_gpus)

    mp.spawn(
        _ddp_worker,
        args=(n_gpus, cfg_kwargs, model_kwargs, shared_bundle, tag_cache),
        nprocs=n_gpus, join=True,
    )
    # Final commit from the parent process catches anything rank 0 wrote after
    # its last incremental commit.
    ckpts_vol.commit()


@app.local_entrypoint()
def main(
    steps: int = DEFAULTS["steps"],
    # 0 = "auto": pick the right default for the path being launched.
    # Single-GPU → DEFAULTS["batch_size"] (64). DDP → DDP_PER_RANK_BATCH
    # (per-rank value chosen so global batch matches the tuned 64). Any
    # explicit positive value overrides this.
    batch_size: int = 0,
    lr: float = DEFAULTS["lr"],
    warmup_steps: int = DEFAULTS["warmup_steps"],
    patience: int = DEFAULTS["patience"],
    eval_batches: int = DEFAULTS["eval_batches"],
    ckpt_subdir: str = DEFAULTS["ckpt_subdir"],
    d_model: int = DEFAULTS["d_model"],
    n_layers: int = DEFAULTS["n_layers"],
    n_heads: int = DEFAULTS["n_heads"],
    d_ff: int = DEFAULTS["d_ff"],
    dropout: float = DEFAULTS["dropout"],
    text_conditioned: bool = True,
    n_gpus: int = 1,
    segment_seconds: float = DEFAULTS["segment_seconds"],
    max_seq_len: int = DEFAULTS["max_seq_len"],
    # On by default — 300M params × 30s segments × bf16 is tight on 80 GB
    # H100s without it. The ~25% step-time cost is worth not having to
    # remember the flag. Pass --no-use-gradient-checkpointing to disable for
    # small-config benchmarks.
    use_gradient_checkpointing: bool = True,
    wandb_project: str = "",
    wandb_run_name: str = "",
):
    if batch_size <= 0:
        batch_size = DDP_PER_RANK_BATCH if n_gpus > 1 else DEFAULTS["batch_size"]
        print(f"[main] batch_size auto-selected = {batch_size} "
              f"(n_gpus={n_gpus}, global batch={batch_size * n_gpus})")
    common = dict(
        steps=steps, batch_size=batch_size, lr=lr, warmup_steps=warmup_steps,
        patience=patience, eval_batches=eval_batches, ckpt_subdir=ckpt_subdir,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
        dropout=dropout, text_conditioned=text_conditioned,
        segment_seconds=segment_seconds,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    if n_gpus > 1:
        fc = train_remote_multi.spawn(**common, n_gpus=n_gpus)
    else:
        fc = train_remote.spawn(**common)
    print(f"training launched (detached) — function call id: {fc.object_id}")
    print("monitor with: modal app logs nano-train")
    print(f"pull checkpoints: modal volume get nano-ckpts /{ckpt_subdir}/best.pt ./checkpoints/best.pt")
    print(f"inspect trajectory: modal run diskrot/modal_inspect_ckpts.py --prefix {ckpt_subdir}")
