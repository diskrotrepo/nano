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
      --batch-size 32 --lr 2.1e-4 --warmup-steps 5000 \\
      --eval-batches 50 --ckpt-subdir v8_sing

Multi-GPU DDP on 8×H100 (per-rank batch_size; global = n_gpus × that):
    modal run --detach diskrot/modal_train.py --n-gpus 8 --batch-size 4

Fine-tune from an existing checkpoint (fresh optimizer/step, new subdir):
    modal run --detach diskrot/modal_train.py --n-gpus 8 \\
      --init-from v8_sing/best.pt --ckpt-subdir v8_ft --lr 5e-5

LoRA-train (frozen base + adapters; single H100 is plenty; see README.finetune.md):
    modal run --detach diskrot/modal_train.py \\
      --init-from v8_sing/best.pt --lora --ckpt-subdir v8_lora \\
      [--data-subdir my_corpus]

Pull checkpoints back when training is done (substitute the subdir):
    modal volume get nano-ckpts /v7_1500m/best.pt ./checkpoints/best.pt
"""
from __future__ import annotations

import threading

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
        # Pinned EXACTLY (not >=): the torch version is part of every
        # torch.compile cache key, so an incidental image rebuild that pulled a
        # newer torch would invalidate the whole nano-compile-cache volume and
        # silently re-pay the ~35-min cold compile. Bump deliberately (and
        # expect one cold compile to reseed when you do). triton ships with
        # torch (3.7.0 with this pair) — no separate pin needed.
        "torch==2.12.0",
        "torchaudio==2.11.0",
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
    # Persist torch.compile artifacts (Inductor FX graphs + Triton kernels) on
    # the nano-compile-cache volume so container restarts — the 24h-timeout
    # auto-retries, preemption resumes, manual relaunches — compile warm
    # (~minutes) instead of re-paying the ~35-min cold window on 8 idle GPUs.
    # Cache keys include code/torch-version/GPU-arch hashes, so a model edit
    # just re-pays one cold compile and reseeds. mp.spawn workers inherit env.
    .env({
        "TORCHINDUCTOR_CACHE_DIR": "/compile-cache/inductor",
        "TRITON_CACHE_DIR": "/compile-cache/triton",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",  # explicit; default varies by torch version
    })
    .add_local_python_source("model", "diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
ckpts_vol = modal.Volume.from_name("nano-ckpts", create_if_missing=True)
# torch.compile cache (see image .env above). All 8 DDP ranks write the same
# dir concurrently; Inductor's file locking is built for concurrent processes —
# worst case on a network volume is duplicated compile work, not corruption.
compile_cache_vol = modal.Volume.from_name("nano-compile-cache", create_if_missing=True)


def _commit_quietly(vol: modal.Volume) -> None:
    """Commit a volume, swallowing transient RPC errors — a cache commit must
    never kill the run (cf. the volume.commit crash-storm in tokenize/transcribe)."""
    try:
        vol.commit()
    except Exception as e:
        print(f"[compile-cache] commit failed (ignored): {e}", flush=True)

# WandB secret: mounted into training containers so train.py's _init_wandb()
# sees WANDB_API_KEY and can call wandb.init(). The secret must already exist;
# create once locally with:
#   modal secret create wandb WANDB_API_KEY=<your-key>
# If you don't want wandb on a particular run, leave --wandb-project empty and
# the env var is ignored. If the secret doesn't exist yet, function calls
# below will fail at launch with a clear NotFoundError -- create it and retry.
wandb_secret = modal.Secret.from_name("wandb")


# ~1.5B model on 8×H100 DDP, served via the sharded mmap dataset path,
# 60s segments with RoPE so the model natively holds ~1-minute single-shots
# (a 60s crop covers a full section + transition; see the section-length
# rationale — 30s clipped the long-tail 16-bar verses). At the recommended
# corpus scale the token count is roughly Chinchilla-matched to 1.5B, so this is
# the right-sized model for the data — the prior 287M was heavily over-data'd.
# Gradient checkpointing is on because 1.5B × 60s segments × bf16 is tight on
# 80 GB H100s without it; the ~25% recompute cost is well worth the headroom.
# Batch is global 32 / per-rank 4 on 8 ranks: 60s doubles the sequence vs the
# old 30s run, so the batch is halved (64→32) to keep the per-rank token load
# (4×5160) equal to the old 8×2580 that fit. Drop to per-rank 2 (global 16) on
# OOM; raise toward per-rank 6 if memory allows (re-sqrt-scale lr if you do).
DEFAULTS = {
    "d_model": 2048,
    "n_layers": 22,
    "n_heads": 16,                 # head_dim = 128 (even, RoPE-safe; 2048 % 16 == 0)
    "d_ff": 8192,
    "dropout": 0.05,
    "max_seq_len": 8192,           # RoPE table covers ~95s at 86 Hz (60s seg = 5168 frames, fits)
    "use_gradient_checkpointing": True,
    "segment_seconds": 60.0,       # 60s = 5160 frames; native ~1-min single-shot
    "batch_size": 32,              # global; per-rank = 4 on 8 ranks. Halved from 64
                                   # because 60s doubles the sequence — per-rank token
                                   # load (4×5160) is identical to the old 30s 8×2580
                                   # that fit, so memory holds. tokens/step unchanged.
    # Stability (2026-06-12): both v8 launches at the sqrt-scaled 2.1e-4 target
    # diverged cb0-first during warmup (onset ~5e-5–1.2e-4, the second one even
    # with the zero-gated conditioning init; Adam beta2 was already 0.95). Fix =
    # QK-norm (the structural piece, see GPTConfig.use_qk_norm) + a lower peak
    # and gentler ramp for margin.
    "lr": 1.5e-4,
    "warmup_steps": 10_000,
    "use_qk_norm": True,
    # Extend QK-norm into the LyricEncoder. The first v8_sing run trained with
    # use_qk_norm=True (decoder) but the encoder lacked it; grad forensics traced
    # that run's gradient explosion (~step 65k) to lyric_encoder.layers.0. See
    # GPTConfig.use_lyric_qk_norm. Checkpoint-incompatible for the encoder weights
    # (adds q_norm/k_norm) — needs a fresh ckpt dir, decoder warm-started.
    "use_lyric_qk_norm": True,
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
    # (frame reorder scrambles sung alignment) but keep tags + co-reordered melody.
    #
    # DEFERRED for the v8 singing run: FIM is feature-flagged OFF here so the lyric
    # path gets full signal (no 15% of batches dropping lyrics) and the run is
    # cleanly attributable to lyrics + structure + melody. The /infill code path
    # (model/fim.py, the train-loop reorder, the server guard, tests) stays intact —
    # flip use_fim back to True for a future fresh start to add infill. Enabling
    # use_fim grows the vocab and is checkpoint-incompatible, so it can't be hot-
    # swapped onto this checkpoint anyway.
    "use_fim": False,
    "fim_prob": 0.0,
}
DDP_PER_RANK_BATCH = DEFAULTS["batch_size"] // 8   # = 4 (global 32 on 8 ranks)


def _build_cfg_kwargs(
    steps: int, batch_size: int, lr: float, warmup_steps: int,
    patience: int, eval_batches: int, ckpt_subdir: str, text_conditioned: bool,
    segment_seconds: float = 10.0,
    fim_prob: float = DEFAULTS["fim_prob"],
    wandb_project: str | None = None, wandb_run_name: str | None = None,
    init_from: str = "",
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_targets: str = "",
    lora_train_text_proj: bool = False,
    data_subdir: str = "",
) -> dict:
    """Shared TrainConfig builder for both single- and multi-GPU paths.
    Returns a plain dict so it survives mp.spawn pickling.

    ``init_from`` is a path relative to the nano-ckpts volume (e.g.
    "v8_sing/best.pt") — fine-tune / LoRA-train from that checkpoint; the
    architecture then comes FROM the checkpoint and the --d-model etc. flags
    are advisory. ``data_subdir`` re-roots the token cache and every
    conditioning path under /tokens/{data_subdir}, so a fine-tune corpus can
    be packed beside the main one (same layout, one directory down)."""
    root = f"/tokens/{data_subdir}" if data_subdir else "/tokens"
    tags_path = f"{root}/tags.json" if text_conditioned else None
    lyrics_path = f"{root}/lyrics" if text_conditioned else None
    structure_path = f"{root}/structure" if text_conditioned else None
    keys_path = f"{root}/keys.json" if text_conditioned else None
    phonemes_path = f"{root}/phonemes" if text_conditioned else None
    from model.lora import DEFAULT_TARGETS

    return dict(
        fim_prob=fim_prob,
        cache_dir=root,
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
        keys_path=keys_path,
        phonemes_path=phonemes_path,
        text_conditioned=text_conditioned,
        segment_seconds=segment_seconds,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        init_from=f"/ckpts/{init_from}" if init_from else None,
        lora=lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_targets=lora_targets or DEFAULT_TARGETS,
        lora_train_text_proj=lora_train_text_proj,
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
    use_qk_norm: bool = DEFAULTS["use_qk_norm"],
    use_lyric_qk_norm: bool = DEFAULTS["use_lyric_qk_norm"],
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
        use_qk_norm=use_qk_norm,
        use_lyric_qk_norm=use_lyric_qk_norm,
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
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol,
             "/compile-cache": compile_cache_vol},
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
    # Fine-tune / LoRA — see _build_cfg_kwargs. init_from is relative to the
    # nano-ckpts volume; with it set, the architecture flags above are
    # advisory (train_run takes GPTConfig from the checkpoint).
    init_from: str = "",
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_targets: str = "",
    lora_train_text_proj: bool = False,
    data_subdir: str = "",
):
    from diskrot.train import TrainConfig, train_run

    cfg_kwargs = _build_cfg_kwargs(
        steps, batch_size, lr, warmup_steps, patience, eval_batches,
        ckpt_subdir, text_conditioned,
        segment_seconds=segment_seconds,
        wandb_project=wandb_project or None,
        wandb_run_name=wandb_run_name or None,
        init_from=init_from, lora=lora, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_targets=lora_targets, lora_train_text_proj=lora_train_text_proj,
        data_subdir=data_subdir,
    )
    cfg_kwargs.pop("text_conditioned")
    model_cfg = _build_model_cfg(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff,
        dropout=dropout, text_conditioned=text_conditioned,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    cfg = TrainConfig(**cfg_kwargs, model=model_cfg)
    try:
        train_run(cfg, ckpt_callback=ckpts_vol.commit)
    finally:
        _commit_quietly(compile_cache_vol)  # persist the seeded compile cache


# Multi-GPU DDP — 8×H100 in a single container, ranks coordinated via mp.spawn.
# batch_size on this path is *per-rank*; the global batch is n_gpus× that. lr
# should be sqrt-scaled against the global batch the same way as the single-GPU
# path (global 32 with per-rank 4 on 8 ranks; lr is sqrt-scaled to 2.1e-4 to match).
@app.function(
    image=image,
    gpu="H100:8",
    timeout=60 * 60 * 24,  # 24 hours
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol,
             "/compile-cache": compile_cache_vol},
    secrets=[wandb_secret],
    # Self-heal on worker preemption — see train_remote(). Auto-resume from
    # latest.pt makes a restart cheap; the DDP ranks re-spawn fresh each time.
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def train_remote_multi(
    steps: int = DEFAULTS["steps"],
    batch_size: int = DDP_PER_RANK_BATCH,  # per-rank; global = 8 × per-rank = 32
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
    # Fine-tune / LoRA — see _build_cfg_kwargs. NOTE: with init_from set, the
    # architecture flags are advisory inside train_run, but the parent's CLAP
    # precompute below still uses --d-model — it must match the base ckpt.
    # Each rank loads the base checkpoint to CPU during startup (transient
    # ~6 GB × n_gpus at 1.5B fp32).
    init_from: str = "",
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_targets: str = "",
    lora_train_text_proj: bool = False,
    data_subdir: str = "",
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
        init_from=init_from, lora=lora, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_targets=lora_targets, lora_train_text_proj=lora_train_text_proj,
        data_subdir=data_subdir,
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
        keys_path=cfg_kwargs["keys_path"],
        phonemes_path=cfg_kwargs["phonemes_path"],
    )
    tag_cache: dict = {}
    if text_conditioned and shared_bundle["tags"]:
        unique_tags = sorted(set(shared_bundle["tags"].values()))
        tag_cache = _precompute_clap_cache(unique_tags, d_model, n_gpus=n_gpus)

    # Periodically persist the torch.compile cache while the ranks run, so the
    # ~35-min cold-compile artifacts survive even if this container dies mid-run
    # (a daemon thread; the join below ends it with the process).
    stop_commits = threading.Event()

    def _cache_committer():
        while not stop_commits.wait(600):
            _commit_quietly(compile_cache_vol)

    threading.Thread(target=_cache_committer, daemon=True).start()
    try:
        mp.spawn(
            _ddp_worker,
            args=(n_gpus, cfg_kwargs, model_kwargs, shared_bundle, tag_cache),
            nprocs=n_gpus, join=True,
        )
    finally:
        stop_commits.set()
        _commit_quietly(compile_cache_vol)
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
    # Fine-tune: start from a checkpoint on nano-ckpts (e.g. "v8_sing/best.pt")
    # with a fresh optimizer/step. Use a NEW --ckpt-subdir for the run (an
    # existing latest.pt there would resume instead). Architecture flags are
    # then advisory — GPTConfig comes from the checkpoint.
    init_from: str = "",
    # LoRA: freeze the base, train adapters only (requires --init-from).
    # Single GPU (the default n_gpus=1) is the recommended LoRA path.
    lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_targets: str = "",
    lora_train_text_proj: bool = False,
    # Train on a fine-tune corpus packed under /tokens/{data_subdir} instead
    # of the main pack (same layout: token_cache shards, tags.json, lyrics/ …).
    data_subdir: str = "",
):
    if lora and not init_from:
        raise SystemExit("--lora requires --init-from (e.g. --init-from v8_sing/best.pt)")
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
        init_from=init_from, lora=lora, lora_r=lora_r, lora_alpha=lora_alpha,
        lora_targets=lora_targets, lora_train_text_proj=lora_train_text_proj,
        data_subdir=data_subdir,
    )
    if n_gpus > 1:
        fc = train_remote_multi.spawn(**common, n_gpus=n_gpus)
    else:
        fc = train_remote.spawn(**common)
    print(f"training launched (detached) — function call id: {fc.object_id}")
    print("monitor with: modal app logs nano-train")
    if lora:
        print(f"merge adapters when done: modal run diskrot/modal_merge_lora.py "
              f"--base {init_from} --lora {ckpt_subdir}/best.pt")
        print(f"then pull: modal volume get nano-ckpts /{ckpt_subdir}/merged_inference.pt ./checkpoints/latest.pt")
    else:
        print(f"pull checkpoints: modal volume get nano-ckpts /{ckpt_subdir}/best.pt ./checkpoints/best.pt")
    print(f"inspect trajectory: modal run diskrot/modal_inspect_ckpts.py --prefix {ckpt_subdir}")
