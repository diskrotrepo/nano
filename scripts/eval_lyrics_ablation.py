"""Diagnostic: does lyric conditioning measurably help?

Evaluates best.pt on val crops that *actually carry lyrics* (after the 30s
window filter), comparing val loss under three conditioning regimes on the
SAME samples:

  full      = tags + lyrics   (what training does)
  tags_only = tags + ""       (lyrics blanked, tags kept)
  uncond    = no conditioning (text_emb=None)

Interpretation:
  loss(tags_only) - loss(full)  > 0  → lyrics lower loss → they help.
  ≈ 0                                → model ignores lyrics → the expensive
                                       Demucs+Whisper pass isn't earning its
                                       compute; consider dropping it.
  loss(uncond) - loss(tags_only)     → isolates how much *tags* contribute.

Only lyric-bearing samples are scored, so the comparison isn't diluted by the
instrumental majority (where every regime sees an empty lyric string anyway).

Run:  modal run scripts/eval_lyrics_ablation.py
"""
from __future__ import annotations

import modal

app = modal.App("nano-eval-lyrics-ablation")

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
    )
    .run_commands("pip install 'protobuf>=4'")
    .add_local_python_source("model", "diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens")
ckpts_vol = modal.Volume.from_name("nano-ckpts")


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 45,
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol},
)
def eval_ablation(
    n_batches: int = 400,
    batch_size: int = 64,
    seed: int = 42,
    val_ratio: float = 0.12,
    segment_seconds: float = 10.0,
):
    import torch
    from torch.utils.data import DataLoader

    from model.codec import DACodec
    from model.nano_audio_gpt import GPTConfig, NanoAudioGPT
    from model.text_encoder import CLAPTextEncoder
    from model.delay_pattern import build_train_inputs
    from diskrot.dataset import TokenDataset, collate_lyrics
    from diskrot.train import TrainConfig, _build_cond, _loss_fn

    device = "cuda"
    ckpt = torch.load("/ckpts/best.pt", map_location=device, weights_only=False)
    model_cfg = GPTConfig(**ckpt["cfg"])
    if not model_cfg.use_lyric_conditioning:
        raise SystemExit("checkpoint has no lyric conditioning — nothing to ablate")

    model = NanoAudioGPT(model_cfg).to(device)
    state_dict = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded best.pt — step={ckpt['step']} best_val_loss={ckpt['best_val_loss']:.4f}")

    text_encoder = CLAPTextEncoder(d_out=model_cfg.d_model, device=device).to(device).eval()
    if "text_proj" in ckpt:
        text_encoder.proj.load_state_dict(ckpt["text_proj"])
        print("loaded text_proj from checkpoint")
    text_encoder._ensure_clap()

    pad_id = model_cfg.pad_id
    segment_frames = int(segment_seconds * DACodec.FRAME_RATE_HZ)

    ds = TokenDataset(
        "/tokens", segment_frames=segment_frames, split="val",
        val_ratio=val_ratio, seed=seed,
        tags_path="/tokens/tags.json", lyrics_path="/tokens/lyrics",
    )
    # Pre-cache CLAP embeddings for every tag string so we don't run the CLAP
    # text tower for tags every batch (matches the training tag_cache path).
    tag_cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for tag in sorted(set(ds._tags.values())):
            if tag:
                tag_cache[tag] = text_encoder._clap.get_text_embeddings([tag]).squeeze(0).to(device)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False, persistent_workers=True,
        collate_fn=collate_lyrics,
    )

    regimes = ("full", "tags_only", "uncond")
    sums = {r: 0.0 for r in regimes}
    n_scored = 0  # lyric-bearing samples actually scored
    n_seen = 0

    @torch.no_grad()
    def regime_loss(inputs, targets, tags_sub, l_ids, l_mask, regime):
        # full = tags + phoneme lyrics; tags_only drops the lyric stream
        # (lyric_ids=None → lyric cross-attn skipped); uncond drops both.
        text_emb = None
        if regime in ("full", "tags_only"):
            text_emb = _build_cond(text_encoder, tags_sub, device, tag_cache)
        lyric_ids = l_ids if regime == "full" else None
        lyric_mask = l_mask if regime == "full" else None
        with torch.amp.autocast(device, enabled=True):
            logits = model(inputs, text_emb=text_emb, lyric_ids=lyric_ids, lyric_mask=lyric_mask)
            total, _ = _loss_fn(logits, targets, pad_id)
        return total.item()

    with torch.no_grad():
        for i, (batch, tags, lyric_ids, lyric_mask) in enumerate(loader):
            if i >= n_batches:
                break
            n_seen += lyric_ids.shape[0]
            # Keep only samples that carry real lyrics (>1 token, i.e. more than
            # the lone BOS) — the only ones where the lyric stream differs from
            # the tags_only regime.
            real_len = lyric_mask.sum(dim=1)  # [B]
            keep = (real_len > 1).nonzero(as_tuple=True)[0].tolist()
            if not keep:
                continue
            idx = torch.tensor(keep, device=device)
            sub = batch.to(device, non_blocking=True).long().index_select(0, idx)
            tags_sub = [tags[j] for j in keep]
            l_ids = lyric_ids.to(device, non_blocking=True).index_select(0, idx)
            l_mask = lyric_mask.to(device, non_blocking=True).index_select(0, idx)
            inputs, targets = build_train_inputs(sub, pad_id)
            n = len(keep)
            for r in regimes:
                sums[r] += regime_loss(inputs, targets, tags_sub, l_ids, l_mask, r) * n
            n_scored += n

    if n_scored == 0:
        raise SystemExit("no lyric-bearing val samples found — check lyrics/ dir")

    means = {r: sums[r] / n_scored for r in regimes}
    print(f"\nscored {n_scored} lyric-bearing samples "
          f"({n_scored / max(n_seen,1):.1%} of {n_seen} seen)")
    for r in regimes:
        print(f"  {r:10} loss {means[r]:.4f}")

    lyric_gain = means["tags_only"] - means["full"]
    tag_gain = means["uncond"] - means["tags_only"]
    print()
    print(f"lyric gain  (tags_only - full)   = {lyric_gain:+.4f}")
    print(f"tag gain    (uncond - tags_only) = {tag_gain:+.4f}")
    print()
    if lyric_gain >= 0.02:
        print("→ lyrics measurably lower loss. Keep lyric conditioning; "
              "Phase 4 speed work is worth it.")
    elif lyric_gain <= 0.005:
        print("→ lyrics barely move loss. The model effectively ignores them — "
              "consider dropping the Demucs+Whisper pass and redirecting compute.")
    else:
        print("→ marginal. Weigh the ~$2k/run transcription cost against the small gain.")


@app.local_entrypoint()
def main(n_batches: int = 400, batch_size: int = 64):
    eval_ablation.remote(n_batches=n_batches, batch_size=batch_size)
