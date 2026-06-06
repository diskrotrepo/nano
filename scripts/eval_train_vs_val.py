"""Diagnostic: compare best.pt loss on train vs val.

If train_loss << val_loss the model is memorizing — data-bound, add more songs.
If train_loss ≈ val_loss the model is at capacity — bigger model.

Run:  modal run scripts/eval_train_vs_val.py
"""
from __future__ import annotations

import modal

app = modal.App("nano-eval-splits")

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
    timeout=60 * 30,
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol},
)
def eval_splits(n_batches: int = 100, batch_size: int = 64, seed: int = 42, val_ratio: float = 0.12):
    import torch
    from torch.utils.data import DataLoader

    from model.codec import DACodec
    from model.nano_audio_gpt import GPTConfig, NanoAudioGPT
    from model.text_encoder import CLAPTextEncoder
    from diskrot.dataset import TokenDataset, collate_lyrics
    from diskrot.train import TrainConfig, _evaluate

    device = "cuda"
    ckpt = torch.load("/ckpts/best.pt", map_location=device, weights_only=False)
    model_cfg = GPTConfig(**ckpt["cfg"])
    model = NanoAudioGPT(model_cfg).to(device)
    state_dict = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(state_dict)
    print(f"loaded best.pt — step={ckpt['step']} best_val_loss={ckpt['best_val_loss']:.4f}")

    segment_frames = int(10.0 * DACodec.FRAME_RATE_HZ)

    text_encoder: CLAPTextEncoder | None = None
    tag_cache: dict[str, torch.Tensor] = {}
    if model_cfg.use_text_conditioning:
        text_encoder = CLAPTextEncoder(d_out=model_cfg.d_model, device=device).to(device).eval()
        if "text_proj" in ckpt:
            text_encoder.proj.load_state_dict(ckpt["text_proj"])
            print("loaded text_proj from checkpoint")
        text_encoder._ensure_clap()

    tcfg = TrainConfig(device=device, model=model_cfg)

    results: dict[str, tuple[float, list[float]]] = {}
    for split in ("train", "val"):
        ds = TokenDataset(
            "/tokens", segment_frames=segment_frames, split=split,
            val_ratio=val_ratio, seed=seed,
            tags_path="/tokens/tags.json", lyrics_path="/tokens/lyrics",
        )
        if text_encoder is not None:
            new_tags = sorted(set(ds._tags.values()) - set(tag_cache))
            if new_tags:
                with torch.no_grad():
                    for tag in new_tags:
                        emb = text_encoder._clap.get_text_embeddings([tag])
                        tag_cache[tag] = emb.squeeze(0).to(device)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
            collate_fn=collate_lyrics,
        )
        mean, per_cb = _evaluate(model, loader, tcfg, n_batches,
                                 text_encoder=text_encoder, tag_cache=tag_cache)
        results[split] = (mean, per_cb)
        cb_str = " ".join(f"{x:.2f}" for x in per_cb)
        print(f"[{split:5}] loss {mean:.4f}  cb[{cb_str}]")

    tr, vl = results["train"][0], results["val"][0]
    gap = vl - tr
    print()
    print(f"gap (val - train) = {gap:+.4f}")
    if gap > 0.15:
        print("→ memorizing: train ≪ val. Likely data-bound — add more songs.")
    elif gap < 0.05:
        print("→ at capacity: train ≈ val. Likely capacity-bound — bigger model.")
    else:
        print("→ mixed signal. Either lever (more data or more params) should help.")


@app.local_entrypoint()
def main(n_batches: int = 100, batch_size: int = 64):
    eval_splits.remote(n_batches=n_batches, batch_size=batch_size)
