"""Depth-limited decode probe: how much of a v9 generation's perceived noise
comes from the still-undertrained DEEP RVQ codebooks?

Generates ONE clip (real conditioning path: sweetened prompt + CFG + per-cb
ladder), then decodes the SAME token matrix at several RVQ depths (SpectroStream
decode accepts any prefix K <= stored depth). A real packed training crop is
decoded at the same depths as the control. If shallow decodes of the generation
sound clearly more musical than the full-depth one, the "random noise" character
is mostly deep-level fizz (training maturity), not coarse-level chaos.

WAVs -> nano-output:/depth_probe/:

    NANO_CODEC=spectrostream modal run scripts/depth_decode_probe.py
    modal volume get nano-output /depth_probe ./depth_probe --force
"""
from __future__ import annotations

import modal

from diskrot.modal_serve import _ss_image

app = modal.App("nano-depth-probe")

tokens_vol = modal.Volume.from_name("nano-tokens")
ckpts_vol = modal.Volume.from_name("nano-ckpts")
out_vol = modal.Volume.from_name("nano-output", create_if_missing=True)

PROMPT = ("A driving techno track with a steady four-on-the-floor kick, hypnotic "
          "synth stabs, deep bass and tight hi-hats. Dark, energetic, instrumental.")


@app.function(
    image=_ss_image,
    gpu="H100",
    timeout=1800,
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol, "/outputs": out_vol},
)
def probe(ckpt: str = "/ckpts/v9_stereo/best_inference.pt", seconds: float = 8.0,
          depths: str = "6,12,18,24", cfg_scale: float = 5.0, seed: int = 0) -> list[str]:
    import os
    import random
    import wave

    import numpy as np
    import torch

    os.environ.setdefault("NANO_DEVICE", "cuda")
    os.environ["NANO_CKPT"] = ckpt
    from server.inference import InferenceEngine
    from diskrot.dataset import TokenDataset, load_mmap_bundle

    eng = InferenceEngine(ckpt_path=ckpt)
    model, codec = eng.model, eng.codec
    K = model.cfg.n_codebooks
    frames = int(seconds * codec.FRAME_RATE_HZ)
    dlist = [int(d) for d in depths.split(",") if 0 < int(d) <= K]

    # COLD_LADDER resampled to K (same shape the sweep uses).
    base_t = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
    base_k = [120, 90, 70, 50, 36, 26, 18, 12, 8]
    def rs(vals):
        n = len(vals)
        return [vals[int(i*(n-1)/(K-1))] + (vals[min(int(i*(n-1)/(K-1))+1, n-1)] - vals[int(i*(n-1)/(K-1))]) * (i*(n-1)/(K-1) - int(i*(n-1)/(K-1))) for i in range(K)]
    temps = rs(base_t)
    topks = [max(1, int(round(v))) for v in rs(base_k)]

    torch.manual_seed(seed)
    text = eng.sweeten_prompt(PROMPT)
    cond_emb, cond_lids, cond_lmask = eng._build_conditioning(text)
    neg_emb, neg_lids, neg_lmask = eng._build_conditioning("")
    with torch.no_grad():
        tokens = model.generate(
            None, num_new_frames=frames,
            temperature=temps, top_k=topks, top_p=0.95,
            text_emb=cond_emb, text_emb_neg=neg_emb,
            lyric_ids=cond_lids, lyric_mask=cond_lmask,
            lyric_ids_neg=neg_lids, lyric_mask_neg=neg_lmask,
            cfg_scale=cfg_scale,
        )
    if tokens.dim() == 3:
        tokens = tokens[0]
    print(f"generated tokens {tuple(tokens.shape)}", flush=True)

    # Control: one real training crop, same depths.
    bundle = load_mmap_bundle("/tokens/packed", segment_frames=frames, val_ratio=0.12, seed=42)
    ds = TokenDataset.from_mmap(bundle, "train", frames, 1024, n_codebooks=K)
    rng = random.Random(seed)
    ridx = next(i for i in rng.sample(range(len(ds.names)), 50) if ds._get(i).shape[1] > frames + 2)
    rt = ds._get(ridx)
    rstart = rng.randint(0, rt.shape[1] - frames)
    real = torch.from_numpy(np.ascontiguousarray(rt[:K, rstart:rstart + frames])).long()

    os.makedirs("/outputs/depth_probe", exist_ok=True)
    written = []
    for label, tok in (("gen", tokens.cpu()), ("real", real)):
        for d in dlist:
            wav = codec.decode(tok[:d])
            if wav.dim() == 3:
                wav = wav[0]
            pcm = (wav.clamp(-1, 1).cpu().numpy().T * 32767).astype(np.int16)
            path = f"/outputs/depth_probe/{label}_depth{d:02d}.wav"
            with wave.open(path, "wb") as w:
                w.setnchannels(pcm.shape[1])
                w.setsampwidth(2)
                w.setframerate(codec.SAMPLE_RATE)
                w.writeframes(pcm.tobytes())
            written.append(path)
            print(f"decoded {label} at depth {d} -> {path}", flush=True)
    out_vol.commit()
    return written


@app.local_entrypoint()
def main(seconds: float = 8.0, depths: str = "6,12,18,24", cfg_scale: float = 5.0):
    for p in probe.remote(seconds=seconds, depths=depths, cfg_scale=cfg_scale):
        print(" ", p)
