"""Decode random crops straight from the packed shards — data-sanity check.

Answers one question: is the audio the model actually TRAINS ON temporally
coherent? A tokenize/pack bug that scrambles frames would teach the model
"radio-station switching" as the true data distribution — teacher-forced val
loss would still fall and tag adherence would still work, so this failure mode
is invisible to every loss-based eval. Decoding a few crops with the exact
slice the dataset serves (``[:n_codebooks, start:start+frames]``,
diskrot/dataset.py ``__getitem__``) and listening is the only direct test.

Reuses the SpectroStream serving image (the codec stack is the deploy-proven
one). WAVs land on the nano-output volume under ``/data_check/``:

    NANO_CODEC=spectrostream modal run scripts/decode_packed_crops.py --n 6
    modal volume get nano-output /data_check ./data_check --force
"""
from __future__ import annotations

import modal

from diskrot.modal_serve import _ss_image

app = modal.App("nano-decode-crops")

tokens_vol = modal.Volume.from_name("nano-tokens")
out_vol = modal.Volume.from_name("nano-output", create_if_missing=True)


@app.function(
    image=_ss_image,
    gpu="L4",
    timeout=1800,
    volumes={"/tokens": tokens_vol, "/outputs": out_vol},
)
def decode_crops(n: int = 6, seconds: float = 20.0, seed: int = 7, n_codebooks: int = 24) -> list[str]:
    import os
    import random
    import wave

    import numpy as np
    import torch

    from model.codec import get_codec
    from diskrot.dataset import TokenDataset, load_mmap_bundle

    frames = int(seconds * 25)
    bundle = load_mmap_bundle("/tokens/packed", segment_frames=frames, val_ratio=0.12, seed=42)
    ds = TokenDataset.from_mmap(bundle, "train", frames, 1024, n_codebooks=n_codebooks)
    codec = get_codec()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds.names)), min(n * 3, len(ds.names)))

    os.makedirs("/outputs/data_check", exist_ok=True)
    written: list[str] = []
    for i in idxs:
        if len(written) >= n:
            break
        t = ds._get(i)  # [K_stored, T_full] int16 mmap view
        T = t.shape[1]
        if T < frames + 2:
            continue
        start = rng.randint(0, T - frames)
        crop = torch.from_numpy(
            np.ascontiguousarray(t[:n_codebooks, start:start + frames])
        ).long()
        wav = codec.decode(crop)
        if wav.dim() == 3:
            wav = wav[0]
        pcm = (wav.clamp(-1, 1).cpu().numpy().T * 32767).astype(np.int16)  # [samples, C]
        name = ds.names[i].replace("/", "_")[:60]
        path = f"/outputs/data_check/{name}__f{start}.wav"
        with wave.open(path, "wb") as w:
            w.setnchannels(pcm.shape[1])
            w.setsampwidth(2)
            w.setframerate(codec.SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        written.append(path)
        print(f"decoded {name} frames [{start}, {start + frames}) -> {path}", flush=True)
    out_vol.commit()
    return written


@app.local_entrypoint()
def main(n: int = 6, seconds: float = 20.0, seed: int = 7):
    paths = decode_crops.remote(n=n, seconds=seconds, seed=seed)
    print(f"{len(paths)} crops decoded to nano-output:/data_check/")
    for p in paths:
        print(" ", p)
