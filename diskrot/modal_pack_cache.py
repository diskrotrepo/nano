"""Modal entrypoint: run the v2 sharded packer over the nano-tokens volume.

Produces /tokens/packed/packed_NNN.bin + packed_NNN.json sidecars + a
packed_index.json. The training side (both single-GPU TokenDataset.__init__
and the DDP train_remote_multi parent process) auto-detects the v2 layout
and uses the mmap-backed fast path. Without this pack step, training spends
~70 minutes reading 33k+ .pt files off the volume on every launch.

Run::

    modal run --detach diskrot/modal_pack_cache.py
    modal run --detach diskrot/modal_pack_cache.py --shard-target-songs 5000

Monitor::

    modal app logs nano-pack
"""
from __future__ import annotations

import modal

app = modal.App("nano-pack")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "numpy>=1.26",
    )
    .add_local_python_source("diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
# Per-song chroma sidecars (<name>.mel.npy) live on their own volume — see
# modal_melody.py for why. Read-only here; only the packed .mel.bin output lands
# back on nano-tokens alongside the token shards.
melody_vol = modal.Volume.from_name("nano-melody", create_if_missing=True)


@app.function(
    image=image,
    cpu=8.0,
    # ~5 GB per shard worth of int16 tensors held in RAM during the per-shard
    # parallel load, plus Python overhead. 16 GB is generous headroom.
    memory=16 * 1024,
    # Packs at ~7 songs/sec via Modal FUSE, so a large corpus can take many
    # hours; the timeout is set to 18 hours to cover a full-scale pack on a
    # single container.
    timeout=60 * 60 * 18,
    # Single container does the whole 12-18h pack, so pin it to a
    # non-preemptible instance — it never restarts mid-run. pack() is also
    # resumable (skips already-valid shards, commits each as it lands via
    # commit_cb below), so even a non-preemption failure + retry resumes from
    # the last committed shard rather than rebuilding from scratch.
    nonpreemptible=True,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
    volumes={"/tokens": tokens_vol, "/melody": melody_vol},
)
def pack_remote(shard_target_songs: int = 5_000, n_workers: int = 16, melody: bool = True):
    from pathlib import Path

    from diskrot.pack_cache import pack

    # Pack the parallel chroma sidecar (packed_NNN.mel.bin) when the per-song
    # <name>.mel.npy files from modal_melody.py are present. Auto-skip if none
    # exist yet (a pre-melody pack) unless explicitly disabled — songs without a
    # chroma file are zero-filled, so a partial extraction is safe.
    mel_cache_dir = None
    if melody:
        has_any = next(Path("/melody").glob("*.mel.npy"), None) is not None
        if has_any:
            mel_cache_dir = "/melody"
            print("[pack] melody: chroma sidecars found — packing parallel .mel.bin", flush=True)
        else:
            print("[pack] melody: no *.mel.npy found on nano-melody — packing tokens "
                  "only (run modal_melody.py first to add melody conditioning)", flush=True)

    out_dir = pack(
        "/tokens",
        shard_target_songs=shard_target_songs,
        n_workers=n_workers,
        verbose=True,
        # Persist each completed shard so a preemption+retry skips it. pack()
        # stays modal-free; the volume commit is injected here.
        commit_cb=tokens_vol.commit,
        mel_cache_dir=mel_cache_dir,
    )
    tokens_vol.commit()  # idempotent safety net if commit_cb was a no-op
    print(f"[done] packed shards live at {out_dir}", flush=True)


@app.local_entrypoint()
def main(shard_target_songs: int = 5_000, n_workers: int = 16, melody: bool = True):
    fc = pack_remote.spawn(
        shard_target_songs=shard_target_songs,
        n_workers=n_workers,
        melody=melody,
    )
    print(f"pack launched (detached) -- function call id: {fc.object_id}")
    print("monitor with: modal app logs nano-pack")
