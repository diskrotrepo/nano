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
# Per-song stem-token sidecars (<name>.stems.npy) live on their own volume too —
# see modal_stems.py. Read-only here; the packed .stem.bin lands on nano-tokens.
stems_vol = modal.Volume.from_name("nano-stems", create_if_missing=True)


@app.function(
    image=image,
    # The 16-thread shard load is FUSE-latency-bound and GIL-serialized, not
    # CPU-bound — 2 reserved cores carry it; bursts above bill actual usage.
    cpu=2.0,
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
    volumes={"/tokens": tokens_vol, "/melody": melody_vol, "/stems": stems_vol},
)
def pack_remote(
    shard_target_songs: int = 5_000, n_workers: int = 16, melody: bool = True,
    stems: bool = True,
):
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

    # Same for the parallel stem sidecar (packed_NNN.stem.bin) from modal_stems.py.
    stem_cache_dir = None
    if stems:
        has_any = next(Path("/stems").glob("*.stems.npy"), None) is not None
        if has_any:
            stem_cache_dir = "/stems"
            print("[pack] stems: stem sidecars found — packing parallel .stem.bin", flush=True)
        else:
            print("[pack] stems: no *.stems.npy found on nano-stems — packing without "
                  "stems (run modal_stems.py first to add /addstem conditioning)", flush=True)

    out_dir = pack(
        "/tokens",
        shard_target_songs=shard_target_songs,
        n_workers=n_workers,
        verbose=True,
        # Persist each completed shard so a preemption+retry skips it. pack()
        # stays modal-free; the volume commit is injected here.
        commit_cb=tokens_vol.commit,
        mel_cache_dir=mel_cache_dir,
        stem_cache_dir=stem_cache_dir,
    )
    tokens_vol.commit()  # idempotent safety net if commit_cb was a no-op
    print(f"[done] packed shards live at {out_dir}", flush=True)


@app.function(
    image=image,
    # Same sizing rationale as pack_remote: FUSE-latency-bound threaded load.
    cpu=2.0,
    memory=16 * 1024,
    # A wave is ~100k songs -> ~20 shards, so this is far quicker than a full
    # pack, but keep the generous ceiling for safety.
    timeout=60 * 60 * 18,
    nonpreemptible=True,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
    volumes={"/tokens": tokens_vol, "/melody": melody_vol, "/stems": stems_vol},
)
def pack_append_remote(
    wave_id: str, shard_target_songs: int = 5_000, n_workers: int = 16,
    melody: bool = True, stems: bool = True,
):
    """Append one wave's tokens (``/tokens/waves/wave_<id>/*.pt``) as NEW shards
    to the existing ``/tokens/packed`` — leaving every prior shard untouched (see
    pack_cache.pack_append). Melody must match the existing pack's has_melody;
    that's honored here and enforced (fail-fast) inside pack_append."""
    from pathlib import Path

    from diskrot.pack_cache import SHARD_INDEX_NAME, load_shard_index, pack_append

    wave_dir = f"/tokens/waves/wave_{wave_id}"
    if next(Path(wave_dir).glob("*.pt"), None) is None:
        raise SystemExit(f"no .pt files in {wave_dir} — tokenize this wave first")

    # Match the existing pack's melody flag when a pack already exists; otherwise
    # (first wave) fall back to whether this wave actually has chroma.
    packed_dir = Path("/tokens/packed")
    existing_has_melody = None
    if (packed_dir / SHARD_INDEX_NAME).exists():
        existing_has_melody = bool(load_shard_index(packed_dir).get("has_melody", False))
    wave_mel_dir = f"/melody/waves/wave_{wave_id}"
    wave_has_mel = next(Path(wave_mel_dir).glob("*.mel.npy"), None) is not None
    want_melody = (
        existing_has_melody if existing_has_melody is not None
        else (melody and wave_has_mel)
    )
    mel_cache_dir = wave_mel_dir if want_melody else None
    if want_melody:
        print(f"[pack-append] melody on — chroma from {wave_mel_dir} "
              f"(missing songs zero-filled)", flush=True)

    # Same gate for stems: match the existing pack's has_stems when a pack exists,
    # else fall back to whether this wave actually has stem sidecars.
    existing_has_stems = None
    if (packed_dir / SHARD_INDEX_NAME).exists():
        existing_has_stems = bool(load_shard_index(packed_dir).get("has_stems", False))
    wave_stem_dir = f"/stems/waves/wave_{wave_id}"
    wave_has_stems = next(Path(wave_stem_dir).glob("*.stems.npy"), None) is not None
    want_stems = (
        existing_has_stems if existing_has_stems is not None
        else (stems and wave_has_stems)
    )
    stem_cache_dir = wave_stem_dir if want_stems else None
    if want_stems:
        print(f"[pack-append] stems on — stem tokens from {wave_stem_dir} "
              f"(missing songs zero-filled + flagged absent)", flush=True)

    out_dir = pack_append(
        wave_dir,
        "/tokens/packed",
        shard_target_songs=shard_target_songs,
        n_workers=n_workers,
        verbose=True,
        commit_cb=tokens_vol.commit,
        mel_cache_dir=mel_cache_dir,
        stem_cache_dir=stem_cache_dir,
    )
    tokens_vol.commit()  # idempotent safety net if commit_cb was a no-op
    print(f"[done] appended wave {wave_id} -> {out_dir}", flush=True)


@app.local_entrypoint()
def main(
    shard_target_songs: int = 5_000, n_workers: int = 16, melody: bool = True,
    stems: bool = True, append: bool = False, wave_id: str = "",
):
    if append:
        if not wave_id:
            raise SystemExit("--append requires --wave-id (e.g. --wave-id 17)")
        fc = pack_append_remote.spawn(
            wave_id=wave_id, shard_target_songs=shard_target_songs,
            n_workers=n_workers, melody=melody, stems=stems,
        )
        print(f"pack-append launched (detached) for wave {wave_id} "
              f"-- function call id: {fc.object_id}")
        print("monitor with: modal app logs nano-pack")
        return
    fc = pack_remote.spawn(
        shard_target_songs=shard_target_songs,
        n_workers=n_workers,
        melody=melody,
        stems=stems,
    )
    print(f"pack launched (detached) -- function call id: {fc.object_id}")
    print("monitor with: modal app logs nano-pack")
