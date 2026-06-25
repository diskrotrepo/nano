"""Modal entrypoint: reclaim a wave's per-song intermediate inodes after packing.

Per-wave ingestion (the v9 unlimited-scale pipeline) writes one ``.pt`` per song
to ``/tokens/waves/wave_<id>`` and one ``.mel.npy`` to
``/melody/waves/wave_<id>``. Once ``pack_append`` has folded that wave into the
packed shards (the only artifact training reads), those loose per-song files are
dead weight — deleting them keeps the live loose-file count bounded to ~one wave
so the corpus can grow past the ~500k-inode volume cap.

Reuses the ``Path.unlink()`` + ``vol.commit()`` reclaim pattern from
``modal_clean_corrupt.py`` / ``modal_prepare.py``. Idempotent: unlink of an
already-gone file is a no-op, so a re-run after an interrupted cleanup finishes
the job. Dry-run by default; pass ``--apply`` to delete.

Run::

    modal run diskrot/modal_wave_cleanup.py --wave-id 17            # dry-run report
    modal run diskrot/modal_wave_cleanup.py --wave-id 17 --apply    # delete .pt+.mel.npy
    modal run diskrot/modal_wave_cleanup.py --wave-id 17 --apply --drop-mp3   # + raw mp3s
"""
from __future__ import annotations

import modal

from diskrot.modal_common import corpus_mount

app = modal.App("nano-wave-cleanup")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy>=1.26")
    .add_local_python_source("diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)
melody_vol = modal.Volume.from_name("nano-melody", create_if_missing=True)
stems_vol = modal.Volume.from_name("nano-stems", create_if_missing=True)
corpus_vol = corpus_mount(read_only=False)  # R2 bucket; only mounted for --drop-mp3


@app.function(
    image=image,
    timeout=60 * 60,
    volumes={"/tokens": tokens_vol, "/melody": melody_vol, "/stems": stems_vol,
             "/corpus": corpus_vol},
)
def cleanup_remote(wave_id: str, apply: bool = False, drop_mp3: bool = False) -> int:
    """Delete wave ``wave_id``'s ``.pt`` (+ ``.mel.npy`` + ``.stems.npy``, optionally
    raw ``.mp3``) and remove the now-empty wave dirs. Returns files deleted."""
    from pathlib import Path

    from diskrot.pack_cache import SHARD_INDEX_NAME

    sub = f"waves/wave_{wave_id}"
    pt_dir = Path("/tokens") / sub
    mel_dir = Path("/melody") / sub
    stem_dir = Path("/stems") / sub
    mp3_dir = Path("/corpus") / sub

    # Safety: never discard intermediates that were never folded into shards.
    # The orchestrator already sequences cleanup AFTER pack_append, but a stray
    # manual invocation must fail loudly rather than lose tokens.
    packed_index = Path("/tokens/packed") / SHARD_INDEX_NAME
    if not packed_index.exists():
        raise SystemExit(
            f"refusing cleanup: no packed index at {packed_index} — pack_append "
            f"wave {wave_id} before reclaiming its intermediates"
        )

    pts = list(pt_dir.glob("*.pt"))
    mels = list(mel_dir.glob("*.mel.npy"))
    stems = list(stem_dir.glob("*.stems.npy"))
    mp3s = list(mp3_dir.glob("*.mp3")) if drop_mp3 else []
    print(f"[cleanup wave {wave_id}] {len(pts)} .pt, {len(mels)} .mel.npy, "
          f"{len(stems)} .stems.npy"
          + (f", {len(mp3s)} .mp3" if drop_mp3 else "")
          + ("" if apply else "  — DRY RUN (pass --apply to delete)"), flush=True)
    if not apply:
        return 0

    deleted = 0
    for p in pts + mels + stems + mp3s:
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            pass  # idempotent: already gone
    # Best-effort: drop the now-empty wave dirs so they don't linger as inodes.
    dirs = [pt_dir, mel_dir, stem_dir] + ([mp3_dir] if drop_mp3 else [])
    for d in dirs:
        try:
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass
    tokens_vol.commit()
    melody_vol.commit()
    stems_vol.commit()
    # corpus is an R2 CloudBucketMount: --drop-mp3 unlinks flush through the mount, no commit().
    print(f"[cleanup wave {wave_id}] deleted {deleted} files — inodes reclaimed",
          flush=True)
    return deleted


@app.local_entrypoint()
def main(wave_id: str = "", apply: bool = False, drop_mp3: bool = False):
    if not wave_id:
        raise SystemExit("--wave-id is required (e.g. --wave-id 17)")
    cleanup_remote.remote(wave_id=wave_id, apply=apply, drop_mp3=drop_mp3)
