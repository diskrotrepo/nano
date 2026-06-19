"""Single-command wave ingestion orchestrator.

Runs ONE wave end-to-end by calling each deployed stage's workhorse function in
sequence:

    prepare -> tokenize -> melody -> auto_tag -> transcribe -> structure
            -> pack_append -> cleanup -> phonemize -> key_detect

Each call blocks until that stage's full pass returns, so only one stage runs at
a time — which means at most one GPU fan-out is ever active and a 50-GPU account
cap is respected automatically (no extra throttling needed).

PREREQUISITE — deploy the stage apps once so they can be looked up by name
(``modal run`` apps are ephemeral and can't be looked up; ``modal deploy``
registers them). For the object-storage source, set NANO_CORPUS_SOURCE=bucket
(+ NANO_AUDIO_*) in the shell you DEPLOY from — the mount is fixed at deploy time::

    for m in prepare tokenize melody auto_tag transcribe structure \
             pack_cache wave_cleanup phonemize key_detect; do
        modal deploy diskrot/modal_$m.py
    done

Then per wave (resumable: a re-run skips stages already marked done in
/tokens/waves/wave_<id>/status.json)::

    modal run --detach diskrot/modal_ingest_wave.py --wave-id 17

Toggle optional conditioning streams with --no-with-melody / --no-with-tags /
--no-with-lyrics / --no-with-structure. Volume-source users (no bucket) pass
--drop-mp3 so cleanup also reclaims the wave's raw mp3 inodes.
"""
from __future__ import annotations

import json

import modal

app = modal.App("nano-ingest-wave")

image = (
    modal.Image.debian_slim(python_version="3.12").add_local_python_source("diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=24 * 60 * 60)
def ingest_wave(
    wave_id: str,
    min_seconds: int = 20,
    tokenize_batch: int = 8,
    with_melody: bool = True,
    with_tags: bool = True,
    with_lyrics: bool = True,
    with_structure: bool = True,
    drop_mp3: bool = False,
):
    import os
    from pathlib import Path

    status_dir = Path("/tokens/waves") / f"wave_{wave_id}"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "status.json"

    def load_status() -> dict:
        tokens_vol.reload()
        if status_path.exists():
            try:
                return json.loads(status_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    status = load_status()

    def _mark(stage: str) -> None:
        status[stage] = "done"
        tmp = status_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, status_path)  # atomic: never a torn status file
        tokens_vol.commit()

    def run(stage: str, app_name: str, fn_name: str, **kwargs) -> None:
        if status.get(stage) == "done":
            print(f"[wave {wave_id}] SKIP {stage} (already done)", flush=True)
            return
        print(f"[wave {wave_id}] >>> {stage}  ({app_name}.{fn_name})", flush=True)
        # from_name resolves a DEPLOYED function; .remote() blocks until its full
        # pass returns, which is what sequences the stages (and the GPU usage).
        fn = modal.Function.from_name(app_name, fn_name)
        fn.remote(**kwargs)
        _mark(stage)
        print(f"[wave {wave_id}] <<< {stage} done", flush=True)

    # CPU/GPU stages serialized by the blocking .remote() calls above.
    run("prepare", "nano-prepare", "run_prepare", apply=True, wave_id=wave_id)
    run("tokenize", "nano-tokenize", "run_tokenize",
        min_seconds=min_seconds, batch_size=tokenize_batch, wave_id=wave_id)
    if with_melody:
        run("melody", "nano-melody", "orchestrate", wave_id=wave_id)
    if with_tags:
        run("auto_tag", "nano-auto-tag", "run_auto_tag",
            batch_size=16, flush_every_batches=4, wave_id=wave_id)
    if with_lyrics:
        run("transcribe", "nano-transcribe", "orchestrate", wave_id=wave_id)
    if with_structure:
        run("structure", "nano-structure", "orchestrate", wave_id=wave_id)
    run("pack", "nano-pack", "pack_append_remote", wave_id=wave_id)
    # Cleanup AFTER pack (needs the wave's .pt/.mel.npy). phonemize/key_detect
    # read the lyrics shards / packed chroma, so they're safe to run after.
    run("cleanup", "nano-wave-cleanup", "cleanup_remote",
        wave_id=wave_id, apply=True, drop_mp3=drop_mp3)
    if with_lyrics:
        run("phonemize", "nano-phonemize", "phonemize_remote")
    run("key_detect", "nano-key-detect", "detect_remote")
    print(f"[wave {wave_id}] ALL STAGES COMPLETE", flush=True)


@app.local_entrypoint()
def main(
    wave_id: str = "",
    min_seconds: int = 20,
    tokenize_batch: int = 8,
    with_melody: bool = True,
    with_tags: bool = True,
    with_lyrics: bool = True,
    with_structure: bool = True,
    drop_mp3: bool = False,
):
    if not wave_id:
        raise SystemExit("--wave-id is required (e.g. --wave-id 17)")
    fc = ingest_wave.spawn(
        wave_id=wave_id, min_seconds=min_seconds, tokenize_batch=tokenize_batch,
        with_melody=with_melody, with_tags=with_tags, with_lyrics=with_lyrics,
        with_structure=with_structure, drop_mp3=drop_mp3,
    )
    print(f"wave {wave_id} ingest launched (detached) — function call id: {fc.object_id}")
    print("monitor: modal app logs nano-ingest-wave")
