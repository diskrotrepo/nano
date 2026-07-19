"""Modal entrypoint: per-song key estimation over the packed chroma sidecars.

Sweeps /tokens/packed/packed_NNN.mel.bin (written by modal_pack_cache.py with
melody enabled) and writes /tokens/keys.json — the per-song ``<key_*>`` header
marker source the dataset loads at train time. Pure CPU + mmap, one container.

Runs any time after pack; a partial/absent keys.json just means those songs get
``<unknown_key>`` at train time, so it can also re-run incrementally after a
corpus grow + repack (resume-by-skip keeps prior estimates).

Run::

    modal run --detach diskrot/modal_key_detect.py

Monitor::

    modal app logs nano-key-detect
"""
from __future__ import annotations

import modal

app = modal.App("nano-key-detect")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",  # model.lyric_encoder import chain (labels only, no GPU)
        "numpy>=1.26",
    )
    .add_local_python_source("diskrot", "model")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(
    image=image,
    cpu=4.0,
    memory=8 * 1024,
    # The sweep streams every .mel.bin once through Modal FUSE (~hundreds of GB
    # at full corpus scale); generous timeout, and detect_keys resumes by skip.
    timeout=60 * 60 * 6,
    retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=5.0),
    volumes={"/tokens": tokens_vol},
)
def detect_remote(data_subdir: str = ""):
    from diskrot.key_detect import detect_keys

    # Keys are derived from the PACKED CHROMA sidecar, so they are frame-rate
    # specific and must be regenerated per codec. detect_keys derives both
    # <root>/packed and <root>/keys.json from this one arg.
    root = f"/tokens/{data_subdir}" if data_subdir else "/tokens"
    out = detect_keys(root, verbose=True, commit_cb=tokens_vol.commit)
    tokens_vol.commit()  # idempotent safety net
    print(f"[done] keys written to {out}", flush=True)


@app.local_entrypoint()
def main(data_subdir: str = ""):
    fc = detect_remote.spawn(data_subdir=data_subdir)
    print(f"key detection launched (detached) -- function call id: {fc.object_id}")
    print("monitor with: modal app logs nano-key-detect")
