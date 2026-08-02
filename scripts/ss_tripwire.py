"""SpectroStream corpus tripwire.

The SS packed corpus at ``/tokens/packed`` is the output of the entire v9 campaign
and exists in exactly ONE place (nano-backup holds conditioning metadata only, no
packed shards). The DAC re-tokenize runs alongside it under ``/tokens/dac`` and is
designed never to resolve ``/tokens/packed`` at all — this script proves that held.

Run it before the DAC work starts (to record a baseline) and after every wave.
It is read-only: it never writes to nano-tokens.

    modal run scripts/ss_tripwire.py            # check against the recorded baseline
    modal run scripts/ss_tripwire.py --record   # print a fresh baseline block

A mismatch means something in the DAC pass wrote into the SS pack. Stop the ingest
loop immediately — the damage is unrecoverable without a backup.
"""
from __future__ import annotations

import modal

app = modal.App("nano-ss-tripwire")

image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("diskrot")
tokens_vol = modal.Volume.from_name("nano-tokens")

# Baseline recorded 2026-07-18, immediately before the DAC re-tokenize began.
BASELINE = {
    "sha256": "b64bc1ee529e40e40f7c5ea0419ccc7815662c3de9323eeb81d553bde55ab46a",
    "bytes": 16288,
    "n_codebooks": 32,
    "n_shards": 180,
    "n_songs_total": 867449,
    "has_melody": True,
    "has_stems": False,
}


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=600)
def check(record: bool = False) -> dict:
    import hashlib
    import json
    from pathlib import Path

    idx = Path("/tokens/packed/packed_index.json")
    if not idx.exists():
        raise RuntimeError(
            "TRIPWIRE: /tokens/packed/packed_index.json is MISSING. The SS pack "
            "should never be touched by the DAC pass. Stop the ingest loop."
        )
    raw = idx.read_bytes()
    d = json.loads(raw)
    cur = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "n_codebooks": d.get("n_codebooks"),
        "n_shards": d.get("n_shards"),
        "n_songs_total": d.get("n_songs_total"),
        "has_melody": d.get("has_melody"),
        "has_stems": d.get("has_stems"),
    }
    if record:
        print("BASELINE = " + json.dumps(cur, indent=4))
        return cur

    diffs = {k: (BASELINE[k], cur[k]) for k in BASELINE if BASELINE[k] != cur[k]}
    if diffs:
        lines = "\n".join(f"    {k}: expected {e!r}, found {f!r}" for k, (e, f) in diffs.items())
        raise RuntimeError(
            f"TRIPWIRE FAILED — the SS pack changed:\n{lines}\n"
            "The DAC pass must never write to /tokens/packed. Stop the ingest loop "
            "and investigate before running another wave."
        )
    print(f"TRIPWIRE OK — SS pack unchanged "
          f"({cur['n_songs_total']} songs, {cur['n_shards']} shards, "
          f"n_codebooks={cur['n_codebooks']})")
    return cur


@app.local_entrypoint()
def main(record: bool = False):
    check.remote(record=record)
