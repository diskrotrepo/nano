"""Modal: shard the legacy tags.json / keys.json on nano-tokens into hash-shard
dirs (tags/ and keys/) for scale.

The single-file stores are loaded whole at startup and rewritten in full on every
metadata flush — at multi-million songs that's hundreds of MB of JSON. This
explodes them into 256 hash-keyed shards (``diskrot.sharded_store``). The dataset
loaders read either layout, so this is a safe, backward-compatible one-time
migration — point training's ``--tags-path``/``--keys-path`` at the dir afterward.
Harmless (and unnecessary) below ~1M songs; the single files are fine there.

Run::

    modal run diskrot/modal_shard_stores.py              # both tags + keys
    modal run diskrot/modal_shard_stores.py --store tags
"""
from __future__ import annotations

import modal

app = modal.App("nano-shard-stores")

image = (
    modal.Image.debian_slim(python_version="3.12").add_local_python_source("diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(image=image, timeout=60 * 60, volumes={"/tokens": tokens_vol})
def shard_remote(store: str = "both") -> None:
    from pathlib import Path

    from diskrot.sharded_store import convert_file_to_shards

    jobs = []
    if store in ("both", "tags"):
        jobs.append(("/tokens/tags.json", "/tokens/tags", "tags"))
    if store in ("both", "keys"):
        jobs.append(("/tokens/keys.json", "/tokens/keys", "keys"))
    for src, out, prefix in jobs:
        if not Path(src).exists():
            print(f"skip {src} (not found)", flush=True)
            continue
        n = convert_file_to_shards(src, out, prefix)
        print(f"sharded {n:,} entries: {src} -> {out}/{prefix}_NNN.json", flush=True)
    tokens_vol.commit()


@app.local_entrypoint()
def main(store: str = "both"):
    shard_remote.remote(store=store)
