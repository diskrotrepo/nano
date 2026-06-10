"""Modal entrypoint: pre-phonemize the lyrics corpus -> /tokens/phonemes shards.

Runs g2p once per song offline (see diskrot/phonemize.py for why: OOV-heavy
Whisper transcripts make lazy in-DataLoader g2p a real 8xH100 bottleneck) and
writes the sharded phonemes/ dir the dataset reads as a pure lookup. CPU-only,
one container, process-parallel across shard buckets.

Runs any time after transcribe, before train; a partial/absent pass is safe
(the dataset falls back to live g2p per song). Re-run after re-transcribing —
songs whose word count changed are detected stale and redone.

Run::

    modal run --detach diskrot/modal_phonemize.py

Monitor::

    modal app logs nano-phonemize
"""
from __future__ import annotations

import modal

app = modal.App("nano-phonemize")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",  # model.lyric_encoder import chain
        "numpy>=1.26",
        "g2p_en>=2.1",
        "nltk>=3.8",
        "tqdm>=4.66",  # diskrot.transcribe_lyrics import chain (shard helpers)
    )
    .run_commands(
        # g2p_en needs these nltk corpora; bake them into the image so the
        # worker processes don't race to download them at runtime (a 16-way
        # concurrent nltk.download() corrupts the zip -> BadZipFile). Both
        # tagger names are required: g2p_en's import guard probes the OLD
        # 'averaged_perceptron_tagger', while nltk>=3.9's pos_tag loads the
        # '_eng' variant at call time.
        "python -c \"import nltk; nltk.download('averaged_perceptron_tagger'); "
        "nltk.download('averaged_perceptron_tagger_eng'); "
        "nltk.download('cmudict')\"",
    )
    .add_local_python_source("diskrot", "model")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(
    image=image,
    cpu=16.0,
    memory=16 * 1024,
    # ~322k songs at ~20-200ms each across 16 worker processes ≈ 1-2 h.
    timeout=60 * 60 * 8,
    retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=5.0),
    volumes={"/tokens": tokens_vol},
)
def phonemize_remote(n_workers: int = 16):
    from diskrot.phonemize import phonemize_corpus

    out = phonemize_corpus(
        "/tokens/lyrics", "/tokens/phonemes",
        n_workers=n_workers, verbose=True, commit_cb=tokens_vol.commit,
    )
    tokens_vol.commit()  # idempotent safety net
    print(f"[done] phoneme shards live at {out}", flush=True)


@app.local_entrypoint()
def main(n_workers: int = 16):
    fc = phonemize_remote.spawn(n_workers=n_workers)
    print(f"phonemize launched (detached) -- function call id: {fc.object_id}")
    print("monitor with: modal app logs nano-phonemize")
