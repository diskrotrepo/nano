"""Single-command wave ingestion orchestrator.

Runs ONE wave end-to-end by calling each deployed stage's workhorse function in
sequence:

    prepare(+quality-gate) -> audio_dedup -> tokenize -> melody -> stems -> auto_tag
            -> transcribe -> filter_lyrics -> align_lyrics -> structure -> tempo
            -> pack_append -> cleanup -> phonemize -> key_detect

Each call blocks until that stage's full pass returns, so only one stage runs at
a time — which means at most one GPU fan-out is ever active and a 50-GPU account
cap is respected automatically (no extra throttling needed).

Data-quality stages (raise the training floor, all toggleable):
  - ``--with-quality-gate`` (default on): prepare also decodes each file and drops
    clipped/silent/dead/low-bitrate audio (cheap CPU, folds into prepare).
  - ``--with-dedup`` (default on): acoustic near-duplicate removal (chromaprint +
    SimHash) after prepare, before tokenize — catches re-uploads SHA-256 misses
    (cheap CPU). Deletes the non-keeper copies, like prepare.
  - ``--with-filter`` (default on): null Whisper-hallucinated lyric entries
    ("Thanks for watching" etc.) after transcribe, before align/phonemize — a cheap
    global shard sweep so junk trains as <instrumental>, not <vocals>+garbage.
  - ``--with-align`` (default ON; disable with ``--no-with-align``): forced-align
    lyric word timestamps after the filter (torchaudio MMS_FA; ``--align-use-demucs``
    for vocal-isolated, sharper but costlier). The one added GPU stage (L4) — it runs
    every wave by default; turn it off per wave if you need to save the GPU cost.

PREREQUISITE — deploy the stage apps once so they can be looked up by name
(``modal run`` apps are ephemeral and can't be looked up; ``modal deploy``
registers them). Set the R2 env (``NANO_AUDIO_BUCKET`` / ``NANO_AUDIO_ENDPOINT``,
+ the ``r2-creds`` secret) in the shell you DEPLOY from — the mount is fixed at
deploy time. RE-DEPLOY ``prepare`` after this change (its ``quality_gate`` param is
new), and deploy the two new apps (``audio_dedup``, ``align_lyrics``)::

    for m in prepare audio_dedup tokenize melody stems auto_tag transcribe filter_lyrics \
             align_lyrics structure tempo pack_cache wave_cleanup phonemize key_detect; do
        modal deploy diskrot/modal_$m.py
    done

Then per wave (resumable: a re-run skips stages already marked done in
/tokens/waves/wave_<id>/status.json)::

    modal run --detach diskrot/modal_ingest_wave.py --wave-id 17

Toggle optional conditioning streams with --no-with-melody / --no-with-tags /
--no-with-lyrics / --no-with-structure / --no-with-tempo. Volume-source users
(no bucket) pass --drop-mp3 so cleanup also reclaims the wave's raw mp3 inodes.

COST: --structure-sample-pct (default 50) runs the expensive allin1 structure
stage (~28% of wave cost) on only a deterministic fraction of songs; the rest
fall back to <no_section> at train time. The cheap dense tempo stage runs at
full coverage so <tempo_*> markers stay 100% (the dataset prefers tempo.json
over the structure bpm) — keep with_tempo on whenever structure is sampled.
COST: --stems-sample-pct (default 50) runs the expensive stems stage (Demucs +
4x codec encode, GPU) on only a deterministic fraction; non-sampled songs are
flagged absent by the packer's present mask and skipped as /addstem targets, so
coverage degrades gracefully (stem_prob=0.15 means only ~15% of train batches use
stems anyway). The stems salt is independent of structure's, so the two samples
don't overlap-correlate.
COST: transcribe is Demucs-free (Whisper on the raw mix; vocal gender comes from
the audio-LLM captioner), so the transcribe stage no longer pays htdemucs vocal
isolation — the dominant former transcribe cost.
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
    tokenize_batch: int = 64,
    with_melody: bool = True,
    with_stems: bool = False,
    with_tags: bool = True,
    with_lyrics: bool = True,
    with_structure: bool = True,
    with_tempo: bool = True,
    with_quality_gate: bool = True,
    with_dedup: bool = True,
    with_filter: bool = True,
    with_align: bool = True,
    align_use_demucs: bool = False,
    structure_sample_pct: int = 50,
    stems_sample_pct: int = 50,
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
    # prepare: validate/dedupe/length-drop (+ optional content quality gate). The
    # quality gate decodes each file (clip/silence/bitrate) and drops garbage.
    run("prepare", "nano-prepare", "run_prepare",
        apply=True, wave_id=wave_id, quality_gate=with_quality_gate)
    if with_dedup:
        # Acoustic near-dup removal BEFORE tokenize so we never tokenize a re-upload
        # SHA-256 missed. Deletes the non-keeper copies (apply=True), like prepare.
        # The fingerprint manifest is global, so cross-wave dups are caught too.
        run("audio_dedup", "nano-audio-dedup", "run_dedup", apply=True, wave_id=wave_id)
    run("tokenize", "nano-tokenize", "run_tokenize",
        min_seconds=min_seconds, batch_size=tokenize_batch, wave_id=wave_id)
    if with_melody:
        run("melody", "nano-melody", "orchestrate", wave_id=wave_id)
    if with_stems:
        # Demucs all 4 stems + codec-tokenize each -> nano-stems (the /addstem
        # conditioning). GPU stage; needs the .pt frame count, so after tokenize,
        # before pack (which folds the stem sidecar in). sample_pct<100 runs it on
        # only a deterministic fraction (cost lever) — non-sampled songs are flagged
        # absent by the packer's present mask and skipped as stem-add targets.
        run("stems", "nano-stems", "orchestrate",
            wave_id=wave_id, sample_pct=stems_sample_pct)
    if with_tags:
        # batch_size=256 (auto_tag's own default), NOT 16: each .map element is
        # run as ONE vLLM continuous batch, internally pipelined in
        # NANO_CAPTION_SUBBATCH(=128)-sized sub-batches so CPU decode overlaps GPU
        # generate. At 16 there's a single 16-song sub-batch and that overlap
        # never engages — the A100 idles during decode. 256 keeps it saturated.
        run("auto_tag", "nano-auto-tag", "run_auto_tag",
            batch_size=256, flush_every_batches=25, wave_id=wave_id)
    if with_lyrics:
        # Transcribe is Demucs-free: Whisper runs on the raw mix and vocal gender
        # comes from the audio-LLM captioner (auto_tag, above), so there is no
        # separation step. Resume-by-skip on carried-over waves.
        run("transcribe", "nano-transcribe", "orchestrate", wave_id=wave_id)
        if with_filter:
            # Null Whisper-hallucinated lyric entries (global shard sweep, cheap CPU,
            # idempotent) BEFORE align/phonemize so junk trains as <instrumental>, not
            # <vocals>+garbage, and alignment doesn't spend GPU on entries it'll null.
            # transcribe is fully done here (sequential), so no race with its flush.
            run("filter_lyrics", "nano-filter-lyrics", "filter_remote", apply=True)
        if with_align:
            # Forced-align lyric word timestamps (GPU, L4). Refinement-only — any
            # failure keeps the original Whisper timestamp, so it never regresses.
            # After the filter (skips nulled entries); idempotent (aligned stamp).
            run("align_lyrics", "nano-align-lyrics", "run_align",
                apply=True, wave_id=wave_id, use_demucs=align_use_demucs)
    if with_structure:
        # sample_pct<100 runs allin1 (the ~28%-of-wave structure stage) on only a
        # deterministic fraction; non-sampled songs fall back to <no_section>.
        run("structure", "nano-structure", "orchestrate",
            wave_id=wave_id, sample_pct=structure_sample_pct)
    if with_tempo:
        # Dense cheap-CPU tempo pass at FULL coverage. Structure (above) is the
        # only other <tempo_*> source, so when it's sampled this keeps tempo
        # markers at 100% — the dataset prefers this tempo.json over the structure
        # bpm. Only needs /corpus, so its order vs structure doesn't matter.
        run("tempo", "nano-tempo", "orchestrate", wave_id=wave_id)
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
    tokenize_batch: int = 64,
    with_melody: bool = True,
    with_stems: bool = False,
    with_tags: bool = True,
    with_lyrics: bool = True,
    with_structure: bool = True,
    with_tempo: bool = True,
    with_quality_gate: bool = True,
    with_dedup: bool = True,
    with_filter: bool = True,
    with_align: bool = True,
    align_use_demucs: bool = False,
    structure_sample_pct: int = 50,
    stems_sample_pct: int = 50,
    drop_mp3: bool = False,
):
    if not wave_id:
        raise SystemExit("--wave-id is required (e.g. --wave-id 17)")
    fc = ingest_wave.spawn(
        wave_id=wave_id, min_seconds=min_seconds, tokenize_batch=tokenize_batch,
        with_melody=with_melody, with_stems=with_stems, with_tags=with_tags,
        with_lyrics=with_lyrics,
        with_structure=with_structure, with_tempo=with_tempo,
        with_quality_gate=with_quality_gate, with_dedup=with_dedup,
        with_filter=with_filter, with_align=with_align,
        align_use_demucs=align_use_demucs,
        structure_sample_pct=structure_sample_pct,
        stems_sample_pct=stems_sample_pct, drop_mp3=drop_mp3,
    )
    print(f"wave {wave_id} ingest launched (detached) — function call id: {fc.object_id}")
    print("monitor: modal app logs nano-ingest-wave")
