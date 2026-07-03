"""Modal entrypoint for captioning the corpus, fanned out across many containers.

Each container loads the audio-LLM captioner (Qwen2-Audio,
[model/audio_llm_captioner.py]) once in ``@modal.enter()``, then captions a batch
of MP3 files over the WHOLE song into rich multi-facet descriptions (the chunked-
CLAP tag path conditions on them in full). The orchestrator lists pending files,
dispatches batches via ``.map()``, and periodically flushes the sharded tags
store (``/tokens/tags/tags_NNN.json``, O(touched-shards) per flush) on the tokens
volume; a merged ``tags.json`` compat file is written once at end-of-stage for
the single-file readers (train startup, audit scripts). Set ``NANO_CAPTIONER=bart``
(+ the matching image) for the legacy single-window LP-MusicCaps captioner.

Run (spawns and returns immediately; --detach keeps the app alive):
    modal run --detach diskrot/modal_auto_tag.py            # caption missing songs
    modal run --detach diskrot/modal_auto_tag.py --redo     # + upgrade legacy short captions (resumable)
    modal run --detach diskrot/modal_auto_tag.py --limit 50 # calibrate the image/cost first
    modal run --detach diskrot/modal_auto_tag.py --wave-id all --redo  # whole-corpus upgrade sweep
                                                            # (all waves/wave_*/; a bare --redo globs
                                                            # only the flat legacy root, empty post-R2)

Watch:
    modal app logs nano-auto-tag -f
"""
# NOTE: do NOT add `from __future__ import annotations` here. Modal's class
# parameter validation crashes on PEP 563 stringified field annotations. The
# Captioner class currently has no modal.parameter() fields, but keeping the
# rule consistent across modal_*.py files makes it safe to add one later.

import json
import os
import time
from pathlib import Path

import modal

from diskrot.modal_common import assert_stage_produced_output, corpus_mount, list_wave_mp3s
from diskrot.sharded_store import load_json_shards, shard_index, write_json_shards

app = modal.App("nano-auto-tag")


_CAPTION_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

# Stamped into each tags.json entry so a re-caption pass can tell a current
# (long, audio-LLM) caption from a legacy one and skip the ones already upgraded
# — the same "treat entries missing the marker as pending" trick the v9 README's
# --redo-missing-language transcribe pass uses. MUST match
# model.audio_llm_captioner.CAPTIONER_MARKER (duplicated as a literal because the
# slim orchestrator image can't import that numpy-heavy module).
# v4: caption entries now carry a ``gender`` field (the audio-LLM's vocal-gender
# judgment, replacing the F0-on-Demucs estimate); a --redo upgrades v3 -> v4.
# v5: entries now also carry a ``stems`` dict (per-stem captions for /addstem); a
# --redo upgrades v4 -> v5.
CAPTIONER_MARKER = "audio_llm_v5"


def _is_current(entry) -> bool:
    """True iff *entry* was produced by the current captioner (skip on --redo)."""
    return isinstance(entry, dict) and entry.get("captioner") == CAPTIONER_MARKER


# With vLLM continuous batching each container saturates its OWN A100 independently,
# so total cost (GPU-seconds = songs × s/song) is ~container-count-independent — the
# vLLM init + KV-cache allocation amortizes via warm-container reuse across .map
# elements regardless of count, leaving only a few $ of per-container startup +
# idle-tail. So more containers just finish the WALL-CLOCK faster at ~the same price:
# default to the account's full 50-GPU allowance. Dial down via env to limit blast
# radius (e.g. while calibrating). Keep <= the account GPU cap (50).
MAX_CONTAINERS = int(os.environ.get("NANO_AUTOTAG_MAX_CONTAINERS", "50"))


# Heavy image for the GPU workers (audio-LLM captioner + vLLM).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        # vLLM owns the torch build (it pins a compatible one) — don't pin torch
        # here or the resolver fights it. torchaudio is dropped (unused: librosa +
        # soundfile do the decode). NOTE: vLLM's multimodal-audio API — the raw
        # (np, sr) tuple in multi_modal_data and limit_mm_per_prompt — is
        # version-sensitive; LOCK the exact vllm version after the --limit
        # calibration confirms caption parity.
        "vllm>=0.10.0",
        "librosa>=0.10",
        "numpy>=1.26",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "transformers>=4.48",  # Qwen2-Audio
        "accelerate>=0.30",
        "huggingface_hub",
    )
    # Bake the audio-LLM weights into the image layer. MUST be run_commands (a
    # pure shell step), NOT run_function: a build-time run_function imports this
    # module to find the callable, but `diskrot` is only added by the
    # add_local_python_source below (copy=False → absent at build time), so the
    # top-level `from diskrot...` import fails with ModuleNotFoundError. A shell
    # command needs no module import and keeps the 16 GB download cached
    # independently of source changes.
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{_CAPTION_MODEL}')\"",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    # FlashInfer's top-p/top-k sampler JIT-compiles a CUDA kernel at vLLM
    # EngineCore init via nvcc, which this debian_slim+pip-vllm image lacks (no
    # CUDA toolkit) -> "Could not find nvcc ... /usr/local/cuda doesn't exist"
    # killed the engine on EVERY caption (captioned: 0). We decode greedily
    # (temperature=0.0), so the FlashInfer sampler buys nothing: disabling it
    # ("FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0")
    # removes the only active nvcc-JIT path — vLLM auto-selects the precompiled,
    # nvcc-free FLASH_ATTN attention backend, and the inductor/VLLM_COMPILE path
    # uses Triton+gcc, not nvcc. Placed AFTER the 16 GB weight bake so this is a
    # cheap env layer, not a re-download. NOTE: do NOT add VLLM_ATTENTION_BACKEND
    # here — v0.23.0 doesn't recognize it (logs "Unknown vLLM environment
    # variable") and it's inert. The real forward-proofing is to LOCK the vllm
    # version above so the resolver can't drift back onto a FlashInfer default.
    .env({
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Throughput knobs, baked from the deploying shell (override there to
        # calibrate). max_num_seqs=32 (the engine default in audio_llm_captioner)
        # left the already-reserved KV cache under-filled on the bandwidth-bound
        # decode; 64 fills it. The decode sub-batch must EXCEED max_num_seqs so
        # the scheduler backfills as sequences finish instead of draining to
        # empty at each generate()'s tail.
        "NANO_VLLM_MAX_NUM_SEQS": os.environ.get("NANO_VLLM_MAX_NUM_SEQS", "64"),
        "NANO_CAPTION_SUBBATCH": os.environ.get("NANO_CAPTION_SUBBATCH", "128"),
    })
    .add_local_python_source("model", "diskrot")
)

# Slim image for the orchestrator — it just lists files, calls .map(), and
# merges results into tags.json. No torch / captioner needed.
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("diskrot")
)

corpus_vol = corpus_mount()  # R2 audio bucket (read-only); see modal_common.corpus_mount
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.cls(
    image=image,
    # Qwen2-Audio-7B in fp16 (~15 GB) — too big for the L4. NANO_AUTOTAG_GPU
    # (read in the deploying/`modal run` shell, like tokenize's NANO_SPIKE_GPU)
    # is the probe lever for H100 / A100-80GB $/song comparisons — identical
    # weights and greedy decode, so outputs are quality-equivalent.
    gpu=os.environ.get("NANO_AUTOTAG_GPU", "A100"),
    timeout=60 * 60 * 4,
    max_containers=MAX_CONTAINERS,
    # Retry an element whose container died under it (preemption, OOM, platform
    # cancellation) so it completes in-run instead of staying pending — the marker
    # resume makes a retry idempotent (already-captioned stems are skipped).
    retries=modal.Retries(max_retries=2, initial_delay=1.0),
    volumes={"/corpus": corpus_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Captioner:
    @modal.enter()
    def load_model(self):
        from model.audio_llm_captioner import load_captioner
        # NANO_CAPTIONER_BACKEND defaults to vllm (continuous batching); set =hf
        # for the transformers fallback (slower, the parity oracle).
        self.model = load_captioner("cuda")

    def _caption_decoded(
        self, decoded: list[tuple[str, list | None, str | None]]
    ) -> list[tuple[str, dict | None, str | None]]:
        """Caption an already-decoded sub-batch. Decode errors pass through; the
        decoded-ok songs run through the batched vLLM path (one continuous batch)
        when available, else the serial HF path."""
        out: list[tuple[str, dict | None, str | None]] = [
            (stem, None, err) for (stem, _w, err) in decoded if err is not None
        ]
        ok = [(stem, w) for (stem, w, err) in decoded if err is None]
        if not ok:
            return out
        # The captioner emits a trailing GENDER tag + four per-stem lines; the
        # *_with_gender_and_stems APIs parse them so tags.json carries
        # {description, gender, stems, captioner}. Gender is the audio-LLM's
        # male/female judgment (None = instrumental); stems is the per-stem caption
        # dict for /addstem (empty until a song is captioned at v5).
        if hasattr(self.model, "caption_many_with_gender_and_stems"):
            try:
                triples = self.model.caption_many_with_gender_and_stems(
                    [w for (_s, w) in ok])
                out.extend(
                    (stem, {"description": d, "gender": g, "stems": s,
                            "captioner": CAPTIONER_MARKER}, None)
                    for (stem, _w), (d, g, s) in zip(ok, triples)
                )
            except Exception as e:
                # A whole-batch generation failure must not crash the element —
                # mark these songs failed (they stay pending, redone next run).
                out.extend((stem, None, f"caption_many: {str(e)[:180]}")
                           for (stem, _w) in ok)
        else:
            for stem, w in ok:
                try:
                    d, g, s = self.model.caption_with_gender_and_stems(w)
                    out.append((stem, {"description": d, "gender": g, "stems": s,
                                       "captioner": CAPTIONER_MARKER}, None))
                except Exception as e:
                    out.append((stem, None, str(e)[:200]))
        return out

    @modal.method()
    def caption_batch(
        self, names: list[str]
    ) -> list[tuple[str, dict | None, str | None]]:
        """Returns one (stem, tags_or_None, error_or_None) per input. Captions each
        WHOLE song (several windows) into a rich multi-facet description, running a
        whole sub-batch as ONE vLLM continuous batch. Audio decodes on CPU threads
        (librosa releases the GIL) and the NEXT sub-batch decodes while the current
        one generates on the GPU, so CPU decode overlaps GPU compute."""
        from concurrent.futures import ThreadPoolExecutor

        from model.audio_llm_captioner import load_song_windows

        n_workers = int(os.environ.get(
            "NANO_DECODE_WORKERS", str(min(16, (os.cpu_count() or 4)))))
        sub = int(os.environ.get("NANO_CAPTION_SUBBATCH", "64"))
        subbatches = [names[i:i + sub] for i in range(0, len(names), sub)]

        def _decode(name: str) -> tuple[str, list | None, str | None]:
            path = Path("/corpus") / name
            try:
                return (path.stem, load_song_windows(str(path)), None)
            except Exception as e:
                return (path.stem, None, str(e)[:200])

        out: list[tuple[str, dict | None, str | None]] = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            def submit(sb):
                return [ex.submit(_decode, nm) for nm in sb]

            next_futs = submit(subbatches[0]) if subbatches else []
            for idx in range(len(subbatches)):
                cur_futs = next_futs
                # Kick off the next sub-batch's decode BEFORE generating this one,
                # so CPU decode runs while the GPU is busy.
                next_futs = (submit(subbatches[idx + 1])
                             if idx + 1 < len(subbatches) else [])
                decoded = [f.result() for f in cur_futs]
                out.extend(self._caption_decoded(decoded))
        return out


@app.function(
    image=orchestrator_image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=60 * 60 * 24,
    # Pin the long-lived coordinator to a non-preemptible instance so it never
    # restarts mid-run (each restart costs a full corpus rescan). The GPU
    # Captioner fan-out stays preemptible — its .map() inputs auto-retry. retries
    # kept as a transient-failure backstop; tags.json is flushed every
    # flush_every_batches so a restart resumes with at most the un-flushed tail lost.
    nonpreemptible=True,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def run_auto_tag(batch_size: int, flush_every_batches: int, wave_id: str = "",
                 redo: bool = False, limit: int = 0) -> None:
    """Full pass: list pending, fan out across Captioner containers, merge into
    tags.json with periodic flushes. Spawned from the local entrypoint so
    the user can launch and walk away. ``wave_id`` scopes the input glob to
    /corpus/waves/wave_<id> ("all" sweeps every wave folder in one launch);
    tags.json keys stay the song stem either way.
    ``redo=True`` ALSO re-captions songs whose existing caption is NOT the current
    format (legacy/short ones, identified by the ``captioner`` marker) — but skips
    those already upgraded, so a killed --redo resumes instead of restarting. A
    bare run still only captions songs missing from tags.json entirely."""
    mp3s = list_wave_mp3s(Path("/corpus"), wave_id)
    tags_path = Path("/tokens/tags.json")
    tags_dir = Path("/tokens/tags")
    existing: dict = {}
    if tags_path.exists():
        existing = json.loads(tags_path.read_text())
    # Overlay the sharded store (the flush target below): on a mid-stage resume
    # the shards are newer than the last end-of-stage compat tags.json, so they
    # win. A never-sharded volume just yields an empty dict here.
    existing.update(load_json_shards(tags_dir, "tags"))

    def _pending(mp3) -> bool:
        if mp3.stem not in existing:
            return True
        return redo and not _is_current(existing[mp3.stem])

    pending = [str(mp3.relative_to("/corpus")) for mp3 in mp3s if _pending(mp3)]
    n_legacy = sum(1 for mp3 in mp3s
                   if mp3.stem in existing and not _is_current(existing[mp3.stem]))
    if redo:
        print(f"REDO (upgrade to {CAPTIONER_MARKER}): {len(pending):,} of "
              f"{len(mp3s):,} songs need (re)captioning — "
              f"{n_legacy:,} legacy/short + {len(pending) - n_legacy:,} missing; "
              f"already-current entries are skipped (resumable)", flush=True)
    elif existing:
        print(f"RESUMING: {len(existing):,} of {len(mp3s):,} already captioned, "
              f"{len(pending):,} still pending ({n_legacy:,} legacy entries kept — "
              f"pass --redo to upgrade them)", flush=True)
    else:
        print(f"fresh run: {len(mp3s):,} mp3s, 0 already captioned, "
              f"{len(pending):,} pending", flush=True)

    if limit and len(pending) > limit:
        print(f"--limit {limit:,}: capping this pass at {limit:,} of "
              f"{len(pending):,} pending songs (calibration mode; captions are "
              f"real output, skipped on the next full pass)", flush=True)
        pending = pending[:limit]

    if not pending:
        print("Nothing to caption — all files already in tags.json", flush=True)
        return

    chunks = [pending[i:i + batch_size]
              for i in range(0, len(pending), batch_size)]
    print(f"dispatching {len(pending):,} files in {len(chunks):,} batches "
          f"of ~{batch_size} across up to {MAX_CONTAINERS} containers...", flush=True)

    touched: set[int] = set()

    def commit_with_retry() -> None:
        # Retry a transient DataLossError so a storage blip on the periodic
        # flush doesn't crash the orchestrator and trigger a full rescan +
        # fleet re-spawn.
        for attempt in range(3):
            try:
                tokens_vol.commit()
                return
            except modal.exception.DataLossError as e:
                if attempt == 2:
                    raise
                print(f"commit failed ({e}); retry {attempt + 1}/2", flush=True)
                time.sleep(2.0 * (attempt + 1))

    def flush() -> None:
        # O(touched) flush: rewrite only the hash shards that gained entries
        # since the last flush (atomic per shard). The monolithic tags.json is
        # deliberately NOT rewritten here — at 1.3M entries each re-serialize is
        # a multi-second O(corpus) stall inside the .map consume loop that
        # backpressures the GPU fleet; the single-file readers get one compat
        # write at end-of-stage instead.
        write_json_shards(tags_dir, "tags", existing, only_shards=touched)
        commit_with_retry()
        touched.clear()

    captioner = Captioner()
    n_done = n_failed = 0
    n_batch_errors = 0
    batch_idx = 0
    # order_outputs=False: a preempted batch must not head-of-line-block the
    # in-order yield (that idles the other containers while they still bill).
    # Results are merged into tags.json by stem, so order doesn't matter.
    # return_exceptions=True: a single batch raising (e.g. a worker that hard-
    # crashes on a poison file, or a transient error) must NOT propagate out and
    # crash this orchestrator — that would re-rescan and re-spin-up the fleet.
    # Count it and continue; those files stay un-captioned and are picked up on
    # the next run via the tags.json skip logic.
    for batch in captioner.caption_batch.map(
        chunks, order_outputs=False, return_exceptions=True
    ):
        if isinstance(batch, Exception):
            n_batch_errors += 1
            if n_batch_errors <= 20:
                print(f"BATCH FAILED (files stay pending, redone next run): "
                      f"{type(batch).__name__}: {str(batch)[:140]}", flush=True)
            batch_idx += 1
            continue
        for stem, tags, error in batch:
            if error is not None:
                print(f"FAILED {stem}: {error}", flush=True)
                n_failed += 1
            else:
                existing[stem] = tags
                touched.add(shard_index(stem))
                n_done += 1
        batch_idx += 1
        if batch_idx % flush_every_batches == 0:
            flush()
            done = n_done + n_failed
            pct = 100.0 * done / len(pending)
            print(f"  progress {done:,}/{len(pending):,} ({pct:.1f}%) "
                  f"— captioned {n_done:,}, failed {n_failed:,} "
                  f"(saved to the tags shard store)", flush=True)

    flush()
    # Compat write for the single-file readers (modal_train's tags_path, the
    # audit/eval scripts): one O(corpus) merged tags.json per stage run instead
    # of per flush. Atomic (temp + os.replace) — tags.json is loaded WHOLE at
    # train startup, and a kill mid-write must never truncate it.
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tags_path.with_suffix(tags_path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing))
    os.replace(tmp, tags_path)
    commit_with_retry()
    print(f"\ncaptioned: {n_done:,}  failed: {n_failed:,}", flush=True)
    assert_stage_produced_output("auto_tag", n_done, len(pending), n_failed)
    if n_batch_errors:
        print(f"batch errors: {n_batch_errors:,} "
              f"(transient — affected files stay pending; re-run to finish them)",
              flush=True)
    print(f"total in tags.json: {len(existing):,}", flush=True)


@app.local_entrypoint()
def main(batch_size: int = 256, flush_every_batches: int = 25, wave_id: str = "",
         redo: bool = False, limit: int = 0):
    # spawn (not remote) — submit the orchestrator and return immediately.
    # Combined with `modal run --detach`, the app stays alive after the local
    # CLI exits, so the user can close their terminal and walk away.
    # --wave-id N scopes captioning to /corpus/waves/wave_N ("all" = every wave
    # folder — the whole-corpus sweep); --redo re-captions legacy/non-current.
    # --limit N caps the pass at N pending songs (calibration mode).
    # batch_size default 256: each .map element is run as ONE vLLM continuous batch
    # (internally pipelined in NANO_CAPTION_SUBBATCH-sized sub-batches so CPU decode
    # overlaps GPU compute), keeping the GPU saturated; the orchestrator still
    # flushes the tags shard store every flush_every_batches elements.
    fc = run_auto_tag.spawn(
        batch_size=batch_size,
        flush_every_batches=flush_every_batches,
        wave_id=wave_id,
        redo=redo,
        limit=limit,
    )
    print(f"auto-tag launched (detached) — function call id: {fc.object_id}")
    print(f"watch:  modal app logs $(modal app list | "
          f"awk '/nano-auto-tag.*ephemeral/{{print $2; exit}}') -f")
    print(f"stop:   modal app stop $(modal app list | "
          f"awk '/nano-auto-tag.*ephemeral/{{print $2; exit}}') -y")
