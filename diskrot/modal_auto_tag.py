"""Modal entrypoint for captioning the corpus, fanned out across many containers.

Each container loads the audio-LLM captioner (Qwen2-Audio,
[model/audio_llm_captioner.py]) once in ``@modal.enter()``, then captions a batch
of MP3 files over the WHOLE song into rich multi-facet descriptions (the chunked-
CLAP tag path conditions on them in full). The orchestrator lists pending files,
dispatches batches via ``.map()``, and periodically flushes ``tags.json`` on the
tokens volume. Set ``NANO_CAPTIONER=bart`` (+ the matching image) for the legacy
single-window LP-MusicCaps captioner.

Run (spawns and returns immediately; --detach keeps the app alive):
    modal run --detach diskrot/modal_auto_tag.py            # caption missing songs
    modal run --detach diskrot/modal_auto_tag.py --redo     # + upgrade legacy short captions (resumable)
    modal run --detach diskrot/modal_auto_tag.py --limit 50 # calibrate the image/cost first

Watch:
    modal app logs nano-auto-tag -f
"""
# NOTE: do NOT add `from __future__ import annotations` here. Modal's class
# parameter validation crashes on PEP 563 stringified field annotations. The
# Captioner class currently has no modal.parameter() fields, but keeping the
# rule consistent across modal_*.py files makes it safe to add one later.

import json
import time
from pathlib import Path

import modal

from diskrot.modal_common import corpus_mount, wave_subdir

app = modal.App("nano-auto-tag")


_CAPTION_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

# Stamped into each tags.json entry so a re-caption pass can tell a current
# (long, audio-LLM) caption from a legacy one and skip the ones already upgraded
# — the same "treat entries missing the marker as pending" trick the v9 README's
# --redo-missing-language transcribe pass uses. MUST match
# model.audio_llm_captioner.CAPTIONER_MARKER (duplicated as a literal because the
# slim orchestrator image can't import that numpy-heavy module).
CAPTIONER_MARKER = "audio_llm_v1"


def _is_current(entry) -> bool:
    """True iff *entry* was produced by the current captioner (skip on --redo)."""
    return isinstance(entry, dict) and entry.get("captioner") == CAPTIONER_MARKER


def _prefetch_captioner():
    """Bake the audio-LLM weights into the image layer (snapshot, no instantiate)."""
    from huggingface_hub import snapshot_download
    snapshot_download(_CAPTION_MODEL)


# Heavy image for the GPU workers (audio-LLM captioner + torch).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4",
        "torchaudio>=2.4",
        "librosa>=0.10",
        "numpy>=1.26",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "transformers>=4.48",  # Qwen2-Audio
        "accelerate>=0.30",
        "huggingface_hub",
    )
    .run_function(_prefetch_captioner, secrets=[modal.Secret.from_name("huggingface-secret")])
    .add_local_python_source("model", "diskrot")
)

# Slim image for the orchestrator — it just lists files, calls .map(), and
# merges results into tags.json. No torch / captioner needed.
orchestrator_image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("diskrot")
)

corpus_vol = corpus_mount()  # nano-corpus Volume, or object storage via NANO_CORPUS_SOURCE=bucket
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100",  # Qwen2-Audio-7B in fp16 (~15 GB) — too big for the L4
    timeout=60 * 60 * 4,
    max_containers=50,
    volumes={"/corpus": corpus_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Captioner:
    @modal.enter()
    def load_model(self):
        from model.audio_llm_captioner import load_captioner
        self.model = load_captioner("cuda")

    @modal.method()
    def caption_batch(
        self, names: list[str]
    ) -> list[tuple[str, dict | None, str | None]]:
        """Returns one (stem, tags_or_None, error_or_None) per input. Captions the
        WHOLE song (several windows) into a rich multi-facet description."""
        from model.audio_llm_captioner import load_song_windows

        out: list[tuple[str, dict | None, str | None]] = []
        for name in names:
            path = Path("/corpus") / name
            stem = path.stem
            try:
                windows = load_song_windows(str(path))
                description = self.model.caption(windows)
                out.append((stem, {"description": description,
                                   "captioner": CAPTIONER_MARKER}, None))
            except Exception as e:
                out.append((stem, None, str(e)[:200]))
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
                 redo: bool = False) -> None:
    """Full pass: list pending, fan out across Captioner containers, merge into
    tags.json with periodic flushes. Spawned from the local entrypoint so
    the user can launch and walk away. ``wave_id`` scopes the input glob to
    /corpus/waves/wave_<id>; tags.json keys stay the song stem either way.
    ``redo=True`` ALSO re-captions songs whose existing caption is NOT the current
    format (legacy/short ones, identified by the ``captioner`` marker) — but skips
    those already upgraded, so a killed --redo resumes instead of restarting. A
    bare run still only captions songs missing from tags.json entirely."""
    mp3s = sorted((Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3"))
    tags_path = Path("/tokens/tags.json")
    existing: dict = {}
    if tags_path.exists():
        existing = json.loads(tags_path.read_text())

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

    if not pending:
        print("Nothing to caption — all files already in tags.json", flush=True)
        return

    chunks = [pending[i:i + batch_size]
              for i in range(0, len(pending), batch_size)]
    print(f"dispatching {len(pending):,} files in {len(chunks):,} batches "
          f"of ~{batch_size} across up to 50 containers...", flush=True)

    def flush() -> None:
        tags_path.parent.mkdir(parents=True, exist_ok=True)
        tags_path.write_text(json.dumps(existing, indent=2))
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
                n_done += 1
        batch_idx += 1
        if batch_idx % flush_every_batches == 0:
            flush()
            done = n_done + n_failed
            pct = 100.0 * done / len(pending)
            print(f"  progress {done:,}/{len(pending):,} ({pct:.1f}%) "
                  f"— captioned {n_done:,}, failed {n_failed:,} "
                  f"(saved to tags.json)", flush=True)

    flush()
    print(f"\ncaptioned: {n_done:,}  failed: {n_failed:,}", flush=True)
    if n_batch_errors:
        print(f"batch errors: {n_batch_errors:,} "
              f"(transient — affected files stay pending; re-run to finish them)",
              flush=True)
    print(f"total in tags.json: {len(existing):,}", flush=True)


@app.local_entrypoint()
def main(batch_size: int = 8, flush_every_batches: int = 4, wave_id: str = "",
         redo: bool = False):
    # spawn (not remote) — submit the orchestrator and return immediately.
    # Combined with `modal run --detach`, the app stays alive after the local
    # CLI exits, so the user can close their terminal and walk away.
    # --wave-id N scopes captioning to /corpus/waves/wave_N; --redo re-captions all.
    # batch_size default 8 (down from 16): the audio-LLM caption is far slower
    # per song than the BART one, so smaller batches keep flushes frequent.
    fc = run_auto_tag.spawn(
        batch_size=batch_size,
        flush_every_batches=flush_every_batches,
        wave_id=wave_id,
        redo=redo,
    )
    print(f"auto-tag launched (detached) — function call id: {fc.object_id}")
    print(f"watch:  modal app logs $(modal app list | "
          f"awk '/nano-auto-tag.*ephemeral/{{print $2; exit}}') -f")
    print(f"stop:   modal app stop $(modal app list | "
          f"awk '/nano-auto-tag.*ephemeral/{{print $2; exit}}') -y")
