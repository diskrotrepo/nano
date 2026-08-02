"""Modal GPU stage: forced-align lyric word timestamps in /tokens/lyrics.

Re-aligns each transcript's EXISTING words to the audio with a CTC forced aligner
(torchaudio MMS_FA), sharpening Whisper's loose word onsets so nano gets better
sung-timing supervision (and the dataset's vocal-crop biasing lands on real words).
Optionally aligns on Demucs-isolated vocals (``--use-demucs``) for extra precision.

In place, idempotent, resumable: each aligned entry is stamped ``aligned=ALIGN_VERSION``
so re-runs skip it; only touched shards are rewritten (atomic temp+rename). Alignment
is a *refinement* — any word/entry that fails keeps its original Whisper timestamp.

Ordering: run AFTER transcribe + filter_lyrics fully finish (in-place lyric edits must
not race the transcribe orchestrator's in-memory flush — same rule as filter_lyrics).
Order vs phonemize is free (phonemize doesn't use timestamps), but running before it is
tidiest. Scope to a wave with ``--wave-id`` (the audio path is then deterministic).

    modal run --detach diskrot/modal_align_lyrics.py --wave-id 7              # dry-run: count eligible
    modal run --detach diskrot/modal_align_lyrics.py --wave-id 7 --apply      # align + rewrite shards
    modal run --detach diskrot/modal_align_lyrics.py --wave-id 7 --apply --use-demucs
"""
from __future__ import annotations

import modal

from diskrot.modal_common import ProgressReporter, corpus_mount, wave_subdir

app = modal.App("nano-align-lyrics")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "torch>=2.4",
        "torchaudio>=2.4",   # MMS_FA forced-alignment bundle
        "numpy>=1.26",
        "demucs",            # optional vocal isolation (reuses transcribe's loader)
        "soundfile",
        # transcribe_lyrics (imported for the Demucs helpers on --use-demucs) does
        # `from tqdm import tqdm` at module top, so tqdm must be present or that
        # import poisons the worker. demucs pulls it transitively, but be explicit.
        "tqdm",
    )
    .add_local_python_source("diskrot", "model")
)

corpus_vol = corpus_mount(read_only=True)
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)

LYRICS_DIR = "/tokens/lyrics"


@app.cls(
    image=image,
    gpu="L4",
    cpu=4.0,
    max_containers=50,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=60 * 60,
)
class Aligner:
    @modal.enter()
    def setup(self):
        import torch

        from diskrot.align_lyrics import ForcedAligner

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.aligner = ForcedAligner(device=self.device)
        self._demucs = None  # lazily loaded on the first --use-demucs batch
        # Warm the wav2vec2 kernels so the first real song isn't a cold-start.
        try:
            self.aligner.align(torch.zeros(16_000), 1.0, ["test"])
        except Exception:
            pass

    def _ensure_demucs(self):
        if self._demucs is None:
            from diskrot.transcribe_lyrics import _load_demucs

            self._demucs = _load_demucs(self.device)  # (model, apply_fn)
        return self._demucs

    def _load_wave(self, rel: str, use_demucs: bool):
        """Return (waveform_16k tensor, audio_seconds) for a corpus-relative mp3."""
        import numpy as np
        import torch

        path = f"/corpus/{rel}"
        if use_demucs:
            import torchaudio.functional as AF

            from diskrot.transcribe_lyrics import _separate_vocals

            model, apply_fn = self._ensure_demucs()
            vocals = _separate_vocals(model, apply_fn, path, self.device)  # ~44.1k mono
            wav = torch.as_tensor(np.ascontiguousarray(vocals), dtype=torch.float32)
            wav = AF.resample(wav, 44_100, 16_000)
        else:
            from diskrot.audio_quality import decode_mono_pcm

            pcm = decode_mono_pcm(path, sr=16_000)
            wav = torch.from_numpy(pcm.copy())
        secs = float(wav.numel()) / 16_000.0
        return wav, secs

    @modal.method()
    def align_batch(self, items: list[dict], use_demucs: bool = False) -> list[dict]:
        """items: [{stem, rel, entry}] → [{stem, entry, changed}]. Each entry is
        re-aligned; failures fall back to original timestamps (refine_entry)."""
        from diskrot.align_lyrics import refine_entry

        out: list[dict] = []
        for it in items:
            updated, changed = refine_entry(
                it["entry"], self.aligner,
                lambda it=it: self._load_wave(it["rel"], use_demucs))
            out.append({"stem": it["stem"], "entry": updated, "changed": changed})
        return out


@app.function(
    image=image,
    cpu=2.0,
    memory=16 * 1024,
    timeout=60 * 60 * 24,
    nonpreemptible=True,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def run_align(
    apply: bool = False,
    wave_id: str = "",
    batch_size: int = 16,
    use_demucs: bool = False,
):
    """List eligible entries (words present, not yet aligned, audio on disk), report,
    and (if --apply) fan out the GPU alignment and rewrite only the touched shards."""
    import json
    import time
    from pathlib import Path

    from diskrot.align_lyrics import is_aligned
    from diskrot.transcribe_lyrics import (
        N_LYRIC_SHARDS,
        _atomic_write_json,
        _lyric_bucket,
        _shard_path,
    )

    mp3s = (Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3")
    stem_to_rel = {p.stem: str(p.relative_to("/corpus")) for p in mp3s}

    # Stream the shards one at a time instead of load_lyrics_shards's full merge:
    # the global store outgrows this container's RAM as the corpus accumulates
    # (a deterministic OOM around ~300-400k transcribed songs that 10 retries
    # only repeat), and only the wave-scoped eligible entries need to stay in
    # memory anyway.
    eligible: list[dict] = []
    n_entries = n_already = 0
    for shard in range(N_LYRIC_SHARDS):
        sp = _shard_path(LYRICS_DIR, shard)
        if not sp.exists():
            continue
        try:
            bucket = json.loads(sp.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # torn/half-written shard: skipped, redone next run
        n_entries += len(bucket)
        for stem, e in bucket.items():
            if is_aligned(e):
                n_already += 1
            elif (isinstance(e, dict) and e.get("words")
                    and stem in stem_to_rel):
                eligible.append(
                    {"stem": stem, "rel": stem_to_rel[stem], "entry": e})

    print("\n=== nano-align-lyrics ===")
    print(f"lyrics entries:   {n_entries:,}")
    print(f"already aligned:  {n_already:,}")
    print(f"eligible now:     {len(eligible):,}  "
          f"(words present, audio on disk, not yet aligned)")
    print(f"mode:             {'demucs-vocals' if use_demucs else 'mix'}")
    if not apply:
        print(f"\n[dry-run] re-run with --apply to align {len(eligible):,} entries (GPU)")
        return
    if not eligible:
        print("nothing to align")
        return

    chunks = [eligible[i:i + batch_size] for i in range(0, len(eligible), batch_size)]
    print(f"aligning {len(eligible):,} entries in {len(chunks):,} batches on L4...")

    touched: set[int] = set()
    pending_by_shard: dict[int, dict] = {}
    n_done = n_changed = 0

    def flush() -> None:
        """O(batch) read-merge-rewrite of the shards holding this run's un-flushed
        results — the modal_transcribe.save_results discipline. Never rewrites a
        shard from a whole-store in-memory copy: each shard also holds other
        waves' entries this run never loaded."""
        if not pending_by_shard:
            return
        tokens_vol.reload()  # see other writers' commits before read-merge-write
        for shard, entries in sorted(pending_by_shard.items()):
            sp = _shard_path(LYRICS_DIR, shard)
            bucket = {}
            if sp.exists():
                try:
                    bucket = json.loads(sp.read_text())
                except (json.JSONDecodeError, OSError):
                    bucket = {}
            bucket.update(entries)
            _atomic_write_json(sp, bucket)
        # Retry a transient DataLossError so a storage blip doesn't crash the
        # orchestrator (a restart redoes only the un-flushed tail — idempotent).
        for attempt in range(3):
            try:
                tokens_vol.commit()
                break
            except modal.exception.DataLossError as e:
                if attempt == 2:
                    raise
                print(f"commit failed ({e}); retry {attempt + 1}/2", flush=True)
                time.sleep(2.0 * (attempt + 1))
        touched.update(pending_by_shard)
        pending_by_shard.clear()

    rep = ProgressReporter(len(eligible), "align_lyrics", unit="entries")
    for batch in Aligner().align_batch.map(
        chunks, kwargs={"use_demucs": use_demucs}, order_outputs=False
    ):
        for r in batch:
            pending_by_shard.setdefault(
                _lyric_bucket(r["stem"]), {})[r["stem"]] = r["entry"]
            n_changed += int(r["changed"])
        n_done += len(batch)
        rep.update(len(batch), extra=f"changed {n_changed:,}")
        # Checkpoint periodically so a preemption doesn't lose hours of alignment.
        if n_done % 5_000 < batch_size:
            flush()

    rep.done(extra=f"changed {n_changed:,}")
    flush()
    print(f"done: aligned {n_done:,} entries, {n_changed:,} had timestamps refined; "
          f"rewrote {len(touched):,} shards")


@app.local_entrypoint()
def main(
    apply: bool = False,
    wave_id: str = "",
    batch_size: int = 16,
    use_demucs: bool = False,
):
    fc = run_align.spawn(
        apply=apply, wave_id=wave_id, batch_size=batch_size, use_demucs=use_demucs)
    mode = "apply" if apply else "dry-run"
    print(f"align-lyrics launched (detached, {mode}) — fc id: {fc.object_id}")
    print(f"watch:  modal app logs $(modal app list | "
          f"awk '/nano-align-lyrics.*ephemeral/{{print $2; exit}}') -f")
