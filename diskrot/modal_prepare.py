"""Validate, dedupe, and filter the raw audio in the R2 (``nano-audio``) bucket.

Cheap CPU pass that runs after the corpus is uploaded and before the GPU steps
(`modal_tokenize.py`, `modal_auto_tag.py`, `modal_transcribe.py`). Catches
files that would crash or be skipped by those steps anyway, so we don't burn
GPU time on garbage.

What it does (user-configured scope):
  - Validate decodability via `ffprobe`. Anything ffprobe rejects is marked
    `undecodable` and deleted.
  - Drop files shorter than 20s (the tokenizer's silent skip threshold) so
    counts surface in the report.
  - Deduplicate by SHA-256 of file bytes. Keeps lexicographically-first name,
    marks the rest `duplicate` and deletes them.
  - **Drop files longer than MAX_DURATION_S** (5:30). This is a song corpus;
    files past that are DJ mixes / full-album rips / hour-long streams, not
    songs. They're marked `too_long` and deleted. (Earlier builds *split* them
    into chunks to dodge the L4 DAC OOM on long mixes, but the per-file crop
    sampler then let mix-derived chunks dominate training — so we drop now.)
  - Prints a stats report (file count, duration distribution, codec / bitrate
    histograms).

Two phases for safety:
  1. Analyze — read-only ffprobe + sha256 across ~20 CPU containers in
     parallel. Writes results to `/tokens/prepare_manifest.json`.
  2. Apply — gated behind `--apply`. Without it, the report still prints and
     the manifest still updates, but `/corpus` is untouched. Default is
     dry-run so a fat-fingered launch can't nuke the corpus.

Resumability: `list_pending()` returns only MP3s whose `Path.stem` is not in
the manifest's `files` dict. Re-runs after a crash continue from where they
left off. Re-runs over a fully-validated corpus regenerate the report from
the manifest with zero validation work.

Setup (one-time):
    # raw audio lives in the R2 nano-audio bucket (NANO_AUDIO_BUCKET / NANO_AUDIO_ENDPOINT)
    modal volume create nano-tokens

Spawns the orchestrator and returns immediately; `--detach` keeps the app
alive after the CLI exits. The report + progress stream to the orchestrator's
logs (the launch command prints the `modal app logs ... -f` line to watch).

Dry-run (default — no deletions):
    modal run --detach diskrot/modal_prepare.py

Apply the deletions:
    modal run --detach diskrot/modal_prepare.py --apply
"""
from __future__ import annotations

from pathlib import Path

import modal

from diskrot.modal_common import corpus_mount, wave_subdir

app = modal.App("nano-prepare")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")  # ffprobe ships with ffmpeg
    # Required for `modal deploy` (used by the ingest orchestrator's from_name
    # lookup): unlike `modal run`, deploy does NOT auto-mount the entrypoint's
    # package, so the top-level `from diskrot...` import below would fail with
    # ModuleNotFoundError without this.
    .add_local_python_source("diskrot")
)

# read_only=False: prepare deletes undecodable/dup/too-long files in place
# (R2 bucket — unlinks flush through the CloudBucketMount, no .commit()).
corpus_vol = corpus_mount(read_only=False)
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)

MIN_DURATION_S = 20.0
# Anything longer than this is DROPPED (not split). This is a song corpus;
# files past ~5.5 min are overwhelmingly DJ mixes, full-album rips, and
# hour-long streams, not songs. Earlier builds *split* long files into chunks
# to dodge the L4 DAC OOM on hour-long mixes, but the per-file crop sampler
# then let mix-derived chunks dominate the corpus (~66% of crops) — the wrong
# distribution for a song generator. So we drop them outright instead.
MAX_DURATION_S = 330.0  # 5:30

MANIFEST_PATH = "/tokens/prepare_manifest.json"


@app.cls(
    image=image,
    cpu=1.0,
    max_containers=20,
    volumes={"/corpus": corpus_vol},
    timeout=60 * 60,
)
class Validator:
    @modal.method()
    def validate_batch(self, names: list[str]) -> list[dict]:
        """ffprobe + sha256 each file. No mutations, no commits."""
        import hashlib
        import json
        import subprocess

        results: list[dict] = []
        for name in names:
            path = Path("/corpus") / name
            entry: dict = {"name": name, "stem": path.stem}

            try:
                proc = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-show_entries",
                        "stream=duration,bit_rate,codec_name:format=duration,bit_rate",
                        "-of", "json",
                        str(path),
                    ],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode != 0 or not proc.stdout.strip():
                    err = (proc.stderr or "ffprobe failed").strip()
                    entry.update(status="undecodable", error=err[:200])
                    results.append(entry)
                    continue
                meta = json.loads(proc.stdout)
                streams = meta.get("streams", [])
                fmt = meta.get("format", {})
                stream = streams[0] if streams else {}
                duration_s = float(
                    stream.get("duration") or fmt.get("duration") or 0
                )
                bit_rate_raw = stream.get("bit_rate") or fmt.get("bit_rate")
                bit_rate = int(bit_rate_raw) if bit_rate_raw else None
                codec = stream.get("codec_name", "unknown")
                size_bytes = path.stat().st_size
            except (subprocess.TimeoutExpired, json.JSONDecodeError,
                    OSError, ValueError) as e:
                entry.update(status="undecodable", error=str(e)[:200])
                results.append(entry)
                continue

            try:
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(1 << 20)  # 1 MiB
                        if not chunk:
                            break
                        h.update(chunk)
                sha = h.hexdigest()
            except OSError as e:
                entry.update(status="undecodable", error=f"hash: {e}"[:200])
                results.append(entry)
                continue

            if duration_s < MIN_DURATION_S:
                status = "too_short"
            elif duration_s > MAX_DURATION_S:
                status = "too_long"
            else:
                status = "ok"
            entry.update(
                status=status,
                duration_s=duration_s,
                bit_rate=bit_rate,
                codec=codec,
                size_bytes=size_bytes,
                sha256=sha,
            )
            results.append(entry)
        return results


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
)
def list_pending(wave_id: str = "") -> tuple[list[str], dict, list[str]]:
    """Return (mp3s not yet in manifest, current manifest, all mp3 names on disk).
    Names are relative to /corpus (wave-prefixed when ``wave_id`` is set); the
    manifest stays GLOBAL on /tokens so duplicates are caught across waves."""
    import json

    mp3s = sorted((Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3"))
    all_names = [str(mp3.relative_to("/corpus")) for mp3 in mp3s]
    manifest_path = Path(MANIFEST_PATH)
    manifest: dict = {"version": 1, "files": {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    known = manifest.get("files", {})
    pending = [str(mp3.relative_to("/corpus")) for mp3 in mp3s
               if mp3.stem not in known]
    print(f"found {len(mp3s):,} mp3s on volume, "
          f"{len(known):,} already in manifest, "
          f"{len(pending):,} pending validation")
    return pending, manifest, all_names


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol},
)
def write_manifest(manifest: dict) -> None:
    import json

    path = Path(MANIFEST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    tokens_vol.commit()


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol},
    timeout=60 * 60,
)
def apply_deletions(names: list[str]) -> int:
    n_deleted = 0
    n_missing = 0
    for name in names:
        path = Path("/corpus") / name
        try:
            path.unlink()
            n_deleted += 1
        except FileNotFoundError:
            n_missing += 1
        except OSError as e:
            print(f"failed to delete {name}: {e}")
    # corpus is an R2 CloudBucketMount: unlinks flush through the FUSE mount; no commit().
    if n_missing:
        print(f"(skipped {n_missing} already-gone files)")
    return n_deleted


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _build_report(
    manifest: dict,
    n_to_delete: int,
    dry_run: bool,
) -> str:
    files = manifest.get("files", {})
    scanned = len(files)
    kept = [v for v in files.values() if v.get("status") == "ok"]
    counts = {
        "undecodable": 0, "too_short": 0, "duplicate": 0, "too_long": 0,
    }
    for v in files.values():
        s = v.get("status")
        if s in counts:
            counts[s] += 1
    total_marked = (
        counts["undecodable"] + counts["too_short"]
        + counts["duplicate"] + counts["too_long"]
    )

    durations = sorted(v["duration_s"] for v in kept if v.get("duration_s"))

    def pct(p: float) -> float:
        if not durations:
            return 0.0
        return durations[min(int(len(durations) * p), len(durations) - 1)]

    total_hours = sum(durations) / 3600 if durations else 0.0

    codec_hist: dict[str, int] = {}
    bitrate_buckets = {"320k": 0, "256k": 0, "192k": 0, "128k": 0, "other": 0}
    for v in kept:
        c = v.get("codec", "unknown")
        codec_hist[c] = codec_hist.get(c, 0) + 1
        br = v.get("bit_rate") or 0
        if 310_000 <= br <= 330_000:
            bitrate_buckets["320k"] += 1
        elif 250_000 <= br <= 262_000:
            bitrate_buckets["256k"] += 1
        elif 188_000 <= br <= 196_000:
            bitrate_buckets["192k"] += 1
        elif 125_000 <= br <= 130_000:
            bitrate_buckets["128k"] += 1
        else:
            bitrate_buckets["other"] += 1

    lines: list[str] = []
    if dry_run and n_to_delete > 0:
        lines.append(
            f"[dry-run] would delete {n_to_delete:,} files — "
            f"re-run with --apply to change anything"
        )
    elif dry_run:
        lines.append("[dry-run] nothing to delete")
    lines.append("")
    lines.append("=== nano-prepare report ===")
    lines.append(f"scanned:           {scanned:,} files")
    lines.append(f"kept:              {len(kept):,}")
    lines.append(
        f"marked for delete: {total_marked:,} "
        f"(undecodable {counts['undecodable']:,} · "
        f"too_short {counts['too_short']:,} · "
        f"duplicate {counts['duplicate']:,} · "
        f"too_long {counts['too_long']:,})"
    )
    if durations:
        lines.append(
            f"duration (kept):   p10/p50/p90  "
            f"{_fmt_duration(pct(0.1))} / "
            f"{_fmt_duration(pct(0.5))} / "
            f"{_fmt_duration(pct(0.9))}   "
            f"total {total_hours:,.1f} h"
        )
    codec_line = " · ".join(
        f"{k} {v:,}" for k, v in sorted(codec_hist.items(), key=lambda x: -x[1])
    )
    if codec_line:
        lines.append(f"codecs (kept):     {codec_line}")
    bitrate_line = " · ".join(
        f"{k} {v:,}" for k, v in bitrate_buckets.items() if v > 0
    )
    if bitrate_line:
        lines.append(f"bit_rates (kept):  {bitrate_line}")
    lines.append(f"manifest:          {MANIFEST_PATH}")
    return "\n".join(lines)


@app.function(
    image=image,
    timeout=60 * 60 * 24,  # 24h cap on the whole orchestration
    # The orchestrator drives the whole pass for hours from a single container.
    # On a preemptible worker a preemption restarts it, and each restart costs a
    # full list_pending() rescan of the corpus (minutes) — observed to stall the
    # run for ~1h+ per preemption. nonpreemptible pins it to a non-preemptible
    # instance so it simply never restarts. The heavy fan-out (Validator.map())
    # stays preemptible — its per-input retries are clean and free. retries is
    # kept as a backstop for non-preemption transient failures; combined with the
    # incremental manifest writes below, even those resume from the last checkpoint.
    nonpreemptible=True,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def run_prepare(
    apply: bool = False,
    batch_size: int = 64,
    wave_id: str = "",
):
    """Full prepare pass: list pending, validate across CPU containers, dedup,
    re-classify, then (if --apply) delete undecodable / too_short / duplicate /
    too_long files. Designed to be `.spawn()`-ed from the local entrypoint so
    the user can launch and walk away — progress and the dry-run report stream
    to this orchestrator's container logs. ``wave_id`` scopes the pass to
    /corpus/waves/wave_<id> (the dedup manifest stays global)."""
    from datetime import datetime, timezone

    pending, manifest, all_names = list_pending.remote(wave_id=wave_id)
    manifest.setdefault("files", {})
    manifest["version"] = 1

    if pending:
        chunks = [pending[i:i + batch_size]
                  for i in range(0, len(pending), batch_size)]
        print(f"validating {len(pending):,} files in {len(chunks):,} batches "
              f"of ~{batch_size} across up to 20 containers...")
        n_done = 0
        last_checkpoint = 0
        # Persist the manifest periodically so a worker preemption (these run
        # on preemptible CPU workers) doesn't discard hours of validation. On
        # restart, list_pending() skips anything already in the manifest, so we
        # resume from the last checkpoint rather than from zero.
        checkpoint_every = 10_000
        # order_outputs=False: results are merged into the manifest by stem, so
        # order is irrelevant. Crucially, it stops a single preempted batch from
        # head-of-line-blocking the in-order yield — which previously froze the
        # counter (and manifest checkpointing) while the other 20 containers sat
        # idle but billing. With it off, completed batches yield immediately and
        # preemptions become invisible.
        for batch in Validator().validate_batch.map(chunks, order_outputs=False):
            for entry in batch:
                manifest["files"][entry["stem"]] = entry
            n_done += len(batch)
            print(f"  validated {n_done:,}/{len(pending):,}")
            if n_done - last_checkpoint >= checkpoint_every:
                manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
                write_manifest.remote(manifest)
                last_checkpoint = n_done
                print(f"  [checkpoint] manifest persisted at "
                      f"{n_done:,}/{len(pending):,} validated")
        # Flush any tail since the last checkpoint before moving on.
        if n_done > last_checkpoint:
            manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest.remote(manifest)
        print("validation complete")
    else:
        print("no new files to validate — reusing existing manifest")

    # Backfill: prior runs may have flagged long files as "ok" (or, before the
    # drop-don't-split change, lowered the threshold). Re-classify based on the
    # current MAX_DURATION_S so the delete pass picks them up without
    # revalidation. This is what catches the 5:30–10:00 band when the cap is
    # lowered from a previous run.
    n_backfilled = 0
    for v in manifest["files"].values():
        if v.get("status") == "ok" and v.get("duration_s", 0) > MAX_DURATION_S:
            v["status"] = "too_long"
            n_backfilled += 1
    if n_backfilled:
        print(f"re-classified {n_backfilled:,} prior 'ok' entries as 'too_long' "
              f"(duration > {int(MAX_DURATION_S)}s)")

    # Re-compute dedup over the whole manifest each run so newly-arrived files
    # are checked against everything already validated. Reset prior
    # "duplicate" markings back to "ok" first so a removed canonical doesn't
    # leave its dupes orphaned.
    sha_to_stems: dict[str, list[str]] = {}
    for stem, v in manifest["files"].items():
        if v.get("status") == "duplicate":
            v["status"] = "ok"
            v.pop("duplicate_of", None)
        if v.get("status") == "ok" and v.get("sha256"):
            sha_to_stems.setdefault(v["sha256"], []).append(stem)

    for stems in sha_to_stems.values():
        if len(stems) <= 1:
            continue
        stems_sorted = sorted(stems, key=lambda s: manifest["files"][s]["name"])
        keeper = stems_sorted[0]
        for s in stems_sorted[1:]:
            manifest["files"][s]["status"] = "duplicate"
            manifest["files"][s]["duplicate_of"] = keeper

    on_disk = set(all_names)
    deletion_names = [
        v["name"] for v in manifest["files"].values()
        if v.get("status") in ("undecodable", "too_short", "duplicate", "too_long")
        and v["name"] in on_disk
    ]

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest.remote(manifest)

    print(_build_report(manifest, len(deletion_names), dry_run=not apply))

    if not apply:
        return

    if deletion_names:
        print(f"\ndeleting {len(deletion_names):,} files from /corpus...")
        n = apply_deletions.remote(deletion_names)
        print(f"deleted {n:,} files")


@app.local_entrypoint()
def main(
    apply: bool = False,
    batch_size: int = 64,
    wave_id: str = "",
):
    # spawn (not remote) — submit the orchestrator and return immediately.
    # Combined with `modal run --detach`, the app stays alive after the local
    # CLI exits, so the user can close their terminal and walk away. The
    # validation report and progress stream to the orchestrator's logs (watch
    # below), not this terminal — unlike the old inline entrypoint.
    # --wave-id N scopes prepare to /corpus/waves/wave_N.
    fc = run_prepare.spawn(apply=apply, batch_size=batch_size, wave_id=wave_id)
    mode = "apply" if apply else "dry-run"
    print(f"prepare launched (detached, {mode}) — function call id: {fc.object_id}")
    print(f"watch:  modal app logs $(modal app list | "
          f"awk '/nano-prepare.*ephemeral/{{print $2; exit}}') -f")
    print(f"stop:   modal app stop $(modal app list | "
          f"awk '/nano-prepare.*ephemeral/{{print $2; exit}}') -y")
