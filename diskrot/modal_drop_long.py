"""Drop all long-file material from the nano-corpus volume.

One-off remediation after a `modal_prepare --apply` split run that we decided
to abandon: the `>=10min` files are DJ mixes / hour-long streams / full-album
rips, not songs. Under the per-file crop sampler they were already
undersampled; splitting them into 5-min chunks *overshot* and made mix-derived
crops ~66% of the corpus — the wrong distribution for a song generator. So we
drop them entirely and keep only the clean short songs.

What it removes from `/corpus`:
  - every split-produced chunk (manifest entry has `derived_from`), and
  - every surviving long original (manifest `status == "split"` or the
    legacy `"too_long"`), and
  - orphan partial chunks left on disk by the disk-full ffmpeg failures
    (filename matches the `__<8hex>_p<NNN>.mp3` segment pattern but never made
    it into the manifest).

It then rewrites `/tokens/prepare_manifest.json`: chunk entries are removed and
the long originals are re-marked `status="dropped_long"` for audit.

Safety: dry-run by default (reports counts, touches nothing). Pass `--apply` to
actually unlink and rewrite the manifest.

Dry-run:
    modal run --detach diskrot/modal_drop_long.py
Apply:
    modal run --detach diskrot/modal_drop_long.py --apply
"""
from __future__ import annotations

import modal

app = modal.App("nano-drop-long")

image = modal.Image.debian_slim(python_version="3.12")

corpus_vol = modal.Volume.from_name("nano-corpus", create_if_missing=True)
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)

MANIFEST_PATH = "/tokens/prepare_manifest.json"

# Matches the segment muxer output names from modal_prepare.split_batch:
#   f"{prefix}__{salt}_p%03d.mp3"  where salt = first 8 sha256 hex chars.
CHUNK_RE = r"__[0-9a-f]{8}_p\d{3}\.mp3$"


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=60 * 60 * 2,
    # The whole pass is one container doing ~250k serial unlinks + an iterdir
    # sweep + a manifest rewrite. nonpreemptible so a preemption can't cancel
    # it mid-run; retries as a backstop. The pass is idempotent — a restart
    # re-reads the (unchanged-until-the-end) manifest, and already-deleted
    # files just count as "already gone" — so a retry resumes cleanly. The
    # delete loop commits periodically, so even a hard failure keeps progress.
    nonpreemptible=True,
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=5.0),
)
def drop_long(apply: bool = False) -> dict:
    import json
    import re
    from pathlib import Path

    corpus = Path("/corpus")
    manifest_p = Path(MANIFEST_PATH)
    manifest = json.loads(manifest_p.read_text())
    files = manifest["files"]

    chunk_stems = [s for s, v in files.items() if v.get("derived_from")]
    original_stems = [
        s for s, v in files.items()
        if v.get("status") in ("split", "too_long")
    ]
    target_names = {files[s]["name"] for s in chunk_stems}
    target_names |= {files[s]["name"] for s in original_stems}

    # How many of those targets are actually still on disk right now?
    on_disk = sum(1 for n in target_names if (corpus / n).exists())

    print(f"manifest entries:        {len(files):,}")
    print(f"chunk entries (drop):    {len(chunk_stems):,}")
    print(f"long originals (drop):   {len(original_stems):,}")
    print(f"distinct target names:   {len(target_names):,}")
    print(f"  of which on disk now:  {on_disk:,}")

    pat = re.compile(CHUNK_RE)

    if not apply:
        # Cheap orphan estimate: scan the volume root for chunk-pattern files
        # not already accounted for by a manifest target name.
        orphans = sum(
            1 for p in corpus.iterdir()
            if p.is_file() and pat.search(p.name) and p.name not in target_names
        )
        print(f"orphan partial chunks:   {orphans:,} (on-disk, not in manifest)")
        survivors = len(files) - len(chunk_stems) - len(original_stems)
        print(f"\n[dry-run] would delete ~{on_disk + orphans:,} files, "
              f"leaving ~{survivors:,} clean songs. Re-run with --apply.")
        return {"targets": len(target_names), "on_disk": on_disk,
                "orphans": orphans, "dry_run": True}

    # --- apply ---
    deleted = missing = errors = 0
    for name in target_names:
        p = corpus / name
        try:
            p.unlink()
            deleted += 1
        except FileNotFoundError:
            missing += 1
        except OSError as e:
            errors += 1
            if errors <= 20:
                print(f"failed to delete {name}: {e}")
        # Commit periodically so a mid-run failure doesn't discard all the
        # freed inodes — and so a retry has less to redo.
        if deleted and deleted % 20_000 == 0:
            corpus_vol.commit()
            print(f"  ...committed at {deleted:,} deleted")
    print(f"deleted {deleted:,} named targets "
          f"({missing:,} already gone, {errors:,} errors)")

    # Orphan sweep: anything left matching the chunk pattern.
    orphans = 0
    for p in corpus.iterdir():
        if p.is_file() and pat.search(p.name):
            try:
                p.unlink()
                orphans += 1
            except OSError as e:
                if orphans <= 20:
                    print(f"failed to delete orphan {p.name}: {e}")
    print(f"swept {orphans:,} orphan partial chunks")

    corpus_vol.commit()

    # Rewrite the manifest: drop chunk entries, re-mark originals. Use
    # "too_long" (not a bespoke status) so it's consistent with prepare's
    # delete list — if one of these long files is ever re-added to the corpus
    # under the same name, prepare will re-drop it instead of skipping it.
    for s in chunk_stems:
        files.pop(s, None)
    for s in original_stems:
        files[s]["status"] = "too_long"
    manifest["files"] = files
    manifest_p.write_text(json.dumps(manifest))
    tokens_vol.commit()
    print(f"manifest rewritten: {len(files):,} entries "
          f"({len(chunk_stems):,} chunk entries removed, "
          f"{len(original_stems):,} originals marked too_long)")

    return {"deleted": deleted, "missing": missing, "errors": errors,
            "orphans_swept": orphans, "dry_run": False}


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=60 * 30,
)
def reconcile(apply: bool = False) -> dict:
    """Reconcile the manifest against on-disk reality.

    Removes phantom `ok` entries — manifest entries marked `ok` whose file is
    no longer on `/corpus` (dedup recompute resets deleted duplicates back to
    `ok`, leaving keepers that don't exist). Intentional deletion records
    (`too_long` / `too_short` / `duplicate` / `undecodable`) are KEPT — they're
    audit + re-add-skip records. Also reports untracked on-disk files (on
    disk but absent from the manifest) without touching them.
    """
    import json
    from pathlib import Path

    corpus = Path("/corpus")
    manifest_p = Path(MANIFEST_PATH)
    manifest = json.loads(manifest_p.read_text())
    files = manifest["files"]

    on_disk = {p.stem for p in corpus.glob("*.mp3")}
    ok_stems = {s for s, v in files.items() if v.get("status") == "ok"}
    phantoms = [s for s in ok_stems if s not in on_disk]
    untracked = on_disk - set(files.keys())

    print(f"manifest entries:     {len(files):,}")
    print(f"on-disk mp3s:         {len(on_disk):,}")
    print(f"ok entries:           {len(ok_stems):,}")
    print(f"phantom ok (no file): {len(phantoms):,}  <- to remove")
    print(f"untracked on disk:    {len(untracked):,}  (left as-is; prepare picks them up)")

    if not apply:
        print("\n[dry-run] re-run with --apply to remove the phantom entries")
        return {"phantoms": len(phantoms), "untracked": len(untracked),
                "dry_run": True}

    for s in phantoms:
        files.pop(s, None)
    manifest["files"] = files
    manifest_p.write_text(json.dumps(manifest, indent=2))
    tokens_vol.commit()
    remaining_ok = sum(1 for v in files.values() if v.get("status") == "ok")
    print(f"removed {len(phantoms):,} phantom ok entries; "
          f"manifest now {len(files):,} entries ({remaining_ok:,} ok == on-disk songs)")
    return {"removed": len(phantoms), "ok_after": remaining_ok,
            "untracked": len(untracked), "dry_run": False}


@app.local_entrypoint()
def reconcile_main(apply: bool = False):
    # Lightweight read-filter-write — runs in seconds, so block inline.
    print(f"result: {reconcile.remote(apply=apply)}")


@app.local_entrypoint()
def main(apply: bool = False):
    if not apply:
        # Dry-run finishes in seconds — block and print the report inline.
        print(f"result: {drop_long.remote(apply=False)}")
        return
    # Apply pass runs for minutes (~250k unlinks). .spawn() + `--detach`
    # decouples it from this client so a disconnect can't cancel it mid-run
    # (the .remote() lifecycle trap that killed the first attempt). Progress
    # and the final counts stream to the orchestrator's logs.
    fc = drop_long.spawn(apply=True)
    print(f"drop_long apply launched (detached) — fc id: {fc.object_id}")
    print("watch:  modal app logs $(modal app list | "
          "awk '/nano-drop-long.*ephemeral/{print $2; exit}') -f")
