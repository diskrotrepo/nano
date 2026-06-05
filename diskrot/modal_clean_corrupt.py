"""Modal entrypoint: remove corrupt .pt token files from the nano-tokens volume.

A corrupt .pt is one ``torch.load`` cannot read — typically a torn/truncated
write from a preempted tokenize worker. ``pack_cache`` already skips these so
they never reach training, but they linger on the volume and force the shards
that contain them to rebuild on every pack run (their membership no longer
matches the live ``*.pt`` glob). Deleting them makes pack fully resumable; a
follow-up ``modal_tokenize`` run regenerates them (it re-tokenizes any mp3 whose
.pt is missing), after which a final pack is clean.

By default this reads the manifest ``packed/corrupt_files.json`` that pack writes
(instant, no re-scan). Pass ``--rescan`` to independently ``torch.load`` every
.pt instead (ground truth, slower). Dry-run unless ``--apply`` is given.

    modal run diskrot/modal_clean_corrupt.py                 # list from manifest
    modal run diskrot/modal_clean_corrupt.py --rescan        # list via full scan
    modal run diskrot/modal_clean_corrupt.py --apply         # delete (manifest)
    modal run diskrot/modal_clean_corrupt.py --rescan --apply # scan + delete
"""
import modal

app = modal.App("nano-clean-corrupt")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4", "numpy>=1.26")
)

tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(
    image=image,
    cpu=8.0,
    memory=8 * 1024,
    timeout=60 * 60 * 6,
    volumes={"/tokens": tokens_vol},
)
def clean(apply: bool, rescan: bool, n_workers: int) -> dict:
    import json
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import torch

    tokens = Path("/tokens")
    manifest_path = tokens / "packed" / "corrupt_files.json"

    if rescan:
        files = sorted(tokens.glob("*.pt"))
        print(f"[clean] --rescan: torch.load-checking {len(files):,} .pt files "
              f"(workers={n_workers})...", flush=True)

        def _check(path: Path):
            try:
                torch.load(path, weights_only=True, map_location="cpu")
                return None
            except Exception as e:  # noqa: BLE001 — any failure = corrupt
                return (path.name, f"{type(e).__name__}: {str(e)[:80]}")

        corrupt_names: list[str] = []
        checked = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for res in ex.map(_check, files):
                checked += 1
                if res is not None:
                    corrupt_names.append(res[0])
                    print(f"  CORRUPT: {res[0]}  ({res[1]})", flush=True)
                if checked % 20_000 == 0:
                    print(f"  …scanned {checked:,}/{len(files):,}, "
                          f"{len(corrupt_names)} corrupt so far", flush=True)
    else:
        if not manifest_path.exists():
            print(f"[clean] no manifest at {manifest_path} — pack hasn't reported "
                  f"any corrupt files (run it first, or use --rescan).", flush=True)
            return {"corrupt": 0, "deleted": 0}
        corrupt_names = json.loads(manifest_path.read_text())
        print(f"[clean] manifest lists {len(corrupt_names):,} corrupt .pt file(s)",
              flush=True)

    if not corrupt_names:
        print("[clean] nothing to remove — no corrupt .pt files.", flush=True)
        return {"corrupt": 0, "deleted": 0}

    # Resolve to real, still-present files (a name may have already been deleted
    # or regenerated since the manifest was written).
    present = [(name, tokens / name) for name in corrupt_names
               if (tokens / name).exists()]
    print(f"[clean] {len(present):,} of {len(corrupt_names):,} listed files "
          f"are still present on the volume", flush=True)

    if not apply:
        print("[clean] DRY RUN — re-run with --apply to delete the files above. "
              "Then: re-run modal_tokenize (regenerates them) and re-pack.",
              flush=True)
        return {"corrupt": len(present), "deleted": 0}

    deleted = 0
    for name, path in present:
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass
    # Clear the manifest so a later pack/clean run starts clean.
    if not rescan and manifest_path.exists():
        manifest_path.unlink()
    tokens_vol.commit()
    print(f"[clean] deleted {deleted:,} corrupt .pt file(s) and committed.\n"
          f"        next: `modal run --detach diskrot/modal_tokenize.py` to "
          f"regenerate them, then `modal run --detach diskrot/modal_pack_cache.py` "
          f"to finish a clean pack.", flush=True)
    return {"corrupt": len(present), "deleted": deleted}


@app.local_entrypoint()
def main(apply: bool = False, rescan: bool = False, n_workers: int = 32):
    clean.remote(apply=apply, rescan=rescan, n_workers=n_workers)
