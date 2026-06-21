"""One-off: delete the flat DAC token artifacts on nano-tokens (frees inodes).

The v9 wave pipeline rebuilds a fresh SpectroStream pack from ``wave_base_*``, so
the legacy flat ``/tokens/<name>.pt`` (~428k files = the bulk of the inode usage)
and the DAC ``/tokens/packed`` are no longer needed and must be cleared so the
v9 wave ``.pt`` fit under the 500k-inode cap. The SOURCE audio is safe in R2
(now under ``wave_base_*``) and v8 *serving* is unaffected (it loads a checkpoint,
not the pack), so this only removes derived artifacts.

Dry-run by default; ``--apply`` deletes. ``--drop-packed`` also removes
``/tokens/packed`` (the DAC pack).

    modal run diskrot/modal_clean_dac_pt.py                       # dry-run count
    modal run --detach diskrot/modal_clean_dac_pt.py --apply --drop-packed
"""
import modal

app = modal.App("nano-clean-dac-pt")
image = modal.Image.debian_slim(python_version="3.12")
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)


@app.function(image=image, volumes={"/tokens": tokens_vol}, timeout=60 * 60 * 3)
def clean(apply: bool = False, drop_packed: bool = False) -> None:
    import shutil
    from pathlib import Path

    tokens = Path("/tokens")
    pts = list(tokens.glob("*.pt"))  # FLAT .pt only (waves/.../*.pt untouched)
    print(f"\n=== flat DAC .pt on nano-tokens: {len(pts):,} files ===", flush=True)
    packed = tokens / "packed"
    print(f"/tokens/packed exists: {packed.exists()} (drop_packed={drop_packed})", flush=True)
    if not apply:
        print("\nDRY RUN — re-run with --apply to delete.", flush=True)
        return

    n = 0
    for p in pts:
        try:
            p.unlink()
            n += 1
            if n % 50_000 == 0:
                print(f"  deleted {n:,}/{len(pts):,} .pt", flush=True)
                tokens_vol.commit()
        except OSError as e:  # noqa: PERF203
            print(f"  skip {p.name}: {e}", flush=True)
    if drop_packed and packed.exists():
        shutil.rmtree(packed, ignore_errors=True)
        print("removed /tokens/packed", flush=True)
    tokens_vol.commit()
    print(f"\ndone: deleted {n:,} flat .pt"
          f"{', dropped packed/' if drop_packed else ''}. Inodes reclaimed.", flush=True)


@app.local_entrypoint()
def main(apply: bool = False, drop_packed: bool = False):
    if apply:
        clean.spawn(apply=True, drop_packed=drop_packed)
        print("cleanup launched (detached).")
    else:
        clean.remote(apply=False)  # dry-run blocks inline
