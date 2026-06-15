"""One-off: find orphan chroma sidecars on nano-melody.

An orphan is a ``<name>.mel.npy`` on the **nano-melody** volume with no matching
``<name>.pt`` on **nano-tokens** — i.e. the song was dropped/re-prepared after
its chroma was extracted, leaving a dead sidecar that only burns an inode.

Read-only by default (lists + counts). Pass ``--apply`` to delete the orphans.

    modal run scripts/find_orphan_melody.py            # report only
    modal run scripts/find_orphan_melody.py --apply    # delete orphans
"""

from pathlib import Path

import modal

app = modal.App("nano-find-orphan-melody")
image = modal.Image.debian_slim()

tokens_vol = modal.Volume.from_name("nano-tokens")
melody_vol = modal.Volume.from_name("nano-melody", create_if_missing=True)

_MEL_EXT = ".mel.npy"


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol, "/melody": melody_vol},
    timeout=60 * 60,
)
def find_orphans(apply: bool = False) -> list[str]:
    tokens_vol.reload()
    melody_vol.reload()

    pt_stems = {p.stem for p in Path("/tokens").glob("*.pt")}

    # Walk EVERYTHING on /melody, not just clean *.mel.npy — interrupted atomic
    # writes leave temp files that still burn an inode.
    all_entries = list(Path("/melody").rglob("*"))
    files = [p for p in all_entries if p.is_file()]
    dirs = [p for p in all_entries if p.is_dir()]
    clean = {p for p in files if p.name.endswith(_MEL_EXT)}
    junk = [p for p in files if p not in clean]  # temp/partial/other

    mel = {p.name[: -len(_MEL_EXT)]: p for p in clean}
    orphans = sorted(set(mel) - pt_stems)
    print(f"tokens .pt:          {len(pt_stems):>7}")
    print(f"melody total files:  {len(files):>7}")
    print(f"melody dirs:         {len(dirs):>7}")
    print(f"clean .mel.npy:      {len(mel):>7}")
    print(f"NON-.mel.npy files:  {len(junk):>7}  (temp/partial leftovers)")
    print(f"orphan .mel.npy:     {len(orphans):>7}  (melody with no matching .pt)")

    from collections import Counter
    ext_hist = Counter(
        p.name[p.name.find(".mel"):] if ".mel" in p.name else p.suffix or "(noext)"
        for p in junk)
    if ext_hist:
        print("non-clean file suffixes:")
        for suf, n in ext_hist.most_common(15):
            print(f"  {n:>7}  {suf!r}")
    for p in junk[:15]:
        print(f"  junk: {p.relative_to('/melody')}")

    for name in orphans[:25]:
        print(f"  orphan: {name}")
    if len(orphans) > 25:
        print(f"  ... and {len(orphans) - 25} more")

    to_delete = [mel[name] for name in orphans] + junk
    if apply and to_delete:
        removed = 0
        for p in to_delete:
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        melody_vol.commit()
        print(f"deleted {removed} files "
              f"({len(orphans)} orphan chroma + {len(junk)} junk); "
              f"committed nano-melody")
    elif to_delete:
        print(f"dry-run — re-run with --apply to delete "
              f"{len(to_delete)} files")

    return orphans


@app.local_entrypoint()
def main(apply: bool = False):
    find_orphans.remote(apply=apply)
