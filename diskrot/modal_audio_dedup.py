"""Modal stage: near-duplicate AUDIO dedup over the R2 corpus (the acoustic dedup
prepare's SHA-256 pass can't do).

Pipeline:
  1. Fingerprint every mp3 with ``fpcalc`` (chromaprint) across CPU containers,
     collapse each to a 64-bit SimHash in the worker (so the manifest stores one int
     per song, not a multi-hundred-int fingerprint), and ffprobe its bitrate.
     Resumable — re-runs skip stems already in the fingerprint manifest.
  2. A single container groups the signatures (LSH bands + Hamming + union-find,
     ``diskrot.audio_dedup``), keeps the best copy per group (highest bitrate), and
     reports the rest.
  3. ``--apply`` deletes the non-keeper near-dups from R2 (batched S3 DeleteObjects
     via boto3, like prepare — ~1000 keys/request, not one FUSE unlink per file).
     Default is dry-run.

Run AFTER prepare (so byte-dupes/garbage are already gone) and before the GPU stages
(don't fingerprint/tokenize/caption a song you're about to delete).

    modal run --detach diskrot/modal_audio_dedup.py                 # dry-run report
    modal run --detach diskrot/modal_audio_dedup.py --apply         # delete near-dups
    modal run --detach diskrot/modal_audio_dedup.py --wave-id 7     # scope to one wave
    modal run --detach diskrot/modal_audio_dedup.py --max-hamming 5 # looser (more recall)
"""
from __future__ import annotations

from pathlib import Path

import modal

from diskrot.modal_common import (
    bulk_delete_r2, corpus_mount, r2_env_secret, wave_subdir,
)

app = modal.App("nano-audio-dedup")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # fpcalc (chromaprint) + ffprobe (ffmpeg). fpcalc decodes the audio itself.
    .apt_install("ffmpeg", "libchromaprint-tools")
    # boto3 drives the batched R2 DeleteObjects in apply_deletions.
    .pip_install("boto3")
    .add_local_python_source("diskrot")
)

corpus_vol = corpus_mount(read_only=False)
tokens_vol = modal.Volume.from_name("nano-tokens", create_if_missing=True)

MANIFEST_PATH = "/tokens/audio_dedup_manifest.json"
FP_LENGTH_S = 120  # fingerprint the first ~2 min (a re-encode matches over that span)


@app.cls(
    image=image,
    # fpcalc and ffprobe run strictly serially per file, single-threaded — a
    # second reserved core never does fingerprint work, it just doubles the bill.
    cpu=1.0,
    max_containers=50,
    volumes={"/corpus": corpus_vol},
    timeout=60 * 60,
)
class Fingerprinter:
    @modal.method()
    def fingerprint_batch(self, names: list[str]) -> list[dict]:
        """fpcalc + ffprobe each file → {stem, name, sig, bit_rate}. The SimHash is
        computed here so the manifest carries one 64-bit int per song, not the raw
        fingerprint. A file fpcalc can't read gets sig=None (skipped at grouping)."""
        import json
        import subprocess

        from diskrot.audio_dedup import parse_fpcalc_raw, simhash64

        out: list[dict] = []
        for name in names:
            path = Path("/corpus") / name
            entry: dict = {"name": name, "stem": path.stem, "sig": None, "bit_rate": None}
            try:
                proc = subprocess.run(
                    ["fpcalc", "-raw", "-length", str(FP_LENGTH_S), str(path)],
                    capture_output=True, text=True, timeout=120,
                )
                fp = parse_fpcalc_raw(proc.stdout) if proc.returncode == 0 else []
                if fp:
                    entry["sig"] = simhash64(fp)
            except (subprocess.TimeoutExpired, OSError, ValueError) as e:
                entry["error"] = str(e)[:200]
            # Bitrate (for keeper choice) — cheap ffprobe; tolerate failure.
            try:
                pr = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "stream=bit_rate:format=bit_rate", "-of", "json", str(path)],
                    capture_output=True, text=True, timeout=30,
                )
                if pr.returncode == 0 and pr.stdout.strip():
                    meta = json.loads(pr.stdout)
                    br = (meta.get("streams", [{}])[0].get("bit_rate")
                          or meta.get("format", {}).get("bit_rate"))
                    entry["bit_rate"] = int(br) if br else None
            except (subprocess.TimeoutExpired, OSError, ValueError, json.JSONDecodeError):
                pass
            out.append(entry)
        return out


@app.function(image=image, volumes={"/corpus": corpus_vol, "/tokens": tokens_vol})
def list_pending(wave_id: str = "") -> tuple[list[str], dict, list[str]]:
    """(mp3s not yet fingerprinted, current manifest, all mp3 names on disk). The
    manifest stays GLOBAL on /tokens so dups are caught across waves."""
    import json

    mp3s = sorted((Path("/corpus") / wave_subdir(wave_id)).glob("*.mp3"))
    all_names = [str(m.relative_to("/corpus")) for m in mp3s]
    manifest: dict = {"version": 1, "files": {}}
    mp = Path(MANIFEST_PATH)
    if mp.exists():
        manifest = json.loads(mp.read_text())
    known = manifest.get("files", {})
    pending = [str(m.relative_to("/corpus")) for m in mp3s if m.stem not in known]
    print(f"found {len(mp3s):,} mp3s, {len(known):,} fingerprinted, "
          f"{len(pending):,} pending")
    return pending, manifest, all_names


@app.function(image=image, volumes={"/tokens": tokens_vol})
def write_manifest(manifest: dict) -> None:
    import json

    p = Path(MANIFEST_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest))
    tokens_vol.commit()


@app.function(
    image=image,
    # No /corpus mount: batch delete goes straight to R2 via boto3, not one
    # FUSE unlink per file. See modal_common.bulk_delete_r2.
    secrets=[modal.Secret.from_name("r2-creds"), r2_env_secret()],
    timeout=60 * 60,
)
def apply_deletions(names: list[str]) -> int:
    n_deleted, errors = bulk_delete_r2(names)
    if errors:
        print(f"({len(errors)} objects failed to delete)")
    return n_deleted


@app.function(
    image=image,
    timeout=60 * 60 * 24,
    nonpreemptible=True,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=5.0),
)
def run_dedup(
    apply: bool = False,
    batch_size: int = 64,
    wave_id: str = "",
    max_hamming: int = 3,
    n_bands: int = 4,
):
    """Fingerprint pending files, group near-dups, report, and (if --apply) delete
    the non-keeper copies. `.spawn()`-ed so the user can walk away — report streams to
    this orchestrator's logs."""
    from datetime import datetime, timezone

    from diskrot.audio_dedup import dedup_verdict, oversized_buckets

    pending, manifest, all_names = list_pending.remote(wave_id=wave_id)
    manifest.setdefault("files", {})
    manifest["version"] = 1

    if pending:
        chunks = [pending[i:i + batch_size] for i in range(0, len(pending), batch_size)]
        print(f"fingerprinting {len(pending):,} files in {len(chunks):,} batches...")
        n_done, last = 0, 0
        for batch in Fingerprinter().fingerprint_batch.map(chunks, order_outputs=False):
            for e in batch:
                manifest["files"][e["stem"]] = e
            n_done += len(batch)
            print(f"  fingerprinted {n_done:,}/{len(pending):,}")
            if n_done - last >= 10_000:
                manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
                write_manifest.remote(manifest)
                last = n_done
        if n_done > last:
            write_manifest.remote(manifest)
        print("fingerprinting complete")
    else:
        print("no new files — reusing existing fingerprints")

    # Group over the whole manifest each run so new files are checked against all.
    on_disk = set(all_names)
    sigs: dict[str, int] = {}
    bit_rates: dict[str, int | None] = {}
    n_unfp = 0
    for stem, v in manifest["files"].items():
        if v.get("name") not in on_disk:
            continue  # already deleted / not in this scope
        if v.get("sig") is None:
            n_unfp += 1
            continue
        sigs[stem] = int(v["sig"])
        bit_rates[stem] = v.get("bit_rate")

    groups, drop_to_keeper = dedup_verdict(
        sigs, bit_rates, max_hamming=max_hamming, n_bands=n_bands)
    n_skipped_buckets = oversized_buckets(sigs, n_bands=n_bands)

    # Map droppable stems → their on-disk names.
    stem_to_name = {s: manifest["files"][s]["name"] for s in drop_to_keeper}
    deletion_names = [stem_to_name[s] for s in drop_to_keeper if stem_to_name[s] in on_disk]

    manifest["verdict"] = {
        "n_groups": len(groups), "n_drop": len(drop_to_keeper),
        "max_hamming": max_hamming, "n_bands": n_bands,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest.remote(manifest)

    dups = sum(len(g) for g in groups)
    print("\n=== nano-audio-dedup report ===")
    print(f"fingerprinted:     {len(sigs):,}  (un-fingerprintable {n_unfp:,})")
    print(f"near-dup groups:   {len(groups):,}  spanning {dups:,} files")
    print(f"would delete:      {len(deletion_names):,}  (keeps 1 best copy per group)")
    if n_skipped_buckets:
        print(f"NOTE: {n_skipped_buckets:,} oversized LSH buckets skipped (uncompared) — "
              f"raise max_bucket or tighten if this is large")
    print(f"manifest:          {MANIFEST_PATH}")
    if not apply:
        print(f"\n[dry-run] re-run with --apply to delete {len(deletion_names):,} near-dups")
        return

    if deletion_names:
        print(f"\ndeleting {len(deletion_names):,} near-dups from /corpus...")
        print(f"deleted {apply_deletions.remote(deletion_names):,} files")


@app.local_entrypoint()
def main(
    apply: bool = False,
    batch_size: int = 64,
    wave_id: str = "",
    max_hamming: int = 3,
    n_bands: int = 4,
):
    fc = run_dedup.spawn(
        apply=apply, batch_size=batch_size, wave_id=wave_id,
        max_hamming=max_hamming, n_bands=n_bands,
    )
    mode = "apply" if apply else "dry-run"
    print(f"audio-dedup launched (detached, {mode}) — fc id: {fc.object_id}")
    print(f"watch:  modal app logs $(modal app list | "
          f"awk '/nano-audio-dedup.*ephemeral/{{print $2; exit}}') -f")
