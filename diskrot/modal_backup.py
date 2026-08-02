"""Modal entrypoint: back up the GPU-expensive derived data to the
``nano-backup`` R2 bucket (rclone, incremental).

What a run copies (CLAUDE.md "Storage" documents the what/why):

- ``/tokens/{tags,lyrics,structure,phonemes}`` — conditioning metadata: $1000s
  of A100 captioning / Whisper transcription, the highest value-per-byte data
- ``/tokens/waves/wave_*/status.json`` — per-wave ingest resume state (NOT the
  wave's transient loose ``.pt`` files)
- ``/tokens/*.json`` (top level) — tags.json / keys.json / tempo.json + manifests
- ``/tokens/packed`` — the packed corpus incl. sidecars (``--no-with-packed`` to skip)
- ``--with-ckpts``: the named nano-ckpts files (best-model insurance; never
  ``step_*`` history)
- the laptop-only bundle (gitignored runbooks, eval logs, Claude memory/plans),
  built locally by the entrypoint via ``diskrot.backup`` and shipped as bytes —
  no R2 credentials needed on the laptop

Safety: both volumes are mounted READ-ONLY (the backup structurally cannot
modify them) and every destination sync is scoped to its own ``tokens/<dir>``
prefix with a ``--max-delete`` guard, so a misconfigured source can never
mass-empty the bucket. Raw audio (``nano-audio``) is deliberately not copied —
single R2 copy, trusted (2026-07-11 decision).

Re-runs are incremental (size+modtime), so the cadence is cheap: re-run after
each wave ingest and after any ``--redo``/``--apply`` sweep that rewrites
``lyrics/`` or ``tags/``. Mid-wave runs are safe — packed shards are written
atomically. ``--verify`` audits volume-vs-bucket (one-way, size-only) instead
of syncing; the report lands in the app logs.

Bucket + endpoint come from the local shell at launch (``NANO_BACKUP_BUCKET``,
default ``nano-backup``; ``NANO_BACKUP_ENDPOINT``, falling back to
``NANO_AUDIO_ENDPOINT``). The bucket is created on first run if the r2-creds
token may; otherwise create it once in the Cloudflare dashboard.

Run::

    modal run --detach diskrot/modal_backup.py                  # metadata + packed + laptop bundle
    modal run --detach diskrot/modal_backup.py --dry-run        # log what would upload, write nothing
    modal run --detach diskrot/modal_backup.py --verify         # size-only volume-vs-bucket audit
    modal run --detach diskrot/modal_backup.py --with-ckpts     # + best-model insurance copies

Monitor::

    modal app logs nano-backup
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

app = modal.App("nano-backup")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("rclone")
    .add_local_python_source("diskrot")
)

# Read-only: the backup must be structurally unable to touch the source data.
tokens_vol = modal.Volume.from_name("nano-tokens").read_only()
ckpts_vol = modal.Volume.from_name("nano-ckpts").read_only()


def _backup_env_secret():
    """NANO_BACKUP_BUCKET/ENDPOINT for the container, captured from the local
    shell at app-build time (mirrors modal_common.r2_env_secret — Modal does
    not forward local env into containers)."""
    endpoint = os.environ.get("NANO_BACKUP_ENDPOINT") or os.environ.get(
        "NANO_AUDIO_ENDPOINT"
    )
    if not endpoint:
        raise ValueError(
            "NANO_BACKUP_ENDPOINT (or NANO_AUDIO_ENDPOINT) must be set in the "
            "launching shell — R2 needs its account endpoint URL"
        )
    return modal.Secret.from_dict(
        {
            "NANO_BACKUP_BUCKET": os.environ.get("NANO_BACKUP_BUCKET", "nano-backup"),
            "NANO_BACKUP_ENDPOINT": endpoint,
        }
    )


@app.function(
    image=image,
    cpu=4.0,
    memory=8 * 1024,
    # First full run streams the whole packed corpus (~hundreds of GB) through
    # volume FUSE once; later runs are metadata-compare + deltas. Idempotent,
    # so retries just resume the sync.
    timeout=60 * 60 * 6,
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0),
    secrets=[modal.Secret.from_name("r2-creds"), _backup_env_secret()],
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol},
)
def backup_remote(
    with_packed: bool = True,
    ckpt_paths: str = "",
    verify: bool = False,
    dry_run: bool = False,
    local_tar: bytes | None = None,
):
    bucket = os.environ["NANO_BACKUP_BUCKET"]
    dst = f"r2:{bucket}"
    # rclone remote defined via env — no config file to bake into the image.
    env = dict(os.environ)
    env.update(
        {
            "RCLONE_CONFIG_R2_TYPE": "s3",
            "RCLONE_CONFIG_R2_PROVIDER": "Cloudflare",
            "RCLONE_CONFIG_R2_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
            "RCLONE_CONFIG_R2_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
            "RCLONE_CONFIG_R2_ENDPOINT": os.environ["NANO_BACKUP_ENDPOINT"],
            "RCLONE_CONFIG_R2_REGION": "auto",
        }
    )
    base_flags = [
        "--transfers", "16", "--checkers", "16", "--fast-list",
        "--s3-chunk-size", "64M",
        "--stats", "60s", "--stats-one-line",  # the progress heartbeat in the logs
    ]

    def rclone(args: list[str], check: bool = True) -> int:
        print(f"[backup] rclone {' '.join(args)}", flush=True)
        return subprocess.run(["rclone", *args], env=env, check=check).returncode

    # Bucket bootstrap — idempotent no-op once it exists.
    try:
        rclone(["mkdir", dst])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"cannot create/see bucket {bucket!r} — if the r2-creds token is "
            f"scoped per-bucket, create the bucket in the Cloudflare dashboard "
            f"and grant the token access, then re-run"
        ) from e

    # (label, src dir, dst suffix, filters, guarded). Guarded steps use `sync`
    # scoped to their own tokens/<dir> prefix + --max-delete; the top-level
    # JSONs use `copy` so nothing is ever deleted at the destination root.
    steps: list[tuple[str, str, str, list[str], bool]] = [
        ("tags", "/tokens/tags", "tokens/tags", [], True),
        ("lyrics", "/tokens/lyrics", "tokens/lyrics", [], True),
        ("structure", "/tokens/structure", "tokens/structure", [], True),
        ("phonemes", "/tokens/phonemes", "tokens/phonemes", [], True),
        ("waves-status", "/tokens/waves", "tokens/waves",
         ["--include", "wave_*/status.json"], True),
        ("top-json", "/tokens", "tokens",
         ["--max-depth", "1", "--include", "*.json"], False),
    ]
    if with_packed:
        steps.append(("packed", "/tokens/packed", "tokens/packed", [], True))

    mismatches: list[str] = []
    for label, src, suffix, filters, guarded in steps:
        if not os.path.isdir(src):
            print(f"[backup] {label}: {src} absent on the volume — skipped", flush=True)
            continue
        target = f"{dst}/{suffix}"
        t0 = time.monotonic()
        if verify:
            rc = rclone(
                ["check", src, target, "--one-way", "--size-only", *filters, *base_flags],
                check=False,
            )
            status = "OK" if rc == 0 else "MISMATCH"
            print(f"[verify] {label}: {status} ({time.monotonic() - t0:.0f}s)", flush=True)
            if rc != 0:
                mismatches.append(label)
        else:
            verb = "sync" if guarded else "copy"
            args = [verb, src, target, *filters, *base_flags]
            if guarded:
                args += ["--max-delete", "2000"]
            if dry_run:
                args.append("--dry-run")
            rclone(args)
            print(f"[backup] {label}: done ({time.monotonic() - t0:.0f}s)", flush=True)

    # Best-model insurance (opt-in): explicit files only, never step_* history.
    # Not covered by --verify (one-shot copyto; re-run the backup to refresh).
    for rel in [p.strip() for p in ckpt_paths.split(",") if p.strip()]:
        src = f"/ckpts/{rel}"
        if not os.path.isfile(src):
            print(f"[backup] ckpt {rel}: absent on nano-ckpts — skipped", flush=True)
            continue
        if verify:
            print(f"[verify] ckpt {rel}: skipped (verify covers tokens/ only)", flush=True)
            continue
        rclone(["copyto", src, f"{dst}/ckpts/{rel}", *base_flags,
                *(["--dry-run"] if dry_run else [])])

    # Laptop bundle: latest (overwritten) + a dated archive copy, both tiny.
    if local_tar and not verify:
        tmp = "/tmp/laptop_bundle.tar.gz"
        with open(tmp, "wb") as f:
            f.write(local_tar)
        stamp = time.strftime("%Y%m%d", time.gmtime())
        for key in ("local/laptop_latest.tar.gz", f"local/archive/laptop_{stamp}.tar.gz"):
            rclone(["copyto", tmp, f"{dst}/{key}",
                    *(["--dry-run"] if dry_run else [])])

    if verify:
        if mismatches:
            raise RuntimeError(
                f"[verify] MISMATCH in: {', '.join(mismatches)} — re-run the backup"
            )
        print("[verify] all prefixes match the bucket (one-way, size-only)", flush=True)
    else:
        rclone(["size", dst], check=False)
        print(f"[done] backup synced to {dst}", flush=True)


@app.local_entrypoint()
def main(
    with_packed: bool = True,
    with_ckpts: bool = False,
    ckpt_paths: str = "v8_sing4/best_inference.pt,v8_sing4/best.pt",
    with_local: bool = True,
    verify: bool = False,
    dry_run: bool = False,
):
    local_tar = None
    if with_local and not verify:
        from diskrot.backup import build_tar_bytes, claude_project_dirs, collect_local_paths

        repo_root = Path(__file__).resolve().parent.parent
        pairs = collect_local_paths(repo_root, extra_dirs=claude_project_dirs(repo_root))
        local_tar = build_tar_bytes(pairs)
        print(f"[backup] laptop bundle: {len(pairs)} files, {len(local_tar) / 1e6:.1f} MB")
    fc = backup_remote.spawn(
        with_packed=with_packed,
        ckpt_paths=ckpt_paths if with_ckpts else "",
        verify=verify,
        dry_run=dry_run,
        local_tar=local_tar,
    )
    print(f"backup launched (detached) -- function call id: {fc.object_id}")
    print("monitor with: modal app logs nano-backup")
