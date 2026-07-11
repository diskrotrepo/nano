"""Pure helpers for the backup stage (modal_backup.py): gather the laptop-only
files — gitignored planning docs (the repo is public, so they can't just be
committed), local Claude settings, eval logs, and the Claude Code memory/plans
dirs — into a tar.gz bundle the Modal wrapper drops into the backup bucket.

Stdlib-only (mirrors diskrot.progress: no modal import), so the local
entrypoint can build the bundle on the laptop and ship it to the container as
plain bytes — no R2 credentials ever needed locally. The volume-side sync is
rclone inside the container (modal_backup.py); nothing here touches the network.
"""
from __future__ import annotations

import io
import tarfile
from collections.abc import Iterable, Sequence
from pathlib import Path

# Laptop-only repo files worth protecting. Missing entries are skipped, so the
# list can name files that only exist sometimes (README.genre.md pattern).
LOCAL_REPO_FILES = (
    "README.v9.md",
    "README.genre.md",
    ".claude/settings.json",
    ".claude/settings.local.json",
)
LOCAL_REPO_GLOBS = ("eval/*.log",)


def claude_project_dirs(repo_root: Path, home: Path | None = None) -> list[Path]:
    """The Claude Code per-project memory dir + the global plans dir for this
    repo. The project dir name is the absolute repo path with '/' → '-'
    (e.g. /Users/x/nano → -Users-x-nano), matching Claude Code's layout."""
    home = home or Path.home()
    proj = "-" + str(Path(repo_root).resolve()).strip("/").replace("/", "-")
    return [
        home / ".claude" / "projects" / proj / "memory",
        home / ".claude" / "plans",
    ]


def collect_local_paths(
    repo_root: Path, extra_dirs: Iterable[Path] = ()
) -> list[tuple[Path, str]]:
    """(absolute file, arcname) pairs for the laptop bundle.

    Repo files land under ``repo/<relpath>``; each extra dir lands under its
    own basename (``memory/…``, ``plans/…``). Anything missing is skipped —
    the bundle is a best-effort snapshot, not a manifest contract."""
    pairs: list[tuple[Path, str]] = []
    for rel in LOCAL_REPO_FILES:
        p = repo_root / rel
        if p.is_file():
            pairs.append((p, f"repo/{rel}"))
    for pattern in LOCAL_REPO_GLOBS:
        for p in sorted(repo_root.glob(pattern)):
            if p.is_file():
                pairs.append((p, f"repo/{p.relative_to(repo_root)}"))
    for d in extra_dirs:
        d = Path(d).expanduser()
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file():
                pairs.append((p, f"{d.name}/{p.relative_to(d)}"))
    return pairs


def build_tar_bytes(pairs: Sequence[tuple[Path, str]]) -> bytes:
    """gzip'd tar of *pairs* as in-memory bytes (the laptop set is a few MB)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, arcname in pairs:
            tf.add(path, arcname=arcname, recursive=False)
    return buf.getvalue()
