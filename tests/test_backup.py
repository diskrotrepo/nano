"""Tests for diskrot/backup.py — the laptop-bundle helpers behind modal_backup.py.

Pure stdlib (tmp files + tar), no Modal."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

from diskrot.backup import (
    build_tar_bytes,
    claude_project_dirs,
    collect_local_paths,
)


def _touch(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_collect_picks_existing_repo_files_and_globs(tmp_path):
    repo = tmp_path / "repo"
    _touch(repo / "README.v9.md", "runbook")
    _touch(repo / ".claude/settings.json", "{}")
    _touch(repo / "eval/a.log", "log-a")
    _touch(repo / "eval/b.log", "log-b")
    _touch(repo / "eval/report.txt", "not a log")  # outside the glob
    pairs = collect_local_paths(repo)
    assert [a for _, a in pairs] == [
        "repo/README.v9.md",
        "repo/.claude/settings.json",
        "repo/eval/a.log",
        "repo/eval/b.log",
    ]


def test_collect_empty_repo_yields_nothing(tmp_path):
    assert collect_local_paths(tmp_path) == []


def test_collect_extra_dirs_land_under_their_basename(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    mem = tmp_path / "memory"
    _touch(mem / "MEMORY.md", "index")
    _touch(mem / "sub/note.md", "note")
    missing = tmp_path / "plans"  # absent → skipped, not an error
    pairs = collect_local_paths(repo, extra_dirs=[mem, missing])
    assert [a for _, a in pairs] == ["memory/MEMORY.md", "memory/sub/note.md"]


def test_tar_roundtrip(tmp_path):
    f = _touch(tmp_path / "eval/a.log", "hello")
    blob = build_tar_bytes([(f, "repo/eval/a.log")])
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        assert tf.getnames() == ["repo/eval/a.log"]
        assert tf.extractfile("repo/eval/a.log").read() == b"hello"


def test_claude_project_dirs_derivation(tmp_path):
    dirs = claude_project_dirs(Path("/Users/x/github/nano"), home=tmp_path)
    assert dirs[0] == tmp_path / ".claude/projects/-Users-x-github-nano/memory"
    assert dirs[1] == tmp_path / ".claude/plans"
