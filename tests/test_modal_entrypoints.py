"""Guard: every Modal pipeline wrapper follows the spawn + --detach pattern.

Fire-and-forget on Modal needs BOTH pieces — ``.spawn()`` so the local
entrypoint exits immediately, and ``modal run --detach`` so the ephemeral app
isn't auto-stopped when the entrypoint completes. Either alone fails: a
blocking ``.remote()`` dies with the terminal even under --detach, and a
``.spawn()`` without --detach is killed when the app stops at entrypoint exit.

The contract pinned down here, for every ``diskrot/modal_*.py`` with a
``@app.local_entrypoint()``:
- pipeline wrappers call ``.spawn()`` (never a blocking ``.remote()``) in the
  entrypoint, and their module docstring documents ``modal run --detach``
- the short interactive tools that intentionally block (print a report or a
  small result inline) are listed in BLOCKING_OK — adding a new wrapper that
  blocks without listing it here is a test failure, so the choice is conscious
- every ``modal run ... diskrot/modal_X.py`` invocation in the docs
  (CLAUDE.md, README*.md, skills) carries ``--detach`` iff X is a detached
  wrapper — so the docs can't drift from the entrypoint's actual behavior
  (this is exactly how modal_filter_lyrics.py drifted in 2026-06)
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Wrappers that intentionally block: quick queries/maintenance whose whole
# point is the report printed to the local terminal. Everything else must
# spawn. Add here ONLY for a short interactive tool, never a corpus sweep.
BLOCKING_OK = {
    "modal_clean_corrupt.py",   # one-off maintenance sweep, inline report
    "modal_export_ckpt.py",     # seconds — prints the pull command
    "modal_inspect_ckpts.py",   # seconds — prints the checkpoint table
    "modal_merge_lora.py",      # minutes — prints the pull command
    "modal_test_transcribe.py", # diagnostic probe, inline output
    "modal_wave_cleanup.py",    # quick per-wave inode reclaim; dry-run reports inline
    "modal_shard_stores.py",    # one-time JSON->shards migration, seconds, inline
    "modal_spectrostream_spike.py",  # v9 codec spike — interactive A/B, inline report
    "modal_duration_audit.py",       # quick read-only duration histogram, inline
    "modal_ipa_coverage.py",         # v9 IPA-coverage spike — read-only, inline report
    "modal_lyrics_lang_audit.py",    # read-only lyrics language-field audit, inline
    "modal_lyrics_stats.py",         # read-only skip-Demucs gate: wave-vs-rest stats, inline
    "modal_r2_wavify.py",            # dry-run blocks inline; apply path spawns
    "modal_clean_dac_pt.py",         # dry-run blocks inline; apply path spawns
    "modal_copy_conditioning.py",    # ~1k-file codec-subdir copy, seconds; dry-run inline
    "modal_make_test_wave.py",       # server-side R2 copy of ~1k objects, seconds; teardown tool
}

MODAL_RUN_RE = re.compile(
    r"modal run\s+((?:--?\S+\s+)*)(?:diskrot/)?(modal_\w+)\.py"
)


def _modal_wrappers():
    return sorted((REPO / "diskrot").glob("modal_*.py"))


def _entrypoint_funcs(tree):
    """Yield FunctionDefs decorated with @app.local_entrypoint()."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            target = call.func if call else dec
            if isinstance(target, ast.Attribute) and target.attr == "local_entrypoint":
                yield node
                break


def _launch_calls(func):
    """Return the set of Modal launch methods (.spawn/.remote) called in func."""
    used = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"spawn", "remote"}
        ):
            used.add(node.func.attr)
    return used


def _classify(path):
    """-> (has_entrypoint, launch_methods, module_docstring)."""
    tree = ast.parse(path.read_text())
    methods = set()
    funcs = list(_entrypoint_funcs(tree))
    for f in funcs:
        methods |= _launch_calls(f)
    return bool(funcs), methods, ast.get_docstring(tree) or ""


def detached_wrappers():
    """Wrapper filenames whose entrypoint spawns (the fire-and-forget set)."""
    names = set()
    for path in _modal_wrappers():
        has_ep, methods, _ = _classify(path)
        if has_ep and path.name not in BLOCKING_OK and "spawn" in methods:
            names.add(path.name)
    return names


@pytest.mark.parametrize("path", _modal_wrappers(), ids=lambda p: p.name)
def test_entrypoint_launch_style(path):
    has_ep, methods, docstring = _classify(path)
    if not has_ep:  # e.g. modal_serve.py (deployed app, no local entrypoint)
        pytest.skip("no local entrypoint")
    if path.name in BLOCKING_OK:
        return  # intentionally interactive — no constraint

    assert "spawn" in methods, (
        f"{path.name}: pipeline wrapper's local entrypoint must .spawn() the "
        f"remote fn (fire-and-forget). If it is genuinely a short interactive "
        f"tool, add it to BLOCKING_OK in {Path(__file__).name} — consciously."
    )
    assert "remote" not in methods, (
        f"{path.name}: local entrypoint mixes blocking .remote() into a "
        f"detached wrapper — the client then dies with the terminal anyway."
    )
    assert "--detach" in docstring, (
        f"{path.name}: docstring must document `modal run --detach ...` — "
        f".spawn() alone is not walk-away-safe (the ephemeral app is "
        f"auto-stopped when the entrypoint exits)."
    )
    for flags, _name in MODAL_RUN_RE.findall(docstring):
        assert "--detach" in flags, (
            f"{path.name}: docstring shows a `modal run` invocation without "
            f"--detach for a detached wrapper."
        )


def test_docs_match_entrypoint_style():
    """Every documented `modal run ... modal_X.py` agrees with X's pattern."""
    detached = detached_wrappers()
    doc_files = (
        [REPO / "CLAUDE.md"]
        + sorted(REPO.glob("README*.md"))
        + sorted(REPO.glob(".claude/skills/*/SKILL.md"))
        + sorted(REPO.glob(".claude/skills/*/references/*.md"))
    )
    problems = []
    for doc in doc_files:
        if not doc.exists():
            continue
        for i, line in enumerate(doc.read_text().splitlines(), 1):
            for flags, name in MODAL_RUN_RE.findall(line):
                fname = f"{name}.py"
                has_detach = "--detach" in flags
                if fname in detached and not has_detach:
                    problems.append(
                        f"{doc.relative_to(REPO)}:{i} — `modal run {name}.py` "
                        f"missing --detach (its entrypoint spawns)"
                    )
                elif fname in BLOCKING_OK and has_detach:
                    problems.append(
                        f"{doc.relative_to(REPO)}:{i} — `modal run --detach "
                        f"{name}.py` but its entrypoint blocks (--detach is "
                        f"useless there; either spawn it or drop the flag)"
                    )
    assert not problems, "docs drift from entrypoint launch style:\n" + "\n".join(problems)
