"""Guard the per-codec data_subdir isolation.

Re-tokenizing the corpus to a second codec (DAC alongside SpectroStream) works by
re-rooting every token-domain path under ``/tokens/<data_subdir>``. The danger is
structural: a ``.pt`` filename is IDENTICAL across codecs — only the tensor's
codebook depth differs (DAC 9 @86 Hz vs SpectroStream 24/32 @25 Hz). So a stage
that forgets to thread ``data_subdir`` does not error; it silently writes one
codec's tokens into the other's tree, and the packer happily builds a mixed,
unusable corpus on top of the survivor.

These tests fail loudly if a future change drops the parameter from a stage, or
adds a codec-scoped stage to the orchestrator without passing it through.

See diskrot/modal_train.py:301 for the matching train-side flag.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DISKROT = REPO / "diskrot"

# (module, function) pairs that write or read token-domain data and therefore
# MUST be per-codec. Adding a codec-scoped stage without adding it here is the
# gap this test exists to close.
CODEC_SCOPED = [
    ("modal_tokenize.py", "run_tokenize"),
    ("modal_tokenize.py", "list_pending"),
    ("modal_melody.py", "orchestrate"),
    ("modal_melody.py", "list_pending"),
    ("modal_pack_cache.py", "pack_remote"),
    ("modal_pack_cache.py", "pack_append_remote"),
    ("modal_key_detect.py", "detect_remote"),
    ("modal_wave_cleanup.py", "cleanup_remote"),
]

# Stages the ingest orchestrator dispatches that are codec-scoped: each must be
# handed data_subdir, or that stage silently operates on the wrong corpus.
ORCHESTRATED_CODEC_STAGES = ["tokenize", "melody", "pack", "cleanup", "key_detect"]


def _module(name: str) -> ast.Module:
    return ast.parse((DISKROT / name).read_text())


def _func(mod: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


@pytest.mark.parametrize("module,func", CODEC_SCOPED,
                         ids=[f"{m}:{f}" for m, f in CODEC_SCOPED])
def test_codec_scoped_fn_takes_data_subdir(module: str, func: str) -> None:
    fn = _func(_module(module), func)
    args = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "data_subdir" in args, (
        f"{module}:{func} must accept data_subdir — without it this stage always "
        f"resolves the default /tokens root and would write one codec's data into "
        f"another codec's tree (filenames are identical across codecs)."
    )


@pytest.mark.parametrize("module,func", CODEC_SCOPED,
                         ids=[f"{m}:{f}" for m, f in CODEC_SCOPED])
def test_data_subdir_defaults_to_empty(module: str, func: str) -> None:
    """Default MUST be '' so every existing (SpectroStream) invocation is
    byte-identical — the isolation is opt-in, never a silent relocation."""
    fn = _func(_module(module), func)
    names = [a.arg for a in fn.args.args]
    defaults = fn.args.defaults
    idx = names.index("data_subdir") - (len(names) - len(defaults))
    assert idx >= 0, f"{module}:{func} data_subdir must have a default"
    node = defaults[idx]
    assert isinstance(node, ast.Constant) and node.value == "", (
        f"{module}:{func} data_subdir must default to '' (the existing /tokens "
        f"layout), got {ast.dump(node)}"
    )


def test_ingest_wave_threads_data_subdir_to_every_codec_stage() -> None:
    """The orchestrator dispatches stages by name; a stage that doesn't receive
    data_subdir runs against the wrong corpus root while reporting success."""
    src = (DISKROT / "modal_ingest_wave.py").read_text()
    tree = ast.parse(src)
    fn = _func(tree, "ingest_wave")

    seen: dict[str, bool] = {}
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "run" and node.args):
            continue
        stage = node.args[0]
        if not (isinstance(stage, ast.Constant) and stage.value in ORCHESTRATED_CODEC_STAGES):
            continue
        seen[stage.value] = any(kw.arg == "data_subdir" for kw in node.keywords)

    missing = [s for s in ORCHESTRATED_CODEC_STAGES if not seen.get(s)]
    assert not missing, (
        f"ingest_wave dispatches these codec-scoped stages without data_subdir: "
        f"{missing}. They would operate on the default /tokens root regardless of "
        f"--data-subdir."
    )


def test_ingest_wave_status_is_per_codec() -> None:
    """status.json must live under the codec root. Sharing it means a second-codec
    ingest reads the first codec's status, sees every stage 'done', skips the
    entire pipeline, and prints ALL STAGES COMPLETE — a perfect silent no-op."""
    fn = _func(ast.parse((DISKROT / "modal_ingest_wave.py").read_text()), "ingest_wave")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "status_dir"):
            assert "data_subdir" in ast.dump(node.value), (
                "status_dir must incorporate data_subdir, else a second-codec "
                "ingest no-ops against the first codec's status.json"
            )
            return
    raise AssertionError("status_dir assignment not found in ingest_wave")


def test_prepare_is_gateable() -> None:
    """prepare mutates the SHARED R2 corpus (apply=True, and the quality gate
    DELETES files). A re-codec pass must be able to skip it, or it can delete
    audio the other codec's pack still references."""
    fn = _func(ast.parse((DISKROT / "modal_ingest_wave.py").read_text()), "ingest_wave")
    args = [a.arg for a in fn.args.args]
    assert "with_prepare" in args, (
        "ingest_wave must expose with_prepare — prepare is destructive to the "
        "shared corpus and must be skippable on a re-codec pass"
    )


def test_cleanup_refuses_drop_mp3_with_data_subdir() -> None:
    """The mp3s are shared across codecs; dropping them during a second-codec
    pass would strand every other codec's corpus."""
    src = (DISKROT / "modal_wave_cleanup.py").read_text()
    assert "drop_mp3 and data_subdir" in src, (
        "modal_wave_cleanup must refuse --drop-mp3 together with --data-subdir"
    )
