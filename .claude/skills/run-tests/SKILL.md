---
name: run-tests
description: >-
  Run the nano pytest suite correctly. Use this skill when the user wants to run
  the tests, run a specific test file, run or skip the slow benchmark tests, or is
  writing or debugging a test and needs the fixtures and conventions.
allowed-tools: Read, Bash
---

# Run the nano tests

pytest is preconfigured in [pyproject.toml](../../../pyproject.toml):
`testpaths = ["tests"]`, `addopts = "-q --tb=short"`, and a `benchmark` marker for
slow perf tests.

## Commands

```bash
pytest                          # everything (from the repo root)
pytest -m "not benchmark"       # skip slow perf tests — the usual dev loop / pre-flight
pytest tests/test_train_loss.py # one file
pytest tests/test_train_loss.py::test_name   # one test
pytest -m benchmark             # only the perf benchmarks
pytest -v                       # verbose (override the quiet default)
```

Run from the repo root so the flat top-level packages (`model`, `diskrot`,
`server`) import correctly. Use the project venv (`.venv/bin/python -m pytest`
or `uv run pytest`). No GPU or real corpus is required — tests use synthetic
tokens.

## Environment gotchas

- **Unset `NANO_CODEC` (or set `=dac`) before running the suite.** A leftover
  `NANO_CODEC=spectrostream` export changes the codec constants module-wide
  (frame rate 25 vs 86, K=32 vs 9) and fails DAC-assuming tests *falsely* —
  the code is fine, the env is wrong.
- **`g2p_en` is not a project dep**, so the lyric/structure tests — including
  the load-bearing train==inference marker guard in
  `test_structure_markers.py` — **silently skip** without it. Install it
  (`uv pip install g2p_en`) when touching anything on the lyric path, then
  check the skip count.

## What's covered

~49 test files (see `ls tests/`), spanning the model core (delay pattern, RoPE,
GPTConfig, qk-norm, cross-KV cache, CFG), the data pipeline (dataset/mmap,
pack + sidecars, tokenize streaming, phonemize, filter/align lyrics, audio
quality/dedup/io), training (loss, param groups, init stability), the
conditioning streams and their **train==inference equivalence guards**
(`test_structure_markers.py`, `test_melody.py`, `test_stem.py`, `test_fim.py`,
`test_chunked_tags.py`), LoRA (`test_lora.py`), MLX backend parity
(`test_mlx_parity.py`), and the server endpoints (generate batch/stream,
sweetener). When you change a conditioning stream, its equivalence guard is
the test that matters.

## Fixtures

[tests/conftest.py](../../../tests/conftest.py) provides **`synth_tokens_dir`** — a
factory fixture that writes synthetic int16 token `.pt` files into a tmp dir (params
include `n_files` for the too-short-skip path). Use it instead of a real corpus.

## Writing a new test

- Put it in `tests/`, named `test_*.py`.
- Reuse `synth_tokens_dir` rather than touching `nano-tokens` or real MP3s.
- Mark anything slow/perf with `@pytest.mark.benchmark` so the default dev loop
  (`pytest -m "not benchmark"`) stays fast. (No benchmark tests exist yet — this is
  the convention for adding one.)

## Tip

`pytest -m "not benchmark"` is a good fast pre-flight before launching a long
**add-songs** or **train-model** run — it exercises the data/pipeline code on
synthetic tokens without spending GPU time.
