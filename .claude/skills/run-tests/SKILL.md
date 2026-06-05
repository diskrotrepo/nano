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
`server`) import correctly. No GPU or real corpus is required — tests use
synthetic tokens.

## What's covered (11 files)

- **Delay pattern** — `test_delay_pattern.py` (MusicGen delay build/revert).
- **Dataset** — `test_dataset.py`, `test_dataset_mmap.py`,
  `test_dataset_v2_autodetect.py` (loading, splits, mmap, sharded-layout autodetect).
- **Model config** — `test_gpt_config.py`.
- **RoPE** — `test_rope_equivalence.py`.
- **Packing** — `test_pack_cache.py` (sharding + atomic/resumable pack).
- **Tokenization** — `test_tokenize_streaming.py` (DAC encode, prefetch, batching).
- **Training loss** — `test_train_loss.py` (per-codebook loss, cosine LR).
- **Pipeline audit** — `test_pipeline_audit.py` (end-to-end correctness invariants).
- **Prompt sweetener** — `test_prompt_sweetener.py`.

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
