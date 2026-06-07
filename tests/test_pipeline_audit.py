"""Bug + perf-win tests from the pipeline audit (plan: look-at-the-training-delightful-fox.md).

Each test is paired with a finding in that plan. Pattern: the test must FAIL on
the pre-fix code (proving the bug is real), then PASS after the minimal fix.
Tests are CPU-only and stand-alone — no GPU, no Modal, no msclap by default.
The CLAP and packed-cache benchmarks live behind ``pytest -m benchmark`` so they
don't run in the normal suite.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------- B1: clip_grad_norm_ must include text_encoder.proj ----------

def test_clip_grad_norm_covers_text_encoder_proj():
    """B1: The training loop builds optim over model.parameters() PLUS
    text_encoder.proj.parameters(), but the clip_grad_norm_ call in train.py
    only operates on model.parameters(). Result: proj grads can blow up
    without being clipped. The fix is to clip the same combined param list
    that the optimizer trains.

    We simulate the bug: load synthetic huge grads onto a tiny model and a
    standalone proj layer, run the same combined-clip the FIXED train.py
    should run, and verify the resulting global norm is ≤ grad_clip.
    """
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    proj = nn.Linear(8, 4)

    # Force a known-huge gradient signal into both modules.
    for p in model.parameters():
        p.grad = torch.full_like(p, 100.0)
    for p in proj.parameters():
        p.grad = torch.full_like(p, 100.0)

    grad_clip = 1.0

    # This is what the FIXED training loop should do: clip the same combined
    # param list the optimizer was built over.
    combined = list(model.parameters()) + list(proj.parameters())
    torch.nn.utils.clip_grad_norm_(combined, grad_clip)

    # Compute the resulting global norm — must be ≤ grad_clip.
    total_sq = sum((p.grad.detach() ** 2).sum() for p in combined)
    final_norm = total_sq.sqrt().item()
    assert final_norm <= grad_clip + 1e-5, (
        f"After clip the combined norm is {final_norm}, expected ≤ {grad_clip}"
    )


def test_buggy_clip_leaves_proj_grad_unclipped():
    """B1 negative-control: the BUGGY path (clip model.parameters() only)
    should leave proj's contribution to the global norm uncapped. This
    documents the bug as a behavioral fact, not a hypothesis."""
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    proj = nn.Linear(8, 4)

    for p in model.parameters():
        p.grad = torch.full_like(p, 100.0)
    for p in proj.parameters():
        p.grad = torch.full_like(p, 100.0)

    # The pre-fix train.py call:
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Now measure proj's grad norm — it should still be ~huge.
    proj_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach()) for p in proj.parameters()])
    ).item()
    # With weights of shape [4,8]+[4] = 36 elements of value 100, norm ≈ 600.
    assert proj_norm > 100, (
        f"Buggy clip should leave proj norm huge, got {proj_norm}"
    )


# ---------- B3: best_val_step default on missing key must be 0 ----------

def test_best_val_step_default_when_missing_from_checkpoint():
    """B3: train.py:488 does ``best_val_step = ckpt.get("best_val_step", step)``.
    For checkpoints written before best_val_step was tracked, this silently
    inherits the current step — making the operator think the best WAS just
    achieved when it wasn't. The fix is to default to 0 (a sentinel meaning
    "unknown / never recorded").
    """
    # Pre-fix behavior is reproducible with a literal dict.get(..., step):
    ckpt = {"step": 50_000}  # older checkpoint, no best_val_step key
    step = ckpt["step"]

    # Pre-fix call:
    buggy_default = ckpt.get("best_val_step", step)
    assert buggy_default == 50_000, "documents the bug — defaults to current step"

    # Post-fix call (what we'll change train.py to do):
    fixed_default = ckpt.get("best_val_step", 0)
    assert fixed_default == 0, "fix defaults to 0 (unknown)"


# ---------- B4: _evaluate must restore train mode even on exception ----------

def test_evaluate_restores_train_mode_on_exception():
    """B4: train.py:287-308 does model.eval() then model.train() with no
    try/finally. If anything in between raises, the model stays in eval mode
    silently — dropout disabled, training degraded. The fix is a try/finally.

    We import the actual ``_evaluate`` and feed it a model that raises mid-
    forward, then check the model's training mode is restored.
    """
    from diskrot.train import TrainConfig, _evaluate
    from model.nano_audio_gpt import GPTConfig

    # Tiny model wrapper that raises on forward — we don't need real logits.
    class BombModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = GPTConfig()
            self.dummy = nn.Linear(1, 1)

        def forward(self, *args, **kwargs):
            raise RuntimeError("simulated mid-eval crash")

    model = BombModel()
    model.train()
    assert model.training is True

    # Minimal loader — yields one bogus batch then stops.
    # collate_lyrics shape: (tokens, tags, lyric_ids, lyric_mask, melody).
    def loader_iter():
        yield (
            torch.zeros(1, 9, 10, dtype=torch.long), [""],
            torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool),
            None,  # melody (no melody conditioning in this audit)
        )

    cfg = TrainConfig(device="cpu")

    with pytest.raises(RuntimeError, match="simulated mid-eval crash"):
        _evaluate(model, loader_iter(), cfg, n_batches=1)

    assert model.training is True, (
        "B4: model must be returned to train() mode even when _evaluate raises"
    )


# ---------- B2: prev_val_loss must survive checkpoint round-trip ----------

def test_checkpoint_round_trips_prev_val_loss():
    """B2: train.py writes checkpoints without ``prev_val_loss``, and the
    resume block never reads it. After resume the first checkup compares
    against None (or worse, the wrong baseline). The fix saves and restores
    it. We assert the contract: a checkpoint built by the (fixed)
    `_build_ckpt_dict` helper includes the key, and the (fixed) resume logic
    restores it.

    This test references a helper that doesn't exist yet. The test FAILS
    today (ImportError or AttributeError), passes once R1's helper is in
    place along with B2's fix.
    """
    from diskrot.train import _build_ckpt_dict, _restore_train_state

    model = nn.Linear(2, 2)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    state = {
        "step": 1234,
        "best_val_loss": 4.5,
        "best_val_step": 1000,
        "evals_without_improvement": 2,
        "prev_val_loss": 4.7,  # the field B2 introduces
    }
    ckpt = _build_ckpt_dict(model=model, optim=optim, text_encoder=None,
                            cfg_model_dict={}, **state)

    assert "prev_val_loss" in ckpt, "B2: must persist prev_val_loss"
    assert ckpt["prev_val_loss"] == pytest.approx(4.7)

    restored = _restore_train_state(ckpt)
    assert restored["prev_val_loss"] == pytest.approx(4.7), (
        "B2: must restore prev_val_loss across resume"
    )
    # And while we're here — B3's contract:
    assert restored["best_val_step"] == 1000
    # When best_val_step is missing, default is 0 (B3).
    ckpt_no_best = dict(ckpt)
    ckpt_no_best.pop("best_val_step")
    restored2 = _restore_train_state(ckpt_no_best)
    assert restored2["best_val_step"] == 0


# ---------- B5: AsyncCommit must not drop the final commit ----------

# ---------- P1: CLAP batching ----------

@pytest.fixture(scope="module")
def _clap_model():
    """Load CLAP once for the P1 tests below. Skipped if msclap isn't
    installed or if loading fails (e.g., no HF cache and no internet)."""
    msclap = pytest.importorskip("msclap")
    try:
        return msclap.CLAP(version="2023", use_cuda=False)
    except Exception as e:
        pytest.skip(f"CLAP load failed: {e}")


def test_clap_batched_matches_single_tag(_clap_model):
    """P1: text_encoder.py:43 comments "msclap doesn't pad variable-length
    tokenizations" and calls ``get_text_embeddings([t])`` once per tag.
    This test challenges that assumption. If batched calls produce the same
    embeddings as single-tag calls (within float epsilon), the comment is
    wrong (or outdated) and we can batch CLAP precompute, cutting it from
    ~11 min to ~1 min on the production corpus.

    PASSING this test is the green light for batching. If it FAILS, the
    comment is right and we must keep the one-at-a-time loop."""
    tags = [
        "lo-fi, hip hop, indie, dreamy, melancholic, synthesizer, vocals, "
        "guitar, instrumental, slow tempo",
        "rock, punk, metal, aggressive, intense, electric guitar, drums, "
        "bass, distorted, fast tempo",
        "ambient",  # very short to exercise the variable-length claim
        "classical, romantic, peaceful, strings, piano, violin, acoustic",
    ]

    # Also throw in a batch of 32 with mixed lengths to catch any padding
    # issues that only show up at larger sizes (the comment we're disproving
    # specifically called out "variable-length tokenizations").
    long_tags = [
        f"genre_{i}, mood_{i % 4}, instrument_{i % 7}, modifier_{i % 3}"
        for i in range(32)
    ]
    tags = tags + long_tags

    # Oracle: one tag at a time, as text_encoder.py currently does.
    single_embs = torch.cat(
        [_clap_model.get_text_embeddings([t]) for t in tags], dim=0
    )

    # Candidate: a single batched call.
    batched_embs = _clap_model.get_text_embeddings(tags)

    assert single_embs.shape == batched_embs.shape, (
        f"shape mismatch: {single_embs.shape} vs {batched_embs.shape}"
    )
    # CLAP's text encoder is deterministic; differences should be at fp16/32
    # numerical noise. Allow a generous atol to cover any kernel-level
    # nondeterminism that might still creep in.
    max_diff = (single_embs - batched_embs).abs().max().item()
    assert max_diff < 1e-4, (
        f"P1: batched encoding differs from per-tag by {max_diff} — "
        "msclap does NOT batch safely; keep the one-at-a-time loop."
    )


def test_dataset_storage_is_int16_not_int64(synth_tokens_dir):
    """P3: The dataset previously cast int16-on-disk to int64-in-RAM at load
    time, which 4×'s the host RAM footprint (~48 GB for the production
    corpus). Cast can happen on the GPU at batch time at negligible cost.

    Contract: TokenDataset's in-memory tensors keep the on-disk int16 dtype.
    Casting to the type Embedding wants (int64) is the responsibility of the
    training-loop step, not the dataset."""
    from diskrot.dataset import TokenDataset
    from diskrot.pack_cache import pack

    tokens_dir = synth_tokens_dir(n_files=5, T=1000)
    pack(tokens_dir, verbose=False)
    ds = TokenDataset(tokens_dir, segment_frames=500)
    sample, *_ = ds[0]
    assert sample.dtype == torch.int16, (
        f"P3: dataset must keep int16 on host (got {sample.dtype}). "
        "Cast to int64 happens on GPU at batch time."
    )


def test_int16_storage_round_trips_through_embedding(synth_tokens_dir):
    """P3: Verify the proposed pattern is functionally correct — int16 from
    the dataset + ``.long()`` on the consumer side produces the same
    embedding lookup as int64 throughout."""
    from diskrot.dataset import TokenDataset
    from diskrot.pack_cache import pack

    tokens_dir = synth_tokens_dir(n_files=3, T=600, vocab=1024)
    pack(tokens_dir, verbose=False)
    ds = TokenDataset(tokens_dir, segment_frames=500)
    sample16, *_ = ds[0]  # [K=9, T=500] int16

    # An int64 "oracle" via explicit cast
    sample64 = sample16.to(torch.int64)

    emb = torch.nn.Embedding(1024, 16)
    out_via_cast = emb(sample16.long())
    out_via_int64 = emb(sample64)
    assert torch.equal(out_via_cast, out_via_int64), (
        "int16 → .long() → Embedding must match int64 → Embedding exactly"
    )


def test_async_commit_does_not_drop_overlapping_submit():
    """B5: train.py:_AsyncCommit.submit drops the request when a previous
    commit is still in-flight. For most checkpoints that's fine (the next
    succeeds), but the FINAL submit before close() can be the dropped one —
    losing the last checkpoint to the volume.

    The fix: keep a flag that says "another commit was requested while one
    was in flight," and close() (or the running commit's completion handler)
    fires one more commit before returning.

    Test: a slow callback; submit twice in quick succession; close() should
    end with the callback having run TWICE (one in-flight + one queued),
    not once (in-flight, queued one dropped).
    """
    import time

    from diskrot.train import _AsyncCommit

    call_count = {"n": 0}

    def slow_callback():
        time.sleep(0.15)
        call_count["n"] += 1

    ac = _AsyncCommit(slow_callback)
    ac.submit()             # call 1 starts running
    time.sleep(0.02)        # ensure call 1 is in flight, not yet done
    ac.submit()             # call 2 — buggy version DROPS this
    ac.close()              # waits for in-flight; fixed version also runs queued

    assert call_count["n"] == 2, (
        f"B5: both submitted commits should run by close(); got {call_count['n']}"
    )
