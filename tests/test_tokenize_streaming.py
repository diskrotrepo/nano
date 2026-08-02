"""Tests for diskrot/tokenize.py — streaming chunked tokenization.

The real DACodec needs a GPU and a network download. A ``FakeCodec`` that
implements the small surface (``SAMPLE_RATE``, ``encode_batch``) is enough to
exercise the chunk/prefetch logic, skip paths, and error handling.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

import diskrot.tokenize as tokmod
from diskrot.tokenize import (
    TokenizeResult,
    tokenize_files_streaming,
    tokenize_one_file,
)


class FakeCodec:
    """Stand-in for DACodec. ``frames_per_file`` controls the codebook length
    returned per input — set per-call by patching the attribute, or fall back
    to a length derived from the audio sample count."""
    SAMPLE_RATE = 44100
    FRAME_RATE_HZ = 86
    N_CODEBOOKS = 9
    N_CHANNELS = 1

    def __init__(self, frames_override: list[int] | None = None, raise_on_encode: bool = False):
        # frames_override: explicit T per file in call order (used to force
        # short outputs). None → derive from audio length the way DAC would.
        self.frames_override = frames_override
        self.raise_on_encode = raise_on_encode
        self.encode_batch_calls = 0
        self.last_batch_size = 0

    def encode(self, source):
        # Single-file path used by tokenize_one_file. Returns shape [9, T].
        if self.raise_on_encode:
            raise RuntimeError("simulated encode failure")
        # source is a path in this codepath; pretend it's "long enough".
        T = (self.frames_override or [200])[0]
        return torch.zeros((self.N_CODEBOOKS, T), dtype=torch.long)

    def encode_batch(self, audios):
        self.encode_batch_calls += 1
        self.last_batch_size = len(audios)
        if self.raise_on_encode:
            raise RuntimeError("simulated batched encode failure")
        out = []
        for i, audio in enumerate(audios):
            if self.frames_override is not None:
                T = self.frames_override[i]
            else:
                # mimic DAC: ceil(samples / 512)
                samples = audio.shape[-1]
                T = (samples + 511) // 512
            out.append(torch.full((self.N_CODEBOOKS, T), i, dtype=torch.long))
        return out


class FakeSSCodec(FakeCodec):
    """SpectroStream-shaped stand-in: stereo, >=48 kHz. ``_max_encode_samples``
    routes these through the (binding) SS int32-overflow ceiling, so the
    pre-encode length guard fires for them."""
    SAMPLE_RATE = 48000
    FRAME_RATE_HZ = 25
    N_CHANNELS = 2


def _make_audio_files(tmp_path: Path, n: int, samples_each: int = 2048) -> list[Path]:
    """Drop ``n`` placeholder audio files in tmp_path. The real librosa-loaded
    audio is replaced by the monkeypatched ``_load_audio``, so the file content
    doesn't matter — they just need to exist."""
    paths = []
    for i in range(n):
        p = tmp_path / f"song_{i:03d}.mp3"
        p.write_bytes(b"\x00")
        paths.append(p)
    return paths


@pytest.fixture
def patched_load_audio(monkeypatch):
    """Bypass librosa: return a deterministic mono tensor for any path."""
    def _fake_load(mp3_path, sample_rate, n_channels=1, normalize=True):
        return torch.zeros((n_channels, 2048), dtype=torch.float32)
    monkeypatch.setattr(tokmod, "_load_audio", _fake_load)
    return _fake_load


# ----- tokenize_one_file -----

def test_one_file_done(tmp_path, patched_load_audio):
    codec = FakeCodec()
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"\x00")
    out = tmp_path / "a.pt"
    result = tokenize_one_file(codec, mp3, out, min_frames=100)
    assert result.status == "done"
    assert result.frames == 200
    assert out.exists()


def test_one_file_skipped_existing(tmp_path, patched_load_audio):
    codec = FakeCodec()
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"\x00")
    out = tmp_path / "a.pt"
    out.write_bytes(b"already here")  # any contents — only existence matters
    result = tokenize_one_file(codec, mp3, out, min_frames=100)
    assert result.status == "skipped_existing"


def test_one_file_short_not_persisted(tmp_path, patched_load_audio):
    codec = FakeCodec(frames_override=[50])
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"\x00")
    out = tmp_path / "a.pt"
    result = tokenize_one_file(codec, mp3, out, min_frames=100)
    assert result.status == "skipped_short"
    assert result.frames == 50
    assert not out.exists()


def test_one_file_codec_failure(tmp_path, patched_load_audio):
    codec = FakeCodec(raise_on_encode=True)
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"\x00")
    out = tmp_path / "a.pt"
    result = tokenize_one_file(codec, mp3, out, min_frames=100)
    assert result.status == "failed"
    assert "simulated encode failure" in (result.error or "")
    assert not out.exists()


# ----- tokenize_files_streaming -----

def test_streaming_empty_input_is_noop(tmp_path, patched_load_audio):
    codec = FakeCodec()
    results = list(tokenize_files_streaming(codec, [], min_frames=100, batch_size=4))
    assert results == []
    assert codec.encode_batch_calls == 0


def test_streaming_yields_done_per_item(tmp_path, patched_load_audio):
    codec = FakeCodec()
    mp3s = _make_audio_files(tmp_path, 3)
    items = [(p, tmp_path / (p.stem + ".pt")) for p in mp3s]
    results = list(tokenize_files_streaming(codec, items, min_frames=1, batch_size=3))
    assert [r.status for r in results] == ["done", "done", "done"]
    for _, out in items:
        assert out.exists()


def test_streaming_results_in_input_order_across_chunks(tmp_path, patched_load_audio):
    """With batch_size=2 over 5 items, the streaming loop processes 3 chunks
    (2+2+1). The yielded TokenizeResult sequence must be in input order
    regardless of chunk boundaries."""
    codec = FakeCodec()
    mp3s = _make_audio_files(tmp_path, 5)
    items = [(p, tmp_path / (p.stem + ".pt")) for p in mp3s]
    results = list(tokenize_files_streaming(codec, items, min_frames=1, batch_size=2))
    assert len(results) == 5
    # Reload each .pt and check the per-item codebook fill value matches the
    # index within its chunk (FakeCodec encodes file i with fill=i in batch).
    # That alone confirms the per-file outputs landed at the right paths.
    for (_, out), _ in zip(items, results):
        assert out.exists()


def test_streaming_skipped_existing_path(tmp_path, patched_load_audio):
    """When out_path already exists, the codec is never called for that item."""
    codec = FakeCodec()
    mp3s = _make_audio_files(tmp_path, 3)
    items = [(p, tmp_path / (p.stem + ".pt")) for p in mp3s]
    # Pre-create the .pt for the middle item.
    items[1][1].write_bytes(b"existing")

    results = list(tokenize_files_streaming(codec, items, min_frames=1, batch_size=3))
    statuses = [r.status for r in results]
    assert statuses == ["done", "skipped_existing", "done"]
    # encode_batch was called once, with only 2 audios in it (the existing one
    # is excluded from the batch).
    assert codec.encode_batch_calls == 1
    assert codec.last_batch_size == 2


def test_streaming_short_outputs_skipped_short(tmp_path, patched_load_audio):
    codec = FakeCodec(frames_override=[10, 200, 10])
    mp3s = _make_audio_files(tmp_path, 3)
    items = [(p, tmp_path / (p.stem + ".pt")) for p in mp3s]
    results = list(tokenize_files_streaming(codec, items, min_frames=100, batch_size=3))
    statuses = [r.status for r in results]
    assert statuses == ["skipped_short", "done", "skipped_short"]
    # The non-short one was persisted.
    assert items[1][1].exists()
    # The short ones were not.
    assert not items[0][1].exists()
    assert not items[2][1].exists()


# ----- length-skip guard (SpectroStream int32-overflow pre-encode skip) -----

def test_one_file_too_long_skipped_short(tmp_path, monkeypatch):
    """A file whose post-resample length exceeds the SS codec's
    ``_max_encode_samples`` ceiling is reported 'skipped_short' (NOT 'failed')
    and writes NO .pt — the pre-encode guard that prevents the uncatchable
    SpectroStream abort()."""
    # Shrink the SS ceiling to 1s (48000 samples) so the fake waveform stays
    # tiny; the module global is read at call time by _max_encode_samples.
    monkeypatch.setattr(tokmod, "_MAX_ENCODE_SECONDS_SS", 1.0)

    def _fake_long_load(mp3_path, sample_rate, n_channels=1, normalize=True):
        # 2s of stereo at 48 kHz -> 96000 samples > 48000 cap.
        return torch.zeros((n_channels, 2 * 48000), dtype=torch.float32)
    monkeypatch.setattr(tokmod, "_load_audio", _fake_long_load)

    codec = FakeSSCodec()
    mp3 = tmp_path / "long.mp3"
    mp3.write_bytes(b"\x00")
    out = tmp_path / "long.pt"
    result = tokenize_one_file(codec, mp3, out, min_frames=1)
    assert result.status == "skipped_short"
    assert "too_long" in (result.error or "")
    assert not out.exists()


def test_streaming_too_long_skipped_short(tmp_path, monkeypatch):
    """Streaming path: an over-length file is skipped pre-encode (status
    'skipped_short', no .pt, not counted as a failure), while a normal file in
    the same chunk still encodes — so the guard never poisons the worker."""
    monkeypatch.setattr(tokmod, "_MAX_ENCODE_SECONDS_SS", 1.0)

    long_name = "song_001.mp3"

    def _selective_load(mp3_path, sample_rate, n_channels=1, normalize=True):
        if mp3_path.name == long_name:
            return torch.zeros((n_channels, 2 * 48000), dtype=torch.float32)  # too long
        return torch.zeros((n_channels, 2048), dtype=torch.float32)
    monkeypatch.setattr(tokmod, "_load_audio", _selective_load)

    codec = FakeSSCodec()
    mp3s = _make_audio_files(tmp_path, 3)
    items = [(p, tmp_path / (p.stem + ".pt")) for p in mp3s]
    results = list(tokenize_files_streaming(codec, items, min_frames=1, batch_size=3))
    statuses = [r.status for r in results]
    assert statuses == ["done", "skipped_short", "done"]
    assert "too_long" in (results[1].error or "")
    # The over-length one wrote no .pt; the others did.
    assert items[0][1].exists()
    assert not items[1][1].exists()
    assert items[2][1].exists()
    # Only the two in-bounds files reached encode_batch.
    assert codec.encode_batch_calls == 1
    assert codec.last_batch_size == 2


def test_max_encode_samples_thresholds():
    """Pin the production length caps + the SS-vs-DAC classifier so a future edit
    to the constants or the is_ss predicate is caught. SS (stereo AND >=48kHz)
    routes through the measured int32 ceiling (360s); everything else gets the
    looser DAC backstop (420s). The multiplier is the codec's own SAMPLE_RATE."""
    # SpectroStream: 360s @ 48kHz — the measured-safe int32 cap.
    assert tokmod._MAX_ENCODE_SECONDS_SS == 360.0
    assert tokmod._max_encode_samples(FakeSSCodec()) == int(360.0 * 48000)
    # DAC: 420s @ 44.1kHz backstop (DAC never approaches int32).
    assert tokmod._MAX_ENCODE_SECONDS_DAC == 420.0
    assert tokmod._max_encode_samples(FakeCodec()) == int(420.0 * 44100)

    # The classifier needs BOTH N_CHANNELS==2 AND SR>=48000. A mono 48kHz codec
    # is NOT SpectroStream -> falls back to the DAC (420s) ceiling.
    class _Mono48k:
        SAMPLE_RATE = 48000
        N_CHANNELS = 1
    assert tokmod._max_encode_samples(_Mono48k()) == int(420.0 * 48000)

    # A codec missing N_CHANNELS entirely defaults (getattr -> 1) to the safe
    # DAC branch rather than mis-detecting as SS.
    class _NoChannels:
        SAMPLE_RATE = 48000
    assert tokmod._max_encode_samples(_NoChannels()) == int(420.0 * 48000)


def test_streaming_batched_encode_failure_marks_all_failed(tmp_path, patched_load_audio):
    """If the whole encode_batch raises, every item that loaded successfully
    in that chunk should be reported as failed (so the caller can still make
    progress)."""
    codec = FakeCodec(raise_on_encode=True)
    mp3s = _make_audio_files(tmp_path, 2)
    items = [(p, tmp_path / (p.stem + ".pt")) for p in mp3s]
    results = list(tokenize_files_streaming(codec, items, min_frames=1, batch_size=2))
    assert [r.status for r in results] == ["failed", "failed"]
    assert "simulated batched encode failure" in (results[0].error or "")
    # Nothing persisted.
    for _, out in items:
        assert not out.exists()


def test_streaming_load_failure_isolated(tmp_path, monkeypatch):
    """If audio load fails for one file, only that one is reported as failed —
    the rest of the chunk still encodes."""
    codec = FakeCodec()
    mp3s = _make_audio_files(tmp_path, 3)
    items = [(p, tmp_path / (p.stem + ".pt")) for p in mp3s]
    bad_path = mp3s[1]

    def _selective_load(mp3_path, sample_rate, n_channels=1, normalize=True):
        if mp3_path == bad_path:
            raise IOError("corrupt mp3")
        return torch.zeros((n_channels, 2048), dtype=torch.float32)
    monkeypatch.setattr(tokmod, "_load_audio", _selective_load)

    results = list(tokenize_files_streaming(codec, items, min_frames=1, batch_size=3))
    assert [r.status for r in results] == ["done", "failed", "done"]
    assert "corrupt mp3" in (results[1].error or "")
    # Encode batch was called once with 2 items (the loadable ones).
    assert codec.encode_batch_calls == 1
    assert codec.last_batch_size == 2


def test_tokenize_result_dataclass_defaults():
    """Sanity: defaults match what callers pattern-match on (frames=0,
    error=None)."""
    r = TokenizeResult(status="done")
    assert r.frames == 0
    assert r.error is None
