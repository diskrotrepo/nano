"""Pre-tokenize the mp3 corpus into cached DAC token files.

Run once on M4 (fast on MPS, ~30-60 min for 339 files). Output is small (~100MB)
and can be uploaded to a Modal volume for training.

Usage:
    python -m diskrot.tokenize
    python -m diskrot.tokenize --corpus /path/to/mp3s --out ./token_cache
"""
from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

import librosa
import numpy as np
import torch
from tqdm import tqdm

from model.codec import DACodec, get_codec


TokenizeStatus = Literal["done", "skipped_existing", "skipped_short", "failed"]

# Pre-encode loudness normalization target (EBU R128 LUFS). -14 LUFS is the common
# streaming reference (Spotify/YouTube); normalizing here homogenizes the corpus's
# wildly varying source levels so the codec tokens have consistent statistics.
# Env-overridable; the same value must be used across the whole corpus.
_LOUDNORM_LUFS = float(os.environ.get("NANO_LOUDNORM_LUFS", "-14.0"))

# Conservative source-quality gate (checked on the RAW pre-normalization signal,
# where level is meaningful — after loudness-norm everything is ~-14 LUFS). Tuned
# to drop only genuinely broken files: near-silent (would train silence) and
# EGREGIOUSLY clipped (a large fraction pinned at digital full-scale = a corrupt /
# destroyed encode, NOT a merely-loud master). Low-bitrate is intentionally KEPT.
_SILENCE_RMS_DBFS = float(os.environ.get("NANO_SILENCE_RMS_DBFS", "-50.0"))
_CLIP_FRACTION = float(os.environ.get("NANO_CLIP_FRACTION", "0.20"))

# Hard per-file length ceiling, in SECONDS at the codec's sample rate, beyond
# which we SKIP the file pre-encode rather than feed it to the codec.
#
# The SpectroStream (TF/JAX) encoder builds an intermediate feature map whose
# flat element count is LINEAR in input length, and TF computes its CUDA launch
# config (`work_element_count`) as a signed int32. A long enough stereo file
# overflows INT_MAX (2,147,483,647): TF CHECK-fails ("F0000 ...
# gpu_launch_config.h: Check failed: work_element_count >= 0 (-1900404736 vs.
# 0)") and calls abort() -> SIGABRT, killing the whole worker. abort() is
# uncatchable from Python, so this MUST be a pre-encode skip, never a try/except.
#
# MEASURED (diskrot/modal_spectrostream_spike.py::measure, depth=32, A100): every
# length 260..360s encodes cleanly, so the overflow length is ABOVE 360s (6:00)
# — well beyond any real song. Working back from the production abort (2.394e9
# elements => the offending file's ACTUAL decoded length was >~400s), the crash
# came from an mp3 whose ffprobe duration was <=330s (so it passed modal_prepare's
# MAX_DURATION_S=330 cap) but whose REAL audio is >~400s: a corrupt/VBR-metadata
# file, NOT a long song. So the binding constraint is metadata-lying files, not
# song length. We cap at 360s: provably safe (360s encoded without overflow), it
# keeps ALL real music (prepare already drops ffprobe>330s and the guard re-checks
# the TRUE decoded length via audio.shape[-1]), and it skips exactly those rare
# liars. Do NOT lower this toward song lengths — that discards huge amounts of
# legitimate 4-6min music to catch a handful of broken files. DAC (mono, 44.1k,
# much smaller intermediates) never approaches int32; its ceiling is a backstop
# only. Both env-overridable.
_MAX_ENCODE_SECONDS_SS = float(os.environ.get("NANO_MAX_ENCODE_SECONDS_SS", "360.0"))
_MAX_ENCODE_SECONDS_DAC = float(os.environ.get("NANO_MAX_ENCODE_SECONDS_DAC", "420.0"))


def _max_encode_samples(codec) -> int:
    """Max input samples (per channel, at codec.SAMPLE_RATE) we'll encode before
    skipping. SpectroStream's TF int32 launch-config overflow is the binding
    constraint; DAC gets a looser ceiling purely as a leak backstop."""
    is_ss = int(getattr(codec, "N_CHANNELS", 1)) == 2 and int(codec.SAMPLE_RATE) >= 48000
    secs = _MAX_ENCODE_SECONDS_SS if is_ss else _MAX_ENCODE_SECONDS_DAC
    return int(secs * codec.SAMPLE_RATE)


class QualitySkip(Exception):
    """Raised by _load_audio when the conservative quality gate rejects a file
    (near-silent / egregiously clipped). Carries the reason string."""


class LengthSkip(Exception):
    """Raised when a file's encoded length would overflow the codec's int32 CUDA
    launch config (the SpectroStream abort()). A pre-encode skip — abort() can't
    be caught after the fact. Carries a human-readable reason string."""


def _audio_quality_reason(y: np.ndarray, sr: int) -> str | None:
    """'silent' / 'clipped' / None for a RAW [C,samples] or [samples] waveform."""
    if y.size == 0:
        return "silent"
    mono = y.mean(axis=0) if y.ndim == 2 else y
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    if rms <= 0.0 or 20.0 * np.log10(rms + 1e-12) < _SILENCE_RMS_DBFS:
        return "silent"
    if float(np.mean(np.abs(mono) >= 0.9995)) > _CLIP_FRACTION:
        return "clipped"
    return None


@dataclass
class TokenizeResult:
    status: TokenizeStatus
    frames: int = 0
    error: str | None = None


def _normalize_loudness(y: np.ndarray, sr: int) -> np.ndarray:
    """Loudness-normalize a [C, samples] or [samples] float32 waveform to
    _LOUDNORM_LUFS (EBU R128 via pyloudnorm; peak-to--1dBFS fallback if pyloudnorm
    is absent or the measure is non-finite), then guard against clipping. Silent
    input is returned unchanged. Mono-summed measurement so L/R are scaled by the
    same gain (stereo image preserved)."""
    peak0 = float(np.abs(y).max()) if y.size else 0.0
    if peak0 <= 0.0:
        return y
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        mono = y.mean(axis=0) if y.ndim == 2 else y
        loud = meter.integrated_loudness(np.ascontiguousarray(mono))
        if np.isfinite(loud):
            y = y * (10.0 ** ((_LOUDNORM_LUFS - loud) / 20.0))
        else:
            y = y * (10.0 ** (-1.0 / 20.0) / peak0)
    except Exception:
        y = y * (10.0 ** (-1.0 / 20.0) / peak0)  # peak-normalize to -1 dBFS
    peak = float(np.abs(y).max())
    if peak > 1.0:
        y = y / peak  # never clip
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_audio(
    mp3_path: Path, sample_rate: int, n_channels: int = 1, normalize: bool = True,
) -> torch.Tensor:
    """CPU-side audio load + resample + loudness-normalize. Safe in a thread.

    Returns ``[n_channels, samples]`` float32: mono -> [1, N]; stereo -> [2, N]
    (a mono source is duplicated to L=R, so it round-trips to a stable centered
    image)."""
    y, _ = librosa.load(str(mp3_path), sr=sample_rate, mono=(n_channels == 1))
    if n_channels == 2 and y.ndim == 1:
        y = np.stack([y, y], axis=0)  # mono source -> L=R
    # Conservative quality gate on the RAW signal (level is meaningful pre-norm).
    reason = _audio_quality_reason(y, sample_rate)
    if reason is not None:
        raise QualitySkip(reason)
    if normalize:
        y = _normalize_loudness(y, sample_rate)
    if y.ndim == 1:
        y = y[None, :]  # [1, samples]
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))


def _save_tokens(tokens: torch.Tensor, out_path: Path) -> int:
    """Persist tokens as int16 .pt. Returns frame count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # int16 is enough for vocab 1024+1 (max idx 1024 < 32767)
    torch.save(tokens.to(torch.int16), out_path)
    return int(tokens.shape[1])


def tokenize_one_file(
    codec: DACodec,
    mp3_path: Path,
    out_path: Path,
    min_frames: int,
) -> TokenizeResult:
    """Encode a single mp3 to a .pt token file. Skips if out_path already exists
    or the encoded sequence is shorter than min_frames. Returns a result record."""
    if out_path.exists():
        return TokenizeResult(status="skipped_existing")
    try:
        # Route through _load_audio so loudness-norm + stereo handling + the
        # quality gate apply on the single-file path too (encode() given a path
        # would skip them).
        audio = _load_audio(mp3_path, codec.SAMPLE_RATE, codec.N_CHANNELS)
        # Pre-encode length guard: a file long enough to overflow the codec's
        # int32 CUDA launch config (SpectroStream) would abort() the worker —
        # uncatchable — so skip it here, before encode().
        max_samples = _max_encode_samples(codec)
        if int(audio.shape[-1]) > max_samples:
            raise LengthSkip(
                f"{audio.shape[-1] / codec.SAMPLE_RATE:.0f}s > "
                f"{max_samples / codec.SAMPLE_RATE:.0f}s cap (int32 overflow guard)"
            )
        tokens = codec.encode(audio)
    except QualitySkip as e:
        return TokenizeResult(status="skipped_short", error=f"quality:{e}")
    except LengthSkip as e:
        return TokenizeResult(status="skipped_short", error=f"too_long:{e}")
    except Exception as e:
        return TokenizeResult(status="failed", error=str(e))
    if tokens.shape[1] < min_frames:
        return TokenizeResult(status="skipped_short", frames=int(tokens.shape[1]))
    frames = _save_tokens(tokens, out_path)
    return TokenizeResult(status="done", frames=frames)


def tokenize_files_streaming(
    codec: DACodec,
    items: list[tuple[Path, Path]],
    min_frames: int,
    batch_size: int = 4,
    prefetch: int = 8,
) -> Iterator[TokenizeResult]:
    """Tokenize a sequence of (mp3_path, out_path) pairs.

    Two knobs, deliberately independent:
      - ``batch_size`` — how many successfully-loaded audios are handed to one
        ``codec.encode_batch`` call. For DAC this is a real padded GPU forward
        (>1 amortizes kernel launches but multiplies peak GPU memory); for
        SpectroStream ``encode_batch`` just loops per file, so batch_size has no
        GPU/memory effect there.
      - ``prefetch`` — how many files' CPU audio (librosa decode + loudness-norm)
        are loaded *ahead* by a background thread pool. This is what hides decode
        latency behind the GPU/TF encode: while the main thread is blocked in
        ``encode_batch``, the pool keeps the next ~``prefetch`` files decoded and
        waiting, so the (often idle) GPU stays fed. Decoupling it from
        ``batch_size`` means even ``batch_size=1`` (the SpectroStream path, where
        batching buys nothing) still overlaps decode with encode.

    Yields one TokenizeResult per input item, in input order. Encode grouping is
    per ``batch_size`` window of input items: the ready items in a window go into a
    single ``encode_batch`` call (skipped/failed items in the window don't split
    it), keeping output deterministic and order-stable."""
    if not items:
        return

    def _maybe_load(
        item: tuple[Path, Path],
    ) -> tuple[tuple[Path, Path], torch.Tensor | None, str, str | None]:
        """Returns (item, audio_or_None, status, error_or_None). status is one of
        'ready', 'skipped_existing', 'quality_skip', 'length_skip', 'failed_load'."""
        mp3_path, out_path = item
        if out_path.exists():
            return item, None, "skipped_existing", None
        try:
            audio = _load_audio(mp3_path, codec.SAMPLE_RATE, codec.N_CHANNELS)
            # Pre-encode length guard: a file long enough to overflow the codec's
            # int32 CUDA launch config (SpectroStream) would abort() this worker —
            # uncatchable by try/except around encode — so skip it now. Dropping
            # it here means simply no .pt is written (same as a quality skip / a
            # decode failure), which keeps packed-corpus membership consistent.
            max_samples = _max_encode_samples(codec)
            if int(audio.shape[-1]) > max_samples:
                reason = (
                    f"{audio.shape[-1] / codec.SAMPLE_RATE:.0f}s > "
                    f"{max_samples / codec.SAMPLE_RATE:.0f}s cap (int32 overflow guard)"
                )
                return item, None, "length_skip", reason
            return item, audio, "ready", None
        except QualitySkip as e:
            return item, None, "quality_skip", str(e)
        except Exception as e:
            return item, None, "failed_load", str(e)

    n = len(items)
    # Persistent loader pool with a sliding submission window: at most
    # ``prefetch`` loads are kept outstanding ahead of the window being encoded,
    # so decoded audio for upcoming files overlaps the current encode without
    # growing memory past ~(prefetch + batch_size) waveforms.
    with ThreadPoolExecutor(max_workers=max(1, prefetch)) as pool:
        futures: dict[int, Future] = {}
        submitted = 0

        def _submit_through(target: int) -> None:
            nonlocal submitted
            target = min(target, n)
            while submitted < target:
                futures[submitted] = pool.submit(_maybe_load, items[submitted])
                submitted += 1

        for win_start in range(0, n, batch_size):
            win_end = min(win_start + batch_size, n)
            # Submit this window plus a prefetch lookahead so the NEXT window's
            # files decode while this window encodes.
            _submit_through(win_end + prefetch)

            # Phase 1: collect this window's loads, in input order.
            loaded = [futures.pop(idx).result() for idx in range(win_start, win_end)]

            # Phase 2: batched encode for everything that loaded successfully.
            encodable = [(idx, audio) for idx, (_, audio, status, _) in enumerate(loaded)
                         if status == "ready" and audio is not None]
            codes_by_idx: dict[int, torch.Tensor] = {}
            encode_error: str | None = None
            if encodable:
                try:
                    codes_list = codec.encode_batch([audio for _, audio in encodable])
                    for (idx, _), codes in zip(encodable, codes_list):
                        codes_by_idx[idx] = codes
                except Exception as e:
                    # If the whole batch fails (OOM, model error), report each
                    # encodable item as failed so the caller can still make progress.
                    encode_error = str(e)
                # Release cached-but-unallocated GPU memory between batches so a
                # long encoder loop doesn't fragment its way into an OOM ~20
                # files in. No-op on CPU; cheap on CUDA.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Phase 3: persist + yield in window order.
            for idx, (item, _, status, err) in enumerate(loaded):
                _, out_path = item
                if status == "skipped_existing":
                    yield TokenizeResult(status="skipped_existing")
                    continue
                if status == "quality_skip":
                    # Near-silent / egregiously-clipped source — skip (don't tokenize).
                    yield TokenizeResult(status="skipped_short", error=f"quality:{err}")
                    continue
                if status == "length_skip":
                    # Too long for the codec's int32 launch config — skip pre-encode
                    # (would otherwise abort() the worker). Counted as skipped_short;
                    # no .pt written, so packed-corpus membership stays consistent.
                    yield TokenizeResult(status="skipped_short", error=f"too_long:{err}")
                    continue
                if status == "failed_load":
                    yield TokenizeResult(status="failed", error=err)
                    continue
                # status == "ready"
                if encode_error is not None:
                    yield TokenizeResult(status="failed", error=encode_error)
                    continue
                codes = codes_by_idx[idx]
                if codes.shape[1] < min_frames:
                    yield TokenizeResult(status="skipped_short", frames=int(codes.shape[1]))
                    continue
                frames = _save_tokens(codes, out_path)
                yield TokenizeResult(status="done", frames=frames)


def tokenize_corpus(
    corpus_dir: str | Path,
    out_dir: str | Path,
    device: str,
    min_seconds: float = 20.0,
    progress_callback: Callable[[int], None] | None = None,
    callback_every: int = 100,
    batch_size: int = 4,
) -> None:
    """Encode all mp3s in corpus_dir to DAC token files in out_dir."""
    corpus_dir = Path(corpus_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mp3s = sorted(corpus_dir.glob("*.mp3"))
    if not mp3s:
        raise SystemExit(f"No mp3s found in {corpus_dir}")

    codec = get_codec(device=device)  # NANO_CODEC: dac (default) or spectrostream
    min_frames = int(min_seconds * codec.FRAME_RATE_HZ)

    print(f"found {len(mp3s)} mp3s | device: {device}")

    items = [(mp3, out_dir / (mp3.stem + ".pt")) for mp3 in mp3s]

    n_done, n_skipped_existing, n_skipped_short, n_failed = 0, 0, 0, 0
    total_frames = 0
    pbar = tqdm(mp3s, desc="encoding", unit="file")
    for mp3, result in zip(pbar, tokenize_files_streaming(codec, items, min_frames, batch_size=batch_size)):
        if result.status == "skipped_existing":
            n_skipped_existing += 1
        elif result.status == "skipped_short":
            n_skipped_short += 1
        elif result.status == "failed":
            tqdm.write(f"FAILED {mp3.name}: {result.error}")
            n_failed += 1
        else:
            n_done += 1
            total_frames += result.frames

        secs = total_frames / codec.FRAME_RATE_HZ
        pbar.set_postfix(done=n_done, skip=n_skipped_existing + n_skipped_short, fail=n_failed,
                         cached=f"{secs/60:.1f}m")
        if (progress_callback is not None
                and result.status == "done"
                and n_done % callback_every == 0):
            progress_callback(n_done)

    print(f"\ndone:                {n_done}")
    print(f"skipped (existing):  {n_skipped_existing}")
    print(f"skipped (too short): {n_skipped_short}")
    print(f"failed:              {n_failed}")
    if n_done > 0:
        secs = total_frames / codec.FRAME_RATE_HZ
        print(f"total audio cached:  {secs/60:.1f} min ({secs/3600:.2f} hours)")


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, required=True,
                   help="Directory containing mp3 files to tokenize.")
    p.add_argument("--out", type=str, default="./token_cache")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--min-seconds", type=float, default=20.0,
                   help="Skip files shorter than this many seconds of audio.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Files per batched DAC forward pass. Higher = less per-file "
                        "kernel overhead, more peak GPU memory. 1 disables batching.")
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tokenize_corpus(args.corpus, args.out, device,
                    min_seconds=args.min_seconds, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
