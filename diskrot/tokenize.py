"""Pre-tokenize the mp3 corpus into cached DAC token files.

Run once on M4 (fast on MPS, ~30-60 min for 339 files). Output is small (~100MB)
and can be uploaded to a Modal volume for training.

Usage:
    python -m diskrot.tokenize
    python -m diskrot.tokenize --corpus /path/to/mp3s --out ./token_cache
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

import librosa
import torch
from tqdm import tqdm

from model.codec import DACodec


TokenizeStatus = Literal["done", "skipped_existing", "skipped_short", "failed"]


@dataclass
class TokenizeResult:
    status: TokenizeStatus
    frames: int = 0
    error: str | None = None


def _load_audio(mp3_path: Path, sample_rate: int) -> torch.Tensor:
    """CPU-side audio load + resample. Safe to call from a background thread."""
    y, _ = librosa.load(str(mp3_path), sr=sample_rate, mono=True)
    return torch.from_numpy(y).unsqueeze(0)


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
        tokens = codec.encode(mp3_path)
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
) -> Iterator[TokenizeResult]:
    """Tokenize a sequence of (mp3_path, out_path) pairs, processing in chunks
    of ``batch_size``. Within each chunk a background thread prefetches the next
    file's audio (CPU librosa load + resample) while the current one is being
    handled; the loaded chunk is then encoded with a single batched DAC forward
    pass to amortize GPU kernel launch overhead. Yields one TokenizeResult per
    input item, in input order.

    Set batch_size=1 to disable encode batching (still benefits from prefetch
    across chunk boundaries less, but useful for memory-constrained devices)."""
    if not items:
        return

    def _maybe_load(
        item: tuple[Path, Path],
    ) -> tuple[tuple[Path, Path], torch.Tensor | None, str, str | None]:
        """Returns (item, audio_or_None, status, error_or_None). status is one of
        'ready', 'skipped_existing', 'failed_load'."""
        mp3_path, out_path = item
        if out_path.exists():
            return item, None, "skipped_existing", None
        try:
            audio = _load_audio(mp3_path, codec.SAMPLE_RATE)
            return item, audio, "ready", None
        except Exception as e:
            return item, None, "failed_load", str(e)

    for chunk_start in range(0, len(items), batch_size):
        chunk = items[chunk_start:chunk_start + batch_size]

        # Phase 1: load every file in the chunk, with a one-ahead background
        # prefetch so the CPU loader overlaps with itself (and, for chunks > 1,
        # the previous chunk's encode tail).
        loaded: list[tuple[tuple[Path, Path], torch.Tensor | None, str, str | None]] = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_maybe_load, chunk[0])
            for i, _ in enumerate(chunk):
                result = future.result()
                if i + 1 < len(chunk):
                    future = pool.submit(_maybe_load, chunk[i + 1])
                loaded.append(result)

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

        # Phase 3: persist + yield in chunk order.
        for idx, (item, _, status, err) in enumerate(loaded):
            _, out_path = item
            if status == "skipped_existing":
                yield TokenizeResult(status="skipped_existing")
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

    codec = DACodec(device=device)
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
