"""Caption a music corpus using LP-MusicCaps.

Generates natural-language descriptions from audio (e.g. "an upbeat indie rock
track with jangly guitars and male vocals") and saves them as tags.json.

Usage:
    python -m diskrot.auto_tag --corpus /path/to/mp3s --out tags.json
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm

from model.captioner import DURATION, N_SAMPLES, SAMPLE_RATE, load_captioner


def _load_audio_clip(path: str | Path, sr: int = SAMPLE_RATE,
                     duration: int = DURATION) -> np.ndarray:
    """Load audio, take a representative 10-second crop, return float32 array.

    Crops from 25 % into the track to avoid intros/outros.  Songs shorter than
    *duration* are zero-padded.
    """
    n_samples = sr * duration
    audio, _ = librosa.load(path, sr=sr, mono=True)
    if audio.shape[-1] > n_samples:
        offset = int(audio.shape[-1] * 0.25)
        offset = min(offset, audio.shape[-1] - n_samples)
        audio = audio[offset:offset + n_samples]
    if audio.shape[-1] < n_samples:
        pad = np.zeros(n_samples, dtype=np.float32)
        pad[:audio.shape[-1]] = audio
        audio = pad
    return audio.astype(np.float32)


def caption_corpus(
    corpus_dir: str | Path,
    out_path: str | Path,
    device: str = "cpu",
    flush_callback=None,
    flush_every: int = 25,
    ckpt_path: str | None = None,
    limit: int | None = None,
) -> None:
    """Caption MP3s in *corpus_dir* and write tags.json."""
    corpus_dir = Path(corpus_dir)
    out_path = Path(out_path)

    mp3s = sorted(corpus_dir.glob("*.mp3"))
    if not mp3s:
        raise SystemExit(f"No mp3s found in {corpus_dir}")
    if limit is not None:
        mp3s = mp3s[:limit]

    print(f"found {len(mp3s)} mp3s | device: {device}")
    print("loading LP-MusicCaps captioner...")
    model = load_captioner(device=device, ckpt_path=ckpt_path)

    tags: dict[str, dict] = {}
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"loaded {len(existing)} existing captions from {out_path}")

    n_done, n_skipped, n_failed = 0, 0, 0
    pbar = tqdm(mp3s, desc="captioning", unit="file")
    for mp3 in pbar:
        key = mp3.stem
        if key in existing:
            tags[key] = existing[key]
            n_skipped += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)
            continue
        try:
            audio = _load_audio_clip(mp3)
            audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
            captions = model.generate(audio_t, num_beams=5)
            description = captions[0]
        except Exception as e:
            tqdm.write(f"FAILED {mp3.name}: {e}")
            n_failed += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)
            continue

        tags[key] = {"description": description}
        n_done += 1
        pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)

        if n_done % flush_every == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(tags, indent=2))
            if flush_callback is not None:
                flush_callback()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tags, indent=2))
    print(f"\ncaptioned: {n_done}  skipped: {n_skipped}  failed: {n_failed}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--out", type=str, default="./tags.json")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=None, help="Max files to caption")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    caption_corpus(args.corpus, args.out, device=device, limit=args.limit)
