"""Transcribe lyrics from a music corpus using Demucs (vocal isolation) + Whisper.

Pipeline: MP3 → Demucs (isolate vocals) → faster-whisper (transcribe) → sharded
lyrics dir (``lyrics/lyrics_NNN.json``, keyed by a stable hash of the song stem).

Usage:
    python -m diskrot.transcribe_lyrics --corpus /path/to/mp3s --out ./lyrics
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Number of shard files lyrics are split across. Keyed by a stable hash of the
# song stem so the same song always lands in the same shard, regardless of
# process or run (builtin hash() is process-salted and must not be used here).
N_LYRIC_SHARDS = 256


def _lyric_bucket(stem: str) -> int:
    """Stable shard index in [0, N_LYRIC_SHARDS) for a song stem."""
    return int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) % N_LYRIC_SHARDS


def _shard_path(lyrics_dir: str | Path, bucket: int) -> Path:
    return Path(lyrics_dir) / f"lyrics_{bucket:03d}.json"


def _atomic_write_json(path: str | Path, obj) -> None:
    """Write JSON to ``path`` atomically (temp file + os.replace).

    Mirrors the temp-then-rename discipline in pack_cache.py: a kill mid-write
    leaves only the ``.tmp`` (which we never read), never a truncated target.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_lyrics_shards(lyrics_dir: str | Path) -> dict[str, dict | None]:
    """Read all ``lyrics_*.json`` shards in ``lyrics_dir`` into one merged dict."""
    lyrics_dir = Path(lyrics_dir)
    merged: dict[str, dict | None] = {}
    if not lyrics_dir.exists():
        return merged
    for shard in sorted(lyrics_dir.glob("lyrics_*.json")):
        merged.update(json.loads(shard.read_text()))
    return merged


def _load_demucs(device: str):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    model = get_model("htdemucs")
    model.to(device)
    model.eval()
    return model, apply_model


def _separate_vocals(demucs_model, apply_fn, audio_path: str | Path, device: str) -> np.ndarray:
    """Isolate vocals from an audio file using Demucs. Returns mono float32 numpy array at 44100Hz."""
    import librosa

    # librosa handles MP3 via soundfile/ffmpeg — no torchcodec dependency
    audio, _ = librosa.load(str(audio_path), sr=44100, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])  # mono to stereo
    wav = torch.from_numpy(audio)  # [2, T]
    wav = wav.unsqueeze(0).to(device)  # [1, 2, T]

    with torch.no_grad():
        sources = apply_fn(demucs_model, wav, device=device)
    # sources shape: [1, n_sources, 2, T] — source order: drums, bass, other, vocals
    vocals = sources[0, -1]  # [2, T] — last source is vocals
    vocals_mono = vocals.mean(dim=0).cpu().numpy()
    return vocals_mono


def _transcribe(whisper_model, vocals: np.ndarray) -> dict | None:
    """Transcribe vocals array at 44100Hz. Returns {text, words} or None if empty."""
    # faster-whisper expects 16kHz
    import librosa

    vocals_16k = librosa.resample(vocals, orig_sr=44100, target_sr=16000)

    segments, info = whisper_model.transcribe(
        vocals_16k,
        language="en",
        word_timestamps=True,
        vad_filter=True,
    )

    words = []
    full_text_parts = []
    for segment in segments:
        full_text_parts.append(segment.text.strip())
        if segment.words:
            for w in segment.words:
                words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})

    full_text = " ".join(full_text_parts).strip()
    if not full_text:
        return None

    return {"text": full_text, "words": words}


def _flush_shards(lyrics_dir: Path, lyrics: dict, dirty: set[int]) -> None:
    """Atomically rewrite only the shard files whose contents changed."""
    by_bucket: dict[int, dict] = {b: {} for b in dirty}
    for key, val in lyrics.items():
        b = _lyric_bucket(key)
        if b in by_bucket:
            by_bucket[b][key] = val
    for b, contents in by_bucket.items():
        _atomic_write_json(_shard_path(lyrics_dir, b), contents)


def transcribe_corpus(
    corpus_dir: str | Path,
    out_path: str | Path,
    device: str = "cuda",
    flush_callback=None,
    flush_every: int = 10,
) -> None:
    """Transcribe lyrics for all mp3s in corpus_dir into a sharded lyrics dir.

    ``out_path`` is a directory holding ``lyrics_NNN.json`` shards (keyed by a
    stable hash of the song stem). Resumes by skipping any stem already present
    in the shards, and flushes only the shards touched since the last flush, so
    a kill mid-write can corrupt at most one shard (atomic temp+rename), never
    the whole corpus.
    """
    corpus_dir = Path(corpus_dir)
    lyrics_dir = Path(out_path)

    mp3s = sorted(corpus_dir.glob("*.mp3"))
    if not mp3s:
        raise SystemExit(f"No mp3s found in {corpus_dir}")

    print(f"found {len(mp3s)} mp3s | device: {device}")

    print("loading Demucs (htdemucs)...")
    demucs_model, apply_fn = _load_demucs(device)

    print("loading Whisper (large-v3)...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("large-v3", device=device, compute_type="float16")

    lyrics = load_lyrics_shards(lyrics_dir)
    if lyrics:
        print(f"loaded {len(lyrics)} existing entries from {lyrics_dir}")

    n_done, n_skipped, n_instrumental, n_failed = 0, 0, 0, 0
    dirty: set[int] = set()
    pbar = tqdm(mp3s, desc="transcribing", unit="file")
    for mp3 in pbar:
        key = mp3.stem
        if key in lyrics:
            n_skipped += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, inst=n_instrumental, fail=n_failed)
            continue
        try:
            vocals = _separate_vocals(demucs_model, apply_fn, mp3, device)
            result = _transcribe(whisper_model, vocals)
        except Exception as e:
            tqdm.write(f"FAILED {mp3.name}: {e}")
            n_failed += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, inst=n_instrumental, fail=n_failed)
            continue

        if result is None:
            lyrics[key] = None  # instrumental
            n_instrumental += 1
        else:
            lyrics[key] = result
            n_done += 1
        dirty.add(_lyric_bucket(key))
        pbar.set_postfix(done=n_done, skip=n_skipped, inst=n_instrumental, fail=n_failed)

        processed = n_done + n_instrumental
        if processed > 0 and processed % flush_every == 0:
            _flush_shards(lyrics_dir, lyrics, dirty)
            dirty.clear()
            if flush_callback is not None:
                flush_callback()

    if dirty:
        _flush_shards(lyrics_dir, lyrics, dirty)
    print(f"\ntranscribed: {n_done}  instrumental: {n_instrumental}  "
          f"skipped: {n_skipped}  failed: {n_failed}")
    print(f"saved → {lyrics_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--out", type=str, default="./lyrics",
                   help="output directory for sharded lyrics_NNN.json files")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    transcribe_corpus(args.corpus, args.out, device=device)
