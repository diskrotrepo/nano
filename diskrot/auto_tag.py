"""Caption a music corpus into natural-language descriptions (tags.json).

By default uses an audio-LLM (Qwen2-Audio — [model/audio_llm_captioner.py]) over
the WHOLE song to produce a rich, multi-facet description (genre/mood, drums, bass,
instruments, vocals, production, arc), which the chunked-CLAP tag path conditions
on in full. Set ``NANO_CAPTIONER=bart`` to fall back to the legacy single-window
LP-MusicCaps captioner ([model/captioner.py], one ~40-word sentence).

Usage:
    python -m diskrot.auto_tag --corpus /path/to/mp3s --out tags.json
    NANO_CAPTIONER=bart python -m diskrot.auto_tag --corpus ... --out ...
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm


def _captioner_kind() -> str:
    return os.environ.get("NANO_CAPTIONER", "audio_llm").strip().lower()


def _marker(kind: str) -> str:
    """The ``captioner`` stamp for this kind — lets --redo skip already-upgraded
    entries and only re-caption legacy ones (resumable)."""
    if kind == "bart":
        return "bart_v1"
    from model.audio_llm_captioner import CAPTIONER_MARKER

    return CAPTIONER_MARKER


def _load_audio_clip(path: str | Path, sr: int, duration: int) -> np.ndarray:
    """Legacy BART path: a single representative 10-second crop (25 % in)."""
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


def _caption_one(
    model, kind: str, mp3: Path, device: str
) -> tuple[str, str | None, dict[str, str]]:
    """Caption a single song -> ``(description, gender, stems)`` with whichever
    captioner is active. The legacy BART captioner has no gender/stem output, so it
    returns ``(desc, None, {})``; the audio-LLM emits all three."""
    if kind == "bart":
        from model.captioner import DURATION, SAMPLE_RATE

        audio = _load_audio_clip(mp3, SAMPLE_RATE, DURATION)
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
        return model.generate(audio_t, num_beams=5)[0], None, {}
    # audio-LLM (whole-song windows): description + gender + per-stem captions.
    from model.audio_llm_captioner import load_song_windows

    return model.caption_with_gender_and_stems(load_song_windows(str(mp3)))


def _atomic_write_json(out_path: Path, obj: object) -> None:
    """Write JSON via temp + os.replace so a kill mid-write can't truncate the
    target. tags.json is loaded WHOLE at train startup, so a partial write is a
    hard startup failure — every sibling stage (key_detect, transcribe, ...) writes
    this way; auto_tag was the lone bare-write."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, out_path)


def caption_corpus(
    corpus_dir: str | Path,
    out_path: str | Path,
    device: str = "cpu",
    flush_callback=None,
    flush_every: int = 25,
    ckpt_path: str | None = None,
    limit: int | None = None,
    redo: bool = False,
) -> None:
    """Caption MP3s in *corpus_dir* and write tags.json.

    ``redo=True`` re-captions only songs whose existing caption is NOT the current
    format — identified by the ``captioner`` marker, so legacy/short entries are
    upgraded while already-current ones are skipped (a killed --redo resumes). A
    bare run skips every song already in tags.json.
    """
    corpus_dir = Path(corpus_dir)
    out_path = Path(out_path)
    kind = _captioner_kind()
    marker = _marker(kind)

    mp3s = sorted(corpus_dir.glob("*.mp3"))
    if not mp3s:
        raise SystemExit(f"No mp3s found in {corpus_dir}")
    if limit is not None:
        mp3s = mp3s[:limit]

    print(f"found {len(mp3s)} mp3s | captioner: {kind} | device: {device}")
    if kind == "bart":
        from model.captioner import load_captioner
        model = load_captioner(device=device, ckpt_path=ckpt_path)
    else:
        from model.audio_llm_captioner import load_captioner
        model = load_captioner(device=device)

    tags: dict[str, dict] = {}
    existing: dict[str, dict] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"loaded {len(existing)} existing captions from {out_path}")

    n_done, n_skipped, n_failed = 0, 0, 0
    pbar = tqdm(mp3s, desc="captioning", unit="file")
    for mp3 in pbar:
        key = mp3.stem
        cur = existing.get(key)
        is_current = isinstance(cur, dict) and cur.get("captioner") == marker
        # skip if it exists AND (we're not redoing, OR it's already current)
        if cur is not None and not (redo and not is_current):
            tags[key] = cur
            n_skipped += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)
            continue
        try:
            description, gender, stems = _caption_one(model, kind, mp3, device)
        except Exception as e:
            tqdm.write(f"FAILED {mp3.name}: {e}")
            n_failed += 1
            pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)
            continue

        entry = {"description": description, "captioner": marker}
        if gender is not None:
            entry["gender"] = gender
        if stems:
            entry["stems"] = stems
        tags[key] = entry
        n_done += 1
        pbar.set_postfix(done=n_done, skip=n_skipped, fail=n_failed)

        if n_done % flush_every == 0:
            _atomic_write_json(out_path, tags)
            if flush_callback is not None:
                flush_callback()

    _atomic_write_json(out_path, tags)
    print(f"\ncaptioned: {n_done}  skipped: {n_skipped}  failed: {n_failed}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, required=True)
    p.add_argument("--out", type=str, default="./tags.json")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit", type=int, default=None, help="Max files to caption")
    p.add_argument("--redo", action="store_true",
                   help="Re-caption songs already in tags.json (e.g. upgrade to the long format)")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    caption_corpus(args.corpus, args.out, device=device, limit=args.limit, redo=args.redo)
