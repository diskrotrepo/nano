"""Per-song global tempo (BPM) estimation -> tempo.json, the cheap dense tempo source.

The ``<tempo_*>`` header marker only needs a coarse bucket: ``bpm_to_id``
(model.lyric_encoder) collapses a BPM float onto a 14-bucket 10-BPM grid
(``TEMPO_BPM_EDGES``). Producing that float with the full allin1 structure pass is
wildly over-precise — allin1's single-threaded madmom DBN beat decode is the per-song
cost bottleneck, and the structure pass now runs on only a sampled fraction of the
corpus (see ``diskrot.structure._sample_keep``). This module recovers DENSE tempo
coverage for ~free: a librosa global-tempo estimate over the waveform, no
torch/demucs/madmom, one scalar per song.

Output is a single ``tempo.json`` mapping ``{name: {"bpm": float}}`` — the same
shape/role as ``keys.json`` (``diskrot.key_detect``), loaded sparse at train time (a
song without an entry gets ``<unknown_tempo>``). Songs whose tempo can't be estimated
(silent / degenerate) are omitted, never guessed.

The dataset prefers tempo.json over the structure pass's bpm where both exist
(``diskrot.dataset.load_mmap_bundle``), so this is the authoritative tempo source.

Resumable: the output is rewritten atomically after every flush, and a re-run skips
songs already present, so a kill mid-sweep only loses the in-flight flush window.

CLI::

    python -m diskrot.tempo_detect --corpus /path/to/mp3s --out ./tempo.json
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

TEMPO_JSON_NAME = "tempo.json"

# librosa's autocorrelation tempo estimator weights the dominant period by a
# log-normal prior centered on ``start_bpm``; centering it in the common band
# preferentially resolves the notorious half/double-tempo ambiguity toward the
# perceptual tempo. 110 ≈ the middle of the dense 90-140 BPM mass.
_TEMPO_PRIOR_BPM = 110.0
# The band TEMPO_BPM_EDGES spans (bucket 0 = [-inf,60), last = [180,+inf)). We fold
# every estimate into [60,180) by octave (*2 / /2) so a half/double mistake still
# lands in the right 10-BPM bucket via the UNCHANGED bpm_to_id.
_TEMPO_FOLD_LO, _TEMPO_FOLD_HI = 60.0, 180.0


def _octave_fold(bpm: float) -> float:
    """Fold ``bpm`` (already finite & >0) into [_TEMPO_FOLD_LO, _TEMPO_FOLD_HI)."""
    while bpm < _TEMPO_FOLD_LO:
        bpm *= 2.0
    while bpm >= _TEMPO_FOLD_HI:
        bpm /= 2.0
    return bpm


def estimate_tempo(y: np.ndarray, sr: int) -> float | None:
    """Single global BPM for a mono waveform, or None when undetectable.

    Uses ``librosa.feature.rhythm.tempo`` (a static, song-global estimate) with a
    prior centered in the common band, then octave-folds into [60,180). Returns
    None for silent/degenerate input (no detectable beat) so the caller omits the
    song -> ``<unknown_tempo>`` rather than guessing. The folded result always
    maps to a real ``TEMPO_BPM_EDGES`` bucket via ``model.lyric_encoder.bpm_to_id``.
    """
    import librosa

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    # Too short or silent -> no reliable beat (the corpus drops <20s clips, so this
    # only guards pathological/test input).
    if y.size < sr // 2 or not np.all(np.isfinite(y)) or float(np.max(np.abs(y))) < 1e-6:
        return None

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if onset_env.size == 0 or float(onset_env.std()) < 1e-8:
        return None  # flat onset envelope carries no tempo information

    # librosa>=0.10 moved tempo to feature.rhythm (the ``rhythm`` submodule isn't
    # auto-exposed as a librosa.feature attribute, so import it explicitly); fall
    # back to feature.tempo, then the deprecated beat.tempo on very old librosa.
    try:
        from librosa.feature.rhythm import tempo as tempo_fn
    except Exception:  # pragma: no cover - very old librosa
        tempo_fn = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
    tempo = tempo_fn(
        onset_envelope=onset_env, sr=sr,
        start_bpm=_TEMPO_PRIOR_BPM, aggregate=np.mean,
    )
    bpm = float(np.asarray(tempo, dtype=np.float64).reshape(-1)[0])
    if not math.isfinite(bpm) or bpm <= 0:
        return None
    return _octave_fold(bpm)


def _atomic_write_json(payload: dict, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, out_path)


def detect_tempo(
    corpus_dir: str | Path,
    out_path: str | Path | None = None,
    verbose: bool = True,
    flush_every: int = 200,
    commit_cb=None,
) -> Path:
    """Estimate tempo for every mp3 in ``corpus_dir`` into a single ``tempo.json``.

    Decodes each song via ``diskrot.melody._load_audio_file`` (ffmpeg -> mono f32),
    estimates a global BPM, and writes ``{name: {"bpm": float}}``. Resumes by
    skipping any stem already present; flushes atomically every ``flush_every``
    new songs (``commit_cb`` — e.g. a Modal volume commit — runs after each flush).
    Returns the output path.
    """
    from diskrot.melody import SAMPLE_RATE, _load_audio_file

    corpus_dir = Path(corpus_dir)
    out_path = Path(out_path) if out_path is not None else corpus_dir / TEMPO_JSON_NAME

    tempo: dict[str, dict] = {}
    if out_path.exists():
        tempo = json.loads(out_path.read_text())
        if verbose:
            print(f"[tempo] resuming: {len(tempo)} songs already in {out_path}", flush=True)

    mp3s = sorted(corpus_dir.glob("*.mp3"))
    if not mp3s:
        raise SystemExit(f"No mp3s found in {corpus_dir}")

    n_done = n_skipped = n_failed = 0
    since_flush = 0
    t0 = time.time()
    for mp3 in mp3s:
        name = mp3.stem
        if name in tempo:
            continue
        try:
            y = _load_audio_file(str(mp3))
            bpm = estimate_tempo(y, SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001 — one bad file must not kill the sweep
            if verbose:
                print(f"[tempo] FAILED {mp3.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            n_failed += 1
            continue
        if bpm is None:
            n_skipped += 1
            continue
        tempo[name] = {"bpm": bpm}
        n_done += 1
        since_flush += 1
        if since_flush >= flush_every:
            _atomic_write_json(tempo, out_path)
            if commit_cb is not None:
                commit_cb()
            since_flush = 0
            if verbose:
                print(f"[tempo] {len(tempo)} total (+{n_done} new, {n_skipped} no-beat, "
                      f"{n_failed} failed, {time.time() - t0:.0f}s)", flush=True)

    _atomic_write_json(tempo, out_path)
    if commit_cb is not None:
        commit_cb()
    if verbose:
        print(f"[tempo] done: {len(tempo)} songs -> {out_path} "
              f"(+{n_done} new, {n_skipped} no-beat, {n_failed} failed, "
              f"{time.time() - t0:.0f}s)", flush=True)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus", type=str, required=True, help="directory of mp3s")
    p.add_argument("--out", type=str, default=None,
                   help="output path (default: <corpus>/tempo.json)")
    args = p.parse_args()
    detect_tempo(args.corpus, out_path=args.out)
