"""Per-codebook sampling probe — an INFERENCE diagnostic for "are my per-codebook
sampling defaults off, and how would I know?"

DAC is a *residual* codec: codebook 0 carries the coarse structure (note/groove),
and each later codebook adds a finer residual. The fine codebooks (≈5–8) encode
near-stochastic high-frequency detail — if you sample them LOOSELY (big top_k /
high temp) they grab noisy tokens and you hear grit/hiss; if you sample the
coarse codebooks too TIGHTLY you get repetition/drone. So a good ladder is
loose→tight (top_k 120→8, temp hot→cold).

This script makes that visible WITHOUT a listening marathon. For each setting it
generates one clip and reports, per codebook:
  - Hnorm    : entropy of the sampled-token histogram, 0..1 (1 = uses all tokens)
  - top%     : how often the single most-used token appears (high → collapse)
  - rep%     : lag-1 repeat rate, token[t]==token[t-1] (high → sticking/drone)
and for the decoded audio: the librosa collapse/beat score + HF energy fraction
(>6 kHz — a proxy for grit) + spectral centroid.

Read it as: fine-codebook Hnorm that stays HIGH while HF% climbs = the ladder is
too loose up top (grit). Coarse-codebook top%/rep% near 1 = too tight (collapse).
Compare your current settings against the loose→tight ladder default.

  python scripts/probe_codebook_sampling.py                 # default: lofi_pop, instrumental, 20s
  python scripts/probe_codebook_sampling.py --mode lyric --genre lofi_indie --seconds 25
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.eval_sampling_sweep import (
    GENRES, _combine, _lyric_field, _features, _librosa_score,
)

# Settings to compare. Each: (temperature, top_k, top_p). The first mirrors the
# user's screenshot (per-cb temp ladder, but top_k FLAT 120 and top_p 0.44); the
# rest restore the loose→tight top_k ladder and a saner top_p to isolate each fix.
_TEMP_DEF = [1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42]
_TOPK_LADDER = [120, 90, 70, 50, 36, 26, 18, 12, 8]
# Warm-but-rolled-off candidates: cb0-2 stay warm (anti-collapse), the FINE
# codebooks (cb3-8, the HF detail) get progressively tighter temp/top_k to tame
# brightness. Goal: lowest spectral centroid / HF% that keeps silence low (no
# collapse). top_p stays 0.95 throughout.
SETTINGS = {
    "default_bright (ref)": (_TEMP_DEF, _TOPK_LADDER, 0.95),
    "warm_mild":   ([1.05, 0.97, 0.88, 0.78, 0.66, 0.54, 0.44, 0.36, 0.30],
                    [120, 84, 58, 38, 24, 15, 10, 6, 4], 0.95),
    "warm_strong": ([1.02, 0.93, 0.82, 0.70, 0.56, 0.44, 0.34, 0.26, 0.20],
                    [110, 70, 46, 28, 16, 9, 6, 4, 3], 0.95),
    "warm_max":    ([1.0, 0.9, 0.78, 0.64, 0.5, 0.38, 0.28, 0.2, 0.16],
                    [100, 60, 38, 22, 12, 7, 4, 3, 2], 0.95),
}


def _cb_stats(tokens: np.ndarray) -> list[dict]:
    """tokens: [K, T] ints. Per-codebook usage entropy / top-token / repeat."""
    out = []
    K, T = tokens.shape
    for k in range(K):
        row = tokens[k]
        counts = np.bincount(row, minlength=1024).astype(float)
        p = counts / counts.sum()
        nz = p[p > 0]
        Hnorm = float(-(nz * np.log(nz)).sum() / np.log(1024))
        top = float(counts.max() / counts.sum())
        rep = float(np.mean(row[1:] == row[:-1])) if T > 1 else 0.0
        out.append({"Hnorm": Hnorm, "top": top, "rep": rep})
    return out


def _hf_fraction(y: np.ndarray, sr: int, cutoff: float = 6000.0) -> float:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tot = S.sum() + 1e-9
    return float(S[freqs > cutoff].sum() / tot)


def main():
    ap = argparse.ArgumentParser(description="Per-codebook sampling probe (inference).")
    ap.add_argument("--ckpt-path", default="checkpoints/latest.pt")
    ap.add_argument("--genre", default="lofi_pop", choices=list(GENRES.keys()))
    ap.add_argument("--mode", default="instrumental", choices=["instrumental", "lyric"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--cfg-scale", type=float, default=7.0)
    ap.add_argument("--lyric-cfg-scale", type=float, default=3.0)
    ap.add_argument("--out-dir", default="sweep_out/cb_probe")
    a = ap.parse_args()

    os.environ["NANO_CKPT"] = a.ckpt_path
    from server.inference import InferenceEngine

    eng = InferenceEngine(ckpt_path=a.ckpt_path)
    sr = eng.codec.SAMPLE_RATE
    frames = int(a.seconds * eng.codec.FRAME_RATE_HZ)

    sweet = eng.sweeten_prompt(GENRES[a.genre]["tag"])
    if a.mode == "instrumental":
        text = _combine(sweet, "[instrumental]")
        lyric_cfg = None
    else:
        text = _combine(sweet, _lyric_field(a.genre))
        lyric_cfg = a.lyric_cfg_scale or None
    print(f"[probe] {a.genre}/{a.mode} {a.seconds:g}s on {eng.device}\n  text -> {text[:90]!r}\n")

    os.makedirs(a.out_dir, exist_ok=True)
    cond_emb, cond_lids, cond_lmask = eng._build_conditioning(text)

    for name, (temp, topk, topp) in SETTINGS.items():
        out = eng.model.generate(
            prompt=None, num_new_frames=frames,
            temperature=temp, top_k=topk, top_p=topp,
            text_emb=cond_emb, lyric_ids=cond_lids, lyric_mask=cond_lmask,
            cfg_scale=a.cfg_scale, lyric_cfg_scale=lyric_cfg,
        )
        tokens = out[:, 1:]  # strip the 1 seed frame (mirrors generate_audio)
        toks_np = tokens.cpu().numpy()
        wav = eng.codec.decode(tokens.cpu())
        y = wav.squeeze().float().cpu().numpy()

        # save a clip to listen to
        import soundfile as sf
        safe = name.split()[0]
        path = os.path.join(a.out_dir, f"{a.genre}_{a.mode}_{safe}.wav")
        sf.write(path, y, sr)

        stats = _cb_stats(toks_np)
        feat = _features(path)
        hf = _hf_fraction(y, sr)
        import librosa
        cen = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        print(f"=== {name} ===")
        print(f"  audio: librosa={_librosa_score(feat):.3f}  beat={feat['beat']:.2f}  "
              f"sil={feat['sil']:.2f}  rms={feat['rms']:.3f}  "
              f"HF>6k={hf*100:.1f}%  centroid={cen:.0f}Hz  -> {path}")
        print(f"  {'cb':>3} {'Hnorm':>6} {'top%':>6} {'rep%':>6}   (cb0=coarse … cb8=fine)")
        for k, s in enumerate(stats):
            bar = "█" * int(s["Hnorm"] * 20)
            print(f"  {k:>3} {s['Hnorm']:>6.3f} {s['top']*100:>5.1f} {s['rep']*100:>5.1f}  {bar}")
        print()


if __name__ == "__main__":
    main()
