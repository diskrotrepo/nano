"""Sanity-check the chosen default sampling ladder (warm_mild) across all 4 lofi
genres in BOTH instrumental and lyric modes — so the default isn't a sample of
one. Renders one clip per genre/mode at the production cfg / lyric_cfg and prints
quick health metrics. Listen to sweep_out/default_check/ — especially the
*_lyric.wav clips (does it sing the [verse]/[chorus] intelligibly?).

  python scripts/check_default.py
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

# The current default ("warm_mild"): warm coarse codebooks (anti-collapse), fine
# codebooks tightened to tame brightness. Keep in sync with server/main.py.
WARM_MILD = ([1.05, 0.97, 0.88, 0.78, 0.66, 0.54, 0.44, 0.36, 0.3],
             [120, 84, 58, 38, 24, 15, 10, 6, 4], 0.95)


def main():
    ap = argparse.ArgumentParser(description="Sanity-check the default ladder across genres + lyrics.")
    ap.add_argument("--ckpt-path", default="checkpoints/latest.pt")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--cfg-scale", type=float, default=7.0)
    ap.add_argument("--lyric-cfg-scale", type=float, default=3.0)
    ap.add_argument("--out-dir", default="sweep_out/default_check")
    a = ap.parse_args()

    os.environ["NANO_CKPT"] = a.ckpt_path
    from server.inference import InferenceEngine
    import soundfile as sf
    import librosa

    eng = InferenceEngine(ckpt_path=a.ckpt_path)
    sr = eng.codec.SAMPLE_RATE
    frames = int(a.seconds * eng.codec.FRAME_RATE_HZ)
    temp, topk, topp = WARM_MILD
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[check] warm_mild default across {len(GENRES)} genres x [instrumental, lyric] "
          f"@ {a.seconds:g}s (cfg {a.cfg_scale:g}, lyric_cfg {a.lyric_cfg_scale:g})\n")

    for genre in GENRES:
        sweet = eng.sweeten_prompt(GENRES[genre]["tag"])
        for mode in ("instrumental", "lyric"):
            if mode == "instrumental":
                text = _combine(sweet, "[instrumental]")
                lcfg = None
            else:
                text = _combine(sweet, _lyric_field(genre))
                lcfg = a.lyric_cfg_scale or None
            cond_emb, cond_lids, cond_lmask = eng._build_conditioning(text)
            out = eng.model.generate(
                prompt=None, num_new_frames=frames,
                temperature=temp, top_k=topk, top_p=topp,
                text_emb=cond_emb, lyric_ids=cond_lids, lyric_mask=cond_lmask,
                cfg_scale=a.cfg_scale, lyric_cfg_scale=lcfg,
            )
            y = eng.codec.decode(out[:, 1:].cpu()).squeeze().float().cpu().numpy()
            path = os.path.join(a.out_dir, f"{genre}_{mode}.wav")
            sf.write(path, y, sr)
            feat = _features(path)
            cen = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            print(f"  {genre:18s} {mode:12s} librosa={_librosa_score(feat):.3f} "
                  f"sil={feat['sil']:.2f} rms={feat['rms']:.3f} beat={feat['beat']:.2f} "
                  f"centroid={cen:.0f}Hz -> {path}")

    print(f"\nlisten: {a.out_dir}/   (especially the *_lyric.wav clips)")


if __name__ == "__main__":
    main()
