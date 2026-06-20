"""Vocal test: does v8_sing4 @ 97k sing the lyrics at ANY setting?

Generates lofi_pop (female) with the [verse]/[chorus] lyric stream while escalating
lyric_cfg_scale (0 -> 12), plus a lower-cfg variant in case high prompt-adherence
is stepping on the lyric stream. Pure ear test — listen to sweep_out/vocal_test/
and tell me if words ever show up. (No Whisper locally; your ear is the judge.)

  python scripts/vocal_test.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.eval_sampling_sweep import GENRES, _combine, _lyric_field, _features, _librosa_score

# warm_mild ladder (the current default), held fixed — we're isolating lyric_cfg.
TEMP = [1.05, 0.97, 0.88, 0.78, 0.66, 0.54, 0.44, 0.36, 0.3]
TOPK = [120, 84, 58, 38, 24, 15, 10, 6, 4]
TOPP = 0.95

# (cfg_scale, lyric_cfg_scale)
TRIALS = [
    (7.0, 0.0),   # control: no lyric guidance -> expect instrumental
    (7.0, 3.0),   # current default
    (7.0, 6.0),
    (7.0, 9.0),
    (7.0, 12.0),
    (4.0, 9.0),   # lower tag-adherence, strong lyric push
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default="checkpoints/latest.pt")
    ap.add_argument("--genre", default="lofi_pop", choices=list(GENRES.keys()))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out-dir", default="sweep_out/vocal_test")
    a = ap.parse_args()

    os.environ["NANO_CKPT"] = a.ckpt_path
    from server.inference import InferenceEngine
    import soundfile as sf
    import librosa

    eng = InferenceEngine(ckpt_path=a.ckpt_path)
    sr = eng.codec.SAMPLE_RATE
    frames = int(a.seconds * eng.codec.FRAME_RATE_HZ)
    os.makedirs(a.out_dir, exist_ok=True)

    sweet = eng.sweeten_prompt(GENRES[a.genre]["tag"])
    text = _combine(sweet, _lyric_field(a.genre))
    print(f"[vocal-test] {a.genre} @ {a.seconds:g}s, lyric stream:\n  {text[:120]!r}\n")
    cond_emb, cond_lids, cond_lmask = eng._build_conditioning(text)
    print(f"  lyric stream length (phoneme ids): "
          f"{None if cond_lids is None else int(cond_lids.shape[-1])}\n")

    for cfg, lcfg in TRIALS:
        out = eng.model.generate(
            prompt=None, num_new_frames=frames,
            temperature=TEMP, top_k=TOPK, top_p=TOPP,
            text_emb=cond_emb, lyric_ids=cond_lids, lyric_mask=cond_lmask,
            cfg_scale=cfg, lyric_cfg_scale=(lcfg or None),
        )
        y = eng.codec.decode(out[:, 1:].cpu()).squeeze().float().cpu().numpy()
        path = os.path.join(a.out_dir, f"lcfg{lcfg:g}_cfg{cfg:g}.wav")
        sf.write(path, y, sr)
        feat = _features(path)
        # weak vocal-band cue: energy fraction in the 300-3400 Hz speech/formant band
        S = np.abs(librosa.stft(y, n_fft=2048)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
        voc = float(S[(fr >= 300) & (fr <= 3400)].sum() / (S.sum() + 1e-9))
        print(f"  cfg{cfg:g} lyric_cfg{lcfg:>4g}  librosa={_librosa_score(feat):.3f} "
              f"sil={feat['sil']:.2f} rms={feat['rms']:.3f} voc-band={voc*100:.0f}%  -> {path}")

    print(f"\nLISTEN: {a.out_dir}/   does any clip actually sing the words?")


if __name__ == "__main__":
    main()
