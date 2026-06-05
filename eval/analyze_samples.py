"""Objective audio-feature read on generated samples (no CLAP — ffmpeg8/torchcodec
crashes locally, librosa is fine). Flags 'noise' vs 'structured music'.

Heuristic flags:
  spectral_flatness high (>~0.35) + weak beat  -> noisy / unstructured
  beat_strength (onset autocorr peak) high      -> has rhythm
"""
import glob, os
import numpy as np
import librosa

rows = []
for path in sorted(glob.glob("eval/samples/*.mp3")) + sorted(glob.glob("eval/samples/*.wav")):
    if "SMOKE" in path:
        continue
    y, sr = librosa.load(path, sr=22050, mono=True)
    if y.size < sr:
        continue
    rms = float(np.sqrt(np.mean(y**2)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    # beat strength: normalized autocorr peak (excluding lag 0) in plausible tempo range
    ac = librosa.autocorrelate(onset - onset.mean())
    ac = ac / (ac[0] + 1e-9)
    beat = float(np.max(ac[4:200])) if ac.size > 200 else 0.0
    tempo = float(librosa.feature.tempo(onset_envelope=onset, sr=sr)[0])
    sil = float(np.mean(np.abs(y) < 0.01))
    name = os.path.basename(path).rsplit(".", 1)[0]
    noisy = flat > 0.35 and beat < 0.15
    rows.append((name, rms, flat, cent, beat, tempo, sil, noisy))

rows.sort(key=lambda r: r[2])  # by flatness, musical first
hdr = f"{'sample':28s} {'rms':>6s} {'flat':>6s} {'centHz':>7s} {'beat':>5s} {'tempo':>6s} {'sil%':>5s}  verdict"
print(hdr); print("-" * len(hdr))
for name, rms, flat, cent, beat, tempo, sil, noisy in rows:
    v = "NOISE" if noisy else ("weak" if beat < 0.2 else "structured")
    print(f"{name:28s} {rms:6.3f} {flat:6.3f} {cent:7.0f} {beat:5.2f} {tempo:6.1f} {sil*100:4.0f}%  {v}")
print("\nflat: 0=tonal/musical .. 1=white noise | beat: onset autocorr peak (rhythm) | sil%: near-silence")
