"""Phase 1 sanity check: DAC encode -> decode round-trip.

Picks an mp3 from the corpus, crops a few seconds, encodes through DAC, decodes,
and writes both the original snippet and the reconstruction next to each other
so you can A/B listen and decide if 44.1kHz/9cb is the right codec for this
material.

Usage:
    python -m scripts.dac_roundtrip
    python -m scripts.dac_roundtrip --mp3 /path/to/file.mp3 --seconds 30
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch

from model.codec import DACodec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mp3",
        type=str,
        default=None,
        help="Path to an mp3 (default: random from corpus).",
    )
    p.add_argument("--corpus", type=str, default="/path/to/files/mp3")
    p.add_argument("--out", type=str, default="out/roundtrip.wav")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    if args.mp3 is None:
        mp3s = sorted(Path(args.corpus).glob("*.mp3"))
        if not mp3s:
            sys.exit(f"No mp3s found in {args.corpus}")
        args.mp3 = str(random.choice(mp3s))
    print(f"input mp3: {args.mp3}")
    print(f"device:    {device}")

    codec = DACodec(device=device)
    print(
        f"codec:     sample_rate={codec.SAMPLE_RATE} codebooks={codec.N_CODEBOOKS} "
        f"frame_rate={codec.FRAME_RATE_HZ}Hz"
    )

    y, _ = librosa.load(args.mp3, sr=codec.SAMPLE_RATE, mono=True)
    n_samples = int(args.seconds * codec.SAMPLE_RATE)
    y = y[:n_samples]
    wav = torch.from_numpy(y).unsqueeze(0)  # [1, samples]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    orig_path = out_path.with_name(out_path.stem + "_orig.wav")
    sf.write(str(orig_path), y, codec.SAMPLE_RATE)
    print(f"orig:      {orig_path}")

    tokens = codec.encode(wav)
    n_frames = tokens.shape[1]
    print(
        f"encoded:   shape={tuple(tokens.shape)} "
        f"({n_frames} frames, {n_frames / codec.FRAME_RATE_HZ:.2f}s)"
    )
    print(
        f"token rate: {n_frames * codec.N_CODEBOOKS / args.seconds:.0f} tokens/sec total"
    )

    audio = codec.decode(tokens)
    if audio.dim() > 1:
        audio = audio.squeeze(0)
    sf.write(str(out_path), audio.numpy(), codec.SAMPLE_RATE)
    print(f"recon:     {out_path}")
    print("\nA/B compare the two files above to judge codec fidelity on your material.")


if __name__ == "__main__":
    main()
