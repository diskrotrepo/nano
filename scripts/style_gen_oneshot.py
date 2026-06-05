"""One-shot: generate a song conditioned on a style-audio reference.

Encodes the style embedding directly from the real .wav path (soundfile
backend) to avoid the engine's .mp3-temp mislabeling + ffmpeg8/torchcodec
crash path. Mirrors InferenceEngine.generate_audio otherwise.
"""
import os
import sys

import torch

from server.inference import InferenceEngine, _encode_audio

STYLE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("STYLE_PATH", "style_reference.wav")
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUT_PATH", "nano_styled.mp3")
SECONDS = 30.0
CFG_SCALE = 3.0
TEMP_LADDER = [0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.3]
TOP_K = 50
TOP_P = 0.95

eng = InferenceEngine(ckpt_path=os.environ.get("NANO_CKPT", "./checkpoints/best.pt"))
if eng.text_encoder is None:
    sys.exit("checkpoint has no text/style conditioning — style_audio unsupported")

# Style embedding straight from the real .wav (no .mp3 temp mislabel).
audio_emb = eng.text_encoder.encode_audio([STYLE_PATH]).to(eng.device)  # [1,1,D]
cond = audio_emb.to(next(eng.model.parameters()).dtype)
print(f"[style] embedding {tuple(cond.shape)} dtype={cond.dtype}")

K = eng.model.cfg.n_codebooks
max_total = eng.model.cfg.max_seq_len - K + 1
new_frames = min(int(SECONDS * eng.codec.FRAME_RATE_HZ), max_total - 1)

with torch.no_grad():
    out = eng.model.generate(
        prompt=None, num_new_frames=new_frames,
        temperature=TEMP_LADDER, top_k=TOP_K, top_p=TOP_P,
        text_emb=cond, text_emb_neg=None, cfg_scale=CFG_SCALE,
    )
out = out[:, 1:]  # strip the random seed frame

wav = eng.codec.decode(out.cpu())
if wav.dim() == 1:
    wav = wav.unsqueeze(0)
body, mime = _encode_audio(wav, eng.codec.SAMPLE_RATE)
with open(OUT_PATH, "wb") as f:
    f.write(body)
print(f"[done] wrote {OUT_PATH} ({len(body)} bytes, {mime})")
