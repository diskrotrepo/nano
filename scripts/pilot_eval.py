"""Evaluate the geometry pilots: identical from-scratch generations per checkpoint.

Loads each pilot checkpoint through the real InferenceEngine (sweetener + CFG +
per-cb ladder resampled to the checkpoint's K), generates the same prompt set,
and writes MP3s to nano-output:/pilot_eval/ for local scoring + listening.

    NANO_CODEC=spectrostream modal run scripts/pilot_eval.py
    modal volume get nano-output /pilot_eval ./pilot_eval --force
"""
from __future__ import annotations

import modal

# Env-selected serve image: NANO_CODEC at `modal run` time picks the SS or DAC
# variant (the SS image BAKES NANO_CODEC=spectrostream into the container, so
# a DAC checkpoint must use the DAC image, not just a runtime env flip).
from diskrot.modal_serve import image as _serve_image

app = modal.App("nano-pilot-eval")

ckpts_vol = modal.Volume.from_name("nano-ckpts")
tokens_vol = modal.Volume.from_name("nano-tokens")
out_vol = modal.Volume.from_name("nano-output", create_if_missing=True)

# name -> (tag prompt, lyrics-or-None). The vocal prompt exercises the lyric
# cross-attention path (markers + words) so sweeps also gate the singing axis.
PROMPTS = {
    "techno": (("A driving techno track with a steady four-on-the-floor kick, hypnotic "
                "synth stabs, deep bass and tight hi-hats. Dark, energetic, instrumental."), None),
    "lofi": (("A relaxing lo-fi hip hop beat with mellow jazzy keys, soft boom-bap drums, "
              "warm bass and vinyl crackle. Chill, nostalgic, instrumental."), None),
    "vocal": (("An upbeat pop song with a clear female lead vocal, catchy melody, bright "
               "synths and punchy drums. Radio-ready, energetic."),
              ("[female] [120bpm] [verse] city lights are calling and I answer in the dark "
               "every step I take is like a spark "
               "[chorus] we run all night we never fall we hear the music through the wall")),
}
COLD_T = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
COLD_K = [120, 90, 70, 50, 36, 26, 18, 12, 8]
WARM_T = [1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42]
# profile -> (temps-or-scalar, topks-or-scalar); ladders resampled to the ckpt's K
PROFILES = {
    "COLD": (COLD_T, COLD_K),
    "WARM": (WARM_T, COLD_K),
    "HOT": (0.9, 50),
}


def _resample(vals, k):
    n = len(vals)
    out = []
    for i in range(k):
        x = i * (n - 1) / (k - 1)
        lo = int(x)
        hi = min(lo + 1, n - 1)
        out.append(vals[lo] + (vals[hi] - vals[lo]) * (x - lo))
    return out


@app.function(
    image=_serve_image,
    gpu="H100",
    timeout=3600,
    volumes={"/ckpts": ckpts_vol, "/tokens": tokens_vol, "/outputs": out_vol},
)
def evaluate(ckpts: str = "pilot_a_k24_300m,pilot_b_k16_300m,pilot_c_cb0w5_300m",
             seconds: float = 8.0, cfg_scale: float = 5.0, takes: int = 2,
             profiles: str = "COLD", cfg_scales: str = "") -> list[str]:
    import gc
    import os

    import torch

    os.environ.setdefault("NANO_DEVICE", "cuda")
    os.makedirs("/outputs/pilot_eval", exist_ok=True)
    written: list[str] = []
    for sub in ckpts.split(","):
        sub = sub.strip()
        # "subdir" -> subdir/best.pt; "subdir/file.pt" -> that exact checkpoint
        # (step_*.pt carry LIVE weights; best*.pt carry the EMA snapshot).
        path = f"/ckpts/{sub}" if sub.endswith(".pt") else f"/ckpts/{sub}/best.pt"
        from server.inference import InferenceEngine
        eng = InferenceEngine(ckpt_path=path)
        # Legacy RANDOM seed, not the silence runway: young checkpoints simply
        # continue a silent seed forever (the v8-era collapse mode), which
        # measures the seed instead of the pilot. Every 2B baseline number was
        # random-seeded, so this also keeps the comparison apples-to-apples.
        eng._silence_seed = None
        K = eng.model.cfg.n_codebooks
        print(f"[{sub}] K={K} params={eng.model.num_params()/1e6:.0f}M", flush=True)
        cfg_list = [float(c) for c in cfg_scales.split(",")] if cfg_scales else [cfg_scale]
        for prof in profiles.split(","):
            t_spec, k_spec = PROFILES[prof.strip().upper()]
            temps = t_spec if isinstance(t_spec, float) else [round(v, 4) for v in _resample(t_spec, K)]
            topks = k_spec if isinstance(k_spec, int) else [max(1, int(round(v))) for v in _resample(k_spec, K)]
            for cs in cfg_list:
                for pname, (prompt, lyr) in PROMPTS.items():
                    text = eng.sweeten_prompt(prompt)  # match the /generate endpoint path
                    for t in range(takes):
                        body, mime = eng.generate_audio(
                            seconds=seconds, text=text, cfg_scale=cs,
                            temperature=temps, top_k=topks, top_p=0.95,
                            lyrics=lyr,
                        )
                        ext = "mp3" if "mpeg" in mime else "wav"
                        tag = sub.replace("/", "_").replace(".pt", "")
                        grid_tag = f"_{prof.strip().lower()}_cfg{cs:g}" if (cfg_scales or profiles != "COLD") else (f"_cfg{cs:g}" if cs != 5.0 else "")
                        out = f"/outputs/pilot_eval/{tag}{grid_tag}__{pname}_t{t}.{ext}"
                        with open(out, "wb") as f:
                            f.write(body)
                        written.append(out)
                        print(f"  wrote {out}", flush=True)
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    out_vol.commit()
    return written


@app.local_entrypoint()
def main(ckpts: str = "pilot_a_k24_300m,pilot_b_k16_300m,pilot_c_cb0w5_300m",
         seconds: float = 8.0, cfg_scale: float = 5.0, takes: int = 2,
         profiles: str = "COLD", cfg_scales: str = ""):
    for p in evaluate.remote(ckpts=ckpts, seconds=seconds, cfg_scale=cfg_scale,
                             takes=takes, profiles=profiles, cfg_scales=cfg_scales):
        print(" ", p)
