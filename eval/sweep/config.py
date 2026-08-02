"""Sweep grid: axes, value sets, eval prompts, stages.

Defaults are PER-MODEL — collapse sensitivity is a function of training state, so
the 287M winner does not transfer to 1.5B. Run the whole sweep per checkpoint.
"""

# ── fixed (held constant so the sweep measures settings, not these) ──────────
SECONDS = 8.0          # long enough for onset-autocorr to find a beat at 86 fps
TOP_P = 0.95           # never the deciding lever in prior runs
CFG_SCALES = [3.0, 4.0, 5.0, 6.0, 7.0]   # sweep the full 3-7 band; collapse
#   sensitivity is per-checkpoint, so don't pre-prune (the v6 "cfg7 too hot"
#   finding does not transfer to v7_1500m — re-measure here).

# ── ladder profiles: temperature + top_k co-vary, so grid the SHAPE, not the
# 9x9 cross product. Each is a (temperature, top_k) pair, scalar or per-codebook
# list. Ladders are authored at K=9 (v8/DAC) and resampled to the target
# checkpoint's codebook count via NANO_SWEEP_K (24 for v9/SpectroStream) — the
# server rejects a per_cb list whose length != the checkpoint's K. ──────────────
import os

SWEEP_K = int(os.environ.get("NANO_SWEEP_K", "9"))


def _resample(vals, k=None):
    """Linearly resample a ladder to k points, preserving shape + endpoints."""
    k = k or SWEEP_K
    n = len(vals)
    if k == n:
        return list(vals)
    out = []
    for i in range(k):
        x = i * (n - 1) / (k - 1)
        lo = int(x)
        hi = min(lo + 1, n - 1)
        out.append(round(vals[lo] + (vals[hi] - vals[lo]) * (x - lo), 4))
    return out


def _resample_topk(vals, k=None):
    return [max(1, int(round(v))) for v in _resample(vals, k)]


PROFILES = {
    "HOT_FLAT":    (0.9, 50),
    "COLD_LADDER": (_resample([0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]),
                    _resample_topk([120, 90, 70, 50, 36, 26, 18, 12, 8])),
    "WARM_LADDER": (_resample([1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42]),
                    _resample_topk([120, 90, 70, 50, 36, 26, 18, 12, 8])),
}

# ── eval prompts: broad genre set so the ranking finds GENERAL-PURPOSE defaults,
# not one vibe's quirks. Full LP-MusicCaps-style sentence descriptions (matches
# the captioner training distribution), no lyrics. Genres favor what the model
# renders well (synthwave / dnb / lofi) plus two contrasting checks. ──────────
PROMPTS = {
    "synthwave":  "A retro 80s synthwave track with lush analog synth pads, a driving gated-reverb drum machine, punchy electronic bass and bright arpeggios. Nostalgic, cinematic, instrumental.",
    "liquid_dnb": "A liquid drum and bass track with fast crisp breakbeats, deep rolling sub bass, warm jazzy chords and smooth atmospheric pads. Energetic, uplifting, instrumental.",
    "lofi":       "A relaxing lo-fi hip hop beat with mellow jazzy keys, soft boom-bap drums, warm bass and vinyl crackle. Chill, nostalgic, instrumental.",
    "techno":     "A driving techno track with a steady four-on-the-floor kick, hypnotic synth stabs, deep bass and tight hi-hats. Dark, energetic, instrumental.",
    "ambient":    "A calm ambient soundscape with slow evolving synth pads, soft drones, gentle texture and no percussion. Spacious, meditative, instrumental.",
}


# ── lyric-conditioned prompts: the same cfg×profile grid is swept over these too,
# so we find non-collapse settings for SUNG generation, not just instrumental.
# tags are caption-style (vocals-forward); lyrics are a clear English line. The
# runner combines them as "tags. lyrics" (server convention) AFTER flattening any
# internal ". " in the tags, so the caption can't spill into the lyric stream. ──
LYRIC_CFG = 3.0   # moderate lyric guidance, held constant while cfg/profile sweep
LYRIC_PROMPTS = {
    "pop_vocal": {
        "tags": "An upbeat pop song with a clear female lead vocal, bright synths, punchy drums and warm bass, catchy and energetic",
        "lyrics": "Hold me close under the city lights tonight, we are electric and we will never fade away",
    },
    "ballad": {
        "tags": "A slow emotional piano ballad with a soft male lead vocal, gentle strings, intimate and warm",
        "lyrics": "I remember every word you said, the quiet way you held my hand in the dark",
    },
}


def all_settings():
    """15 settings: cfg(5) x profile(3). Each is a dict ready for
    InferenceEngine.generate_audio + an id string."""
    out = []
    for cfg in CFG_SCALES:
        for pname, (temp, topk) in PROFILES.items():
            sid = f"cfg{cfg}_{pname}"
            out.append({
                "id": sid,
                "cfg_scale": cfg,
                "profile": pname,
                "temperature": temp,
                "top_k": topk,
                "top_p": TOP_P,
            })
    return out
