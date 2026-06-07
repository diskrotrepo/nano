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
# 9x9 cross product. Each is a (temperature, top_k) pair, scalar or length-9. ──
PROFILES = {
    "HOT_FLAT":    (0.9, 50),
    "COLD_LADDER": ([0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25],
                    [120, 90, 70, 50, 36, 26, 18, 12, 8]),
    "WARM_LADDER": ([1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42],
                    [120, 90, 70, 50, 36, 26, 18, 12, 8]),
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
