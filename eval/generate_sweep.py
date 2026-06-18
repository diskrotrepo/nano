"""Genre x settings generation sweep on the current best checkpoint.

Writes 10s mp3 samples to eval/samples/. Set SMOKE=1 for a single short gen
to validate the pipeline before the full run.

Per-codebook ladders (recommended for this undertrained-ish ckpt): cb0 samples
hot, later codebooks model fine residuals and must sample cold or they hiss.
"""
import os, time, json
import torch
from server.inference import InferenceEngine

OUT = "eval/samples"
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"

eng = InferenceEngine(ckpt_path="checkpoints/best.pt")

# descending ladders, length 9
TEMP_LADDER = [1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42]
TOPK_LADDER = [120, 90, 70, 50, 36, 26, 18, 12, 8]
TOPP = 0.95

# (well-represented genres + deliberately thin ones the user expects to be noise)
GENRES = [
    ("electronic_edm", "An energetic electronic EDM track with pulsing synth leads, a four-on-the-floor beat, deep bass and bright arpeggios. Instrumental, danceable."),
    ("techno",         "A driving techno track with a relentless kick drum, hypnotic synth stabs, analog bass and shimmering hi-hats. Dark, hypnotic, instrumental."),
    ("chiptune",       "An upbeat chiptune 8-bit video game track with square wave melodies, fast arpeggios and lo-fi drums. Energetic and nostalgic."),
    ("ambient_cinematic","A cinematic ambient piece with lush evolving synth pads, soft strings and a slow emotional swell. Atmospheric, no drums."),
    ("hiphop",         "A laid-back hip hop beat with a boom-bap drum loop, jazzy piano sample, deep 808 bass and vinyl crackle. Instrumental."),
    ("rock",           "An energetic rock song with distorted electric guitars, driving drums, bass guitar and a male vocal. Powerful and anthemic."),
    ("metal",          "An aggressive heavy metal track with palm-muted distorted guitars, double-kick drums, growling male vocals and fast riffing."),
    ("pop_female",     "An upbeat pop song with a catchy female vocal, bright synths, punchy drums and a danceable groove. Polished and radio-friendly."),
    ("classical_piano","A gentle classical solo piano piece, expressive and emotional, with delicate melodic phrasing. Acoustic, no drums."),
    ("jazz",           "A smooth jazz tune with brushed drums, upright bass, warm piano chords and a muted trumpet solo. Relaxed and groovy."),
    ("country",        "A country song with acoustic guitar, banjo, fiddle, pedal steel guitar and a heartfelt male vocal. Warm and twangy."),
    ("lofi",           "A relaxing lo-fi hip hop beat with mellow jazzy keys, soft drums, warm bass and vinyl crackle. Chill and instrumental."),
]

# settings sweep applied to two genres (one well-rep, one thin) to show effect
SETTINGS = [
    ("ladder_cfg3",  dict(temperature=TEMP_LADDER, top_k=TOPK_LADDER, top_p=TOPP, cfg_scale=3.0)),
    ("ladder_cfg5",  dict(temperature=TEMP_LADDER, top_k=TOPK_LADDER, top_p=TOPP, cfg_scale=5.0)),
    ("flat_t0.9_cfg3", dict(temperature=0.9, top_k=50, top_p=0.95, cfg_scale=3.0)),
    ("cold_cfg1.5",  dict(temperature=[0.8,0.7,0.6,0.5,0.45,0.4,0.35,0.3,0.25], top_k=TOPK_LADDER, top_p=0.9, cfg_scale=1.5)),
]

def gen(name, text, settings, seconds=10.0):
    t0 = time.time()
    audio, mime = eng.generate_audio(seconds=seconds, text=text, **settings)
    ext = "mp3" if mime == "audio/mpeg" else "wav"
    path = f"{OUT}/{name}.{ext}"
    open(path, "wb").write(audio)
    dt = time.time() - t0
    print(f"[{dt:5.1f}s] {path}  ({len(audio)//1024} KB)")
    return path

if SMOKE:
    gen("SMOKE_techno", GENRES[1][1], SETTINGS[0][1], seconds=3.0)
    print("smoke ok")
    raise SystemExit

manifest = []
# 1) one canonical good gen per genre
for name, text in GENRES:
    p = gen(f"g_{name}", text, SETTINGS[0][1])
    manifest.append({"genre": name, "setting": SETTINGS[0][0], "file": p, "prompt": text})

# 2) settings sweep on techno (well-rep) and country (thin)
for gname, gtext in [("techno", GENRES[1][1]), ("country", GENRES[10][1])]:
    for sname, sset in SETTINGS:
        p = gen(f"s_{gname}__{sname}", gtext, sset)
        manifest.append({"genre": gname, "setting": sname, "file": p, "prompt": gtext})

json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
print(f"\n[done] {len(manifest)} samples -> {OUT}/  (+ manifest.json)")
