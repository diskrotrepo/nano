"""Genre/style distribution sweep over tags.json (LP-MusicCaps free-text captions).

The captions rarely name a genre outright ("this is country"); they describe
instrumentation, production, and mood. So we count two things:
  1. explicit genre words when present
  2. instrument / production / mood signals that proxy for genre coverage
Output: ranked hit counts + co-occurrence, written to eval/genre_report.txt.
"""
import json, re
from collections import Counter

d = json.load(open("eval/tags.json"))
descs = []
for v in d.values():
    s = v.get("description", "") if isinstance(v, dict) else str(v)
    descs.append(s.lower())
N = len(descs)
blob_join = "\n".join(descs)

# --- genre / style terms (word-boundary regex, grouped synonyms) ---
GENRES = {
    "rock":          r"\brock\b",
    "pop":           r"\bpop\b",
    "jazz":          r"\bjazz\b",
    "classical":     r"\bclassical\b|\borchestral\b|\bsymphon|\bconcerto\b|\bsonata\b",
    "electronic/edm":r"\belectronic\b|\bedm\b|\bsynth(esi[sz]er| pop|wave)?\b",
    "hip hop/rap":   r"\bhip[- ]?hop\b|\brap\b|\btrap\b|\bboom bap\b",
    "house":         r"\bhouse\b",
    "techno":        r"\btechno\b",
    "trance":        r"\btrance\b",
    "dnb/jungle":    r"\bdrum and bass\b|\bdrum'n'bass\b|\bd&b\b|\bjungle\b",
    "dubstep":       r"\bdubstep\b|\bbass music\b",
    "ambient":       r"\bambient\b|\bdrone\b|\bsoundscape\b",
    "lo-fi":         r"\blo[- ]?fi\b",
    "country":       r"\bcountry\b|\bbluegrass\b|\bhonky[- ]?tonk\b",
    "folk":          r"\bfolk\b|\bsinger[- ]?songwriter\b",
    "blues":         r"\bblues\b",
    "metal":         r"\bmetal\b|\bdeath metal\b|\bblack metal\b|\bdjent\b",
    "punk":          r"\bpunk\b|\bhardcore\b",
    "reggae/dub":    r"\breggae\b|\bdub\b|\bska\b|\bdancehall\b",
    "funk":          r"\bfunk\b",
    "soul/r&b":      r"\bsoul\b|\br&b\b|\brhythm and blues\b|\bmotown\b",
    "disco":         r"\bdisco\b",
    "gospel":        r"\bgospel\b|\bchoir\b|\bworship\b",
    "latin":         r"\blatin\b|\bsalsa\b|\bbossa\b|\bsamba\b|\btango\b|\bcumbia\b|\breggaeton\b|\bflamenco\b",
    "afro":          r"\bafrobeat\b|\bafro[- ]?pop\b|\bhighlife\b|\bamapiano\b",
    "indie/alt":     r"\bindie\b|\balternative\b|\bshoegaze\b|\bpost[- ]?rock\b",
    "cinematic":     r"\bcinematic\b|\bsoundtrack\b|\bfilm score\b|\bepic\b|\btrailer\b",
    "chiptune":      r"\bchiptune\b|\b8[- ]?bit\b|\bchip ?break\b|\bvideo game\b",
    "vaporwave":     r"\bvaporwave\b|\bvapor\b|\bsynthwave\b|\bretrowave\b",
    "experimental":  r"\bexperimental\b|\bavant[- ]?garde\b|\bnoise\b|\bglitch\b",
    "world/ethnic":  r"\bworld music\b|\bsitar\b|\btabla\b|\bkoto\b|\bethnic\b|\bfolk traditional\b",
}

# --- instrument / production proxies (what's actually IN the audio) ---
PROXIES = {
    "male vocal":     r"\bmale (vocal|voice|singer|sing)",
    "female vocal":   r"\bfemale (vocal|voice|singer|sing)",
    "instrumental":   r"\binstrumental\b|\bno vocal|\bno voice|\bwithout vocal",
    "acoustic guitar":r"\bacoustic guitar\b",
    "electric guitar":r"\belectric guitar\b",
    "distorted gtr":  r"\bdistort",
    "piano":          r"\bpiano\b",
    "synth":          r"\bsynth",
    "strings":        r"\bstrings?\b|\bviolin\b|\bcello\b",
    "brass/horns":    r"\bbrass\b|\btrumpet\b|\bsaxophone\b|\bhorn\b",
    "drum machine":   r"\bdrum machine\b|\b808\b|\b909\b|\bprogrammed drum",
    "acoustic drums": r"\bacoustic drum|\blive drum|\bdrum kit\b",
    "banjo/fiddle":   r"\bbanjo\b|\bfiddle\b|\bmandolin\b|\bpedal steel\b|\bslide guitar\b",
    "turntable/dj":   r"\bturntable\b|\bscratch\b|\bsample[ds]?\b|\bbeat\b",
    "low quality/lofi recording": r"\blow quality\b|\bnoisy\b|\bmono\b",
}

def sweep(table):
    out = []
    for name, pat in table.items():
        rx = re.compile(pat)
        c = sum(1 for s in descs if rx.search(s))
        out.append((name, c, 100.0 * c / N))
    out.sort(key=lambda x: -x[1])
    return out

lines = []
def p(s=""):
    lines.append(s); print(s)

p(f"# Genre/style sweep over {N:,} captioned songs (tags.json)")
p(f"# Note: counts are caption keyword hits, not curated labels. A song can hit several.\n")

p("## Explicit genre/style terms (ranked by # of songs mentioning)")
p(f"{'genre':22s} {'songs':>8s} {'% corpus':>9s}")
for name, c, pct in sweep(GENRES):
    bar = "#" * int(pct / 2)
    p(f"{name:22s} {c:>8,d} {pct:>8.2f}%  {bar}")

p("\n## Instrument / production proxies (what the audio actually contains)")
p(f"{'proxy':30s} {'songs':>8s} {'% corpus':>9s}")
for name, c, pct in sweep(PROXIES):
    bar = "#" * int(pct / 3)
    p(f"{name:30s} {c:>8,d} {pct:>8.2f}%  {bar}")

# crude overall mood read
p("\n## Mood signals")
for mood in ["energetic","emotional","aggressive","relaxing|chill|mellow","dark|melancholic|sad","happy|uplifting|joyful","danceable|groovy"]:
    rx = re.compile(mood); c = sum(1 for s in descs if rx.search(s))
    p(f"{mood:28s} {c:>8,d} {100.0*c/N:>7.2f}%")

open("eval/genre_report.txt","w").write("\n".join(lines))
print("\n[written eval/genre_report.txt]")
