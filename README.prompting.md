# Prompting nano

A practical guide to getting good songs out of nano. The centerpiece is the
**[example library](#example-library)** — ready-to-paste field values you can drop
straight into the web UI. The reference sections above it explain *why* the examples
look the way they do.

Format note: examples are written as **web-UI form fields** (`prompt:`, `lyrics:`,
etc.). The same values work over the HTTP API — see the [curl appendix](#curl-appendix)
and the **serve-model** skill for server mechanics.

---

## The mental model

nano conditions on **three independent axes**. They never compete — each one steers a
different thing, and each can be left blank.

| Axis | Field(s) | What it steers | How it's encoded |
|---|---|---|---|
| **Tags** | `prompt` | The *vibe* — genre, instrumentation, mood, production | A natural-language caption → frozen CLAP → one pooled vector |
| **Lyrics + markers** | `lyrics` (+ `gender`/`bpm`) | The *words to sing* and per-song attributes (gender, tempo, key, vocals, section) | A phoneme sequence the decoder cross-attends to |
| **Melody** | `melody_audio` (`/cover`) | The *contour* — the tune to follow | A time-aligned chromagram added per-frame |

Each axis drops independently for classifier-free guidance, so `cfg_scale` (and the
optional `lyric_cfg_scale` / `melody_cfg_scale`) pushes adherence on each axis on its own.

---

## Tags: write a caption, not keywords

CLAP was trained on **LP-MusicCaps prose captions** (~40-word descriptive sentences),
so a terse keyword list conditions *weakly*. Write — or let the sweetener write — a
short paragraph describing the instruments, the groove, the mood, and the production.

**Sweeten is ON by default.** It rewrites a terse `prompt` into caption style with a
small local model before it ever reaches CLAP, so `prompt: "lofi beat to study to"`
becomes something like *"A mellow lofi hip hop instrumental with a jazzy electric piano,
warm bass, soft boom-bap drums and vinyl crackle; relaxed and nostalgic."* You usually
**want this on** — keep it on unless you've hand-written a full caption and want it sent
verbatim (`sweeten: false`).

The register to aim for (real captions from the training set):

> *"The low quality recording features a passionate male vocal singing over punchy kick
> and snare hits, shimmering hi hats, synth pad and groovy bass. It sounds energetic,
> groovy and hypnotic."*

> *"This music is instrumental. The tempo is fast with a groovy bass line, rhythmic
> acoustic guitar strumming. The music is upbeat, catchy, punchy and vivacious."*

Useful vocabulary to reach for: instruments (*electric piano, analog synth pad, distorted
guitar, upright bass, brushed drums, strings*), groove (*steady boom-bap, four-on-the-floor,
driving, laid-back, syncopated*), mood (*energetic, melancholic, dreamy, aggressive, warm,
hypnotic*), production (*lo-fi, vinyl crackle, lush reverb, punchy, mono, tape saturation*).

`negative_prompt` discourages a vibe (it is **not** sweetened) — e.g. `negative_prompt: "muffled, noisy, amateur recording"`.

---

## Lyrics + markers

The `lyrics` field carries the actual words to sing **plus** inline `[bracket]` markers
that set per-song attributes. Markers are **case-insensitive** and
space/dash/underscore-interchangeable (`[a minor]` = `[A-Minor]` = `[a_minor]`).

### The five marker families

| Family | Canonical | Aliases / forms | Default | Examples |
|---|---|---|---|---|
| **Gender** | `[male]` `[female]` | male: `m man men boy guy` · female: `f woman women girl` · `[unknown]`/`[none]` | `<unknown_gender>` | `[female]` |
| **Tempo** | `[120bpm]` | `[120]` · `[tempo:120]` · words `[slow]`(65) `[medium]`/`[mid]`/`[moderate]`(100) `[fast]`(160) | `<unknown_tempo>` | `[85bpm]` `[fast]` |
| **Key** | `[a minor]` | `[Am]` · `[key:Am]` · `[f# major]` · `[Bb]` (flats fold to sharps) · 24 keys | `<unknown_key>` | `[c major]` `[f# minor]` |
| **Vocals** | `[vocals]` `[instrumental]` | vocals: `vocal voice sung` · instrumental: `no vocals`/`no_vocal` | `<vocals>` if words present, else `<unknown_vocals>` | `[instrumental]` |
| **Section** | `[verse]` `[chorus]` | `intro verse chorus bridge outro break inst solo` | `<no_section>` | `[verse]` … `[chorus]` |

### Placement rules (important)

- **Gender, tempo, key, vocals are prefix-only.** Put them at the **start** of the
  lyrics. They're collected into a fixed header — order among them doesn't matter
  (`[female] [85bpm]` ≡ `[85bpm] [female]`); the model always sees
  `<gender> <tempo> <key> <vocals> <section>`. A second one mid-stream is dropped.
- **Only section markers work inline.** Put `[verse]`, `[chorus]`, etc. *between* lines
  to mark structure boundaries as the song unfolds.
- Dedicated **`gender` and `bpm` form fields** do the same thing as the brackets — set
  `gender: female` / `bpm: 85` instead of typing `[female] [85bpm]` if the UI exposes them.

### Worked example

```
lyrics:  [female] [85bpm] [a minor] [verse]
         city lights blur in the falling rain
         i trace your name on a window pane
         [chorus]
         and we run, we run till the morning comes
         chasing every spark of the rising sun
```

This sings female vocals, ~85 BPM, in A minor, with a verse→chorus structure.

**Two marker gotchas:**
- `[m]` and `[f]` are **gender**, not keys. For the keys, write `[c major]`, `[f minor]`, etc.
- `[inst]` is a **section** label (instrumental break), *not* "no vocals". To request no
  vocals use `[instrumental]`.

---

## Sampling (you can usually leave it alone)

`/generate` ships the sweep-winning **WARM ladder** by default — a per-codebook
temperature/top-k ramp that runs the coarse codebooks hot and the fine DAC-residual
codebooks cold. It usually sounds better than one flat temperature, so **the defaults
are a good starting point.**

- **Defaults** (`/generate`): `per_cb_temperature = 1.05,0.98,0.9,0.82,0.74,0.66,0.58,0.5,0.42`,
  `per_cb_top_k = 120,90,70,50,36,26,18,12,8`, `top_p = 0.95`, `cfg_scale = 7.0`.
- **Other endpoints** (`/extend`, `/cover`, `/infill`) default `cfg_scale = 3.0` and a
  flat scalar (`temperature 0.9`, `top_k 50`, `top_p 0.95`).
- `temperature` / `top_k` / `top_p` are **scalars only**. For per-codebook control use
  the separate `per_cb_temperature` / `per_cb_top_k` / `per_cb_top_p` fields — each a
  bare comma-separated list of **9** values (no brackets). A simple decreasing ladder
  like `0.9,0.9,0.7,0.7,0.5,0.5,0.4,0.4,0.3` is a safe hand-tuned alternative.
- **Push a single axis harder:** raise `cfg_scale` for stronger tag adherence;
  `lyric_cfg_scale` (>0) for cleaner words; `melody_cfg_scale` (>0, `/cover`) for tighter
  melody following. Too high gets brittle/robotic — nudge in steps.

---

## Example library

Paste these into the matching fields. Blank fields are omitted. Each notes *why it's
robust*.

### `/generate` — instrumentals

**Lofi study beat**
```
prompt:  mellow lofi hip hop instrumental, jazzy electric piano, warm sub bass, soft
         boom-bap drums, vinyl crackle, relaxed and nostalgic
lyrics:  [instrumental] [85bpm]
```
*Why:* caption-style tags + explicit `[instrumental]` so the model commits to no vocals.

**Ambient pad**
```
prompt:  slow evolving ambient soundscape, lush analog synth pads, deep reverb, no
         drums, calm and meditative, cinematic
lyrics:  [instrumental] [slow]
```
*Why:* "no drums" + slow tempo steers away from beats; ambient is well-represented data.

**Aggressive metal**
```
prompt:  aggressive heavy metal, fast double-kick drums, distorted down-tuned guitars,
         driving bass, intense and powerful
lyrics:  [instrumental] [fast] [e minor]
```
*Why:* fast tempo + minor key reinforce the genre; metal is a strong data cluster.

**Cinematic strings**
```
prompt:  epic cinematic orchestral score, soaring string section, swelling brass,
         timpani hits, dramatic and emotional
lyrics:  [instrumental] [medium] [d minor]
```

**House groove**
```
prompt:  energetic house track, four-on-the-floor kick, crisp hi hats, deep rolling
         bassline, bright synth stabs, danceable and hypnotic
lyrics:  [instrumental] [125bpm]
```

**Acoustic folk**
```
prompt:  warm acoustic folk, fingerpicked steel-string guitar, gentle brushed drums,
         soft upright bass, intimate and earthy
lyrics:  [instrumental] [95bpm] [g major]
```

**Boom-bap hip hop instrumental**
```
prompt:  classic boom-bap hip hop beat, dusty vinyl drum break, chopped soul sample,
         warm bass, head-nodding and gritty
lyrics:  [instrumental] [90bpm]
```

**Jazz trio**
```
prompt:  laid-back jazz trio, brushed drums, walking upright bass, smooth piano
         comping, late-night and smoky
lyrics:  [instrumental] [110bpm]
```

**Synthwave**
```
prompt:  retro synthwave, pulsing analog arpeggios, gated reverb drums, neon synth
         lead, nostalgic and driving
lyrics:  [instrumental] [115bpm] [a minor]
```

### `/generate` — vocal songs

**Female pop ballad**
```
prompt:  tender pop ballad, emotional female vocal, soft piano, swelling strings,
         gentle drums, heartfelt and intimate
lyrics:  [female] [72bpm] [a minor] [verse]
         i still hear the echo of your voice
         in the quiet of the empty room
         [chorus]
         but i'll carry the light you left behind
         till the morning breaks through the gloom
```
*Why:* slow tempo + minor key + verse/chorus structure match the ballad form.

**Male rock anthem**
```
prompt:  uplifting rock anthem, passionate male vocal, driving electric guitars,
         punchy drums, big chorus, energetic and triumphant
lyrics:  [male] [128bpm] [e major] [verse]
         we were born in the static and the noise
         chasing something we could never name
         [chorus]
         so raise it up, let the whole world hear
         we are louder than the fading flame
```

**Rap verse**
```
prompt:  hard-hitting hip hop, confident male rap vocal, booming 808 bass, crisp trap
         hi hats, dark and energetic
lyrics:  [male] [140bpm] [verse]
         started from the bottom of a borrowed dream
         now the city lights flicker on my team
```
*Why:* fast tempo + trap production cue the rap delivery.

**Dreamy female indie**
```
prompt:  dreamy indie pop, airy female vocal, shimmering reverb-drenched guitars,
         steady drums, warm bass, wistful and hazy
lyrics:  [female] [108bpm] [c major] [verse]
         sunlight pooling on the kitchen floor
         we don't talk about it anymore
         [chorus]
         oh, let it go, let it drift away
         we'll find another ordinary day
```

**Explicitly instrumental (vocal-genre tags)**
```
prompt:  smooth r&b groove, electric piano, finger-snaps, mellow bass, no vocals
lyrics:  [instrumental] [95bpm]
```
*Why:* even with an R&B prompt that usually implies singing, `[instrumental]` forces no vocals.

### `/extend` — continue a clip

Upload the clip as `audio`. Set `add_seconds` for how much to add; leave `from_seconds`
blank for a seamless append from the tail.

**Seamless continuation**
```
audio:        my_clip.mp3
add_seconds:  20
prompt:       (same tags you generated the clip with, for consistency)
```
*Why:* matching tags keeps the timbre coherent across the seam.

**Continue into a new section**
```
audio:        my_verse.mp3
add_seconds:  20
prompt:       (same tags)
lyrics:       [chorus]
              and we run, we run till the morning comes
```
*Why:* a leading `[chorus]` tells the continuation to lift into the hook.

### `/cover` — re-render a melody *(needs a melody-trained checkpoint)*

Upload your hum/tune as `melody_audio`; its audio never appears in the output — only its
contour. Use `prompt` for the timbre to render it in.

**Hum → solo violin**
```
melody_audio:     my_hum.mp3
prompt:           expressive solo violin, warm tone, legato phrasing, subtle hall reverb
melody_cfg_scale: 2.0
```
*Why:* `melody_cfg_scale > 0` tightens adherence to your tune while the tags swap the timbre.

**Hum → synth lead**
```
melody_audio:     my_hum.mp3
prompt:           bright analog synth lead, punchy plucky envelope, retro synthwave backing
melody_cfg_scale: 2.5
```

### `/infill` — bridge two clips *(needs a FIM-trained checkpoint)*

> ⚠️ The current v8 checkpoint ships with FIM **disabled**, so `/infill` returns HTTP 400
> on it. This recipe applies once a FIM-trained checkpoint is served.

```
before_audio:  intro.mp3
after_audio:   outro.mp3
gap_seconds:   10
prompt:        warm analog synth pads, steady groove, smooth transition
```
*Why:* lyrics are ignored by infill; `prompt` is the only steering for the bridge's timbre.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Output is a single-pitch **drone** | Not the prompt — it's the fp16-on-MPS bug. Run bf16 on Apple Silicon (see **serve-model**). |
| Weak adherence to the prompt | Keep `sweeten` **on**; raise `cfg_scale` a notch. Terse keyword prompts condition weakly. |
| Wanted no vocals, got singing | Add `[instrumental]` to `lyrics`. `[inst]` is a *section*, not "no vocals". |
| Key marker ignored | `[m]`/`[f]` are **gender**. Use `[c major]`, `[f minor]`, etc. for keys. |
| Lyrics box content leaked into the tags / vibe | With `sweeten: false`, a `". "` inside a hand-written `prompt` splits the rest into the lyrics slot. Keep sweeten on (it collapses internal `". "` → `"; "`), or avoid `". "` in raw tag prompts. |
| Gender/tempo marker mid-song does nothing | Those are **prefix-only** — move them to the start of `lyrics`. Only section markers work inline. |
| `/cover` or `/infill` returns HTTP 400 | The served checkpoint wasn't trained for that axis (`use_melody_conditioning` / `use_fim`). |

---

## Curl appendix

The fields map 1:1 to multipart form fields. See the **serve-model** skill and
[README.md](README.md) for full server mechanics (deploy, dtype, quantization).

```bash
# Instrumental generate (relies on the WARM-ladder defaults)
curl -X POST http://localhost:8000/generate \
  -F prompt="mellow lofi hip hop instrumental, jazzy electric piano, warm sub bass, soft boom-bap drums, vinyl crackle, relaxed and nostalgic" \
  -F lyrics="[instrumental] [85bpm]" \
  -F seconds=30 \
  --output lofi.mp3

# Vocal song with markers
curl -X POST http://localhost:8000/generate \
  -F prompt="tender pop ballad, emotional female vocal, soft piano, swelling strings, heartfelt and intimate" \
  -F lyrics="[female] [72bpm] [a minor] [verse] i still hear the echo of your voice [chorus] but i'll carry the light you left behind" \
  -F cfg_scale=7.0 \
  --output ballad.mp3

# Extend a clip (seamless append)
curl -X POST http://localhost:8000/extend \
  -F audio=@lofi.mp3 \
  -F add_seconds=20 \
  -F prompt="mellow lofi hip hop instrumental, jazzy electric piano, warm sub bass" \
  --output lofi_longer.mp3

# Cover a hum as solo violin (needs a melody-trained checkpoint)
curl -X POST http://localhost:8000/cover \
  -F melody_audio=@my_hum.mp3 \
  -F prompt="expressive solo violin, warm tone, legato phrasing" \
  -F melody_cfg_scale=2.0 \
  --output cover.mp3
```
