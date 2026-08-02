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
| **Tags** | `prompt` | The *vibe* — genre, instrumentation, mood, production | A natural-language caption → split into ≤77-token chunks → frozen CLAP → a *sequence* of pooled vectors the decoder cross-attends to |
| **Lyrics + markers** | `lyrics` (+ `gender`/`bpm`) | The *words to sing* and per-song attributes (gender, tempo, key, vocals, section) | A phoneme sequence the decoder cross-attends to |
| **Melody** | `melody_audio` (`/cover`) | The *contour* — the tune to follow | A time-aligned chromagram added per-frame |
| **Stem** | `audio` + `target_stem` (`/addstem`) | A *new stem to add* to an existing song (e.g. a bassline) | The song's other stems' codec tokens, added per-frame |

Each axis drops independently for classifier-free guidance, so `cfg_scale` (and the
optional `lyric_cfg_scale` / `melody_cfg_scale` / `stem_cfg_scale`) pushes adherence on each axis on its own.

---

## Tags: write a caption, not keywords

CLAP was trained on **LP-MusicCaps prose captions**, so a terse keyword list conditions
*weakly*. Write — or let the sweetener write — a paragraph describing the instruments,
the groove, the mood, and the production. **Long descriptions are fine now:** the prompt
is split into ≤77-token chunks and each pooled by CLAP, so the decoder reads the *whole*
caption (up to ~3000 chars) rather than a truncated single vector. A multi-sentence,
free-form, or even multilingual description works — there's no internal-`". "` gotcha
anymore (tags and lyrics are separate fields). For the richest conditioning, give one
theme per sentence (genre/mood · drums · bass · instruments · vocals · production · arc),
so each chunk pools a distinct facet.

**Sweeten is ON by default.** It rewrites a terse `prompt` into caption style with a
small local model before it ever reaches CLAP, so `prompt: "lofi beat to study to"`
becomes something like *"A mellow lofi hip hop instrumental with a jazzy electric piano,
warm bass, soft boom-bap drums and vinyl crackle. It sounds relaxed and nostalgic."* You
usually **want this on** for terse prompts — but a long / already-detailed prompt
(>~60 words) is passed through **verbatim** (sweetening would only discard your detail),
and you can force verbatim any time with `sweeten: false`.

The register to aim for (real v5 audio-LLM captions from the training set):

> *"The track has a chill, relaxing mood with a blend of electronic and instrumental
> elements. It features a groovy bass line and a soothing melody played on a piano. The
> drums and percussion add a subtle rhythm to the background. Vocals are sparse, with
> occasional crooned notes adding texture to the sound. The production and mix character
> is clean, with no overpowering elements."*

> *"The music is a lively punk rock track … The guitars are raw and distorted … the bass
> follows a simple yet effective bassline … The drums play a steady and driving beat …
> Vocals are sparse, featuring occasional shouting and screaming … The production is
> lo-fi, with a raw and unpolished sound that adds to the punk rock aesthetic."*

The captioner writes one facet per sentence in a fixed arc — **genre/mood → drums →
bass → harmony/lead → vocals → production → build** — so a prompt that follows the same
arc pools one clean facet per CLAP chunk.

Vocabulary with real weight in the corpus (share of captions; 13.6k-caption random
sample of the v5 tag store, 2026-07-11):

- **genres**: *electronic* 37%, *guitar* 24%, *lo-fi* 19%, *pop* 17%, *rock* 17%,
  *dance* 17%, *ambient* 8%, *techno* 6%, *rap* 5%, *folk* 5%, *jazz* 5%
- **moods**: *haunting* 28%, *energetic* 20%, *upbeat* 20%, *groovy* 10%,
  *aggressive* 10%, *dreamy* 9%, *uplifting* 8%, *eerie* 7%, *melancholic* 5%
- **vocal styles the captioner actually names**: *screaming, crooned, soulful, rapped,
  belted, harmonized*
- **blends it names**: *pop+electronic, classical+electronic, electronic+rock,
  rap+electronic, jazz+electronic, orchestral+electronic, strings+synthesizers*
- **color instruments with real presence**: flute, bells, trumpet, cello, accordion,
  harp, harmonica, banjo, sitar, tabla

Two consequences. (1) Prompts built from these words adhere strongest — and because
*haunting*, *lo-fi*, and *sparse vocals* saturate the data, counter them explicitly
("clean, polished production", "bright and joyful") when you don't want that default
vibe. (2) Words the captions never use condition weakly: e.g. *vocoder* and *talkbox*
appear **zero** times in the sample; the nearest real handles are "auto-tune" (rare) and
rap+electronic phrasing.

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

Paste these into the matching fields. Blank fields are omitted. The `/generate` examples
are **grounded in the live corpus captions** (the same 13.6k-caption v5 sample as above,
2026-07-11): each is written in the captioner's own register — one facet per sentence,
its actual vocabulary, several near-verbatim from real captions — so it conditions on
phrasing CLAP saw at train time. All are >60 words, so the sweetener passes them through
**verbatim**. Instrumental prompts say "there are no vocals" in the prose *and* carry
`[instrumental]` in lyrics — belt and suspenders against unwanted singing.

### `/generate` — the corpus's center of gravity

**Haunting electronic with crooned female vocal** — *haunting* (28%) + *crooned* + sparse
vocals is the single most common vibe in the data
```
prompt:  The track has a haunting, hypnotic mood built on electronic elements. The drums and
         percussion carry a slow, steady rhythm. The bass line is deep and resonates throughout.
         A synth adds an eerie melody that lingers. Vocals are sparse, with a haunting female
         lead delivered in a crooned style. The production is clean and balanced, and the track
         builds from a foreboding opening to an intense finale.
lyrics:  [female] [90bpm] [a minor] [verse]
         shadows gather where you used to stand
         i hold the cold light in my hand
```

**Classical + electronic blend** — one of the top blends the captioner names
```
prompt:  The music is a blend of classical and electronic elements with a melancholic, dreamy
         mood. Soaring strings and a delicate piano carry the lead melody over a deep electronic
         bass line. The drums are subtle and smooth. There are no vocals. The production is
         top-notch, with a clear and balanced sound, and the track builds from an introspective
         opening into a sweeping, emotional climax.
lyrics:  [instrumental] [medium] [d minor]
```

**Groovy lo-fi chill with piano** — near-verbatim from a real caption
```
prompt:  The track has a chill, relaxing mood blending electronic and instrumental elements. It
         features a groovy bass line and a soothing melody played on a piano. The drums and
         percussion add a subtle rhythm in the background. There are no vocals. The production
         has a warm lo-fi character with a raw, unpolished charm, and the track maintains a
         consistent tempo and tranquil energy throughout.
lyrics:  [instrumental] [82bpm]
```

**Energetic dance-pop + electronic** — the #1 blend in the corpus
```
prompt:  The track is an upbeat blend of pop and electronic music with a danceable rhythm. The
         drums drive a steady four-on-the-floor beat with crisp hi-hats. The bass line is
         prominent and groovy. Bright synths carry a catchy lead melody. The vocals feature a
         strong, belted female lead with harmonized backing. The production is clean and punchy,
         and the track builds into a euphoric final chorus.
lyrics:  [female] [124bpm] [c major] [verse]
         heartbeat racing down the boulevard
         we came too far to fall apart
         [chorus]
         so turn it up, we're not going home
```

**Rap + electronic**
```
prompt:  The track is a blend of rap and electronic music with a dark, energetic mood. The drums
         hit hard with a booming kick and crisp, rolling hi-hats. The bass is deep and heavy,
         providing a solid foundation. A sparse, eerie synth melody floats over the beat. The
         vocals are rapped with a confident, aggressive delivery. The production is clean and
         modern with plenty of low end.
lyrics:  [male] [142bpm] [verse]
         concrete dreams under a static sky
         count the reasons, watch the seasons fly
```

### `/generate` — guitar-land (24% of captions)

**Post-hardcore / post-rock in B minor** — near-verbatim from a real caption
```
prompt:  The track is a fast-paced post-hardcore and post-rock track with a dynamic, intense
         mood. The drums are intense with heavy cymbal work. The bass follows a repetitive
         pattern under a complex guitar arrangement that shifts between tension and release.
         Vocals are sparse, with occasional screaming. The production is clean and well-mixed,
         focused on the instruments, and the track moves between quiet passages and explosive
         climaxes.
lyrics:  [male] [fast] [b minor] [verse]
         we carved our names in falling ash
         and waited for the sky to crash
```

**Nu-metal / hard rock**
```
prompt:  The track is a fast-paced hard rock track with elements of nu metal and a gritty,
         aggressive mood. The drums drive a complex, heavy beat. The bass line is deep and heavy.
         The harmony is a mix of down-tuned guitars and synthesizers, creating an expansive,
         intense sonic landscape. Vocals feature aggressive delivery with occasional screaming.
         The production is clear and balanced, building into an intense middle section.
lyrics:  [male] [fast] [e minor] [verse]
         bite down on the wire, swallow the spark
         everything you promised got lost in the dark
```

**Raw lo-fi punk**
```
prompt:  The music is a lively punk rock track with raw, distorted guitars and a driving beat.
         The bass follows a simple, effective line that adds to the energy. The drums are steady
         and relentless. Vocals feature shouted, aggressive delivery with gang backing. The
         production is lo-fi, with a raw and unpolished sound that adds to the punk aesthetic.
lyrics:  [male] [130bpm] [a major] [chorus]
         no more waiting, no more lies
         tear it down before it dies
```

**Dreamy guitar-pop, harmonized**
```
prompt:  The track has a dreamy, nostalgic mood blending indie pop and electronic textures.
         Shimmering, reverb-washed guitars carry the harmony over a warm, round bass line. The
         drums are soft and steady. Vocals feature a gentle female lead with lush harmonized
         backing vocals. The production is hazy and warm, and the track drifts from a sparse
         verse into a blissful, layered chorus.
lyrics:  [female] [105bpm] [c major] [verse]
         polaroids fading on the wall
         i kept the summer, kept it all
         [chorus]
         and if you call, i'll drift away
```

### `/generate` — genre blends the captioner names

**Electronic + Arabic strings** — near-verbatim from a real caption
```
prompt:  The track is a blend of electronic and Arabic music elements with a melancholic mood.
         The drums and percussion have a subtle, smooth rhythm. The bass line is deep and
         resonates throughout the song. The harmony features a beautiful blend of Arabic strings
         and plucked instruments carrying a captivating melody. Vocals are sparse, with a
         haunting female voice. The track builds from a slow, introspective opening into a more
         energetic section.
lyrics:  [female] [100bpm] [d minor]
```
*Why the odd lyrics:* `[female]` with no words leaves the vocal slot `<unknown_vocals>`,
matching the caption's *sparse, wordless* vocals — neither forced singing nor forced
instrumental.

**Brazilian funk-rock with belted female lead** — near-verbatim from a real caption
```
prompt:  The music is a lively Brazilian funk track with a danceable rhythm and a distinct rock
         influence. It has an upbeat mood, with a prominent bassline and driving drums. The
         vocals are delivered in a belted style by a strong, dynamic female lead. Guitars and
         keyboards provide the harmony with a prominent lead melody. The production is lo-fi,
         raw and unpolished, which adds to its charm.
lyrics:  [female] [112bpm] [chorus]
         dança comigo até o sol chegar
         não deixa a noite acabar
```

**Indian classical + electronic** — sitar/tabla have real corpus presence
```
prompt:  The track is a blend of traditional Indian and electronic music with a hypnotic,
         meditative mood. Tabla percussion carries an intricate rhythm over a deep electronic
         bass line. A sitar leads with an ornamented, winding melody, answered by warm synth
         pads. There are no vocals. The production is clean and spacious, and the track builds
         slowly in intensity without ever breaking its trance.
lyrics:  [instrumental] [95bpm]
```

**Jazz + electronic**
```
prompt:  The track is a blend of jazz and electronic music with a late-night, groovy mood. The
         drums mix brushed acoustic textures with programmed hi-hats. A walking upright bass
         anchors the harmony while a smoky trumpet carries the lead melody over warm electric
         piano chords. There are no vocals. The production is clean and intimate, like a dim
         club after midnight.
lyrics:  [instrumental] [98bpm]
```

**Orchestral + electronic epic**
```
prompt:  The track is a blend of orchestral and electronic music with an epic, suspenseful mood.
         Thundering percussion and a deep electronic pulse drive the rhythm. Soaring strings and
         brass carry the harmony while a choir adds a haunting texture. There are no vocals in
         the lead. The production is massive and cinematic, and the track builds relentlessly
         from an ominous opening to a triumphant, explosive finale.
lyrics:  [instrumental] [medium] [d minor]
```

### `/generate` — color instruments & quirky corners

**French accordion waltz**
```
prompt:  The music is a charming French folk waltz with a nostalgic, romantic mood. An accordion
         carries the lead melody, supported by a gently strummed acoustic guitar and a soft
         upright bass. The percussion is light and brushed. There are no vocals. The production
         is warm and intimate, like a street café recording, and the track sways gracefully from
         start to finish.
lyrics:  [instrumental] [slow] [g major]
```

**Banjo & harmonica folk stomp**
```
prompt:  The track is a lively folk and country stomp with an upbeat, festive mood. A banjo
         drives the rhythm with rapid picking while a harmonica trades lead lines with an
         acoustic guitar. The bass is a simple, bouncing upright line and the drums are a
         stomping kick and clap. Vocals feature a raspy male lead with harmonized gang choruses.
         The production is raw and energetic.
lyrics:  [male] [140bpm] [g major] [chorus]
         raise your glass to the river town
         we ain't ever gonna slow it down
```

**Cello + harp chamber piece**
```
prompt:  The track is a delicate neoclassical chamber piece with a somber, haunting mood. A solo
         cello carries the lead melody with long, mournful phrases. A harp plays gentle arpeggios
         beneath it, and soft bells add sparse, glassy accents. There are no drums and no vocals.
         The production is intimate and spacious with a natural hall reverb, and the piece swells
         gently before fading to silence.
lyrics:  [instrumental] [slow] [e minor]
```

**Reggaeton**
```
prompt:  The track is an energetic reggaeton song with a danceable, festive mood. The drums lock
         into the signature dembow rhythm with a punchy kick and crisp snare. The bass is deep
         and round. Bright synth plucks and a marimba-like melody carry the harmony. The vocals
         are a confident male lead delivered in a rhythmic, half-rapped style. The production is
         modern, clean and loud.
lyrics:  [male] [96bpm] [verse]
         baila conmigo bajo la luna llena
         la noche es nuestra, vale la pena
```

**8-bit / chiptune** — a real cluster (~1% of captions)
```
prompt:  The track is an upbeat chiptune piece built from 8-bit video game sounds with a playful,
         energetic mood. Square-wave leads carry a catchy, fast melody over an arpeggiated pulse
         bass. The percussion is crunchy programmed noise-channel drums. There are no vocals. The
         production is intentionally lo-fi and bright, and the track loops through rising,
         triumphant phrases like a level theme.
lyrics:  [instrumental] [150bpm] [c major]
```

**Dark suspenseful electronic-rock hybrid** — near-verbatim from a real caption
```
prompt:  The track has an ominous, suspenseful mood with a blend of electronic and rock elements.
         The drums have a driving rhythm that creates urgency and tension. The bass underpins the
         percussion with a steady pulse. A synth adds a haunting melody that lingers in the
         listener's mind while distorted guitars swell underneath. Vocals are sparse, with a
         haunting female lead in a crooned style. The track builds from a foreboding tone to a
         thrilling, intense finale.
lyrics:  [female] [110bpm] [f# minor] [verse]
         sirens sleeping in the wires
         city breathing through the fires
```

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

**Worked sequence: build a full song from sections**

nano can generate up to **~5.4 min in one shot** (a full ~3 min song fits), so you no
longer *have* to stitch — but for a very long or tightly structured song you can still
assemble it by chaining `/extend` and switching the **vocals** and **section** markers per
call. Each call returns `[everything so far | new part]`, so the **last file is the whole
song.** Restate
`[tempo]`/`[key]` (and gender) on *every* call — the marker header is per-call, not
inherited; the audio tail is what carries timbre across the seam.

*Step 1 — instrumental intro (`/generate`)*
```
prompt:   dreamy indie pop, shimmering reverb-drenched guitars, warm bass, soft
          brushed drums, wistful and hazy
lyrics:   [instrumental] [108bpm] [c major] [intro]
seconds:  12
```
→ save as `01.mp3`. `[instrumental]` forces no vocals; `[intro]` sets the section — a
section marker rides the header slot, so it registers even with no words.

*Step 2 — verse/chorus, bring the vocals in (`/extend` `01.mp3`)*
```
audio:        01.mp3
add_seconds:  24
prompt:       (same tags as step 1)
lyrics:       [female] [108bpm] [c major] [vocals] [verse]
              sunlight pooling on the kitchen floor
              we don't talk about it anymore
              [chorus]
              oh, let it go, let it drift away
```
→ save as `02.mp3`. The `[female] [vocals]` header turns singing on for the appended part.

*Step 3 — instrumental break / solo (`/extend` `02.mp3`)*
```
audio:        02.mp3
add_seconds:  16
prompt:       dreamy indie pop, soaring reverb-guitar solo, instrumental break
lyrics:       [instrumental] [108bpm] [c major] [solo]
```
→ save as `03.mp3`. Back to `[instrumental]` to drop the vocals for the solo.

*Step 4 — outro, wind down (`/extend` `03.mp3`)*
```
audio:        03.mp3
add_seconds:  16
prompt:       dreamy indie pop, sparse, winding down, soft fade
lyrics:       [instrumental] [108bpm] [c major] [outro]
```
→ `04.mp3` is the finished **intro → verse/chorus → solo → outro** arrangement.

*Why it works:* `[instrumental]` ↔ `[vocals]` toggles singing per section, the section
marker sets each part's role, and a consistent `[tempo]`/`[key]` plus the seed-from-tail
seam keep it coherent. *Caveats:* this is a **stitching workflow, not one-shot song-form** —
keep the prompt consistent across calls, and expect some drift over long chains (re-anchor
the prompt if the vibe wanders).

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

> ⚠️ The current v9 checkpoint ships with FIM **disabled** (`use_fim=False`), so `/infill`
> returns HTTP 400 on it. This recipe applies once a FIM-trained checkpoint is served.

```
before_audio:  intro.mp3
after_audio:   outro.mp3
gap_seconds:   10
prompt:        warm analog synth pads, steady groove, smooth transition
```
*Why:* lyrics are ignored by infill; `prompt` is the only steering for the bridge's timbre.

---

### `/addstem` — add a stem to a song *(needs a stem-trained checkpoint)*

Give it a finished song and a stem to add; it generates a NEW isolated stem that
fits, steered by `prompt`. The generative inverse of `/stem` (which only removes).

```
audio:          my_song.mp3
target_stem:    bass            # drums | bass | vocals | other
prompt:         funky 70s warbly synth bassline, syncopated, round low end
output:         mix             # mix = song + new stem | stem = the new stem alone
stem_cfg_scale: 4               # >0: push the new stem to fit the song tighter
```
*Why:* the model Demucs-separates your upload, conditions on its OTHER stems +
`prompt`, and renders the target. `prompt` describes the *stem* you want (not the
whole song); lyrics/melody are unused. Use `output=stem` to get the raw stem to
mix yourself, or `output=mix` (default) for the song with it layered in.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Output is a single-pitch **drone** | Not the prompt — it's the fp16-on-MPS bug. Run bf16 on Apple Silicon (see **serve-model**). |
| Weak adherence to the prompt | Keep `sweeten` **on**; raise `cfg_scale` a notch. Terse keyword prompts condition weakly. |
| Wanted no vocals, got singing | Add `[instrumental]` to `lyrics`. `[inst]` is a *section*, not "no vocals". |
| Key marker ignored | `[m]`/`[f]` are **gender**. Use `[c major]`, `[f minor]`, etc. for keys. |
| Tags / lyrics cross-contaminating | Fixed — tags (`prompt`) and lyrics (`lyrics`) are separate fields now (no `". "` join/split), so a multi-sentence prose `prompt` stays entirely in the tags slot. |
| Gender/tempo marker mid-song does nothing | Those are **prefix-only** — move them to the start of `lyrics`. Only section markers work inline. |
| `/cover`, `/infill`, or `/addstem` returns HTTP 400 | The served checkpoint wasn't trained for that axis (`use_melody_conditioning` / `use_fim` / `use_stem_conditioning`). |
| `/addstem` prompt steers the stem weakly | Songs captioned before `CAPTIONER_MARKER` v5 have no per-stem caption (training falls back to the song caption) — re-run `auto_tag --redo` to backfill; meanwhile raise `stem_cfg_scale` and be specific in `prompt`. |

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

# Build a structured song by chaining /extend, toggling [instrumental] <-> [vocals]
# and the section marker per call (each call returns "[so far | new part]").
TAGS="dreamy indie pop, shimmering reverb-drenched guitars, warm bass, soft brushed drums"
# 1) instrumental intro
curl -sX POST http://localhost:8000/generate \
  -F prompt="$TAGS" -F lyrics="[instrumental] [108bpm] [c major] [intro]" \
  -F seconds=12 --output 01.mp3
# 2) verse/chorus with vocals
curl -sX POST http://localhost:8000/extend \
  -F audio=@01.mp3 -F add_seconds=24 -F prompt="$TAGS" \
  -F lyrics="[female] [108bpm] [c major] [vocals] [verse] sunlight pooling on the kitchen floor [chorus] oh, let it go, let it drift away" \
  --output 02.mp3
# 3) instrumental solo
curl -sX POST http://localhost:8000/extend \
  -F audio=@02.mp3 -F add_seconds=16 -F prompt="$TAGS, soaring guitar solo, instrumental break" \
  -F lyrics="[instrumental] [108bpm] [c major] [solo]" --output 03.mp3
# 4) instrumental outro -> 04.mp3 is the finished arrangement
curl -sX POST http://localhost:8000/extend \
  -F audio=@03.mp3 -F add_seconds=16 -F prompt="$TAGS, sparse, winding down, soft fade" \
  -F lyrics="[instrumental] [108bpm] [c major] [outro]" --output 04.mp3

# Cover a hum as solo violin (needs a melody-trained checkpoint)
curl -X POST http://localhost:8000/cover \
  -F melody_audio=@my_hum.mp3 \
  -F prompt="expressive solo violin, warm tone, legato phrasing" \
  -F melody_cfg_scale=2.0 \
  --output cover.mp3

# Add a bassline to an existing song (needs a stem-trained checkpoint)
curl -X POST http://localhost:8000/addstem \
  -F audio=@my_song.mp3 \
  -F target_stem=bass \
  -F prompt="funky 70s warbly synth bassline, syncopated, round low end" \
  -F stem_cfg_scale=4.0 -F output=mix \
  --output with_bass.mp3
```
