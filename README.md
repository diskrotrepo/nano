# nano

Nano-scale audio generation model. Tokenizes audio with a neural codec — the default v9 path uses SpectroStream (48 kHz, joint stereo, 24 codebooks) — then trains a MusicGen-style delayed-sequence transformer to predict next frames autoregressively. (The Descript Audio Codec (DAC) — 44.1 kHz mono, 9 codebooks — is the code default, selected via `NANO_CODEC`.) Supports text conditioning (genre/mood descriptions via pooled CLAP), sung lyrics (a phoneme `LyricEncoder`, not CLAP), and melody conditioning (a time-aligned chromagram).

> **New to ML?** [README.explained.md](README.explained.md) walks through the whole project from first principles for engineers with a CS background but no ML experience — what a transformer actually is, why audio gets turned into integers, what the delay pattern accomplishes, and what the training log lines mean.

## Model

nano is a single bespoke model. Its shape is the `DEFAULTS` dict in [diskrot/modal_train.py](diskrot/modal_train.py) — the source of truth.

| | |
|---|---|
| Parameters | ~2.08B (2,077M, with text/lyric/melody conditioning, on by default; ~1.2B / 1,208M without) |
| d_model / layers / heads / d_ff | 2048 / 22 / 16 / 8192 |
| Positional encoding | RoPE (`rope_base` 10000), `max_seq_len` 8192 |
| Training segments | 180s (Modal default) |
| Training steps | 400,000 (Modal default; early stopping at `patience=20`) |
| Corpus | your own MP3s, served from a sharded mmap layout (~50k songs minimum recommended) |
| Audio codec | SpectroStream — 24 codebooks, 1024 vocab each, 25 Hz frame rate, joint stereo, 48 kHz (the v9 path; select with `NANO_CODEC=spectrostream`). DAC — 9 codebooks, 86 Hz, mono, 44.1 kHz — is the code default fallback |
| Max single-shot generation | ~5.4 minutes on SpectroStream — a full song fits in one shot (via the 8192-token RoPE table; ~95s under DAC's 86 Hz) |
| Inference | CPU, MPS (Apple Silicon), or CUDA — no GPU required |

This is a bespoke model: it is trained at scale on one kind of data. More of the same data helps; variety does not — the corpus is not curated for genre/style diversity. You supply your own MP3s (stored in the R2 `nano-audio` bucket); **plan on at least ~50,000 songs** for coherent musical output (below ~10,000 the model mostly produces noise, useful only for validating the pipeline), with quality improving as you add more of the same kind of data — object storage has no inode cap, so the corpus can grow without a hard ceiling.

## What to expect during training

Random chance on a 1024-vocab codebook is `ln(1024) ~ 6.93`. Per-codebook losses converge in order (codebook 0 first, then 1, etc.). Watch codebooks 2-4 in particular — they carry the most perceptually important detail.

| Val loss range | What it sounds like |
|---|---|
| ~6.5-6.9 | Noise with vague tonal hints |
| ~5.5-6.0 | Recognizable as "audio" — rhythm and rough pitch emerge, still very distorted |
| ~4.5-5.5 | Sounds like music played through a broken speaker. Beats, some melodic fragments |
| ~3.5-4.5 | Recognizably musical. Short coherent phrases, audible instruments, noticeable DAC artifacts |
| <3.5 | Best this model size can likely achieve. Coherent 3-5 second passages |

## Training

Pick the path that matches your hardware:

- **[Modal](README.modal.md)** — cloud GPUs, easiest for large corpora and parallel tagging/transcription.
- **[RTX 5090](README.5090.md)** — single-GPU Linux box, full pipeline local.
- **[M4 Max](README.m4max.md)** — Apple Silicon via MPS, good for small corpora and experimentation.

Once training is done, all paths land a checkpoint at `./checkpoints/latest.pt` and inference is identical.

## Inference

Start the server locally. It loads `./checkpoints/latest.pt` by default (override with the `NANO_CKPT` env var):

```bash
source .venv/bin/activate
uv pip install -e .
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

### Web UI

A Flutter web front-end in [webapp/](webapp/) exposes every generation mode and
parameter the server accepts. With the inference server running (above), start it
in a second terminal:

```bash
cd webapp
flutter pub get      # first run only — fetch dependencies
flutter run -d chrome
```

It defaults to `http://127.0.0.1:8000`; point the **server url** field at any
reachable host (e.g. a deployed Modal endpoint). See
[webapp/README.md](webapp/README.md) for the full parameter list, `flutter build
web` deploy bundle, and Modal hosting.

### API

Generate audio from scratch:

```bash
curl -X POST http://localhost:8000/generate \
  -F seconds=10 \
  --output generated.mp3
```

Generate with tags (genre/mood/instrument labels):

```bash
curl -X POST http://localhost:8000/generate \
  -F prompt="electronic, ambient, calm, synthesizer, piano, slow tempo" \
  -F seconds=10 \
  --output ambient.mp3
```

Generate with tags + lyrics:

```bash
curl -X POST http://localhost:8000/generate \
  -F prompt="rock, energetic, guitar, drums, vocals" \
  -F lyrics="walking through the city lights tonight" \
  -F seconds=10 \
  --output rock.mp3
```

Lyrics can be steered with inline `[marker]` brackets — gender, tempo, key,
vocals, and song section, e.g. `[female] [120bpm] [a minor] [chorus] walking
through the city lights tonight`. See [README.prompting.md](README.prompting.md)
for the full marker syntax.

Generate with a style reference audio:

```bash
curl -X POST http://localhost:8000/generate \
  -F style_audio=@reference.mp3 \
  -F seconds=10 \
  --output styled.mp3
```

Blend text + audio conditioning (style_weight controls the mix, 0.0 = all text, 1.0 = all audio):

```bash
curl -X POST http://localhost:8000/generate \
  -F prompt="ambient electronic" \
  -F style_audio=@reference.mp3 \
  -F style_weight=0.7 \
  -F seconds=10 \
  --output blended.mp3
```

Extend a clip — continue forward from a point in time (returns the original up to
the cut point, then newly generated audio):

```bash
curl -X POST http://localhost:8000/extend \
  -F audio=@clip.mp3 \
  -F add_seconds=10 \
  -F overlap_seconds=5 \
  --output extended.mp3
```

By default the cut point is the clip's tail, so this just appends 10s onto the end.
Pass `-F from_seconds=30` to instead keep the original up to 0:30, regenerate from
there, and discard whatever came after. Chain `/extend` calls to grow clips past the
~5.4 minute single-shot limit. All endpoints accept optional `prompt`, `lyrics`,
`style_audio`, and `style_weight` parameters.

### More endpoints

Beyond `/generate` and `/extend`, the server also exposes `/cover` (re-render a
hummed/uploaded melody in the prompt's timbre, via its chromagram), `/infill`
(fill the gap between two clips), `/stem` (Demucs source separation — no model),
and `/addstem` (generate a new stem that fits an existing song). `/infill` and
`/addstem` need a checkpoint trained for them; the v9 checkpoint disables both
(`use_fim=False`, `use_stem_conditioning=False`), so they return HTTP 400.

See [README.prompting.md](README.prompting.md) for the full lyric `[marker]`
syntax and ready-to-paste recipes for every endpoint.
