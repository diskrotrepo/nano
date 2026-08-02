# nano web UI

A minimalist Flutter web front-end for the nano inference server — white-on-black
with hot-pink highlights. Exposes every parameter the server accepts across all
four generation modes.

## Run

```bash
# 1. start the inference server (from the repo root)
uvicorn server.main:app --host 127.0.0.1 --port 8000

# 2. start the web UI (from this dir)
cd webapp
flutter pub get      # first run only — fetch dependencies
flutter run -d chrome
```

The server already sends permissive CORS headers (`allow_origins=["*"]`), so the
Flutter dev server can call it from its own localhost port. Point the **server
url** field at any reachable host — e.g. a deployed Modal endpoint.

## What's exposed

- **Modes**: `generate` (from scratch), `extend` (append onto the end via an
  overlap window), `cover` (re-render an uploaded melody's chromagram in the
  prompt's timbre), `stem` (pure Demucs source separation — keep/drop stems).
- **Model picker**: `GET /models` lists the server's switchable checkpoints
  (`NANO_MODELS`); the chosen id rides every request as the `model` field.
- **Conditioning** (all modes): tags/style prompt, lyrics, negative prompt,
  `sweeten` toggle, optional style-audio upload + `style_weight`.
- **Sampling** (all modes): `temperature`, `top_k`, `top_p`, `cfg_scale`,
  `lyric_cfg_scale`, plus per-codebook comma-separated overrides for
  temperature / top_k / top_p (under "per-codebook overrides").
- **generate**: `seconds`, `score_clap`.
- **extend**: input audio, `add_seconds`, `overlap_seconds`.
- **cover**: input audio (the melody to cover — its audio never appears in the
  output, only its chromagram conditions generation) + `melody_cfg_scale`.
- **stem**: input audio + which Demucs stems to keep (drums/bass/other/vocals) —
  no model, the server just separates and remixes.
- **Streaming**: generate plays progressively via `GET /generate_stream`; extend
  and cover stream over `POST /extend_stream` / `POST /cover_stream`.

The result is played inline, downloadable as `.mp3`/`.wav`, and surfaces the
`X-Nano-Sweetened-Prompt` and `X-Nano-Clap-Score` response headers when present.

## Build for deploy

```bash
flutter build web
# output in build/web/
```

## Serve on Modal

The Modal entrypoint ([../diskrot/modal_serve.py](../diskrot/modal_serve.py))
serves this UI as a second, CPU-only endpoint next to the GPU inference API:

```bash
cd webapp && flutter build web        # the bundle is mounted into the image
modal serve diskrot/modal_serve.py    # dev URLs (from the repo root)
modal deploy diskrot/modal_serve.py   # persistent URLs
```

UI:  `https://<workspace>--nano-serve-ui[-dev].modal.run`
API: `https://<workspace>--nano-serve-serve[-dev].modal.run`

When the UI is loaded from its Modal URL it pre-fills the **server url** field
with the sibling API URL (`defaultServerUrl` in [lib/api.dart](lib/api.dart));
anywhere else it defaults to `http://127.0.0.1:8000`. The field stays editable
either way.

## Tests

Widget tests touch web-only libraries (`package:web`, `dart:js_interop`), so run
them on the Chrome platform:

```bash
flutter test --platform chrome
```
