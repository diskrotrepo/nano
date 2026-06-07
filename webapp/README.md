# nano web UI

A minimalist Flutter web front-end for the nano inference server — white-on-black
with hot-pink highlights. Exposes every parameter the server accepts across all
three generation modes.

## Run

```bash
# 1. start the inference server (from the repo root)
uvicorn server.main:app --host 127.0.0.1 --port 8000

# 2. start the web UI (from this dir)
cd webapp
flutter run -d chrome
```

The server already sends permissive CORS headers (`allow_origins=["*"]`), so the
Flutter dev server can call it from its own localhost port. Point the **server
url** field at any reachable host — e.g. a deployed Modal endpoint.

## What's exposed

- **Modes**: `generate` (from scratch), `continue` (extend an uploaded clip from
  a prompt window), `extend` (append onto the end via an overlap window).
- **Conditioning** (all modes): tags/style prompt, lyrics, negative prompt,
  `sweeten` toggle, optional style-audio upload + `style_weight`.
- **Sampling** (all modes): `temperature`, `top_k`, `top_p`, `cfg_scale`,
  `lyric_cfg_scale`, plus per-codebook comma-separated overrides for
  temperature / top_k / top_p (under "per-codebook overrides").
- **generate**: `seconds`, `score_clap`.
- **continue**: input audio, `add_seconds`, `prompt_seconds`.
- **extend**: input audio, `add_seconds`, `overlap_seconds`.

The result is played inline, downloadable as `.mp3`/`.wav`, and surfaces the
`X-Nano-Sweetened-Prompt` and `X-Nano-Clap-Score` response headers when present.

## Build for deploy

```bash
flutter build web
# output in build/web/
```

## Tests

Widget tests touch web-only libraries (`package:web`, `dart:js_interop`), so run
them on the Chrome platform:

```bash
flutter test --platform chrome
```
