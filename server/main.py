"""FastAPI inference server.

Run:
    uvicorn server.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health
    POST /generate       from scratch — random seed → fresh clip
    POST /extend         continue forward from a point T → [original 0→T | new]
    POST /cover          re-render a hummed melody (chroma) in the prompt's timbre

    /generate, /extend accept optional text (tags), lyrics, gender, bpm, and
style_audio conditioning (gender/bpm ride the lyric stream as leading [male]/
[120bpm] markers). /extend seeds from the overlap_seconds before a cut point T, keeps
the original up to T, and generates forward (T defaults to the clip end, a
seamless grow-the-clip). /cover conditions on the uploaded melody's chromagram
(its audio never appears in the output) and needs a melody-trained checkpoint.
See each endpoint's docstring (surfaced in /docs).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from server.inference import InferenceEngine

engine: InferenceEngine | None = None


def _combine_text_lyrics(text: str, lyrics: str) -> str | None:
    """Combine tags and lyrics into a single conditioning string."""
    text = text.strip()
    lyrics = lyrics.strip()
    if text and lyrics:
        return f"{text}. {lyrics}"
    return text or lyrics or None


def _norm_gender(gender: str) -> str | None:
    """Normalize the optional vocal-gender selector to "male"/"female"/None.

    Folds a few aliases; anything unrecognized (incl. "" / "auto") -> None, which
    leaves the lyric stream's gender slot at <unknown_gender> (the train-time
    fallback). The engine only acts on exactly "male"/"female"."""
    g = (gender or "").strip().lower()
    if g in ("m", "man", "boy", "guy"):
        g = "male"
    elif g in ("f", "woman", "girl"):
        g = "female"
    return g if g in ("male", "female") else None


def _maybe_sweeten(prompt: str, sweeten: bool) -> tuple[str, dict[str, str]]:
    """Rewrite the tags prompt into LP-MusicCaps caption style via the local
    sweetener. On by default (CLAP was trained on ~40-word prose captions, so a
    terse prompt conditions weakly — sweetening lifts adherence markedly); pass
    sweeten=false to send the prompt verbatim. Returns the (possibly rewritten)
    prompt and any response headers exposing the result. Only the tags prompt is
    sweetened — lyrics and the negative prompt are passed through verbatim by the
    caller."""
    assert engine is not None
    if sweeten and prompt.strip():
        rewritten = engine.sweeten_prompt(prompt)
        # HTTP headers are latin-1; drop any non-ASCII so uvicorn won't choke.
        header_val = rewritten.encode("ascii", "ignore").decode()
        return rewritten, {"X-Nano-Sweetened-Prompt": header_val}
    return prompt, {}


def _parse_per_cb_temp(raw: str, scalar: float) -> float | list[float]:
    raw = raw.strip()
    if not raw:
        return scalar
    return [float(x) for x in raw.split(",")]


def _parse_per_cb_topk(raw: str, scalar: int) -> int | None | list[int | None]:
    raw = raw.strip()
    if not raw:
        return scalar if scalar > 0 else None
    return [int(x) if int(x) > 0 else None for x in raw.split(",")]


def _parse_per_cb_topp(raw: str, scalar: float) -> float | None | list[float | None]:
    raw = raw.strip()
    if not raw:
        return scalar if 0.0 < scalar < 1.0 else None
    out: list[float | None] = []
    for x in raw.split(","):
        v = float(x)
        out.append(v if 0.0 < v < 1.0 else None)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = InferenceEngine()
    yield


app = FastAPI(lifespan=lifespan, title="nano")

# Allow the Flutter web dev server to call us from a different localhost port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    assert engine is not None
    return {
        "ok": True,
        "device": engine.device,
        "ckpt_path": engine.ckpt_path,
        "ckpt_step": engine.ckpt_step,
        "codec": {
            "sample_rate": engine.codec.SAMPLE_RATE,
            "n_codebooks": engine.codec.N_CODEBOOKS,
            "frame_rate_hz": engine.codec.FRAME_RATE_HZ,
        },
        "model_params": engine.model.num_params(),
        "text_conditioning": engine.text_encoder is not None,
    }


@app.post("/generate")
async def generate_endpoint(
    seconds: float = Form(30.0),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    per_cb_temperature: str = Form(""),
    per_cb_top_k: str = Form(""),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(3.0),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    gender: str = Form(""),
    bpm: float = Form(0.0),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    style_audio: UploadFile | None = File(None),
    style_weight: float = Form(0.5),
    lyric_cfg_scale: float = Form(0.0),
    score_clap: bool = Form(False),
) -> Response:
    """Generate audio from scratch — no audio input. Returns a fresh clip.

    Bootstraps the autoregressive loop from a random DAC seed and produces
    `seconds` of audio, steered entirely by the optional text / lyrics /
    style_audio conditioning. This is the starting point; /extend builds on an
    existing clip instead.

    score_clap: when true and a text prompt is present, the CLAP text<->audio
        adherence of the generated clip is returned in the X-Nano-Clap-Score
        response header (used by the inference sweep to rank prompt adherence,
        which collapse-only librosa scoring can't see).

    per_cb_temperature / per_cb_top_k / per_cb_top_p: optional comma-separated
        list (length = n_codebooks) overriding the scalar. Later codebooks model
        high-entropy DAC residuals; a decreasing ladder (e.g.
        per_cb_temperature="0.9,0.9,0.7,0.7,0.5,0.5,0.4,0.4,0.3") usually sounds
        better than a single temperature applied across all 9.
    """
    assert engine is not None
    style_bytes = (await style_audio.read()) if style_audio else None
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    combined = _combine_text_lyrics(prompt, lyrics)
    try:
        result = engine.generate_audio(
            seconds=seconds,
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=combined,
            negative_text=negative_prompt.strip() or None,
            style_audio_bytes=style_bytes or None,
            style_weight=style_weight,
            lyric_cfg_scale=lyric_cfg_scale or None,
            gender=_norm_gender(gender),
            bpm=bpm or None,
            score_clap=score_clap,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    headers = dict(sweet_headers)
    if score_clap:
        body, mime, clap = result
        if clap is not None:
            headers["X-Nano-Clap-Score"] = f"{clap:.6f}"
    else:
        body, mime = result
    return Response(content=body, media_type=mime, headers=headers)


@app.post("/extend")
async def extend_endpoint(
    audio: UploadFile = File(...),
    add_seconds: float = Form(20.0),
    overlap_seconds: float = Form(8.0),
    from_seconds: float = Form(-1.0),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    per_cb_temperature: str = Form(""),
    per_cb_top_k: str = Form(""),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(3.0),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    gender: str = Form(""),
    bpm: float = Form(0.0),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    style_audio: UploadFile | None = File(None),
    style_weight: float = Form(0.5),
    lyric_cfg_scale: float = Form(0.0),
) -> Response:
    """Continue a clip forward from a point in time. Returns [original 0→T | new].

    `from_seconds` (T) is the cut point: the original is kept verbatim from the
    start up to T, the model generates add_seconds forward from there, and whatever
    the clip had after T is discarded. The seed is the `overlap_seconds` just
    before T. Omit from_seconds (or pass <0) to use the clip's tail — then nothing
    is dropped and you get [full original | new], the seamless grow-the-clip case.

    Only the small seed window counts against the context budget, so call this
    repeatedly to chain a clip past the model's single-shot length cap.
    """
    assert engine is not None
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    style_bytes = (await style_audio.read()) if style_audio else None
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    combined = _combine_text_lyrics(prompt, lyrics)
    try:
        body, mime = engine.extend_audio(
            data,
            add_seconds=add_seconds,
            overlap_seconds=overlap_seconds,
            from_seconds=from_seconds if from_seconds >= 0 else None,
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=combined,
            negative_text=negative_prompt.strip() or None,
            style_audio_bytes=style_bytes or None,
            style_weight=style_weight,
            lyric_cfg_scale=lyric_cfg_scale or None,
            gender=_norm_gender(gender),
            bpm=bpm or None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content=body, media_type=mime, headers=sweet_headers)


@app.post("/cover")
async def cover_endpoint(
    melody_audio: UploadFile = File(...),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    per_cb_temperature: str = Form(""),
    per_cb_top_k: str = Form(""),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(3.0),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    gender: str = Form(""),
    bpm: float = Form(0.0),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    melody_cfg_scale: float = Form(0.0),
    lyric_cfg_scale: float = Form(0.0),
) -> Response:
    """Cover a hummed/uploaded melody in the prompt's timbre.

    `melody_audio` is the melody to follow (a hum, whistle, or any clip). It is
    converted to a chromagram and used to condition generation — its audio/tokens
    never appear in the output. `prompt` (tags) drives the timbre/instrumentation
    (e.g. "solo violin") and `lyrics` the words to sing; the hum's length sets the
    output length. `melody_cfg_scale` (>0) pushes melody adherence with its own
    guidance scale, independent of `cfg_scale`.

    Requires a checkpoint trained with melody conditioning (use_melody_conditioning).
    """
    assert engine is not None
    data = await melody_audio.read()
    if not data:
        raise HTTPException(400, "empty melody_audio upload")
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    combined = _combine_text_lyrics(prompt, lyrics)
    try:
        body, mime = engine.cover_audio(
            data,
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=combined,
            negative_text=negative_prompt.strip() or None,
            melody_cfg_scale=melody_cfg_scale or None,
            lyric_cfg_scale=lyric_cfg_scale or None,
            gender=_norm_gender(gender),
            bpm=bpm or None,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return Response(content=body, media_type=mime, headers=sweet_headers)


@app.post("/infill")
async def infill_endpoint(
    before_audio: UploadFile = File(...),
    after_audio: UploadFile = File(...),
    gap_seconds: float = Form(10.0),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    per_cb_temperature: str = Form(""),
    per_cb_top_k: str = Form(""),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(3.0),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    melody_audio: UploadFile | None = File(None),
    melody_cfg_scale: float = Form(0.0),
) -> Response:
    """Fill the gap between two clips — returns ``[before | middle | after]``.

    `before_audio` and `after_audio` are kept verbatim; the model generates a
    `gap_seconds` bridge that flows out of the first and into the second.
    `prompt` (tags) drives the timbre of the fill. An optional `melody_audio` hum
    guides the gap's melodic contour. Lyrics are not used by infill.

    Requires a checkpoint trained with FIM (use_fim — a v8+ model).
    """
    assert engine is not None
    before = await before_audio.read()
    after = await after_audio.read()
    if not before or not after:
        raise HTTPException(400, "both before_audio and after_audio are required")
    mel = await melody_audio.read() if melody_audio is not None else None
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    try:
        body, mime = engine.infill_audio(
            before, after,
            gap_seconds=gap_seconds,
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=prompt.strip() or None,
            negative_text=negative_prompt.strip() or None,
            melody_audio_bytes=mel or None,
            melody_cfg_scale=melody_cfg_scale or None,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return Response(content=body, media_type=mime, headers=sweet_headers)
