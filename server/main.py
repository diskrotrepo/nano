"""FastAPI inference server.

Run:
    uvicorn server.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health
    POST /generate       form params, optional text + style_audio conditioning
    POST /continue       multipart: audio file + optional text/style_audio conditioning
    POST /extend         multipart: audio file + optional text/style_audio conditioning
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


@app.post("/continue")
async def continue_endpoint(
    audio: UploadFile = File(...),
    add_seconds: float = Form(25.0),
    prompt_seconds: float = Form(8.0),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    per_cb_temperature: str = Form(""),
    per_cb_top_k: str = Form(""),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(3.0),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    style_audio: UploadFile | None = File(None),
    style_weight: float = Form(0.5),
    lyric_cfg_scale: float = Form(0.0),
) -> Response:
    assert engine is not None
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    style_bytes = (await style_audio.read()) if style_audio else None
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    combined = _combine_text_lyrics(prompt, lyrics)
    try:
        body, mime = engine.continue_audio(
            data,
            add_seconds=add_seconds,
            prompt_seconds=prompt_seconds if prompt_seconds > 0 else None,
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=combined,
            negative_text=negative_prompt.strip() or None,
            style_audio_bytes=style_bytes or None,
            style_weight=style_weight,
            lyric_cfg_scale=lyric_cfg_scale or None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content=body, media_type=mime, headers=sweet_headers)


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
    seed_mode: str = Form("random"),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    style_audio: UploadFile | None = File(None),
    style_weight: float = Form(0.5),
    lyric_cfg_scale: float = Form(0.0),
    score_clap: bool = Form(False),
) -> Response:
    """Generate audio from scratch. Optional text and/or style_audio conditioning.

    score_clap: when true and a text prompt is present, the CLAP text<->audio
        adherence of the generated clip is returned in the X-Nano-Clap-Score
        response header (used by the inference sweep to rank prompt adherence,
        which collapse-only librosa scoring can't see).

    seed_mode='random' (default): random DAC seed token. Works for any prompt
        character — the model uses CFG + tight sampling to find a coherent
        trajectory regardless of seed energy.
    seed_mode='silence': 1 second of encoded silence as seed. Better for
        quiet/ambient/slow prompts where the audio context aligns with the
        prompt. Locks high-energy prompts into silence.

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
            seed_mode=seed_mode,
            text=combined,
            negative_text=negative_prompt.strip() or None,
            style_audio_bytes=style_bytes or None,
            style_weight=style_weight,
            lyric_cfg_scale=lyric_cfg_scale or None,
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
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    per_cb_temperature: str = Form(""),
    per_cb_top_k: str = Form(""),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(3.0),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    style_audio: UploadFile | None = File(None),
    style_weight: float = Form(0.5),
    lyric_cfg_scale: float = Form(0.0),
) -> Response:
    """Append more audio onto the end of an existing clip. Returns original + new."""
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
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=combined,
            negative_text=negative_prompt.strip() or None,
            style_audio_bytes=style_bytes or None,
            style_weight=style_weight,
            lyric_cfg_scale=lyric_cfg_scale or None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content=body, media_type=mime, headers=sweet_headers)
