"""FastAPI inference server.

Run:
    uvicorn server.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health
    POST /generate       from scratch — random seed → fresh clip
    POST /extend         continue forward from a point T → [original 0→T | new]
    POST /cover          re-render a hummed melody (chroma) in the prompt's timbre
    POST /infill         fill the gap between two clips → [before | middle | after]
    POST /stem           remove/isolate stems via Demucs (no model — any ckpt)

    /generate, /extend accept optional text (tags), lyrics, gender, bpm, key,
vocal-presence, and style_audio conditioning (gender/bpm/key/vocals ride the
lyric stream as leading [male]/[120bpm]/[a minor]/[instrumental] markers — key
also accepts [key:Am]/[f# major] forms, and [instrumental]/[vocals] requests
no-vocals/vocals; section markers like [chorus] may appear inline). /extend
seeds from the overlap_seconds before a cut point T, keeps
the original up to T, and generates forward (T defaults to the clip end, a
seamless grow-the-clip). /cover conditions on the uploaded melody's chromagram
(its audio never appears in the output) and needs a melody-trained checkpoint.
See each endpoint's docstring (surfaced in /docs).
"""
from __future__ import annotations

import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import glob

import base64

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from server.inference import InferenceEngine

# Upper bound on a single batched generation. Decode is memory-bandwidth-bound so
# the wall-clock barely grows with B, but the KV cache scales linearly with it;
# 8 fits a 30s clip on an H100 with huge headroom (see the plan's budget).
MAX_BATCH = int(os.environ.get("NANO_MAX_BATCH", "8"))

# --- Model registry (switch between checkpoints at serve time) ---------------
# The server can hold several named checkpoints (e.g. "full"=v8_sing4 and
# "fast"=the distilled student) and pick one per request via the `model` form
# field. Engines are LAZY-loaded on first use, so a cold container only pays for
# the default model — the others load on their first request. The single active
# engine lives in the module global `engine` (every endpoint selects it via
# `_get_engine` first); this is safe because the server serializes requests
# (Modal `max_inputs=1`, and the static KV cache already forbids concurrent
# generations). MODELS maps id -> checkpoint path (None = let InferenceEngine
# resolve NANO_CKPT / its default).
MODELS: dict[str, str | None] = {}
DEFAULT_MODEL: str = "default"
ACTIVE_MODEL: str | None = None
ENGINES: dict[str, InferenceEngine] = {}
engine: InferenceEngine | None = None

# When set, every generation is also written here (e.g. the nano-output volume
# on Modal, mounted at /outputs). Empty = response-only, nothing persisted.
OUTPUT_DIR = os.environ.get("NANO_OUTPUT_DIR", "")


def _parse_models_env() -> tuple[dict[str, str | None], str]:
    """Parse NANO_MODELS ("id=path,id2=path2") -> ({id: path}, default_id).

    The default is NANO_DEFAULT_MODEL if set, else the first listed id. When
    NANO_MODELS is unset, fall back to a single "default" model that
    InferenceEngine resolves from NANO_CKPT / its own default — so existing
    single-checkpoint deployments keep working unchanged."""
    raw = os.environ.get("NANO_MODELS", "").strip()
    if not raw:
        return {"default": os.environ.get("NANO_CKPT") or None}, "default"
    models: dict[str, str | None] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"NANO_MODELS entry '{part}' must be 'id=path'")
        mid, path = part.split("=", 1)
        models[mid.strip()] = path.strip()
    if not models:
        return {"default": os.environ.get("NANO_CKPT") or None}, "default"
    default = os.environ.get("NANO_DEFAULT_MODEL", "").strip() or next(iter(models))
    if default not in models:
        raise ValueError(
            f"NANO_DEFAULT_MODEL '{default}' is not one of NANO_MODELS ({', '.join(models)})"
        )
    return models, default


def _get_engine(model_id: str | None) -> InferenceEngine:
    """Select (lazy-loading if needed) the engine for `model_id` and make it the
    active one. Empty/None -> the default model. Raises 400 on an unknown id."""
    global engine, ACTIVE_MODEL
    mid = (model_id or "").strip() or DEFAULT_MODEL
    if mid not in MODELS:
        raise HTTPException(
            400, f"unknown model '{mid}'; available: {', '.join(MODELS) or '(none)'}"
        )
    eng = ENGINES.get(mid)
    if eng is None:
        print(f"[models] lazy-loading '{mid}' from {MODELS[mid] or '(default ckpt)'}", flush=True)
        eng = InferenceEngine(ckpt_path=MODELS[mid])
        ENGINES[mid] = eng
    engine = eng
    ACTIVE_MODEL = mid
    return eng


def _output_name(mode: str, prompt: str, ext: str = "mp3", uid: str | None = None) -> str:
    """Build a unique, sortable output filename. `uid` (a client-supplied req_id)
    is used as the trailing token when given, so the client can fetch the saved
    clip by id afterwards (see GET /outputs/{name}); otherwise a random suffix."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48] or "untitled"
    tail = re.sub(r"[^a-z0-9]+", "", (uid or "").lower())[:24] or uuid.uuid4().hex[:6]
    return f"{stamp}_{mode}_{slug}_{tail}.{ext}"


def _save_named(body: bytes, name: str) -> None:
    """Write bytes to OUTPUT_DIR/name; never fatal to the response."""
    if not OUTPUT_DIR:
        return
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Direct write, no temp+rename: filenames are unique per request, and on
        # a Modal volume the background commit can capture the temp file while
        # the rename's delete never propagates (orphaned .tmp entries).
        with open(os.path.join(OUTPUT_DIR, name), "wb") as f:
            f.write(body)
        print(f"[output] saved {name} ({len(body)} bytes)")
    except Exception as e:  # disk-full etc. must not break the response
        print(f"[output] save failed: {e}")


def _save_output(body: bytes, mime: str, mode: str, prompt: str) -> None:
    """Persist a generation to OUTPUT_DIR; never fatal to the response."""
    if not OUTPUT_DIR:
        return
    ext = "mp3" if "mpeg" in mime else "wav"
    _save_named(body, _output_name(mode, prompt, ext))


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


# Demucs (htdemucs) source order — the four stems the /stem endpoint can keep/drop.
STEM_NAMES = ("drums", "bass", "other", "vocals")
_STEM_ALIASES = {
    "vocal": "vocals", "vox": "vocals", "voice": "vocals", "voices": "vocals",
    "drum": "drums", "perc": "drums", "percussion": "drums",
    "instrument": "other", "instruments": "other",
}


def _parse_stem_list(raw: str) -> list[str]:
    """Comma-separated stem names -> canonical Demucs stems, de-duplicated.

    Folds a few aliases (``vox`` -> ``vocals`` etc.); raises 400 on an unknown
    name so a typo fails loudly instead of silently keeping/dropping nothing."""
    out: list[str] = []
    for tok in raw.split(","):
        s = _STEM_ALIASES.get(tok.strip().lower(), tok.strip().lower())
        if not s:
            continue
        if s not in STEM_NAMES:
            raise HTTPException(400, f"unknown stem '{tok.strip()}'; valid: {', '.join(STEM_NAMES)}")
        if s not in out:
            out.append(s)
    return out


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
    global MODELS, DEFAULT_MODEL
    MODELS, DEFAULT_MODEL = _parse_models_env()
    print(f"[models] registry: {', '.join(f'{k}={v or 'default'}' for k, v in MODELS.items())} "
          f"(default={DEFAULT_MODEL})", flush=True)
    # Eager-load only the default model; others lazy-load on first request.
    _get_engine(DEFAULT_MODEL)
    # Warm up at container init so the first real request doesn't pay
    # torch.compile tracing + CUDA-graph capture + the sweetener load on the
    # request path. The compiled decode graph is keyed on sequence length, so warm
    # at the UI's default (NANO_WARMUP_SECONDS, 30s) — that length is then fast;
    # other lengths recompile once on first use. Set NANO_WARMUP_SECONDS=0 to skip.
    warm_s = float(os.environ.get("NANO_WARMUP_SECONDS", "30"))
    if warm_s > 0:
        try:
            print(f"[warmup] sweetener + compiling decode graph at {warm_s:.0f}s...")
            import time as _time
            t0 = _time.time()
            try:
                engine.sweeten_prompt("warm up the cache")
            except Exception as e:
                print(f"[warmup] sweetener skipped: {e}")
            warm_cfg = float(os.environ.get("NANO_WARMUP_CFG", "7.0"))
            engine.generate_audio(seconds=warm_s, text="warm up", cfg_scale=warm_cfg)
            print(f"[warmup] done in {_time.time() - t0:.1f}s")
        except Exception as e:
            print(f"[warmup] skipped: {e}")
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
        # Active/available models (the `model` request field selects among these).
        "model": ACTIVE_MODEL,
        "default_model": DEFAULT_MODEL,
        "available_models": list(MODELS),
    }


@app.get("/models")
def list_models() -> dict:
    """List the available checkpoints the `model` request field can select.

    Reports each id, whether it's currently loaded, and (when loaded) its
    checkpoint path / step / param count. The UI populates its model picker from
    this; unloaded models load lazily on their first generation request."""
    out = []
    for mid in MODELS:
        eng = ENGINES.get(mid)
        entry: dict = {"id": mid, "loaded": eng is not None, "default": mid == DEFAULT_MODEL}
        if eng is not None:
            entry.update(
                ckpt_path=eng.ckpt_path,
                ckpt_step=eng.ckpt_step,
                model_params=eng.model.num_params(),
            )
        out.append(entry)
    return {"models": out, "default_model": DEFAULT_MODEL, "active_model": ACTIVE_MODEL}


@app.post("/generate")
async def generate_endpoint(
    seconds: float = Form(30.0),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    # Defaults = sweep winner cfg7.0_WARM_LADDER (eval/sweep/config.py). The WARM
    # ladder rides per_cb_temperature/top_k; top_p (0.95) and cfg (7.0) below match.
    per_cb_temperature: str = Form("1.05,0.98,0.9,0.82,0.74,0.66,0.58,0.5,0.42"),
    per_cb_top_k: str = Form("120,90,70,50,36,26,18,12,8"),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(7.0),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    gender: str = Form(""),
    bpm: float = Form(0.0),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    style_audio: UploadFile | None = File(None),
    style_weight: float = Form(0.5),
    lyric_cfg_scale: float = Form(8.0),
    score_clap: bool = Form(False),
    model: str = Form(""),
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
    _get_engine(model)
    assert engine is not None
    style_bytes = (await style_audio.read()) if style_audio else None
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    try:
        result = engine.generate_audio(
            seconds=seconds,
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=prompt or None,
            lyrics=(lyrics or "").strip() or None,
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
    _save_output(body, mime, "generate", prompt)
    return Response(content=body, media_type=mime, headers=headers)


class BatchItem(BaseModel):
    """One clip in a /generate_batch request. Any field left unset inherits the
    batch-level shared default of the same name."""
    prompt: str | None = None
    lyrics: str | None = None
    gender: str | None = None
    bpm: float | None = None
    negative_prompt: str | None = None
    req_id: str | None = None


class BatchRequest(BaseModel):
    """Generate several clips in ONE batched forward ("8 at once" on one GPU).

    Two ways to specify the batch:
    - ``items``: an explicit list of per-clip params (many DIFFERENT prompts), each
      inheriting the shared fields below where unset.
    - ``count`` (with no ``items``): N identical TAKES of the shared prompt/lyrics.

    Sampling, cfg, and ``seconds`` are shared across the batch (the decode loop
    applies one set batch-wide)."""
    items: list[BatchItem] | None = None
    count: int = 1
    # shared conditioning (defaults for items, or the prompt replicated `count`x)
    prompt: str = ""
    lyrics: str = ""
    gender: str = ""
    bpm: float = 0.0
    negative_prompt: str = ""
    # shared sampling / guidance
    seconds: float = 30.0
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    per_cb_temperature: str = "1.05,0.98,0.9,0.82,0.74,0.66,0.58,0.5,0.42"
    per_cb_top_k: str = "120,90,70,50,36,26,18,12,8"
    per_cb_top_p: str = ""
    cfg_scale: float = 7.0
    lyric_cfg_scale: float = 8.0
    sweeten: bool = True
    model: str = ""


@app.post("/generate_batch")
def generate_batch_endpoint(req: BatchRequest) -> dict:
    """Generate up to NANO_MAX_BATCH clips from scratch in a SINGLE batched run.

    Because autoregressive decode is memory-bandwidth-bound, B clips cost ≈ the
    wall-clock of one — so this is the cheap "generate 8 at once" path (one GPU,
    one batched forward) as opposed to fanning N requests across N containers.

    Returns a JSON manifest: ``{"items": [{req_id, mime, sweetened_prompt,
    audio_b64}, ...], "count", "seconds"}``. Each clip is returned inline as base64
    (works with no server-side persistence) AND saved to OUTPUT_DIR when configured
    (fetchable later via GET /outputs/{req_id})."""
    _get_engine(req.model)
    assert engine is not None
    if req.seconds <= 0:
        raise HTTPException(400, "seconds must be > 0")

    # Resolve the batch into a flat list of per-clip param dicts.
    raw_items = req.items if req.items else [BatchItem() for _ in range(max(1, req.count))]
    if not raw_items:
        raise HTTPException(400, "empty batch")
    if len(raw_items) > MAX_BATCH:
        raise HTTPException(400, f"batch too large: {len(raw_items)} > NANO_MAX_BATCH={MAX_BATCH}")

    def _pick(item_val, shared_val):
        return item_val if item_val is not None else shared_val

    # Sweeten each DISTINCT prompt once (N identical takes -> one LLM call).
    sweet_cache: dict[str, tuple[str, dict]] = {}

    def _sweet(p: str) -> str:
        if p not in sweet_cache:
            sweet_cache[p] = _maybe_sweeten(p, req.sweeten)
        return sweet_cache[p][0]

    manifest: list[dict] = []
    gen_requests: list[dict] = []
    for n, item in enumerate(raw_items):
        prompt = _pick(item.prompt, req.prompt)
        lyrics = _pick(item.lyrics, req.lyrics)
        gender = _pick(item.gender, req.gender)
        bpm = _pick(item.bpm, req.bpm)
        neg = _pick(item.negative_prompt, req.negative_prompt)
        sweetened = _sweet(prompt)
        gen_requests.append({
            "text": sweetened or None,
            "lyrics": (lyrics or "").strip() or None,
            "negative_text": neg.strip() or None,
            "gender": _norm_gender(gender),
            "bpm": bpm or None,
        })
        req_id = (item.req_id or "").strip() or f"b{n}{uuid.uuid4().hex[:10]}"
        manifest.append({"req_id": req_id, "sweetened_prompt": sweetened, "_prompt": prompt})

    def _on_item(i: int, body: bytes, mime: str) -> None:
        ext = "mp3" if "mpeg" in mime else "wav"
        name = _output_name("generate", manifest[i]["_prompt"], ext, uid=manifest[i]["req_id"])
        _save_named(body, name)
        manifest[i]["mime"] = mime
        manifest[i]["audio_b64"] = base64.b64encode(body).decode("ascii")
        manifest[i]["file"] = name

    try:
        engine.generate_audio_batch(
            gen_requests,
            seconds=req.seconds,
            temperature=_parse_per_cb_temp(req.per_cb_temperature, req.temperature),
            top_k=_parse_per_cb_topk(req.per_cb_top_k, req.top_k),
            top_p=_parse_per_cb_topp(req.per_cb_top_p, req.top_p),
            cfg_scale=req.cfg_scale,
            lyric_cfg_scale=req.lyric_cfg_scale or None,
            on_item=_on_item,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    for m in manifest:
        m.pop("_prompt", None)
    return {"items": manifest, "count": len(manifest), "seconds": req.seconds}


def _generate_stream_response(
    *, seconds, temperature, top_k, top_p, per_cb_temperature, per_cb_top_k,
    per_cb_top_p, cfg_scale, prompt, lyrics, gender, bpm, negative_prompt,
    sweeten, lyric_cfg_scale, req_id, model,
) -> StreamingResponse:
    """Shared body for the GET and POST /generate_stream endpoints."""
    _get_engine(model)
    assert engine is not None
    if seconds <= 0:
        raise HTTPException(400, "seconds must be > 0")
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    name = _output_name("generate", prompt, "mp3", uid=req_id or None)

    stream = engine.generate_audio_stream(
        seconds=seconds,
        temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
        top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
        top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
        cfg_scale=cfg_scale,
        text=prompt or None,
        lyrics=(lyrics or "").strip() or None,
        negative_text=negative_prompt.strip() or None,
        lyric_cfg_scale=lyric_cfg_scale or None,
        gender=_norm_gender(gender),
        bpm=bpm or None,
        on_complete=lambda body, mime: _save_named(body, name),
    )
    headers = dict(sweet_headers)
    headers["X-Nano-Output-File"] = name
    headers["Access-Control-Expose-Headers"] = (
        "X-Nano-Output-File, X-Nano-Sweetened-Prompt"
    )
    return StreamingResponse(stream, media_type="audio/mpeg", headers=headers)


@app.get("/generate_stream")
def generate_stream_endpoint(
    seconds: float = 30.0,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 0.95,
    per_cb_temperature: str = "1.05,0.98,0.9,0.82,0.74,0.66,0.58,0.5,0.42",
    per_cb_top_k: str = "120,90,70,50,36,26,18,12,8",
    per_cb_top_p: str = "",
    cfg_scale: float = 7.0,
    prompt: str = "",
    lyrics: str = "",
    gender: str = "",
    bpm: float = 0.0,
    negative_prompt: str = "",
    sweeten: bool = True,
    lyric_cfg_scale: float = 8.0,
    req_id: str = "",
    model: str = "",
) -> StreamingResponse:
    """Streaming counterpart to POST /generate — emits MP3 bytes progressively.

    A native <audio src> can point straight at this URL for progressive playback;
    because bytes start flowing within ~one chunk, the response stays "active" and
    slips under Modal's 150 s web-endpoint timeout (which kills the buffered POST
    path on long clips). Query params mirror POST /generate minus style_audio (no
    file upload on GET) and score_clap (needs the whole clip).

    Pass a client-generated `req_id`; the canonical gapless clip is saved under a
    filename ending in that id, so the client can fetch it from GET /outputs/{id}
    after playback for download / accurate duration / a gapless re-listen.

    Lyrics longer than a GET URL can carry should use POST /generate_stream.
    """
    return _generate_stream_response(
        seconds=seconds, temperature=temperature, top_k=top_k, top_p=top_p,
        per_cb_temperature=per_cb_temperature, per_cb_top_k=per_cb_top_k,
        per_cb_top_p=per_cb_top_p, cfg_scale=cfg_scale, prompt=prompt,
        lyrics=lyrics, gender=gender, bpm=bpm, negative_prompt=negative_prompt,
        sweeten=sweeten, lyric_cfg_scale=lyric_cfg_scale, req_id=req_id, model=model,
    )


@app.post("/generate_stream")
def generate_stream_post_endpoint(
    seconds: float = Form(30.0),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    per_cb_temperature: str = Form("1.05,0.98,0.9,0.82,0.74,0.66,0.58,0.5,0.42"),
    per_cb_top_k: str = Form("120,90,70,50,36,26,18,12,8"),
    per_cb_top_p: str = Form(""),
    cfg_scale: float = Form(7.0),
    prompt: str = Form(""),
    lyrics: str = Form(""),
    gender: str = Form(""),
    bpm: float = Form(0.0),
    negative_prompt: str = Form(""),
    sweeten: bool = Form(True),
    lyric_cfg_scale: float = Form(8.0),
    req_id: str = Form(""),
    model: str = Form(""),
) -> StreamingResponse:
    """POST form variant of GET /generate_stream — same progressive MP3 stream,
    but lyrics ride the request body, so long lyrics that would overflow a GET URL
    still stream (instead of dropping to a no-progress buffered POST /generate)."""
    return _generate_stream_response(
        seconds=seconds, temperature=temperature, top_k=top_k, top_p=top_p,
        per_cb_temperature=per_cb_temperature, per_cb_top_k=per_cb_top_k,
        per_cb_top_p=per_cb_top_p, cfg_scale=cfg_scale, prompt=prompt,
        lyrics=lyrics, gender=gender, bpm=bpm, negative_prompt=negative_prompt,
        sweeten=sweeten, lyric_cfg_scale=lyric_cfg_scale, req_id=req_id, model=model,
    )


@app.get("/outputs/{name}")
def get_output(name: str) -> FileResponse:
    """Serve a saved generation from OUTPUT_DIR. Accepts either the exact filename
    or a req_id (the streaming path saves <stamp>_generate_<slug>_<req_id>.mp3, so
    the client fetches the canonical clip knowing only the id it generated)."""
    if not OUTPUT_DIR:
        raise HTTPException(404, "outputs not persisted on this server")
    safe = os.path.basename(name)
    if not safe or safe != name:
        raise HTTPException(400, "bad name")
    path = os.path.join(OUTPUT_DIR, safe)
    if not os.path.isfile(path):
        token = re.sub(r"[^a-z0-9]+", "", safe.lower())[:24]
        matches = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"*_{token}.mp3"))) if token else []
        if not matches:
            raise HTTPException(404, "not found")
        path = matches[-1]
    mime = "audio/mpeg" if path.endswith(".mp3") else "audio/wav"
    return FileResponse(path, media_type=mime)


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
    lyric_cfg_scale: float = Form(8.0),
    model: str = Form(""),
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
    _get_engine(model)
    assert engine is not None
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    style_bytes = (await style_audio.read()) if style_audio else None
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
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
            text=prompt or None,
            lyrics=(lyrics or "").strip() or None,
            negative_text=negative_prompt.strip() or None,
            style_audio_bytes=style_bytes or None,
            style_weight=style_weight,
            lyric_cfg_scale=lyric_cfg_scale or None,
            gender=_norm_gender(gender),
            bpm=bpm or None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    _save_output(body, mime, "extend", prompt)
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
    lyric_cfg_scale: float = Form(8.0),
    model: str = Form(""),
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
    _get_engine(model)
    assert engine is not None
    data = await melody_audio.read()
    if not data:
        raise HTTPException(400, "empty melody_audio upload")
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    try:
        body, mime = engine.cover_audio(
            data,
            temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
            top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
            top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
            cfg_scale=cfg_scale,
            text=prompt or None,
            lyrics=(lyrics or "").strip() or None,
            negative_text=negative_prompt.strip() or None,
            melody_cfg_scale=melody_cfg_scale or None,
            lyric_cfg_scale=lyric_cfg_scale or None,
            gender=_norm_gender(gender),
            bpm=bpm or None,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    _save_output(body, mime, "cover", prompt)
    return Response(content=body, media_type=mime, headers=sweet_headers)


@app.post("/extend_stream")
async def extend_stream_endpoint(
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
    lyric_cfg_scale: float = Form(8.0),
    req_id: str = Form(""),
    model: str = Form(""),
) -> StreamingResponse:
    """Streaming /extend: yields the kept original instantly, then the generated
    continuation as it's produced (progressive MSE playback; bypasses the 150s
    wall). The finished clip is saved to OUTPUT_DIR (fetch via /outputs/{req_id})."""
    _get_engine(model)
    assert engine is not None
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    name = _output_name("extend", prompt, "mp3", uid=req_id or None)
    stream = engine.extend_audio_stream(
        data,
        add_seconds=add_seconds,
        overlap_seconds=overlap_seconds,
        from_seconds=from_seconds if from_seconds >= 0 else None,
        temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
        top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
        top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
        cfg_scale=cfg_scale,
        text=prompt or None,
        lyrics=(lyrics or "").strip() or None,
        negative_text=negative_prompt.strip() or None,
        lyric_cfg_scale=lyric_cfg_scale or None,
        gender=_norm_gender(gender),
        bpm=bpm or None,
        on_complete=lambda body, mime: _save_named(body, name),
    )
    headers = dict(sweet_headers)
    headers["X-Nano-Output-File"] = name
    headers["Access-Control-Expose-Headers"] = (
        "X-Nano-Output-File, X-Nano-Sweetened-Prompt"
    )
    return StreamingResponse(stream, media_type="audio/mpeg", headers=headers)


@app.post("/cover_stream")
async def cover_stream_endpoint(
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
    lyric_cfg_scale: float = Form(8.0),
    req_id: str = Form(""),
    model: str = Form(""),
) -> StreamingResponse:
    """Streaming /cover: re-render the hum's melody in the prompt's timbre,
    streaming the result as it generates. Needs a melody-trained checkpoint."""
    _get_engine(model)
    assert engine is not None
    data = await melody_audio.read()
    if not data:
        raise HTTPException(400, "empty melody_audio upload")
    prompt, sweet_headers = _maybe_sweeten(prompt, sweeten)
    name = _output_name("cover", prompt, "mp3", uid=req_id or None)
    stream = engine.cover_audio_stream(
        data,
        temperature=_parse_per_cb_temp(per_cb_temperature, temperature),
        top_k=_parse_per_cb_topk(per_cb_top_k, top_k),
        top_p=_parse_per_cb_topp(per_cb_top_p, top_p),
        cfg_scale=cfg_scale,
        text=prompt or None,
        lyrics=(lyrics or "").strip() or None,
        negative_text=negative_prompt.strip() or None,
        melody_cfg_scale=melody_cfg_scale or None,
        lyric_cfg_scale=lyric_cfg_scale or None,
        gender=_norm_gender(gender),
        bpm=bpm or None,
        on_complete=lambda body, mime: _save_named(body, name),
    )
    headers = dict(sweet_headers)
    headers["X-Nano-Output-File"] = name
    headers["Access-Control-Expose-Headers"] = (
        "X-Nano-Output-File, X-Nano-Sweetened-Prompt"
    )
    return StreamingResponse(stream, media_type="audio/mpeg", headers=headers)


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
    model: str = Form(""),
) -> Response:
    """Fill the gap between two clips — returns ``[before | middle | after]``.

    `before_audio` and `after_audio` are kept verbatim; the model generates a
    `gap_seconds` bridge that flows out of the first and into the second.
    `prompt` (tags) drives the timbre of the fill. An optional `melody_audio` hum
    guides the gap's melodic contour. Lyrics are not used by infill.

    Requires a checkpoint trained with FIM (use_fim — a v8+ model).
    """
    _get_engine(model)
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
    _save_output(body, mime, "infill", prompt)
    return Response(content=body, media_type=mime, headers=sweet_headers)


@app.post("/stem")
async def stem_endpoint(
    audio: UploadFile = File(...),
    remove: str = Form("vocals"),
    keep: str = Form(""),
    model: str = Form(""),
) -> Response:
    """Remove or isolate instrument stems from an upload via Demucs separation.

    Pure source separation — the nano model is NOT used, so this works with any
    checkpoint (and needs the `demucs` package installed). Demucs splits the audio
    into drums / bass / other / vocals; the result is the kept stems summed back
    into one mono mixdown.

    - `remove` (default `vocals`): comma-separated stems to drop — e.g.
      `remove=vocals` -> instrumental, `remove=drums,bass` -> drop the rhythm
      section.
    - `keep`: when set, OVERRIDES `remove` and keeps exactly these — e.g.
      `keep=vocals` -> a-cappella, `keep=drums` -> drums only.

    Stem names: drums, bass, other, vocals (aliases like `vox` fold in).
    Generating a brand-new stem (the complementary "add" direction) needs a
    stem-conditioned checkpoint and is not this endpoint.
    """
    _get_engine(model)
    assert engine is not None
    data = await audio.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    keep_list = _parse_stem_list(keep)
    remove_list = _parse_stem_list(remove)
    # `keep` wins when given; otherwise keep everything not in `remove`.
    if keep_list:
        keep_set = [s for s in STEM_NAMES if s in keep_list]
    else:
        keep_set = [s for s in STEM_NAMES if s not in remove_list]
    if not keep_set:
        raise HTTPException(400, "nothing left to keep — remove fewer stems")
    try:
        body, mime = engine.separate_stems(data, keep=keep_set)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    _save_output(body, mime, "stem", "+".join(keep_set))
    return Response(content=body, media_type=mime)
