"""Realistic lyric sweep: WER + listenable clips across cfg x lyric_cfg x ladder
x {lyrics, instrumental}, through the EXACT server inference path.

Unlike scripts/eval_lyric_wer.py (which calls model.generate() directly, with no
sweetener and a fixed ladder), this drives server.inference.InferenceEngine the
same way the /generate endpoint does:

    tags --sweeten--> "sweetened_tags. <lyric|[instrumental]>" --> generate_audio
    (per-cb temperature/top_k ladder, cfg_scale, lyric_cfg_scale)

so the clips match what a user actually hears. Each cell then transcribes its
output back (Demucs + faster-whisper, the same pipeline that built the training
lyrics) for WER (lyric cells) or a leaked-word count (instrumental cells, which
should be ~0 if the <instrumental> vocal-presence marker works). One
representative clip per cell is saved to the nano-output volume to listen to.

Grid (4 axes):
  cfg_scale          x {3.0, 5.0}
  lyric_cfg_scale    x {0.0, 3.0, 6.0}   (lyric cells only)
  ladder             x {HOT_FLAT, COLD_LADDER}
  conditioning       x {lyrics, instrumental, none}
-> 12 lyric cells + 4 instrumental + 4 none (tags-only, no lyric stream at all)
   = 20 cells, n_clips each. "none" is the genuine lyric-free baseline:
   instrumental still feeds the <instrumental> marker header, "none" feeds nothing.

Run:  modal run scripts/eval_lyric_sweep.py --ckpt-path /ckpts/v8_sing2/best_inference.pt
      modal run scripts/eval_lyric_sweep.py --n-clips 3 --seconds 12
"""
from __future__ import annotations

import modal

app = modal.App("nano-eval-lyric-sweep")


def _prefetch_g2p() -> None:
    import nltk

    for res in ("averaged_perceptron_tagger_eng", "cmudict", "averaged_perceptron_tagger"):
        nltk.download(res, quiet=True)
    from g2p_en import G2p

    G2p()("warm up the cache")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # espeak-ng + phonemizer: required by the v10/IPA-256 lyric path — without
    # them text_to_phoneme_ids silently drops every word and the WER cells
    # measure a wordless header (the 2026-07-23 no-vocals eval artifact).
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng")
    .pip_install("phonemizer>=3.2")
    # Pin torch to CUDA 12.1 — faster-whisper's ctranslate2 links libcublas.so.12
    # (an unpinned torch>=2.4 pulls a CUDA-13 wheel and crashes WhisperModel init).
    .pip_install(
        "torch==2.4.1",
        "torchaudio==2.4.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "librosa>=0.10",
        "descript-audio-codec>=1.0.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "msclap",
        "transformers>=4.40",
        "accelerate>=0.30",
        "g2p_en==2.1.0",
        "demucs",
        "faster-whisper",
        "jiwer",
    )
    .run_commands("pip install 'protobuf>=4'")
    .run_function(_prefetch_g2p)
    .add_local_python_source("model", "diskrot", "server")
)

tokens_vol = modal.Volume.from_name("nano-tokens")
ckpts_vol = modal.Volume.from_name("nano-ckpts")
output_vol = modal.Volume.from_name("nano-output", create_if_missing=True)

# Ladder profiles (temperature, top_k), reused from eval/sweep/config.py — scalar
# = flat, length-9 = per-codebook (coarse hot / fine cold).
LADDERS = {
    # COLD_LADDER first: it's the realistic serving config, so its clips land
    # before the HOT_FLAT control (which sounds like noise on a 9-codebook model).
    "COLD_LADDER": (
        [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25],
        [120, 90, 70, 50, 36, 26, 18, 12, 8],
    ),
    "HOT_FLAT": (0.9, 50),
}


def _slice_lyric(words: list[dict], max_words: int) -> str:
    toks = [w["word"].strip() for w in words if w.get("word", "").strip()]
    return " ".join(toks[:max_words])


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol, "/outputs": output_vol},
)
def sweep(
    ckpt_path: str = "/ckpts/v8_sing2/best_inference.pt",
    n_clips: int = 5,
    seconds: float = 12.0,
    cfg_scales: str = "3.0,5.0",
    lyric_cfgs: str = "0.0,3.0,6.0",
    max_words: int = 12,
    seed: int = 42,
    whisper_size: str = "large-v3",
):
    import os
    import random
    import tempfile

    import jiwer
    import soundfile as sf

    from server.inference import InferenceEngine
    from diskrot.dataset import _load_lyrics, _load_tags
    from diskrot.transcribe_lyrics import _load_demucs, _separate_vocals, _transcribe

    cfgs = [float(x) for x in cfg_scales.split(",")]
    lcfgs = [float(x) for x in lyric_cfgs.split(",")]

    # Real inference path (sweetener + generate_audio + per-cb ladder).
    os.environ["NANO_CKPT"] = ckpt_path
    eng = InferenceEngine(ckpt_path=ckpt_path)
    print(f"loaded engine on {eng.device} from {ckpt_path}")

    # Held-out lyric-bearing songs — SAME set across every cell (seed) so cell
    # differences are attributable to the settings, not the sample.
    lyrics = _load_lyrics("/tokens/lyrics", verbose=False)
    tags = _load_tags("/tokens/tags.json", verbose=False)
    names = [n for n, e in lyrics.items() if e.get("words")]
    random.Random(seed).shuffle(names)
    names = names[:n_clips]
    if not names:
        raise SystemExit("no lyric-bearing songs under /tokens/lyrics")

    # Pre-sweeten each song's tags ONCE (deterministic across cells, far cheaper
    # than re-running Qwen per cell) and pre-slice its reference lyric.
    refs = {}
    for name in names:
        ref = _slice_lyric(lyrics[name]["words"], max_words)
        tag_str = tags.get(name, "")
        sweet = eng.sweeten_prompt(tag_str) if tag_str else ""
        refs[name] = {"ref": ref, "sweet": sweet}
        print(f"  [{name}] tags->{sweet[:60]!r} ref={ref!r}")

    print("loading Demucs + faster-whisper for transcribe-back...")
    demucs_model, apply_fn = _load_demucs(eng.device)
    from faster_whisper import WhisperModel

    whisper = WhisperModel(whisper_size, device=eng.device, compute_type="float16")

    norm = jiwer.Compose([
        jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
    ])

    # Build the cell grid: lyric cells (cfg x lyric_cfg x ladder) + instrumental
    # controls (cfg x ladder; lyric_cfg is irrelevant with no lyric stream).
    cells = []
    for ladder_name in LADDERS:
        for cfg in cfgs:
            for lcfg in lcfgs:
                cells.append({"cond": "lyrics", "cfg": cfg, "lyric_cfg": lcfg, "ladder": ladder_name})
            cells.append({"cond": "instrumental", "cfg": cfg, "lyric_cfg": 0.0, "ladder": ladder_name})
            # "none" = tags only, NO lyric stream at all (not even the marker
            # header) — the genuine lyric-free baseline. Metric is leaked_words,
            # same as instrumental, but here nothing suppresses vocals.
            cells.append({"cond": "none", "cfg": cfg, "lyric_cfg": 0.0, "ladder": ladder_name})

    tag = ckpt_path.rstrip("/").split("/")[-1].replace(".pt", "")
    out_dir = f"/outputs/lyric_sweep/{tag}"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    temp_t, temp_k = None, None
    for ci, cell in enumerate(cells):
        temp, topk = LADDERS[cell["ladder"]]
        cid = f"{cell['cond']}_cfg{cell['cfg']:g}_lcfg{cell['lyric_cfg']:g}_{cell['ladder']}"
        wers, leaked, saved = [], [], False
        for i, name in enumerate(names):
            r = refs[name]
            if cell["cond"] == "lyrics":
                if not r["ref"]:
                    continue
                lyric_field = r["ref"]
            elif cell["cond"] == "instrumental":
                lyric_field = "[instrumental]"
            else:  # "none" — tags only, no lyric field appended
                lyric_field = None
            if lyric_field is None:
                # Tags-only: sanitize internal ". " so no tag prose spills past
                # the engine's first-". " tags|lyrics split into the lyric slot.
                combined = r["sweet"].replace(". ", ", ") if r["sweet"] else ""
            else:
                combined = f"{r['sweet']}. {lyric_field}" if r["sweet"] else lyric_field

            audio, mime = eng.generate_audio(
                seconds=seconds, text=combined,
                temperature=temp, top_k=topk, top_p=0.95,
                cfg_scale=cell["cfg"],
                lyric_cfg_scale=(cell["lyric_cfg"] or None),
            )
            ext = "mp3" if mime == "audio/mpeg" else "wav"
            with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
                clip_path = f.name
                f.write(audio)
            # Keep the first clip of each cell to listen to.
            if not saved:
                with open(f"{out_dir}/{cid}.{ext}", "wb") as fo:
                    fo.write(audio)
                saved = True

            vocals = _separate_vocals(demucs_model, apply_fn, clip_path, eng.device)
            hyp = (_transcribe(whisper, vocals) or {}).get("text", "")
            h = norm(hyp)
            if cell["cond"] == "lyrics":
                rr = norm(r["ref"])
                if rr:
                    wers.append(jiwer.wer(rr, h))
            else:
                leaked.append(len(h.split()))

        if cell["cond"] == "lyrics":
            metric = sum(wers) / len(wers) if wers else None
            rows.append((cid, cell, f"WER={metric:.3f}" if metric is not None else "WER=NA", len(wers)))
        else:
            metric = sum(leaked) / len(leaked) if leaked else None
            rows.append((cid, cell, f"leaked_words={metric:.1f}" if metric is not None else "NA", len(leaked)))
        print(f"[{ci+1}/{len(cells)}] {cid:48s} {rows[-1][2]}  (n={rows[-1][3]})")
        output_vol.commit()

    print("\n==================== LYRIC SWEEP RESULTS ====================")
    print(f"ckpt={tag}  n_clips={n_clips}  seconds={seconds}  (sweetener ON, real /generate path)")
    print(f"{'cell':50s} {'metric':18s} n")
    for cid, cell, metric, n in rows:
        print(f"{cid:50s} {metric:18s} {n}")
    print(f"\nclips saved to nano-output:/lyric_sweep/{tag}/  (one per cell)")
    print(f"pull:  modal volume get nano-output /lyric_sweep/{tag} ./sweep_clips --force")
    output_vol.commit()


@app.local_entrypoint()
def main(
    ckpt_path: str = "/ckpts/v8_sing2/best_inference.pt",
    n_clips: int = 5,
    seconds: float = 12.0,
    cfg_scales: str = "3.0,5.0",
    lyric_cfgs: str = "0.0,3.0,6.0",
):
    sweep.remote(
        ckpt_path=ckpt_path, n_clips=n_clips, seconds=seconds,
        cfg_scales=cfg_scales, lyric_cfgs=lyric_cfgs,
    )
