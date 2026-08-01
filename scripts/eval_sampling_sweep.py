"""Two-phase sampling/CFG sweep to dial in nano's DEFAULT generation settings.

Goal: find one set of default sampling settings that sounds best BY EAR across
four lofi sub-genres (pop / hiphop / electronica / indie), in both lyric (using
[intro][verse][chorus] markers) and instrumental modes, at 60 seconds. The knobs
are exactly the webapp "advanced" panel: cfg_scale, the sampling shape
(temperature / top_k / top_p, scalar OR per-codebook length-9 ladders), and
lyric_cfg_scale.

Strategy (confirmed with the user): AUTO-NARROW -> EAR-PICK.
  Phase 1  renders the full grid at SHORT clips and auto-scores each clip with a
           librosa collapse/rhythm gate + a CLAP genre-match score, to drop the
           obvious failures and rank (profile, cfg) by CROSS-GENRE robustness.
  Phase 2  re-renders only the top finalists at full 60s, organized per
           genre/mode, for the user to choose by ear. CLAP is a coarse filter
           only -- the final pick is the user's.

Curated profiles bundle the per-cb ladder shape AND top_p into one axis (the two
the user most wants nailed, alongside cfg and lyric_cfg). lyric_cfg is NOT
auto-scored (CLAP/librosa can't measure word intelligibility, which is exactly
what lyric_cfg trades off) -- it is swept in the Phase-2 ear test instead.

TWO WAYS TO RUN (same grid, same code via run_sweep_core):

  Modal (H100, CLAP genre-match scoring ON):
    modal run --detach scripts/eval_sampling_sweep.py \
      --ckpt-path /ckpts/v8_sing4/best_inference.pt
    modal run scripts/eval_sampling_sweep.py --smoke
    # pull: modal volume get nano-output /sampling_sweep/<tag> ./sweep_clips --force

  Local (Apple Silicon / MPS, CLAP scoring OFF -- torchcodec crashes on ffmpeg8
  locally, so ranking is librosa-only; clips land in ./sweep_out/):
    python scripts/eval_sampling_sweep.py --ckpt-path checkpoints/latest.pt
    python scripts/eval_sampling_sweep.py --smoke        # quick end-to-end + timing
"""
from __future__ import annotations

import modal

app = modal.App("nano-eval-sampling-sweep")


# --- image: mirror diskrot/modal_serve.py (prefetch DAC + CLAP + Qwen sweetener
#     + g2p_en so a cold container never blocks on a CDN download) -------------
def _prefetch_weights() -> None:
    import dac
    from msclap import CLAP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Must match prompt_sweetener.DEFAULT_MODEL (hardcoded: this runs at
    # image-build time, before add_local_python_source mounts `server`).
    sweetener_model = "Qwen/Qwen2.5-1.5B-Instruct"
    dac.utils.download(model_type="44khz")
    CLAP(version="2023", use_cuda=False)
    AutoTokenizer.from_pretrained(sweetener_model)
    AutoModelForCausalLM.from_pretrained(sweetener_model)


def _prefetch_g2p() -> None:
    import nltk

    for res in ("averaged_perceptron_tagger_eng", "cmudict", "averaged_perceptron_tagger"):
        nltk.download(res, quiet=True)
    from g2p_en import G2p

    G2p()("warm up the cache")


image = (
    modal.Image.debian_slim(python_version="3.12")
    # espeak-ng + phonemizer: the v10/IPA-256 lyric path phonemizes request lyrics
    # via phonemizer's EspeakBackend — WITHOUT them text_to_phoneme_ids silently
    # drops every word and "lyric" cells run on a wordless header (the 2026-07-23
    # no-vocals eval artifact). g2p_en stays only for legacy v8/ARPABET ckpts.
    .apt_install("ffmpeg", "libsndfile1", "espeak-ng")
    .pip_install(
        "phonemizer>=3.2",
        "torch>=2.4",
        "torchaudio>=2.4",
        "librosa>=0.10",
        "descript-audio-codec>=1.0.0",
        "numpy>=1.26",
        "soundfile>=0.12",
        "fastapi>=0.115",
        "python-multipart>=0.0.12",
        "msclap",
        "transformers>=4.35",
        "g2p_en==2.1.0",
    )
    .run_commands("pip install 'protobuf>=4.25,<5'")
    .run_function(_prefetch_weights)
    .run_function(_prefetch_g2p)
    .add_local_python_source("model", "diskrot", "server")
)

ckpts_vol = modal.Volume.from_name("nano-ckpts", create_if_missing=True)
output_vol = modal.Volume.from_name("nano-output", create_if_missing=True)


# ── Fixed content (held identical across every cell so score differences are
# attributable to the SETTINGS, not the content). ────────────────────────────
GENRES = {
    "lofi_pop": {
        "tag": "warm lo-fi bedroom pop, soft female vocals, mellow electric piano, "
               "gentle drums, vinyl crackle, hazy and nostalgic",
        "gender": "[female]",
    },
    "lofi_hiphop": {
        "tag": "chilled lo-fi hip hop beat, boom-bap drums, dusty jazzy piano, "
               "mellow bass, vinyl crackle, relaxed head-nod groove",
        "gender": "",
    },
    "lofi_electronica": {
        "tag": "lo-fi electronica, downtempo, warm analog synth pads, soft "
               "four-on-the-floor, tape saturation, dreamy and atmospheric",
        "gender": "",
    },
    "lofi_indie": {
        "tag": "lo-fi indie, jangly reverb guitar, soft male vocals, lazy drums, "
               "warm cassette hiss, wistful and intimate",
        "gender": "[male]",
    },
}

# One fixed lyric, structure markers the user asked for. gender + tempo header
# markers are prepended per genre (see _lyric_field).
LYRIC_BODY = """[intro]
[verse]
neon hums against the rain
counting taxis down the lane
every window holds a name
nothing here will feel the same
[chorus]
slow it down let the night unfold
hold the quiet leave the noise untold
[verse]
coffee going cold again
turning pages one to ten
[chorus]
slow it down let the night unfold
hold the quiet leave the noise untold
[outro]"""

# ── Curated sampling profiles: name -> (temperature, top_k, top_p). scalar =
# flat, length-9 = per-codebook ladder. Spans the per-cb ladder SHAPE (the
# user's flagged-untuned axis) and top_p (their current 0.44 is unusually low). ─
_DEF_TEMP = [1.05, 0.98, 0.9, 0.82, 0.74, 0.66, 0.58, 0.5, 0.42]
_DEF_TOPK = [120, 90, 70, 50, 36, 26, 18, 12, 8]
PROFILES = {
    "default_ladder": (_DEF_TEMP, _DEF_TOPK, 0.95),                       # current server default
    "user_ladder":    ([0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.3], _DEF_TOPK, 0.95),  # screenshot override
    "tight_ladder":   ([0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25],
                       [100, 70, 50, 36, 26, 18, 12, 8, 6], 0.95),        # cleaner / tighter
    "low_topp":       (_DEF_TEMP, _DEF_TOPK, 0.44),                       # user's current note-confidence
    "mid_topp":       (_DEF_TEMP, _DEF_TOPK, 0.70),                       # top_p mid-point
    "open_flat":      (0.95, 80, 0.95),                                   # wilder, flat control
}

MODES = ["lyric", "instrumental"]
PHASE1_LYRIC_CFG = 3.0   # held constant for lyric cells in Phase 1 (swept in Phase 2)


# ── librosa collapse/rhythm scoring. Copied verbatim from eval/sweep/score.py
# (that tree is not an importable package); keep in sync. CLAP can't run locally
# (torchcodec vs ffmpeg8) but DOES on Modal CUDA, so it's blended in when
# available (score_clap) and skipped otherwise. ──────────────────────────────
def _features(path: str, sr: int = 22050) -> dict:
    import numpy as np
    import librosa

    y, _sr = librosa.load(path, sr=sr, mono=True)
    if y.size < sr:  # < 1s -> treat as collapsed
        return {"rms": 0.0, "beat": 0.0, "sil": 1.0, "centroid": 0.0}
    rms = float(np.sqrt(np.mean(y**2)))
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    ac = librosa.autocorrelate(onset - onset.mean())
    ac = ac / (ac[0] + 1e-9)
    beat = float(np.max(ac[4:200])) if ac.size > 200 else 0.0
    sil = float(np.mean(np.abs(y) < 0.01))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return {"rms": rms, "beat": beat, "sil": sil, "centroid": centroid}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _librosa_score(feat: dict) -> float:
    """0..1. Collapse is a multiplicative gate; within non-collapsed clips,
    0.6*beat + 0.4*rms_band."""
    collapse_gate = _clip(1 - feat["sil"] / 0.30, 0.0, 1.0)
    beat_term = min(feat["beat"], 1.0)
    rms = feat["rms"]
    if rms < 0.02 or rms > 0.35:
        rms_term = 0.0
    else:
        rms_term = _clip(1 - abs(rms - 0.10) / 0.10, 0.0, 1.0)
    return collapse_gate * (0.60 * beat_term + 0.40 * rms_term)


def _aggregate(scores: list[float]) -> dict:
    """Rank on mean - 0.5*std so collapse-prone (high-variance) settings are
    penalized as risky defaults."""
    import numpy as np

    a = np.array(scores, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "rank_score": float(a.mean() - 0.5 * a.std())}


def _lyric_field(genre_key: str) -> str:
    """Per-genre lyric stream: gender + tempo header markers, then the body."""
    gender = GENRES[genre_key]["gender"]
    header = " ".join(m for m in (gender, "[80bpm]") if m)
    return f"{header} {LYRIC_BODY}"


def _combine(sweet: str, lyric_field: str) -> str:
    """Reproduce the engine's tags|lyrics join. Sanitize the sweetened tag's
    internal '. ' -> ', ' so the engine's split on the FIRST '. ' lands exactly
    at the tags|lyric boundary (see the inference_tags_lyrics_delimiter note)."""
    sweet = (sweet or "").replace(". ", ", ").strip()
    if not sweet:
        return lyric_field
    return f"{sweet}. {lyric_field}"


def run_sweep_core(
    eng,
    out_dir,
    *,
    ckpt_label,
    phase1_seconds=25.0,
    phase2_seconds=60.0,
    phase1_seeds=2,
    cfg_scales="4,6,8,10",
    lyric_cfgs="0,3,6",
    top_k_per_mode=2,
    smoke=False,
    score_clap=True,
    commit=lambda: None,
):
    """Device-agnostic sweep body, shared by the Modal function (score_clap=True,
    commit=volume.commit) and the local CLI (score_clap=False -- CLAP's
    torchcodec loader crashes on ffmpeg8 locally -- commit=no-op)."""
    import json
    import os
    import time

    import numpy as np

    cfgs = [float(x) for x in str(cfg_scales).split(",") if str(x).strip()]
    lcfgs = [float(x) for x in str(lyric_cfgs).split(",") if str(x).strip()]
    genres = list(GENRES.keys())
    profiles = list(PROFILES.keys())

    if smoke:  # minimal grid to validate the whole pipeline end-to-end
        genres = genres[:1]
        profiles = profiles[:1]
        cfgs = cfgs[:1]
        lcfgs = lcfgs[:1]
        phase1_seeds = 1
        phase1_seconds = min(phase1_seconds, 6.0)
        phase2_seconds = min(phase2_seconds, 6.0)
        print("[smoke] reduced grid:", genres, profiles, cfgs, "lcfgs", lcfgs)

    print(f"[sweep] engine on {eng.device}; score_clap={score_clap}; out_dir={out_dir}")

    # Pre-sweeten each genre tag ONCE (deterministic, reused byte-identically in
    # every cell -- honors "sweetener always on" while keeping content constant).
    sweet = {}
    for g in genres:
        s = eng.sweeten_prompt(GENRES[g]["tag"])
        sweet[g] = s
        print(f"  [{g}] tag -> {s[:80]!r}")

    os.makedirs(f"{out_dir}/phase1", exist_ok=True)

    # Resume: reload any already-recorded Phase-1 scores so a preemption only
    # loses the in-flight cell.
    scores_path = f"{out_dir}/phase1/scores.json"
    records: dict[str, dict] = {}
    if os.path.exists(scores_path):
        try:
            with open(scores_path) as f:
                for r in json.load(f).get("records", []):
                    records[r["id"]] = r
            print(f"[sweep] resumed {len(records)} Phase-1 records from {scores_path}")
        except Exception as e:
            print(f"[sweep] could not reload scores ({e!r}); starting fresh")

    def _persist():
        with open(scores_path, "w") as f:
            json.dump({"records": list(records.values())}, f, indent=2)
        commit()

    def _gen(genre, mode, profile, cfg, lcfg, seconds):
        """One generation -> (audio_bytes, mime, clap_or_None)."""
        temp, topk, topp = PROFILES[profile]
        if mode == "instrumental":
            lyric_field = "[instrumental]"
            lyric_cfg = None
        else:
            lyric_field = _lyric_field(genre)
            lyric_cfg = lcfg or None
        text = _combine(sweet[genre], lyric_field)
        res = eng.generate_audio(
            seconds=seconds, text=text,
            temperature=temp, top_k=topk, top_p=topp,
            cfg_scale=cfg, lyric_cfg_scale=lyric_cfg,
            score_clap=score_clap,
        )
        if len(res) == 3:
            return res[0], res[1], res[2]
        return res[0], res[1], None

    # ── Phase 1: full grid at short clips, auto-scored ───────────────────────
    cells = [(g, m, p, c)
             for g in genres for m in MODES for p in profiles for c in cfgs]
    print(f"\n[phase1] {len(cells)} cells x {phase1_seeds} seeds = "
          f"{len(cells) * phase1_seeds} clips @ {phase1_seconds:g}s")
    t0 = time.time()
    for ci, (genre, mode, profile, cfg) in enumerate(cells):
        cell_dir = f"{out_dir}/phase1/{genre}/{mode}"
        os.makedirs(cell_dir, exist_ok=True)
        for seed in range(phase1_seeds):
            rid = f"{genre}/{mode}/{profile}/cfg{cfg:g}/s{seed}"
            if rid in records:
                continue
            try:
                body, mime, clap = _gen(genre, mode, profile, cfg, PHASE1_LYRIC_CFG, phase1_seconds)
                ext = "mp3" if mime == "audio/mpeg" else "wav"
                path = f"{cell_dir}/{profile}__cfg{cfg:g}__s{seed}.{ext}"
                with open(path, "wb") as fo:
                    fo.write(body)
                feat = _features(path)
                records[rid] = {
                    "id": rid, "genre": genre, "mode": mode, "profile": profile,
                    "cfg": cfg, "lyric_cfg": PHASE1_LYRIC_CFG, "seed": seed,
                    "path": path, "librosa": _librosa_score(feat),
                    "clap": clap, "sil": feat["sil"], "beat": feat["beat"],
                    "rms": feat["rms"],
                }
            except Exception as e:
                print(f"  !! gen failed {rid}: {e!r}")
                records[rid] = {
                    "id": rid, "genre": genre, "mode": mode, "profile": profile,
                    "cfg": cfg, "lyric_cfg": PHASE1_LYRIC_CFG, "seed": seed,
                    "path": None, "librosa": 0.0, "clap": None,
                    "sil": 1.0, "beat": 0.0, "rms": 0.0, "error": repr(e),
                }
        _persist()
        if (ci + 1) % 8 == 0 or ci + 1 == len(cells):
            dt = time.time() - t0
            print(f"  [{ci+1}/{len(cells)}] {dt:.0f}s elapsed "
                  f"(~{dt / (ci + 1) * (len(cells) - ci - 1):.0f}s left)")

    # ── Rank: blend CLAP (normalized across the run) into the librosa score,
    # aggregate per cell, then rank (profile,cfg) by CROSS-GENRE robustness. ──
    recs = list(records.values())
    claps = [r["clap"] for r in recs if r.get("clap") is not None]
    lo, hi = (min(claps), max(claps)) if claps else (0.0, 0.0)
    rng = hi - lo
    for r in recs:
        ls = r["librosa"]
        c = r.get("clap")
        if rng > 1e-9 and c is not None:
            r["combined"] = ls * (0.5 + 0.5 * (c - lo) / rng)  # genre-match boosts up to 2x
        else:
            r["combined"] = ls  # local / no-CLAP: librosa-only

    cell_groups: dict[tuple, list] = {}
    for r in recs:
        cell_groups.setdefault((r["genre"], r["mode"], r["profile"], r["cfg"]), []).append(r["combined"])
    cell_scores = {k: _aggregate(v) for k, v in cell_groups.items()}

    combo_genres: dict[tuple, list] = {}
    for (genre, mode, profile, cfg), agg in cell_scores.items():
        combo_genres.setdefault((mode, profile, cfg), []).append(agg["rank_score"])
    combo_rank = {}
    for k, vals in combo_genres.items():
        a = np.array(vals, dtype=float)
        combo_rank[k] = float(a.mean() - 0.5 * a.std())  # robust across genres

    finalists = {}
    for mode in MODES:
        ranked = sorted([(k, v) for k, v in combo_rank.items() if k[0] == mode],
                        key=lambda x: -x[1])
        finalists[mode] = [(k[1], k[2]) for k, _ in ranked[:top_k_per_mode]]  # (profile, cfg)
        print(f"\n[phase1] top {mode} finalists:")
        for (m, p, c), v in ranked[:top_k_per_mode]:
            print(f"    {p:16s} cfg{c:g}  cross-genre={v:.3f}")

    # ── Phase 2: render finalists at full length, organized for the ear test ─
    print(f"\n[phase2] rendering finalists @ {phase2_seconds:g}s")
    manifest = {"ckpt": ckpt_label, "phase1": recs, "phase2": []}
    for mode in MODES:
        for profile, cfg in finalists[mode]:
            lcfg_set = lcfgs if mode == "lyric" else [0.0]
            for genre in genres:
                for lcfg in lcfg_set:
                    cell_dir = f"{out_dir}/phase2_finalists/{genre}/{mode}"
                    os.makedirs(cell_dir, exist_ok=True)
                    if mode == "lyric":
                        fname = f"{profile}__cfg{cfg:g}__lcfg{lcfg:g}"
                    else:
                        fname = f"{profile}__cfg{cfg:g}"
                    try:
                        body, mime, clap = _gen(genre, mode, profile, cfg, lcfg, phase2_seconds)
                        ext = "mp3" if mime == "audio/mpeg" else "wav"
                        path = f"{cell_dir}/{fname}.{ext}"
                        with open(path, "wb") as fo:
                            fo.write(body)
                        manifest["phase2"].append({
                            "genre": genre, "mode": mode, "profile": profile,
                            "cfg": cfg, "lyric_cfg": (lcfg if mode == "lyric" else None),
                            "path": path, "clap": clap,
                            "temperature": PROFILES[profile][0],
                            "top_k": PROFILES[profile][1], "top_p": PROFILES[profile][2],
                        })
                        print(f"    {genre}/{mode}/{fname}")
                    except Exception as e:
                        print(f"  !! phase2 gen failed {genre}/{mode}/{fname}: {e!r}")
            commit()

    with open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    _write_report(out_dir, finalists, combo_rank, cell_scores, recs, ckpt_label,
                  phase1_seconds, phase2_seconds, phase1_seeds, score_clap)
    commit()
    print("\n==================== SAMPLING SWEEP DONE ====================")
    print(f"outputs in {out_dir}  (audition phase2_finalists/, then read REPORT.md)")
    return out_dir


def _write_report(out_dir, finalists, combo_rank, cell_scores, recs, ckpt_label,
                  p1s, p2s, seeds, score_clap=True):
    import numpy as np

    lines = [
        f"# Sampling sweep — {ckpt_label}", "",
        f"Phase 1: {p1s:g}s clips, {seeds} seed(s)/cell, "
        f"{'CLAP+librosa' if score_clap else 'librosa-only (local, CLAP off)'} auto-score. "
        f"Phase 2 finalists: {p2s:g}s.", "",
        "**The auto-ranking is a coarse filter (collapse/rhythm"
        + (" + genre-match)" if score_clap else ")") +
        ". Pick the final default BY EAR from `phase2_finalists/`.** `lyric_cfg` was "
        "NOT auto-scored — judge it by ear in the lyric finalists.", "",
    ]
    # per-clip avg clap per combo for context
    clap_by_combo: dict[tuple, list] = {}
    for r in recs:
        if r.get("clap") is not None:
            clap_by_combo.setdefault((r["mode"], r["profile"], r["cfg"]), []).append(r["clap"])

    for mode in MODES:
        lines += [f"## {mode}", "",
                  "| rank | profile | cfg | cross-genre | per-genre (rank_score) | avg CLAP |",
                  "|---|---|---|---|---|---|"]
        ranked = sorted([(k, v) for k, v in combo_rank.items() if k[0] == mode],
                        key=lambda x: -x[1])
        for i, ((m, profile, cfg), v) in enumerate(ranked):
            pg = []
            for genre in GENRES:
                agg = cell_scores.get((genre, mode, profile, cfg))
                pg.append(f"{genre.split('_')[1][:4]}={agg['rank_score']:.2f}" if agg else "—")
            cl = clap_by_combo.get((mode, profile, cfg))
            cl_s = f"{np.mean(cl):.3f}" if cl else "—"
            star = " ⭐" if (profile, cfg) in finalists[mode] else ""
            lines.append(f"| {i+1}{star} | {profile} | {cfg:g} | {v:.3f} | "
                         f"{' '.join(pg)} | {cl_s} |")
        lines += ["", f"**Auto-recommended {mode} default:** "
                  f"`{finalists[mode][0][0]}` @ cfg {finalists[mode][0][1]:g} "
                  f"(⭐ = rendered at {p2s:g}s in `phase2_finalists/`).", ""]

    lines += [
        "## Listening guide", "",
        "1. Open `phase2_finalists/<genre>/<mode>/` and A/B the ⭐ profiles per genre.",
        "2. For lyric cells, compare `__lcfg0/3/6` — pick the lyric_cfg where words "
        "are clearest without the music degrading.",
        "3. Pick the profile+cfg that holds up across ALL four genres (a robust default).",
        "", "## Apply the winner", "",
        "Set the chosen values as defaults in:",
        "- `server/main.py` `/generate` endpoint — `per_cb_temperature`, `per_cb_top_k`, "
        "`per_cb_top_p`/`top_p`, `cfg_scale`, `lyric_cfg_scale`.",
        "- the webapp 'advanced' panel defaults.", "",
        "Profile definitions (temperature, top_k, top_p):", "",
    ]
    for name, (t, k, p) in PROFILES.items():
        lines.append(f"- `{name}`: temp={t}, top_k={k}, top_p={p}")
    with open(f"{out_dir}/REPORT.md", "w") as f:
        f.write("\n".join(lines) + "\n")


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,  # ~3-4h expected for the exhaustive grid; headroom for slow decode
    volumes={"/ckpts": ckpts_vol, "/outputs": output_vol},
)
def sweep(
    ckpt_path: str = "/ckpts/v8_sing4/best_inference.pt",
    phase1_seconds: float = 25.0,
    phase2_seconds: float = 60.0,
    phase1_seeds: int = 2,
    cfg_scales: str = "4,6,8,10",
    lyric_cfgs: str = "0,3,6",
    top_k_per_mode: int = 2,
    smoke: bool = False,
):
    import os

    os.environ["NANO_CKPT"] = ckpt_path
    os.environ.setdefault("NANO_DEVICE", "cuda")
    os.environ.setdefault("NANO_OUTPUT_DIR", "/outputs")
    os.environ.setdefault("NANO_COMPILE", "default")  # ~2x decode, matches prod server

    from server.inference import InferenceEngine

    eng = InferenceEngine(ckpt_path=ckpt_path)
    tag = ckpt_path.rstrip("/").split("/")[-1].replace(".pt", "")
    ck_dir = ckpt_path.rstrip("/").split("/")[-2] if "/" in ckpt_path.rstrip("/") else "ckpt"
    out_dir = f"/outputs/sampling_sweep/{ck_dir}_{tag}"
    run_sweep_core(
        eng, out_dir, ckpt_label=ckpt_path,
        phase1_seconds=phase1_seconds, phase2_seconds=phase2_seconds,
        phase1_seeds=phase1_seeds, cfg_scales=cfg_scales, lyric_cfgs=lyric_cfgs,
        top_k_per_mode=top_k_per_mode, smoke=smoke,
        score_clap=True, commit=output_vol.commit,
    )
    print(f"pull:  modal volume get nano-output {out_dir.replace('/outputs', '')} ./sweep_clips --force")


@app.local_entrypoint()
def main(
    ckpt_path: str = "/ckpts/v8_sing4/best_inference.pt",
    phase1_seconds: float = 25.0,
    phase2_seconds: float = 60.0,
    phase1_seeds: int = 2,
    cfg_scales: str = "4,6,8,10",
    lyric_cfgs: str = "0,3,6",
    top_k_per_mode: int = 2,
    smoke: bool = False,
):
    # .spawn() (NOT .remote()): fire-and-forget so the hours-long sweep runs
    # independently of this launcher. With `modal run --detach`, a blocking
    # .remote() ties the function to the client and CANCELS it when the launcher
    # disconnects; .spawn() + --detach is the robust fire-and-forget pattern.
    handle = sweep.spawn(
        ckpt_path=ckpt_path, phase1_seconds=phase1_seconds,
        phase2_seconds=phase2_seconds, phase1_seeds=phase1_seeds,
        cfg_scales=cfg_scales, lyric_cfgs=lyric_cfgs,
        top_k_per_mode=top_k_per_mode, smoke=smoke,
    )
    print(f"spawned sweep: function call {handle.object_id} "
          f"(detached — monitor via `modal app logs` or the nano-output volume)")


if __name__ == "__main__":
    # ── Local runner (no Modal). Same grid + code as the Modal path via
    # run_sweep_core, but CLAP scoring is OFF (torchcodec crashes on ffmpeg8
    # locally) so ranking is librosa-only, and clips land in a local folder.
    # Default ckpt is checkpoints/latest.pt = v8_sing4 @ step 97000 (canonical).
    import argparse
    import os
    import sys

    # Make the repo root importable when invoked as `python scripts/...`.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    ap = argparse.ArgumentParser(description="Run the nano sampling sweep locally (no Modal).")
    ap.add_argument("--ckpt-path", default="checkpoints/latest.pt")
    ap.add_argument("--out-dir", default=None, help="default: sweep_out/local_<ckpt>")
    ap.add_argument("--phase1-seconds", type=float, default=25.0)
    ap.add_argument("--phase2-seconds", type=float, default=60.0)
    ap.add_argument("--phase1-seeds", type=int, default=2)
    ap.add_argument("--cfg-scales", default="4,6,8,10")
    ap.add_argument("--lyric-cfgs", default="0,3,6")
    ap.add_argument("--top-k-per-mode", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="tiny grid + short clips to validate + time it")
    a = ap.parse_args()

    os.environ["NANO_CKPT"] = a.ckpt_path
    # bf16 on MPS is the correct local dtype (fp16 collapses the rollout); the
    # engine already defaults to it. Decode runs eager locally (NANO_COMPILE is a
    # CUDA-only speedup).
    from server.inference import InferenceEngine

    eng = InferenceEngine(ckpt_path=a.ckpt_path)
    tag = os.path.basename(a.ckpt_path.rstrip("/")).replace(".pt", "")
    out_dir = a.out_dir or os.path.join("sweep_out", f"local_{tag}")
    run_sweep_core(
        eng, out_dir, ckpt_label=a.ckpt_path,
        phase1_seconds=a.phase1_seconds, phase2_seconds=a.phase2_seconds,
        phase1_seeds=a.phase1_seeds, cfg_scales=a.cfg_scales, lyric_cfgs=a.lyric_cfgs,
        top_k_per_mode=a.top_k_per_mode, smoke=a.smoke,
        score_clap=False, commit=lambda: None,
    )
    print(f"\nAudition: open {out_dir}/phase2_finalists/  then read {out_dir}/REPORT.md")
