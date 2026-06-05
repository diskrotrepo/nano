"""Post-training eval: generate a fixed prompt pack, score with audio metrics,
optionally sweep sampling configs.

Outputs a markdown report ranking sampling configs by a composite of
(CLAP score, spectral flatness, silence ratio, onset density). The score
filters obvious failure modes (silence, white-noise buzz) so you only
listen to plausible candidates. Subjective listening on the top-N is still
required for the final call.

Local usage:
    python -m scripts.eval_checkpoint --ckpt ./checkpoints/best.pt --out ./eval/v4_40m
    python -m scripts.eval_checkpoint --ckpt ./checkpoints/best.pt --out ./eval/v4_40m --sweep

Output:
    ./eval/v4_40m/
      report.md
      metrics.json
      baseline/<prompt-id>_seed<N>.wav
      sweep/<prompt-id>_seed<N>_<config-id>.wav   (only with --sweep)
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from server.inference import InferenceEngine


DEFAULT_PROMPTS = [
    {"id": "ambient",  "tags": "ambient vaporwave",                 "seed": 42},
    {"id": "lofi",     "tags": "lo-fi hip hop slow tempo piano",    "seed": 42},
    {"id": "glitch",   "tags": "glitch electronic granular",        "seed": 42},
    {"id": "uncond1",  "tags": "",                                  "seed": 42},
    {"id": "uncond2",  "tags": "",                                  "seed": 7},
]


# Sampling configs. cb0–2 model coarse pitch/rhythm and tolerate higher temp;
# cb3–8 model fine residuals where high temp produces white-noise buzz.
SAMPLING_CONFIGS = [
    {"id": "default",         "temp": 0.9,                                            "top_k": 50,                                         "top_p": 0.95, "cfg": 3.0},
    {"id": "warm_topk",       "temp": 1.0,                                            "top_k": 250,                                        "top_p": 0.95, "cfg": 3.0},
    {"id": "tight_fine",      "temp": [0.9]*3 + [0.6]*3 + [0.4]*3,                    "top_k": [250]*3 + [100]*3 + [50]*3,                 "top_p": 0.95, "cfg": 3.0},
    {"id": "very_tight_fine", "temp": [1.0]*2 + [0.7]*3 + [0.4]*4,                    "top_k": [250]*2 + [100]*3 + [50]*4,                 "top_p": 0.95, "cfg": 3.0},
    {"id": "high_cfg",        "temp": 0.9,                                            "top_k": 50,                                         "top_p": 0.95, "cfg": 7.0},
    {"id": "low_cfg",         "temp": 0.9,                                            "top_k": 50,                                         "top_p": 0.95, "cfg": 1.5},
    {"id": "tight_high_cfg",  "temp": [0.9]*3 + [0.5]*6,                              "top_k": [250]*3 + [50]*6,                           "top_p": 0.95, "cfg": 5.0},
]


def _build_text_cond(engine: InferenceEngine, tags: str, lyrics: str) -> torch.Tensor | None:
    if engine.text_encoder is None:
        return None
    parts = []
    if tags:
        parts.append(engine.text_encoder.encode([tags]).to(engine.device))
    if lyrics:
        parts.append(engine.text_encoder.encode([lyrics]).to(engine.device))
    if not parts:
        return None
    cond = torch.cat(parts, dim=1)
    return cond.to(next(engine.model.parameters()).dtype)


@torch.no_grad()
def generate_clip(engine: InferenceEngine, prompt: dict, sampling: dict, seconds: float) -> np.ndarray:
    """Generate one clip with the given sampling config → mono float32 numpy."""
    torch.manual_seed(prompt.get("seed", 42))

    K = engine.model.cfg.n_codebooks
    max_total = engine.model.cfg.max_seq_len - K + 1
    new_frames = min(int(seconds * engine.codec.FRAME_RATE_HZ), max_total - 1)

    cond_emb = _build_text_cond(engine, prompt.get("tags", ""), prompt.get("lyrics", ""))

    out_tokens = engine.model.generate(
        prompt=None,
        num_new_frames=new_frames,
        temperature=sampling["temp"],
        top_k=sampling["top_k"],
        top_p=sampling.get("top_p"),
        text_emb=cond_emb,
        cfg_scale=sampling["cfg"] if cond_emb is not None else 1.0,
    )
    # model.generate prepends a 1-frame random seed when prompt is None
    out_tokens = out_tokens[:, 1:]
    wav = engine.codec.decode(out_tokens.cpu())
    if wav.dim() > 1:
        wav = wav.squeeze(0)
    return wav.float().numpy()


def _compute_clap(wav: np.ndarray, sr: int, text_emb: torch.Tensor | None, clap) -> float | None:
    if text_emb is None or clap is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
        sf.write(tmp, wav, sr)
    try:
        audio_emb = clap.get_audio_embeddings([tmp]).cpu()  # [1, 1024]
    finally:
        os.unlink(tmp)
    a = audio_emb / (audio_emb.norm(dim=-1, keepdim=True) + 1e-9)
    t = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-9)
    return float((a @ t.T).squeeze().item())


def compute_metrics(wav: np.ndarray, sr: int, text_emb: torch.Tensor | None, clap) -> dict:
    """Audio metrics for a single clip. All cheap to compute."""
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=wav)))
    centroid = librosa.feature.spectral_centroid(y=wav, sr=sr)[0]
    onsets = librosa.onset.onset_detect(y=wav, sr=sr, units="time")
    rms = librosa.feature.rms(y=wav, frame_length=2048, hop_length=512)[0]
    return {
        "spectral_flatness": flatness,
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "onset_density": float(len(onsets) / max(len(wav) / sr, 1e-6)),
        "silence_ratio": float(np.mean(rms < 1e-3)),
        "rms_mean": float(np.mean(rms)),
        "clap_score": _compute_clap(wav, sr, text_emb, clap),
    }


def composite_score(m: dict) -> float:
    """Higher = better. Tuned so silence and white-noise score low; on-prompt music scores high."""
    clap = m["clap_score"] if m["clap_score"] is not None else 0.0
    onset_norm = min(m["onset_density"] / 5.0, 1.0)
    return clap - 0.5 * m["spectral_flatness"] - 0.8 * m["silence_ratio"] + 0.2 * onset_norm


def _fmt(v, prec=3) -> str:
    return "—" if v is None else f"{v:.{prec}f}"


def _sampling_repr(cfg: dict) -> str:
    """Short string for the report table."""
    def short(v):
        if isinstance(v, list):
            # collapse runs of equal values
            uniq = []
            for x in v:
                if not uniq or uniq[-1][0] != x:
                    uniq.append([x, 1])
                else:
                    uniq[-1][1] += 1
            return "[" + " ".join(f"{x}×{n}" if n > 1 else str(x) for x, n in uniq) + "]"
        return str(v)
    return f"t={short(cfg['temp'])} k={short(cfg['top_k'])} p={cfg.get('top_p')} cfg={cfg['cfg']}"


def write_report(out_path: Path, engine: InferenceEngine, rows: list[dict], top_n: int, args) -> None:
    lines: list[str] = []
    lines.append("# Eval report\n")
    lines.append(f"- checkpoint: `{args.ckpt}`")
    lines.append(f"- step: {engine.ckpt_step}")
    lines.append(f"- device: {engine.device}")
    lines.append(f"- clip duration: {args.seconds:.1f}s")
    lines.append(f"- model: {engine.model.cfg.d_model}d × {engine.model.cfg.n_layers}L × {engine.model.cfg.n_heads}H, "
                 f"text_cond={'on' if engine.text_encoder is not None else 'off'}")
    lines.append("")

    baseline = [r for r in rows if r["section"] == "baseline"]
    if baseline:
        lines.append("## Baseline (default sampling)\n")
        lines.append("| prompt | seed | flatness | silence | onsets/s | CLAP | score | wav |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in baseline:
            lines.append(
                f"| {r['prompt']} | {r['seed']} | {_fmt(r['spectral_flatness'])} | "
                f"{_fmt(r['silence_ratio'], 2)} | {_fmt(r['onset_density'], 2)} | "
                f"{_fmt(r['clap_score'])} | **{_fmt(r['score'])}** | `{r['wav_path']}` |"
            )

    sweep = [r for r in rows if r["section"] == "sweep"]
    if sweep:
        ranked = sorted(sweep, key=lambda r: r["score"], reverse=True)
        lines.append(f"\n## Sweep — top {top_n} of {len(ranked)} clips\n")
        lines.append("| rank | sampling | prompt | seed | flatness | silence | onsets/s | CLAP | score | wav |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(ranked[:top_n], 1):
            lines.append(
                f"| {i} | `{r['sampling_id']}` | {r['prompt']} | {r['seed']} | "
                f"{_fmt(r['spectral_flatness'])} | {_fmt(r['silence_ratio'], 2)} | "
                f"{_fmt(r['onset_density'], 2)} | {_fmt(r['clap_score'])} | "
                f"**{_fmt(r['score'])}** | `{r['wav_path']}` |"
            )

        # average score per config — useful to see which sampling config wins across prompts
        from collections import defaultdict
        by_cfg = defaultdict(list)
        for r in sweep:
            by_cfg[r["sampling_id"]].append(r["score"])
        agg = sorted(((cid, sum(s) / len(s)) for cid, s in by_cfg.items()), key=lambda x: -x[1])
        lines.append("\n### Mean score per sampling config (across prompts)\n")
        lines.append("| config | mean score | params |")
        lines.append("|---|---|---|")
        cfg_lookup = {c["id"]: c for c in SAMPLING_CONFIGS}
        for cid, mean_s in agg:
            lines.append(f"| `{cid}` | {mean_s:.3f} | `{_sampling_repr(cfg_lookup[cid])}` |")

    lines.append("\n## Interpretation cheat-sheet\n")
    lines.append("- **spectral_flatness ≈ 1.0** — white-noise-like. Likely the buzz failure mode.")
    lines.append("- **spectral_flatness ≈ 0** — highly tonal. Either music or one sustained tone — listen.")
    lines.append("- **silence_ratio > 0.5** — silence-attractor collapse. CFG / random seed didn't escape it.")
    lines.append("- **onset_density < 0.5/s** — no rhythmic content. Either ambient or stuck.")
    lines.append("- **CLAP < 0.10** — generated audio doesn't reflect the prompt.")
    lines.append("- **score** is a composite; use it to pick the top 3–5 to listen to, not to decide alone.")

    lines.append("\n## Decision matrix\n")
    lines.append("Pair this report with the per-codebook CE numbers from training logs:")
    lines.append("- cb0–1 dropped a lot, cb3–8 near log(1024)≈6.93 → buzz is codec-floor. 150M won't help.")
    lines.append("- cb0–1 dropped, cb3–8 clearly below 6.93 → learnable structure left. 150M plausibly helps if sampling sweep isn't enough.")
    lines.append("- cb0–1 still near 6.93 → silence-attractor failure. Debug config, not capacity.")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to checkpoint .pt")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--prompts", default=None, help="path to prompts.json (default: built-in pack)")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detect by default)")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--sweep", action="store_true", help="also run the sampling sweep")
    p.add_argument("--top-n", type=int, default=5, help="how many sweep clips to highlight in report")
    args = p.parse_args()

    out_dir = Path(args.out)
    (out_dir / "baseline").mkdir(parents=True, exist_ok=True)
    if args.sweep:
        (out_dir / "sweep").mkdir(parents=True, exist_ok=True)

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = json.loads(Path(args.prompts).read_text())

    engine = InferenceEngine(ckpt_path=args.ckpt, device=args.device)
    sr = engine.codec.SAMPLE_RATE

    # Reuse engine's CLAP (heavy ~500MB) for audio↔text scoring
    clap = None
    prompt_text_embs: dict[str, torch.Tensor] = {}
    if engine.text_encoder is not None:
        engine.text_encoder._ensure_clap()
        clap = engine.text_encoder._clap
        for pr in prompts:
            text = (pr.get("tags", "") + " " + pr.get("lyrics", "")).strip()
            if text:
                prompt_text_embs[pr["id"]] = clap.get_text_embeddings([text]).cpu()

    rows: list[dict] = []
    default_cfg = next(c for c in SAMPLING_CONFIGS if c["id"] == "default")

    print("=== baseline (default sampling) ===")
    for pr in prompts:
        wav = generate_clip(engine, pr, default_cfg, args.seconds)
        wav_path = out_dir / "baseline" / f"{pr['id']}_seed{pr.get('seed', 42)}.wav"
        sf.write(wav_path, wav, sr)
        m = compute_metrics(wav, sr, prompt_text_embs.get(pr["id"]), clap)
        score = composite_score(m)
        rows.append({
            "section": "baseline", "prompt": pr["id"], "seed": pr.get("seed", 42),
            "sampling_id": "default", "wav_path": str(wav_path), "score": score, **m,
        })
        print(f"  {pr['id']:8s} seed={pr.get('seed', 42):>3}  "
              f"flat={m['spectral_flatness']:.3f} sil={m['silence_ratio']:.2f} "
              f"ons={m['onset_density']:.2f}/s clap={_fmt(m['clap_score'])} → {score:+.3f}")

    if args.sweep:
        print("\n=== sweep ===")
        configs_to_run = [c for c in SAMPLING_CONFIGS if c["id"] != "default"]
        total = len(configs_to_run) * len(prompts)
        i = 0
        for cfg in configs_to_run:
            for pr in prompts:
                i += 1
                wav = generate_clip(engine, pr, cfg, args.seconds)
                wav_path = out_dir / "sweep" / f"{pr['id']}_seed{pr.get('seed', 42)}_{cfg['id']}.wav"
                sf.write(wav_path, wav, sr)
                m = compute_metrics(wav, sr, prompt_text_embs.get(pr["id"]), clap)
                score = composite_score(m)
                rows.append({
                    "section": "sweep", "prompt": pr["id"], "seed": pr.get("seed", 42),
                    "sampling_id": cfg["id"], "wav_path": str(wav_path), "score": score, **m,
                })
                print(f"  [{i:>3}/{total}] {cfg['id']:18s} × {pr['id']:8s}  "
                      f"flat={m['spectral_flatness']:.3f} sil={m['silence_ratio']:.2f} "
                      f"clap={_fmt(m['clap_score'])} → {score:+.3f}")

    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    write_report(out_dir / "report.md", engine, rows, args.top_n, args)
    print(f"\nreport: {out_dir / 'report.md'}")
    print(f"metrics: {out_dir / 'metrics.json'}")
    if args.sweep:
        print(f"\nlisten to top {args.top_n} sweep clips in: {out_dir / 'sweep'}")


if __name__ == "__main__":
    main()
