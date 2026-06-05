"""Drive the sampling sweep over HTTP against a live inference server.

Same grid + manifest layout as eval/sweep/run_sweep.py, but instead of calling
InferenceEngine in-process it POSTs each setting to a running server (local or
Modal). This lets the 1.5B model run the sweep on a GPU while we score locally.

Two scoring signals are combined per clip:
  - collapse/musicality  -> eval.sweep.score.score()  (librosa, local, 0..1)
  - prompt adherence      -> CLAP text<->audio cosine, computed server-side and
                             returned in the X-Nano-Clap-Score header (CLAP
                             can't run locally; see eval/sweep/score.py docstring)
The manifest `score` field is the BLEND (so rank.py ranks on both); raw
`collapse` and `clap` are kept alongside for the report.

Usage:
    python -m eval.sweep.sweep_http --base https://xxxx.modal.run --stage 1
    python -m eval.sweep.sweep_http --base https://xxxx.modal.run --stage 2 --top-n 4
    python -m eval.sweep.sweep_http --base http://127.0.0.1:8000 --smoke

Writes:
    eval/sweep/samples/<ckpt_tag>/<setting>__<prompt>__c<clip>.{mp3,wav}
    eval/sweep/results/<ckpt_tag>/run_manifest.json   (settings + features, appended)
"""
import argparse, json, time
from pathlib import Path

import requests

from eval.sweep import config as C
from eval.sweep.score import features, score, aggregate

ROOT = Path("eval/sweep")

# CLAP audio<->text cosines for on-prompt music sit ~0.2-0.45; map to ~0..1.
CLAP_NORM = 0.5


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def blended_score(collapse: float, clap: float | None) -> float:
    """Collapse score is the gate (0 if the clip died); adherence modulates it
    +/-50%. A high-CLAP but collapsed clip still scores 0; among healthy clips,
    the one that follows the prompt wins."""
    clap_norm = _clip((clap or 0.0) / CLAP_NORM, 0.0, 1.0)
    return collapse * (0.5 + 0.5 * clap_norm)


def _form_for_profile(setting: dict) -> dict:
    """Map a profile's temperature/top_k (scalar or length-9 list) onto the
    server's scalar vs per_cb_* form fields."""
    form: dict[str, str] = {}
    temp, topk = setting["temperature"], setting["top_k"]
    if isinstance(temp, list):
        form["per_cb_temperature"] = ",".join(str(x) for x in temp)
    else:
        form["temperature"] = str(temp)
    if isinstance(topk, list):
        form["per_cb_top_k"] = ",".join(str(x) for x in topk)
    else:
        form["top_k"] = str(topk)
    return form


def ckpt_tag(base: str) -> str:
    h = requests.get(f"{base}/health", timeout=30).json()
    if not h.get("text_conditioning"):
        raise SystemExit(
            "server reports text_conditioning: false -> CFG is a no-op; "
            "the whole cfg sweep would be meaningless. Aborting."
        )
    stem = Path(h["ckpt_path"]).stem
    return f"{stem}_step{h.get('ckpt_step', -1)}"


def _post_with_retry(base, form, attempts=3):
    """POST /generate, retrying transient failures (a one-off 500 or timeout
    shouldn't abort a 75-gen sweep). Returns the response or raises after the
    last attempt."""
    last = None
    for i in range(attempts):
        try:
            r = requests.post(f"{base}/generate", data=form, timeout=600)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 — log + retry any HTTP/network error
            last = e
            print(f"   [retry {i+1}/{attempts}] {type(e).__name__}: {e}")
            time.sleep(5)
    raise last


def gen_one(base, setting, prompt_key, clip_idx, tag):
    sdir = ROOT / "samples" / tag
    sdir.mkdir(parents=True, exist_ok=True)
    form = {
        "seconds": str(C.SECONDS),
        "prompt": C.PROMPTS[prompt_key],
        "seed_mode": setting["seed_mode"],
        "top_p": str(setting["top_p"]),
        "cfg_scale": str(setting["cfg_scale"]),
        "score_clap": "true",
        **_form_for_profile(setting),
    }
    t0 = time.time()
    r = _post_with_retry(base, form)
    ext = "mp3" if "mpeg" in r.headers.get("content-type", "") else "wav"
    path = sdir / f"{setting['id']}__{prompt_key}__c{clip_idx}.{ext}"
    path.write_bytes(r.content)

    clap_hdr = r.headers.get("X-Nano-Clap-Score")
    clap = float(clap_hdr) if clap_hdr is not None else None
    feat = features(str(path))
    collapse = score(feat)
    blended = blended_score(collapse, clap)
    rec = {
        "setting": setting["id"], "cfg_scale": setting["cfg_scale"],
        "seed_mode": setting["seed_mode"], "profile": setting["profile"],
        "prompt": prompt_key, "clip": clip_idx, "file": str(path),
        **feat, "collapse": collapse, "clap": clap, "score": blended,
        "gen_sec": round(time.time() - t0, 1),
    }
    print(f"[{rec['gen_sec']:5.1f}s] {setting['id']:24s} {prompt_key:10s} "
          f"beat={feat['beat']:.2f} sil={feat['sil']*100:3.0f}% "
          f"clap={clap if clap is None else round(clap, 3)} score={blended:.3f}")
    return rec


def manifest_path(tag):
    rdir = ROOT / "results" / tag
    rdir.mkdir(parents=True, exist_ok=True)
    return rdir / "run_manifest.json"


def write_manifest(tag, recs):
    """Overwrite the manifest with the full accumulated record list. Called after
    every gen so a mid-run crash never loses completed work."""
    manifest_path(tag).write_text(json.dumps(recs, indent=2))


def try_gen(base, setting, prompt_key, clip_idx, tag):
    """gen_one wrapped so a single failed setting is skipped, not fatal."""
    try:
        return gen_one(base, setting, prompt_key, clip_idx, tag)
    except Exception as e:  # noqa: BLE001
        print(f"   [SKIP] {setting['id']} {prompt_key}: {type(e).__name__}: {e}")
        return None


def top_settings_from_manifest(tag, n):
    recs = json.loads(manifest_path(tag).read_text())
    by_setting = {}
    for r in recs:
        by_setting.setdefault(r["setting"], []).append(r["score"])
    ranked = sorted(by_setting.items(),
                    key=lambda kv: -aggregate(kv[1])["rank_score"])
    return [sid for sid, _ in ranked[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="server base URL, e.g. https://xxx.modal.run")
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="one 3s gen, sanity only")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    tag = ckpt_tag(base)
    print(f"[sweep] base={base}  ckpt_tag={tag}")

    if args.smoke:
        s = C.all_settings()[len(C.all_settings()) // 2]
        sdir = ROOT / "samples" / tag
        sdir.mkdir(parents=True, exist_ok=True)
        form = {"seconds": "3", "prompt": C.PROMPTS["techno"],
                "seed_mode": s["seed_mode"], "top_p": str(s["top_p"]),
                "cfg_scale": str(s["cfg_scale"]), "score_clap": "true",
                **_form_for_profile(s)}
        r = requests.post(f"{base}/generate", data=form, timeout=600)
        r.raise_for_status()
        (sdir / "SMOKE.mp3").write_bytes(r.content)
        print(f"smoke ok — clap={r.headers.get('X-Nano-Clap-Score')}")
        return

    # Stage 1 starts a fresh manifest; stage 2 appends to it (refinement clips).
    recs = [] if args.stage == 1 else json.loads(manifest_path(tag).read_text())
    n_before = len(recs)

    if args.stage == 1:
        jobs = [(s, pk, 0) for s in C.all_settings() for pk in C.PROMPTS]
    elif args.stage == 2:
        keep = set(top_settings_from_manifest(tag, args.top_n))
        settings = [s for s in C.all_settings() if s["id"] in keep]
        print(f"[stage2] refining {len(settings)} settings x 2 clips: {sorted(keep)}")
        jobs = [(s, pk, clip) for s in settings for pk in C.PROMPTS for clip in (1, 2)]
    else:
        raise SystemExit(f"unknown --stage {args.stage}")

    for s, pk, clip in jobs:
        rec = try_gen(base, s, pk, clip, tag)
        if rec is not None:
            recs.append(rec)
            write_manifest(tag, recs)  # persist after every gen

    added = len(recs) - n_before
    print(f"\n[manifest] +{added}/{len(jobs)} records -> {manifest_path(tag)}")
    print("Now rank:  python -m eval.sweep.rank")


if __name__ == "__main__":
    main()
