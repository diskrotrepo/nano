"""Generate the sweep grid and record per-clip features.

Usage:
    SMOKE=1 python -m eval.sweep.run_sweep            # single 3s gen, sanity only
    python -m eval.sweep.run_sweep --stage 1          # coarse: 15 settings x 5 prompts x 1 clip
    python -m eval.sweep.run_sweep --stage 2          # fine: top-N from stage1 x 2 clips
    NANO_CKPT=checkpoints/best_1500m.pt python -m eval.sweep.run_sweep --stage 1

Writes:
    eval/sweep/samples/<ckpt_tag>/<setting>__<prompt>__c<clip>.{mp3,wav}
    eval/sweep/results/<ckpt_tag>/run_manifest.json   (settings + features, appended)
"""
import argparse, json, os, time
from pathlib import Path

from server.inference import InferenceEngine
from eval.sweep import config as C
from eval.sweep.score import features, score

ROOT = Path("eval/sweep")


def ckpt_tag(eng) -> str:
    base = Path(eng.ckpt_path).stem
    return f"{base}_step{eng.ckpt_step}"


def _flatten_tags(s: str) -> str:
    """generate_audio splits text on the FIRST ". " into tags|lyrics, so flatten
    any internal ". " in a caption to ", " — otherwise the caption tail spills
    into the lyric stream (instrumental prompts would get sung)."""
    return s.replace(". ", ", ").strip()


def gen_one(eng, setting, tags, lyrics, prompt_key, clip_idx, tag):
    """One clip. lyrics=None -> instrumental (tags only); else sung (tags+lyrics,
    moderate lyric_cfg). Collapse features/score apply to both modes."""
    sdir = ROOT / "samples" / tag
    sdir.mkdir(parents=True, exist_ok=True)
    mode = "lyric" if lyrics else "instr"
    text = _flatten_tags(tags)
    if lyrics:
        text = f"{text}. {lyrics}"
    t0 = time.time()
    audio, mime = eng.generate_audio(
        seconds=C.SECONDS,
        text=text,
        temperature=setting["temperature"],
        top_k=setting["top_k"],
        top_p=setting["top_p"],
        cfg_scale=setting["cfg_scale"],
        lyric_cfg_scale=(C.LYRIC_CFG if lyrics else None),
    )
    ext = "mp3" if mime == "audio/mpeg" else "wav"
    path = sdir / f"{setting['id']}__{mode}_{prompt_key}__c{clip_idx}.{ext}"
    path.write_bytes(audio)
    feat = features(str(path))
    rec = {
        "setting": setting["id"], "cfg_scale": setting["cfg_scale"],
        "profile": setting["profile"], "mode": mode,
        "prompt": prompt_key, "clip": clip_idx, "file": str(path),
        **feat, "score": score(feat), "gen_sec": round(time.time() - t0, 1),
    }
    print(f"[{rec['gen_sec']:5.1f}s] {setting['id']:24s} {mode:5s} {prompt_key:10s} "
          f"beat={feat['beat']:.2f} sil={feat['sil']*100:3.0f}% score={rec['score']:.3f}")
    return rec


def append_manifest(tag, recs):
    rdir = ROOT / "results" / tag
    rdir.mkdir(parents=True, exist_ok=True)
    mpath = rdir / "run_manifest.json"
    existing = json.loads(mpath.read_text()) if mpath.exists() else []
    existing.extend(recs)
    mpath.write_text(json.dumps(existing, indent=2))
    print(f"\n[manifest] {len(recs)} records -> {mpath}")


def top_settings_from_manifest(tag, n):
    """Re-rank stage1 manifest, return the top-n setting ids by rank_score."""
    from eval.sweep.score import aggregate
    mpath = ROOT / "results" / tag / "run_manifest.json"
    recs = json.loads(mpath.read_text())
    by_setting = {}
    for r in recs:
        by_setting.setdefault(r["setting"], []).append(r["score"])
    ranked = sorted(by_setting.items(),
                    key=lambda kv: -aggregate(kv[1])["rank_score"])
    return [sid for sid, _ in ranked[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--top-n", type=int, default=4)
    args = ap.parse_args()

    eng = InferenceEngine(ckpt_path=os.environ.get("NANO_CKPT", "checkpoints/best.pt"))
    tag = ckpt_tag(eng)
    print(f"[sweep] ckpt_tag = {tag}")

    if os.environ.get("SMOKE") == "1":
        s = C.all_settings()[6]  # an arbitrary mid setting
        sdir = ROOT / "samples" / tag
        sdir.mkdir(parents=True, exist_ok=True)
        audio, mime = eng.generate_audio(
            seconds=3.0, text=C.PROMPTS["techno"],
            temperature=s["temperature"], top_k=s["top_k"], top_p=s["top_p"],
            cfg_scale=s["cfg_scale"])
        (sdir / "SMOKE.mp3").write_bytes(audio)
        print("smoke ok")
        return

    recs = []
    if args.stage == 1:
        settings = C.all_settings()
        for s in settings:
            for pk, tags in C.PROMPTS.items():               # instrumental
                recs.append(gen_one(eng, s, tags, None, pk, 0, tag))
            for pk, lp in C.LYRIC_PROMPTS.items():            # sung
                recs.append(gen_one(eng, s, lp["tags"], lp["lyrics"], pk, 0, tag))
    elif args.stage == 2:
        keep = set(top_settings_from_manifest(tag, args.top_n))
        settings = [s for s in C.all_settings() if s["id"] in keep]
        print(f"[stage2] refining {len(settings)} settings x 2 clips: {sorted(keep)}")
        for s in settings:
            for pk, tags in C.PROMPTS.items():
                for clip in (1, 2):
                    recs.append(gen_one(eng, s, tags, None, pk, clip, tag))
            for pk, lp in C.LYRIC_PROMPTS.items():
                for clip in (1, 2):
                    recs.append(gen_one(eng, s, lp["tags"], lp["lyrics"], pk, clip, tag))
    append_manifest(tag, recs)
    print("\nNow rank:  python -m eval.sweep.rank")


if __name__ == "__main__":
    main()
