"""Rank sweep settings from the run manifest -> results/<tag>/REPORT.md + a
recommended-default block to paste into server defaults.

Usage:
    python -m eval.sweep.rank                          # auto-pick newest tag
    python -m eval.sweep.rank --tag best_step135000
"""
import argparse, json
from pathlib import Path

from eval.sweep import config as C
from eval.sweep.score import aggregate

ROOT = Path("eval/sweep")


def pick_tag(tag):
    rdir = ROOT / "results"
    if tag:
        return tag
    cands = [p.parent.name for p in rdir.glob("*/run_manifest.json")]
    if not cands:
        raise SystemExit("no run_manifest.json found — run run_sweep first")
    # newest by mtime
    return max(cands, key=lambda t: (rdir / t / "run_manifest.json").stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = pick_tag(args.tag)
    recs = json.loads((ROOT / "results" / tag / "run_manifest.json").read_text())

    by_setting = {}
    for r in recs:
        by_setting.setdefault(r["setting"], {"scores": [], "rows": []})
        by_setting[r["setting"]]["scores"].append(r["score"])
        by_setting[r["setting"]]["rows"].append(r)

    def _avg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    table = []
    for sid, d in by_setting.items():
        agg = aggregate(d["scores"])
        ex = d["rows"][0]
        beat = _avg(d["rows"], "beat")
        sil = _avg(d["rows"], "sil")
        clap = _avg(d["rows"], "clap")
        n = len(d["scores"])
        table.append((agg["rank_score"], sid, ex["cfg_scale"],
                      ex["profile"], agg["mean"], agg["min"], beat, sil, clap, n))
    table.sort(key=lambda x: -x[0])

    # `mean - 0.5*std` is only comparable at equal n. After a stage-2 refinement
    # the finalists have many more clips than the rest, which both regresses their
    # lucky means and inflates their std — so a naive sort floats UNDER-tested
    # settings to the top. Only recommend from the most-tested tier (max n), and
    # surface n in the table so the bias is never silent.
    max_n = max(row[9] for row in table)
    finalists = [row for row in table if row[9] == max_n]

    lines = [f"# Inference sweep — {tag}", "",
             f"Ranked on `mean - 0.5*std` of the blended score over {len(recs)} clips "
             f"(blend = collapse-gate x CLAP adherence). beat=rhythm (higher better), "
             f"sil%=collapse (lower better), clap=prompt adherence (higher better). "
             f"**n = clips per setting; only compare settings at equal n.**", "",
             "| rank | setting | cfg | profile | rankscore | mean | min | beat | sil% | clap | n |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (rk, sid, cfg, prof, mean, mn, beat, sil, clap, n) in enumerate(table, 1):
        beat_s = "—" if beat is None else f"{beat:.2f}"
        sil_s = "—" if sil is None else f"{sil*100:.0f}%"
        clap_s = "—" if clap is None else f"{clap:.3f}"
        lines.append(f"| {i} | {sid} | {cfg} | {prof} | {rk:.3f} | "
                     f"{mean:.3f} | {mn:.3f} | {beat_s} | {sil_s} | {clap_s} | {n} |")

    win = finalists[0]  # finalists is a slice of the rankscore-sorted table
    _, wid, wcfg, wprof, *_ = win
    temp, topk = C.PROFILES[wprof]
    lines += ["", "## Recommended default",
              f"From the {len(finalists)} fully-refined finalists (n={max_n}); the "
              f"under-tested n<{max_n} rows above are NOT comparable and are excluded.",
              "```python",
              f'# winner: {wid}',
              f'cfg_scale={wcfg}',
              f'temperature={temp}', f'top_k={topk}', f'top_p={C.TOP_P}',
              "```",
              "",
              "Scores are close and collapse-variance is high — confirm by ear "
              "(samples/<tag>/) before committing. Wire into server/inference.py "
              "generate_audio defaults + server/main.py Form() fallbacks. Defaults are "
              "per-model — re-run this sweep on each new checkpoint."]

    out = ROOT / "results" / tag / "REPORT.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
