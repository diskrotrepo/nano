"""Focused cfg_scale x temperature sweep over the live HTTP server.

Unlike eval.sweep.run_sweep (which fixes SECONDS=8 and bundles temperature into
3 fixed (temp, top_k) ladder profiles), this sweeps a clean scalar-temperature
axis against cfg_scale at 10s, holding everything else constant:

  - sweeten=False           -> identical conditioning across every cell
  - comma-only prompt       -> no ". " so nothing leaks into the lyrics slot
                               (engine splits conditioning on the first ". ")
  - fixed prompt + top_p    -> only cfg_scale and temperature vary

Seed caveat: /generate draws a fresh random DAC seed per call, so each cell is
one draw, not a fixed-seed comparison. Trends across the grid are meaningful; a
single cell winning by a hair is within seed noise.

Usage:
    uv run --no-sync python -m eval.sweep.cfg_temp_sweep
    BASE=http://127.0.0.1:8000 SECONDS=10 uv run --no-sync python -m eval.sweep.cfg_temp_sweep
"""
import os
import subprocess
from pathlib import Path

from eval.sweep.score import features, score

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
SECONDS = float(os.environ.get("SECONDS", "10"))
OUT = Path("eval/sweep/cfg_temp")

# Comma-only so the whole string lands in the tags slot (no ". " split).
PROMPT = ("driving techno, four-on-the-floor kick, hypnotic synth stabs, "
          "deep bass, tight hi-hats, dark, energetic, instrumental")

CFG_SCALES = [1.5, 3.0, 5.0, 7.0]
TEMPERATURES = [0.7, 0.9, 1.1]
TOP_P = 0.95


def gen(cfg: float, temp: float, path: Path) -> bool:
    """One /generate call -> mp3 at path. Returns True on HTTP 200 + non-tiny file."""
    code = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{BASE}/generate",
         "-F", f"prompt={PROMPT}",
         "-F", f"seconds={SECONDS}",
         "-F", f"temperature={temp}",
         "-F", f"cfg_scale={cfg}",
         "-F", f"top_p={TOP_P}",
         "-F", "sweeten=false",
         "-o", str(path), "-w", "%{http_code}"],
        capture_output=True, text=True,
    ).stdout.strip()
    ok = code == "200" and path.exists() and path.stat().st_size > 2048
    return ok


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = [(c, t) for c in CFG_SCALES for t in TEMPERATURES]
    for i, (cfg, temp) in enumerate(cells, 1):
        name = f"cfg{cfg}_temp{temp}.mp3"
        path = OUT / name
        print(f"[{i}/{len(cells)}] cfg={cfg} temp={temp} ...", flush=True)
        if not gen(cfg, temp, path):
            print(f"    FAILED (http error or tiny file): {name}", flush=True)
            rows.append((-1.0, cfg, temp, name, {}))
            continue
        feat = features(str(path))
        sc = score(feat)
        rows.append((sc, cfg, temp, name, feat))
        print(f"    score={sc:.3f}  beat={feat['beat']:.2f} rms={feat['rms']:.3f} "
              f"sil={feat['sil']:.2f} centroid={feat['centroid']:.0f}", flush=True)

    rows.sort(key=lambda r: r[0], reverse=True)
    report = OUT / "REPORT.md"
    lines = [
        "# cfg_scale x temperature sweep",
        "",
        f"prompt: `{PROMPT}`",
        f"seconds={SECONDS}, top_p={TOP_P}, sweeten=off (1 clip/cell).",
        "Score = collapse-gated 0.6*beat + 0.4*rms (higher = more musical, less silence/noise).",
        "",
        "| rank | score | cfg | temp | beat | rms | sil | centroid | file |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, (sc, cfg, temp, name, feat) in enumerate(rows, 1):
        if feat:
            lines.append(f"| {rank} | {sc:.3f} | {cfg} | {temp} | {feat['beat']:.2f} | "
                         f"{feat['rms']:.3f} | {feat['sil']:.2f} | {feat['centroid']:.0f} | {name} |")
        else:
            lines.append(f"| {rank} | FAIL | {cfg} | {temp} | - | - | - | - | {name} |")
    report.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {report}")
    print("\n".join(lines[6:]))


if __name__ == "__main__":
    main()
