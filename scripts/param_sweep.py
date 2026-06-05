#!/usr/bin/env python3
"""Parameter sweep against the deployed nano inference server.

Sweeps cfg_scale x temperature x top_k for a fixed text prompt, saving each
generation plus a manifest.json into an output dir. Requests serialize on the
Modal container (max_inputs=1); this just walks the grid sequentially.
"""
import itertools
import json
import os
import sys
import time
import urllib.request

URL = os.environ.get("NANO_URL", "http://localhost:8000") + "/generate"
OUTDIR = os.environ.get("NANO_SWEEP_OUT", "./nano_sweep")

PROMPT = os.environ.get(
    "NANO_PROMPT",
    "energetic electronic dance track, driving synth bass, four-on-the-floor kick, bright arpeggios",
)
SECONDS = float(os.environ.get("NANO_SECONDS", "15"))

CFG_SCALES = [1.5, 3.0, 5.0, 7.0]
TEMPERATURES = [0.7, 0.9, 1.1]
TOP_KS = [50, 250]
SEED_MODE = "random"


def _multipart(fields: dict) -> tuple[bytes, str]:
    boundary = "----nanosweep7f3a2b"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        parts.append(f"{v}\r\n".encode())
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def run_one(cfg_scale, temperature, top_k, idx, total):
    fields = {
        "seconds": SECONDS,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": 0.95,
        "cfg_scale": cfg_scale,
        "seed_mode": SEED_MODE,
        "prompt": PROMPT,
    }
    body, boundary = _multipart(fields)
    req = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    tag = f"cfg{cfg_scale}_temp{temperature}_topk{top_k}"
    t0 = time.time()
    print(f"[{idx}/{total}] {tag} ...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = resp.read()
            mime = resp.headers.get("Content-Type", "audio/mpeg")
    except Exception as e:
        print(f"[{idx}/{total}] {tag} FAILED: {e}", flush=True)
        return {"tag": tag, "cfg_scale": cfg_scale, "temperature": temperature,
                "top_k": top_k, "ok": False, "error": str(e)}
    ext = "mp3" if "mpeg" in mime or "mp3" in mime else ("wav" if "wav" in mime else "bin")
    fname = f"{idx:02d}_{tag}.{ext}"
    with open(os.path.join(OUTDIR, fname), "wb") as f:
        f.write(data)
    dt = time.time() - t0
    print(f"[{idx}/{total}] {tag} -> {fname} ({len(data)} bytes, {dt:.0f}s)", flush=True)
    return {"tag": tag, "cfg_scale": cfg_scale, "temperature": temperature,
            "top_k": top_k, "ok": True, "file": fname, "bytes": len(data), "seconds_wall": dt}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    combos = list(itertools.product(CFG_SCALES, TEMPERATURES, TOP_KS))
    total = len(combos)
    print(f"sweep: {total} combos, prompt={PROMPT!r}, {SECONDS}s each -> {OUTDIR}", flush=True)
    results = []
    for i, (cfg_scale, temperature, top_k) in enumerate(combos, 1):
        results.append(run_one(cfg_scale, temperature, top_k, i, total))
        manifest = {
            "url": URL, "prompt": PROMPT, "seconds": SECONDS,
            "axes": {"cfg_scale": CFG_SCALES, "temperature": TEMPERATURES, "top_k": TOP_KS},
            "results": results,
        }
        with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
    ok = sum(1 for r in results if r.get("ok"))
    print(f"DONE: {ok}/{total} succeeded. Output in {OUTDIR}", flush=True)


if __name__ == "__main__":
    main()
