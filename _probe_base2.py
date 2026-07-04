"""TEMP diagnostic: why is base_2 tokenize ~99% 'short'? CPU-only.

For a spread sample of base_2 mp3s, compare the ffprobe HEADER duration against
the REAL decoded audio produced by the EXACT tokenize decode command
(ffmpeg -ac 2 -ar 48000, i.e. the SpectroStream path), plus RMS and the frame
count at 25 Hz (short = frames < 20*25 = 500). Delete this file after.

    NANO_CODEC=spectrostream modal run _probe_base2.py
"""
import os

import modal

app = modal.App("nano-tokenize-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("numpy")
)

# Inline the R2 mount (== diskrot.modal_common.corpus_mount) so the container
# needs no diskrot package — just ffmpeg + numpy.
_mount_kwargs = {"secret": modal.Secret.from_name("r2-creds"), "read_only": True}
_endpoint = os.environ.get("NANO_AUDIO_ENDPOINT")  # set locally; absent in container
if _endpoint:
    _mount_kwargs["bucket_endpoint_url"] = _endpoint
corpus_vol = modal.CloudBucketMount(
    os.environ.get("NANO_AUDIO_BUCKET", "nano-audio"), **_mount_kwargs
)
tokens_vol = modal.Volume.from_name("nano-tokens")

SR = 48000
HOP = 1920          # 48000/25
MIN_FRAMES = 500    # 20s * 25Hz


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    timeout=900,
)
def probe(wave_id: str = "base_2", n: int = 40):
    import json
    import math
    import subprocess
    from pathlib import Path

    import numpy as np

    d = Path("/corpus") / f"waves/wave_{wave_id}"
    mp3s = sorted(d.glob("*.mp3"))
    total = len(mp3s)
    if total == 0:
        return {"error": f"no mp3s under {d}"}

    # PENDING = mp3s with no matching .pt (exactly list_pending's logic). These
    # are the files that keep classifying 'short' — the population we must sample.
    tok_dir = Path("/tokens") / f"waves/wave_{wave_id}"
    existing = {p.stem for p in tok_dir.glob("*.pt")}
    pending = [p for p in mp3s if p.stem not in existing]
    n_pending = len(pending)
    if n_pending == 0:
        return {"error": f"no pending mp3s (all {total} tokenized)"}

    # Spread sample across the PENDING set: first few + an even stride.
    idxs = list(range(min(8, n_pending)))
    stride = max(1, n_pending // n)
    idxs += list(range(0, n_pending, stride))
    idxs = sorted(set(idxs))[:n]
    sample = [pending[i] for i in idxs]

    def header_dur(p):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_entries", "stream=duration:format=duration", str(p)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
            ).stdout
            j = json.loads(out)
            fmt = (j.get("format") or {}).get("duration")
            strm = None
            for s in j.get("streams", []):
                if s.get("duration"):
                    strm = s["duration"]; break
            return float(fmt or strm or 0.0)
        except Exception as e:
            return f"ERR:{type(e).__name__}"

    def decode(p):
        """EXACT tokenize decode: ffmpeg -ac 2 -ar 48000 f32le. Returns
        (real_seconds, rms_dbfs, clip_frac, n_samples, retcode/err)."""
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(p),
               "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(SR), "-"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            return (None, None, None, 0, f"rc={proc.returncode}:{proc.stderr.decode()[:120]}")
        buf = np.frombuffer(proc.stdout, dtype=np.float32)
        if buf.size == 0:
            return (0.0, None, None, 0, "empty")
        y = buf.reshape(-1, 2).T           # [2, N]
        n_samp = y.shape[1]
        mono = y.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        rms_db = 20.0 * math.log10(rms + 1e-12)
        clip = float(np.mean(np.abs(mono) >= 0.9995))
        return (n_samp / SR, rms_db, clip, n_samp, "ok")

    rows = []
    for p in sample:
        hd = header_dur(p)
        real, rms_db, clip, n_samp, status = decode(p)
        frames = (n_samp + HOP - 1) // HOP if n_samp else 0
        short = frames < MIN_FRAMES
        verdict = "decode_fail" if status.startswith("rc=") else (
            "silent" if (rms_db is not None and rms_db < -50.0) else
            "clipped" if (clip is not None and clip > 0.20) else
            "short" if short else "OK")
        rows.append({
            "name": p.name, "hdr_s": hd, "real_s": real, "frames": frames,
            "rms_db": None if rms_db is None else round(rms_db, 1),
            "clip": None if clip is None else round(clip, 3),
            "short": short, "verdict": verdict, "status": status[:60],
        })

    # Aggregate
    agg = {"total_in_wave": total, "pending_no_pt": n_pending, "sampled": len(rows)}
    for k in ("OK", "short", "silent", "clipped", "decode_fail"):
        agg[k] = sum(1 for r in rows if r["verdict"] == k)
    reals = [r["real_s"] for r in rows if isinstance(r["real_s"], (int, float))]
    hdrs = [r["hdr_s"] for r in rows if isinstance(r["hdr_s"], (int, float))]
    if reals:
        agg["real_s_min/med/max"] = [round(min(reals), 1),
                                     round(sorted(reals)[len(reals)//2], 1),
                                     round(max(reals), 1)]
    if hdrs:
        agg["hdr_s_min/med/max"] = [round(min(hdrs), 1),
                                    round(sorted(hdrs)[len(hdrs)//2], 1),
                                    round(max(hdrs), 1)]
    return {"agg": agg, "rows": rows}


@app.local_entrypoint()
def main(wave_id: str = "base_2", n: int = 40):
    import json
    res = probe.remote(wave_id=wave_id, n=n)
    print(json.dumps(res, indent=2, default=str))
