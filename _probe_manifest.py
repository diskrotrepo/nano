"""TEMP diagnostic: why didn't prepare drop base_2's too_short clips?

Cross-references base_2's on-disk mp3s against the GLOBAL prepare manifest
(/tokens/prepare_manifest.json, keyed by bare stem). Reports, for base_2's
files: manifest status distribution, how many carry a DELETE status yet are
still on disk, and — the smoking gun — how many have a stored `name` pointing
to a DIFFERENT wave (so prepare's delete, which matches v['name'] against THIS
wave's on_disk, silently skips them). Delete this file after.

    modal run _probe_manifest.py
"""
import os

import modal

app = modal.App("nano-prepare-manifest-probe")

image = modal.Image.debian_slim(python_version="3.12")

_mk = {"secret": modal.Secret.from_name("r2-creds"), "read_only": True}
_ep = os.environ.get("NANO_AUDIO_ENDPOINT")
if _ep:
    _mk["bucket_endpoint_url"] = _ep
corpus_vol = modal.CloudBucketMount(os.environ.get("NANO_AUDIO_BUCKET", "nano-audio"), **_mk)
tokens_vol = modal.Volume.from_name("nano-tokens")

DELETE_STATUSES = {"undecodable", "too_short", "duplicate", "too_long", "low_quality"}


@app.function(
    image=image,
    volumes={"/corpus": corpus_vol, "/tokens": tokens_vol},
    memory=16384,
    timeout=900,
)
def probe(wave_id: str = "base_2"):
    import json
    from collections import Counter
    from pathlib import Path

    wave_prefix = f"waves/wave_{wave_id}"
    mp3s = sorted((Path("/corpus") / wave_prefix).glob("*.mp3"))
    on_disk_names = {str(p.relative_to("/corpus")) for p in mp3s}
    stems = {p.stem: str(p.relative_to("/corpus")) for p in mp3s}

    man = json.loads(Path("/tokens/prepare_manifest.json").read_text())
    files = man.get("files", {})

    status_dist = Counter()
    in_manifest = 0
    missing = 0
    should_be_deleted = 0          # delete-status but still on disk (this wave)
    cross_wave_name = 0            # delete-status + stored name is a DIFFERENT wave
    cross_wave_any = 0            # ANY status where stored name != this wave path
    too_short_here = 0
    samples = []

    for stem, disk_name in stems.items():
        e = files.get(stem)
        if e is None:
            missing += 1
            status_dist["<not-in-manifest>"] += 1
            continue
        in_manifest += 1
        st = e.get("status", "<none>")
        status_dist[st] += 1
        stored = e.get("name", "")
        stored_wave = stored.rsplit("/", 1)[0] if "/" in stored else ""
        name_mismatch = stored != disk_name
        if name_mismatch:
            cross_wave_any += 1
        if st in DELETE_STATUSES:
            should_be_deleted += 1
            if name_mismatch:
                cross_wave_name += 1
            if st == "too_short":
                too_short_here += 1
            if len(samples) < 15:
                samples.append({
                    "stem": stem[:55], "status": st,
                    "dur_s": e.get("duration_s"),
                    "stored_name_wave": stored_wave,
                    "disk_wave": wave_prefix,
                    "name_mismatch": name_mismatch,
                })

    return {
        "wave": wave_id,
        "manifest_total_entries": len(files),
        "wave_mp3s_on_disk": len(mp3s),
        "in_manifest": in_manifest,
        "not_in_manifest": missing,
        "status_dist": dict(status_dist.most_common()),
        "should_be_deleted_still_on_disk": should_be_deleted,
        "  of_those_cross_wave_name_mismatch": cross_wave_name,
        "too_short_still_on_disk": too_short_here,
        "any_status_name_points_elsewhere": cross_wave_any,
        "samples": samples,
    }


@app.local_entrypoint()
def main(wave_id: str = "base_2"):
    import json
    print(json.dumps(probe.remote(wave_id=wave_id), indent=2, default=str))
