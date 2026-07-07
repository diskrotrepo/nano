"""Read-only audit: phoneme-store coverage of each wave's usable-word songs.

Answers "did the phonemize OOM (modal_phonemize.py, SIGKILL 137) leave gaps?"
per wave. Phonemize (diskrot/phonemize.py) must cover every song with >=1
transcribed valid word; a deterministic static-memory OOM at ~bucket 5/256
silently skips the rest, which then fall back to (slow) live g2p at train time.
Because /tokens/phonemes is a SINGLE global store (256 stem-hash shards, not
per-wave) and every wave's ingest re-runs phonemize over ALL lyrics, coverage is
cumulative — this reports the CURRENT global coverage restricted to each wave's
stems, which is exactly what you want to know before trusting waves 0-3.

Reports, per wave and in total:
  usable    songs with >=1 valid word (the phonemize denominator)
  covered   of those, present in /tokens/phonemes
  MISSING   of those, absent  -> the OOM gap (falls back to live g2p)
  stale     present but group-count != word-count (phonemize would redo)

Purely read-only (no writes, no volume commit). Safe to run mid-pipeline.

Run:
    modal run scripts/phoneme_coverage.py                      # base_0..base_3
    modal run scripts/phoneme_coverage.py --waves base_0,base_5
    modal run scripts/phoneme_coverage.py --waves all          # every wave dir
"""
from __future__ import annotations

import modal

app = modal.App("nano-phoneme-coverage")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_python_source("diskrot")  # for diskrot.modal_common.corpus_mount
)

from diskrot.modal_common import corpus_mount

tokens_vol = modal.Volume.from_name("nano-tokens")
corpus_vol = corpus_mount()  # read-only R2 bucket


@app.function(
    image=image,
    volumes={"/tokens": tokens_vol, "/corpus": corpus_vol},
    timeout=60 * 30,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
)
def coverage(waves: str = "base_0,base_1,base_2,base_3"):
    import json
    from pathlib import Path

    # is_valid_word: the SAME predicate diskrot.phonemize._load_lyric_words uses
    # (transcribe_lyrics.is_valid_word) to decide a song has usable words. Inlined
    # to avoid importing transcribe_lyrics (which drags in torch/whisper).
    def is_valid_word(w):
        return (
            isinstance(w, dict)
            and isinstance(w.get("word"), str)
            and isinstance(w.get("start"), (int, float)) and not isinstance(w["start"], bool)
            and isinstance(w.get("end"), (int, float)) and not isinstance(w["end"], bool)
        )

    # --- usable-word song set + word count, mirroring _load_lyric_words ---
    word_count: dict[str, int] = {}
    for shard in sorted(Path("/tokens/lyrics").glob("lyrics_*.json")):
        for name, val in json.loads(shard.read_text()).items():
            if not isinstance(val, dict):
                continue
            words = val.get("words")
            clean = [w for w in words if is_valid_word(w)] if isinstance(words, list) else []
            if clean:
                word_count[name] = len(clean)
    print(f"lyrics store: {len(word_count):,} songs with usable words (global)")

    # --- phoneme store: stem -> stored group count ---
    ph_groups: dict[str, int] = {}
    ph_dir = Path("/tokens/phonemes")
    if ph_dir.exists():
        for shard in sorted(ph_dir.glob("phonemes_*.json")):
            for name, groups in json.loads(shard.read_text()).items():
                ph_groups[name] = len(groups) if isinstance(groups, list) else 0
    print(f"phoneme store: {len(ph_groups):,} songs phonemized (global)\n")

    # --- resolve wave dirs -> stems ---
    corpus = Path("/corpus")
    if waves.strip() == "all":
        wave_dirs = sorted(corpus.glob("waves/wave_*"))
    else:
        wave_dirs = [corpus / "waves" / f"wave_{w.strip()}" for w in waves.split(",") if w.strip()]

    hdr = f"{'wave':<16}{'mp3s':>10}{'usable':>10}{'covered':>10}{'MISSING':>10}{'stale':>8}{'cov%':>7}"
    print(hdr)
    print("-" * len(hdr))

    tot_mp3 = tot_usable = tot_cov = tot_missing = tot_stale = 0
    for wd in wave_dirs:
        stems = {p.stem for p in wd.glob("*.mp3")}
        if not stems:
            print(f"{wd.name:<16}{'(no mp3s found)':>40}")
            continue
        usable = stems & word_count.keys()
        covered = usable & ph_groups.keys()
        missing = usable - ph_groups.keys()
        stale = sum(1 for s in covered if ph_groups[s] != word_count[s])
        pct = 100.0 * len(covered) / max(len(usable), 1)
        print(f"{wd.name:<16}{len(stems):>10,}{len(usable):>10,}{len(covered):>10,}"
              f"{len(missing):>10,}{stale:>8,}{pct:>6.1f}%")
        tot_mp3 += len(stems); tot_usable += len(usable); tot_cov += len(covered)
        tot_missing += len(missing); tot_stale += stale

    print("-" * len(hdr))
    tpct = 100.0 * tot_cov / max(tot_usable, 1)
    print(f"{'TOTAL':<16}{tot_mp3:>10,}{tot_usable:>10,}{tot_cov:>10,}"
          f"{tot_missing:>10,}{tot_stale:>8,}{tpct:>6.1f}%")
    if tot_missing == 0 and tot_stale == 0:
        print("\n=> clean: every usable-word song in these waves is phonemized and current.")
    else:
        print(f"\n=> gap: {tot_missing:,} missing + {tot_stale:,} stale — re-run the "
              "(32GB-fixed) phonemize to backfill; lyrics persist so it repairs these waves.")


@app.local_entrypoint()
def main(waves: str = "base_0,base_1,base_2,base_3"):
    coverage.remote(waves=waves)
