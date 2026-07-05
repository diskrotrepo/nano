"""Workstream-0 spike: does Google's SpectroStream codec load standalone, and how
does it sound vs DAC on our material?

This is the v9 codec de-risk (plan: "Workstream 0 — codec spike"). SpectroStream is
the discrete RVQ codec inside Magenta RealTime: 48 kHz, JOINT stereo, 25 Hz frame
rate, up to 64 RVQ codebooks x 1024 vocab (codes are plain per-codebook indices in
[0, 1024) — embed + cross-entropy directly, no offset). We must replace mono DAC
with it for native-stereo + full-song v9.

The v1_legacy magenta-rt stack is JAX + TensorFlow + T5X, Linux+CUDA only (no macOS),
so this MUST run on Modal. We base the image on magenta's prebuilt GPU Docker image
rather than reproducing the patched t5x build.

What it does (one container, no fan-out):
  - load `spectrostream.SpectroStream(max_rvq_depth=64)` standalone
  - encode/decode a synthetic stereo tone (zero external deps) + a few real corpus
    mp3s, at RVQ depths {8, 16, 24, 64}
  - report token shape/dtype/min-max, frame rate, vocab, decode shape
  - check encode@16 == (encode@64)[:, :16] equivalence (RVQ prefix-stability)
  - write orig + recon stereo WAVs to the nano-output volume for A/B listening

Run:
    modal run --detach diskrot/modal_spectrostream_spike.py
    modal run diskrot/modal_spectrostream_spike.py --n-songs 6 --gpu A100-40GB

Pull the WAVs to A/B locally:
    modal volume get nano-output /ss_spike/ ./ss_spike/
"""
# NOTE: no `from __future__ import annotations` (Modal class param validation — see
# modal_tokenize.py). Not strictly needed here (no @app.cls) but kept consistent.

import os
from pathlib import Path

import modal

from diskrot.modal_common import corpus_mount

app = modal.App("nano-ss-spike")

# Magenta RT's prebuilt GPU image already has python3.12 + magenta_rt + the full
# JAX/TF/T5X stack installed (saves us the patched t5x-from-source build).
MAGENTA_GPU_IMAGE = "us-docker.pkg.dev/brain-magenta/magenta-rt/magenta-rt:gpu"

image = (
    modal.Image.from_registry(MAGENTA_GPU_IMAGE)
    .apt_install("ffmpeg", "libsndfile1")
    # soundfile/librosa are magenta-rt deps already, but pin them present so the
    # corpus-loading path can't fail on a missing decoder backend.
    .pip_install("soundfile>=0.12", "librosa>=0.10")
    # CPU-only torch: model.codec imports torch for tensor plumbing (the wrapper
    # verify path). CPU build avoids contending with JAX/TF for the GPU/CUDA libs.
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cpu")
    .env(
        {
            # JAX and TF both try to grab ~all GPU memory; cap so they coexist.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.4",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            # Cache HF-fetched SavedModel weights on a volume so re-runs skip the
            # multi-hundred-MB download. magenta_rt.asset defaults to the HF repo
            # google/magenta-realtime; point every plausible cache dir at /cache.
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "XDG_CACHE_HOME": "/cache",
        }
    )
    .add_local_python_source("model", "diskrot")
)

# Read the mp3 corpus from R2 (rglob picks up the waves/wave_*/ layout). If the
# bucket is empty/unreachable the spike falls back to a synthetic-only round-trip.
corpus_vol = corpus_mount()  # read-only R2 CloudBucketMount
out_vol = modal.Volume.from_name("nano-output", create_if_missing=True)
cache_vol = modal.Volume.from_name("nano-ss-cache", create_if_missing=True)

OUT_DIR = "/outputs/ss_spike"
DEPTHS = (8, 16, 24, 64)
SECONDS = 12.0  # per-song crop for the A/B clips


@app.function(
    image=image,
    gpu=os.environ.get("NANO_SPIKE_GPU", "A100-40GB"),
    timeout=60 * 30,
    volumes={"/corpus": corpus_vol, "/outputs": out_vol, "/cache": cache_vol},
)
def spike(n_songs: int = 6) -> None:
    import numpy as np
    import soundfile as sf

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print("=== loading SpectroStream (max_rvq_depth=64) ===", flush=True)
    from magenta_rt import audio, spectrostream

    codec = spectrostream.SpectroStream(max_rvq_depth=64)
    cfg = codec.config
    print(
        f"config: sample_rate={cfg.sample_rate} num_channels={cfg.num_channels} "
        f"frame_rate={cfg.frame_rate}Hz embedding_dim={cfg.embedding_dim} "
        f"rvq_depth={cfg.rvq_depth} vocab={cfg.rvq_codebook_size}",
        flush=True,
    )
    sr = int(cfg.sample_rate)

    def _wave(samples_n2: "np.ndarray") -> "audio.Waveform":
        # Waveform expects [num_samples, num_channels] float32 at `sr`.
        return audio.Waveform(np.ascontiguousarray(samples_n2, dtype=np.float32), sr)

    def _roundtrip(name: str, wav: "audio.Waveform") -> None:
        wav = wav.resample(sr).as_stereo()
        x = wav.samples  # [N, 2]
        sf.write(str(out / f"{name}_orig.wav"), x, sr)
        full = codec.encode(wav)  # [S, 64] int32
        print(
            f"\n[{name}] encoded full: shape={tuple(full.shape)} dtype={full.dtype} "
            f"min={int(full.min())} max={int(full.max())} "
            f"frames={full.shape[0]} ({full.shape[0] / cfg.frame_rate:.2f}s) "
            f"channels_in={x.shape[1]}",
            flush=True,
        )
        for k in DEPTHS:
            toks_k = full[:, :k]  # RVQ residual prefix → coarser reconstruction
            recon = codec.decode(toks_k)  # stereo Waveform
            y = recon.samples  # [M, 2]
            sf.write(str(out / f"{name}_recon_k{k:02d}.wav"), y, sr)
            kbps = k * cfg.frame_rate * np.log2(cfg.rvq_codebook_size) / 1000.0
            print(
                f"  k={k:>2}: recon shape={tuple(y.shape)} (~{kbps:.1f} kbps) "
                f"-> {name}_recon_k{k:02d}.wav",
                flush=True,
            )

    # 1) synthetic stereo tone — validates the codec with zero external deps.
    t = np.linspace(0, SECONDS, int(sr * SECONDS), endpoint=False, dtype=np.float32)
    tone = np.stack(
        [0.2 * np.sin(2 * np.pi * 220.0 * t), 0.2 * np.sin(2 * np.pi * 277.18 * t)],
        axis=-1,
    )
    _roundtrip("synthetic", _wave(tone))

    # 2) equivalence check: encode@16 vs (encode@64)[:, :16] (RVQ prefix-stability).
    print("\n=== truncation equivalence (encode@16 vs encode@64[:, :16]) ===", flush=True)
    codec16 = spectrostream.SpectroStream(max_rvq_depth=16)
    full64 = codec.encode(_wave(tone))
    direct16 = codec16.encode(_wave(tone))
    sliced16 = full64[:, :16]
    if direct16.shape == sliced16.shape:
        eq = bool(np.array_equal(direct16, sliced16))
        mism = int((direct16 != sliced16).sum())
        print(f"  shapes match {direct16.shape}; identical={eq} mismatched_codes={mism}",
              flush=True)
    else:
        print(f"  SHAPE DIFFERS: direct={direct16.shape} sliced={sliced16.shape}",
              flush=True)

    # 3) real corpus songs (skip gracefully if the mount is empty).
    import librosa

    corpus = Path("/corpus")
    mp3s = sorted(corpus.rglob("*.mp3"))[:n_songs]
    if not mp3s:
        print("\n(no corpus mp3s found — synthetic-only spike)", flush=True)
    else:
        print(f"\n=== {len(mp3s)} corpus songs (first {SECONDS:.0f}s each) ===", flush=True)
        for i, mp3 in enumerate(mp3s):
            try:
                y, _ = librosa.load(str(mp3), sr=sr, mono=False)  # [2, N] or [N]
                if y.ndim == 1:
                    y = np.stack([y, y], axis=0)
                y = y[:, : int(SECONDS * sr)]
                _roundtrip(f"song{i:02d}", _wave(y.T))
            except Exception as e:  # noqa: BLE001 — spike: surface, don't crash
                print(f"  song{i:02d} FAILED ({mp3.name}): {type(e).__name__}: {e}",
                      flush=True)

    out_vol.commit()
    print(f"\nWAVs written to nano-output:{OUT_DIR}", flush=True)
    print("pull:  modal volume get nano-output /ss_spike/ ./ss_spike/", flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("NANO_SPIKE_GPU", "A100-40GB"),
    timeout=60 * 20,
    volumes={"/corpus": corpus_vol, "/outputs": out_vol, "/cache": cache_vol},
)
def verify_wrapper(depth: int = 16) -> None:
    """Validate the model.codec.SpectroStreamCodec FACADE on Modal (it can't run
    locally): the transpose [S,K]<->[K,T] and stereo [2,samples] logic, the
    get_codec("spectrostream") switch, and a real round-trip WAV."""
    import numpy as np
    import soundfile as sf
    import torch

    from model.codec import SpectroStreamCodec, get_codec

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    print(f"=== SpectroStreamCodec wrapper @ depth={depth} ===", flush=True)
    codec = SpectroStreamCodec(device="cuda", depth=depth)
    assert codec.N_CODEBOOKS == depth
    assert codec.SAMPLE_RATE == 48000 and codec.FRAME_RATE_HZ == 25 and codec.VOCAB_SIZE == 1024
    sr = codec.SAMPLE_RATE

    # synthetic stereo [2, samples] tensor in -> [K, T] codes
    t = np.linspace(0, 8.0, sr * 8, endpoint=False, dtype=np.float32)
    stereo = torch.from_numpy(
        np.stack([0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 330 * t)], 0)
    )  # [2, N]
    codes = codec.encode(stereo)
    print(f"encode([2,{stereo.shape[1]}]) -> {tuple(codes.shape)} dtype={codes.dtype} "
          f"min={int(codes.min())} max={int(codes.max())}", flush=True)
    assert codes.dim() == 2 and codes.shape[0] == depth, codes.shape
    assert codes.dtype == torch.long
    assert 0 <= int(codes.min()) and int(codes.max()) < codec.VOCAB_SIZE
    expected_T = (stereo.shape[1] + (sr // 25) - 1) // (sr // 25)
    assert codes.shape[1] == expected_T, (codes.shape[1], expected_T)

    # decode [K, T] -> stereo [2, samples]
    wav = codec.decode(codes)
    print(f"decode -> {tuple(wav.shape)} dtype={wav.dtype}", flush=True)
    assert wav.dim() == 2 and wav.shape[0] == 2, wav.shape
    sf.write(str(out / f"wrapper_synthetic_k{depth:02d}.wav"), wav.T.numpy(), sr)

    # batched decode [B, K, T] -> [B, 2, samples]
    batched = codec.decode(codes.unsqueeze(0))
    assert batched.dim() == 3 and batched.shape[1] == 2, batched.shape
    print(f"batched decode([1,K,T]) -> {tuple(batched.shape)} OK", flush=True)

    # get_codec("spectrostream") switch + a real corpus song round-trip
    codec2 = get_codec(device="cuda", codec="spectrostream")
    assert isinstance(codec2, SpectroStreamCodec)
    import librosa
    mp3s = sorted(Path("/corpus").rglob("*.mp3"))[:1]
    if mp3s:
        y, _ = librosa.load(str(mp3s[0]), sr=sr, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y], 0)
        seg = torch.from_numpy(np.ascontiguousarray(y[:, : sr * 8]))
        c = codec.encode(seg)
        r = codec.decode(c)
        sf.write(str(out / f"wrapper_song_k{depth:02d}.wav"), r.T.numpy(), sr)
        print(f"corpus round-trip: encode {tuple(c.shape)} -> decode {tuple(r.shape)} "
              f"-> wrapper_song_k{depth:02d}.wav", flush=True)

    out_vol.commit()
    print("WRAPPER OK — model.codec.SpectroStreamCodec validated end-to-end.", flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("NANO_SPIKE_GPU", "A100-40GB"),
    timeout=60 * 30,
    volumes={"/cache": cache_vol},
    retries=0,  # the overflow CHECK aborts (SIGABRT) — do NOT re-run the sweep
)
def measure_overflow(start_s: int = 180, stop_s: int = 480, step_s: int = 10,
                     depth: int = 32) -> None:
    """Pin the SpectroStream encoder's int32 CUDA launch-config overflow length.

    Encodes synthetic stereo audio (content-irrelevant; only LENGTH drives the
    overflowing front-end feature map) at increasing lengths via the PRODUCTION
    path (model.codec.SpectroStreamCodec, depth=32 == NANO_SS_DEPTH) until TF's
    `Check failed: work_element_count >= 0` aborts the container with SIGABRT.

    Two readings come out of the flushed logs:
      - BRACKET: the largest "OK secs=X" before the abort, and the "ATTEMPT
        secs=Y" with no matching OK -> crossover C is in (X, Y].
      - EXACT: pair the abort's F0000 work_element_count (a wrapped int32) with
        the attempted sample count to solve elements-per-sample k, hence the
        exact C = INT_MAX / k. (Printed pre-encode so it survives the abort.)
    """
    import torch

    from model.codec import SpectroStreamCodec

    INT_MAX = 2**31 - 1
    sr = 48000
    print(f"=== SS overflow measurement: depth={depth} sr={sr} INT_MAX={INT_MAX} ===",
          flush=True)
    codec = SpectroStreamCodec(device="cuda", depth=depth)
    last_ok = None
    for secs in range(start_s, stop_s + 1, step_s):
        n = secs * sr
        # Pure silence: the int32 overflow is a TENSOR-SIZE (length) effect, fully
        # content-independent, and the silence quality-gate lives in tokenize, not
        # in codec.encode — so feeding zeros here is correct and cheap.
        stereo = torch.zeros((2, n), dtype=torch.float32)
        print(f"ATTEMPT secs={secs} samples={n} "
              f"(if this aborts, k=(2**32+work_element_count)/{n})", flush=True)
        codes = codec.encode(stereo)
        frames = int(codes.shape[1])
        last_ok = secs
        elems_per_sample_lo = INT_MAX / n  # lower bound on k while still OK
        print(f"OK secs={secs} samples={n} frames={frames} "
              f"(k < {elems_per_sample_lo:.2f} elems/sample so far)", flush=True)
        del codes, stereo
    print(f"=== NO overflow up to {stop_s}s (last_ok={last_ok}); widen stop_s ===",
          flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("NANO_SPIKE_GPU", "A100-40GB"),
    timeout=60 * 30,
    volumes={"/cache": cache_vol},
)
def bucket_check(depth: int = 32, bucket_frames: int = 256) -> None:
    """Validate the frame-alignment crash fix + gate the cuFFT-thrash speedup.

    (1) ALIGNMENT (the crash fix): encode several NON-frame-aligned lengths (the
        remainder that made the ffmpeg-decoded corpus fail with the internal
        "(1,T,256) vs (1,T-1,256)" mismatch). With `encode` now padding the tail up to
        a whole frame, each must SUCCEED and return exactly ceil(samples/hop) frames.
        A raised exception here means aligning did NOT fix the codec off-by-one.
    (2) EQUIVALENCE (the speed gate): a COARSER grid (bucket_frames) must give
        BYTE-IDENTICAL codes to minimal 1-frame alignment — i.e. tail padding beyond
        the true frames never perturbs the real codes — before enabling
        NANO_SS_BUCKET_FRAMES for cuFFT-plan reuse.
    (3) TIMING: encode a spread of DISTINCT lengths at grid=1 vs grid=bucket_frames;
        the coarse grid should be faster once the encoder stops "constantly creating
        new plans" (the cuFFT-cache thrash seen in the live logs).
    """
    import time

    import numpy as np
    import torch

    from model.codec import SpectroStreamCodec, bucket_target_samples

    sr = 48000
    hop = sr // 25  # 1920
    codec = SpectroStreamCodec(device="cuda", depth=depth)

    def _stereo(secs: float) -> torch.Tensor:
        n = int(secs * sr)
        t = np.linspace(0, secs, n, endpoint=False, dtype=np.float32)
        return torch.from_numpy(
            np.stack([0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 277 * t)], 0)
        )

    # Lengths whose final frame is PARTIAL (n not a multiple of hop) — the class that
    # crashed. 250.7s ≈ a real failure (ceil has a partial tail frame).
    non_aligned = (7.3, 31.7, 63.1, 100.02, 156.55, 250.7)
    print(f"=== (1) alignment / crash fix (depth={depth}) ===", flush=True)
    codec._bucket_frames = 1  # minimal frame-alignment (prod default)
    align_ok = True
    aligned_codes = {}
    for secs in non_aligned:
        n = int(secs * sr)
        want = (n + hop - 1) // hop
        try:
            c = codec.encode(_stereo(secs))
            got = int(c.shape[1])
            good = got == want and n % hop != 0  # confirm it WAS non-aligned
            aligned_codes[secs] = c
            print(f"  {secs:>7}s: want_frames={want} got={got} non_aligned={n % hop != 0} "
                  f"{'OK' if good else 'BAD'}", flush=True)
            align_ok = align_ok and (got == want)
        except Exception as e:  # noqa: BLE001 — the whole point is to catch the crash
            align_ok = False
            print(f"  {secs:>7}s: STILL CRASHES — {type(e).__name__}: {str(e)[:120]}", flush=True)
    print("ALIGNMENT: " + ("PASS — frame-align fixes the off-by-one crash"
                           if align_ok else "FAIL — aligning did NOT fix it"), flush=True)

    print(f"\n=== (2) bucket equivalence (grid=1 vs {bucket_frames}) ===", flush=True)
    ok = True
    for secs in non_aligned:
        x = _stereo(secs)
        a = aligned_codes.get(secs)
        if a is None:
            continue
        codec._bucket_frames = bucket_frames
        b = codec.encode(x)  # coarser grid, SAME production path
        codec._bucket_frames = 1
        same = tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a, b))
        mism = int((a != b).sum()) if a.shape == b.shape else -1
        n = int(secs * sr)
        tgt_frames = bucket_target_samples(n, hop, bucket_frames, codec._max_encode_samples) // hop
        print(f"  {secs:>7}s: true_frames={(n + hop - 1)//hop} padded_frames={tgt_frames} "
              f"identical={same} mismatched_codes={mism}", flush=True)
        ok = ok and same
    print("EQUIVALENCE: " + ("PASS — safe to set NANO_SS_BUCKET_FRAMES"
                             if ok else "FAIL — coarse bucketing drifts (keep grid=1)"), flush=True)

    # Timing: a spread of DISTINCT lengths (what thrashes the plan cache in prod).
    xs = [_stereo(30 + (i * 7) % 300) for i in range(40)]

    def _run(grid: int) -> float:
        codec._bucket_frames = grid
        codec.encode(xs[0])  # warm this config
        t0 = time.perf_counter()
        for x in xs:
            codec.encode(x)
        return time.perf_counter() - t0

    grid1 = _run(1)  # minimal frame-align (prod default) — distinct length per song
    coarse = _run(bucket_frames)  # coarse grid → few distinct lengths → plan reuse
    codec._bucket_frames = 1
    print(f"\n=== (3) TIMING over {len(xs)} distinct-length encodes ===", flush=True)
    print(f"grid=1={grid1:.1f}s  grid={bucket_frames}={coarse:.1f}s  "
          f"speedup={grid1 / max(1e-9, coarse):.2f}x", flush=True)


@app.function(
    image=image,
    gpu=os.environ.get("NANO_SPIKE_GPU", "A100-40GB"),
    timeout=60 * 30,
    volumes={"/corpus": corpus_vol, "/cache": cache_vol},
)
def bench_gpu(n_songs: int = 16, gpu_label: str = "") -> None:
    """Fixed-file GPU encode benchmark — the clean cross-GPU $/file instrument.

    Encodes the SAME deterministic set of real base_13 songs (so runs on different
    GPUs are apples-to-apples, unlike a fan-out over a resumable/picked-over wave)
    at grid=1 (the prod default), timing ONLY codec.encode(). Decode is the prod
    ffmpeg path (audio_io.decode_pcm); loudness-norm is skipped since it changes
    sample VALUES not COUNT, so it can't affect encode time. Warms once (like the
    prod @enter warmup) so file-1's initial XLA graph build isn't charged to it;
    each real file then pays its own per-length cuFFT/XLA recompile — the real cost.
    Per-file (frames, encode_s) is printed so steady-state (encode_s/frames ~const)
    vs recompile-dominated (encode_s ~const regardless of frames) is visible. Set
    NANO_SPIKE_GPU to pick the card."""
    import time

    import numpy as np
    import torch

    from diskrot.audio_io import decode_pcm
    from model.codec import SpectroStreamCodec

    # NANO_SPIKE_GPU is a LOCAL var (picks the gpu= decorator at modal-run parse);
    # it is NOT in the container env, so the label must come in as an arg.
    gpu = gpu_label or os.environ.get("NANO_SPIKE_GPU", "A100-40GB")
    depth = int(os.environ.get("NANO_SS_DEPTH", "32"))
    codec = SpectroStreamCodec(device="cuda", depth=depth)
    sr, ch = int(codec.SAMPLE_RATE), int(codec.N_CHANNELS)

    # Deterministic pick: sort the wave, stride across it (spans the wave, so a
    # natural length mix), take up to n_songs successful encodes. Identical set on
    # every GPU because the corpus + ordering are static. No os.stat() (per-file
    # HEAD over R2 is slow at 100k files); over-sample the stride to absorb dregs.
    all_mp3 = sorted(Path("/corpus/waves/wave_base_13").glob("*.mp3"))
    if not all_mp3:
        print("no base_13 mp3s found", flush=True)
        return
    step = max(1, len(all_mp3) // (n_songs * 3))
    cands = all_mp3[::step]
    print(f"=== bench_gpu on {gpu}: target {n_songs} songs "
          f"(of {len(all_mp3)} in wave, striding {len(cands)} candidates) depth={depth} ===",
          flush=True)

    # Warm once — pay the initial XLA graph build off the measured path.
    codec.encode(torch.zeros((ch, sr * 30), dtype=torch.float32))
    print("[warmup] done", flush=True)

    rows = []
    for p in cands:
        if len(rows) >= n_songs:
            break
        try:
            y = decode_pcm(p, sr, ch)  # [C, N] float32, prod decode
            x = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
            t0 = time.perf_counter()
            codes = codec.encode(x)  # TF/JAX blocks until the host array is ready
            dt = time.perf_counter() - t0
            frames = int(codes.shape[1])
            if frames < 500:  # < ~20s: prod min_frames skip; don't count dregs
                continue
            rows.append((frames, dt))
            print(f"  {frames:>6}f ({frames / codec.FRAME_RATE_HZ:>5.0f}s)  "
                  f"encode {dt:6.2f}s  {1000 * dt / frames:6.2f} ms/frame  {p.name[:48]}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 — surface + skip (the off-by-one crash etc.)
            print(f"  FAILED {p.name[:48]}: {type(e).__name__}: {str(e)[:80]}", flush=True)

    if not rows:
        print(f"=== {gpu}: NO successful encodes ===", flush=True)
        return
    tot = sum(dt for _, dt in rows)
    totf = sum(f for f, _ in rows)
    n = len(rows)
    per = sorted(dt for _, dt in rows)
    med = per[n // 2]
    print(f"\n=== {gpu} SUMMARY: {n} songs | total encode {tot:.1f}s | "
          f"mean {tot / n:.2f}s/file | median {med:.2f}s/file | "
          f"{1000 * tot / totf:.2f} ms/frame overall ({totf} frames, "
          f"{totf / codec.FRAME_RATE_HZ / 60:.1f} min audio) ===", flush=True)


@app.function(image=image, timeout=60 * 10, volumes={"/cache": cache_vol})
def inspect_ss(lo: int = 100, hi: int = 200) -> None:
    """Dump SpectroStream.encode source + the module lines around the assertion so
    we can root-cause the grid=1 off-by-one crash (`Expected (1,T,256) got
    (1,T-1,256)`). CPU-only (no gpu on this fn) — just reads source, no model build."""
    import inspect

    from magenta_rt import spectrostream

    print("===== SpectroStream.encode source =====", flush=True)
    try:
        print(inspect.getsource(spectrostream.SpectroStream.encode), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(couldn't getsource of .encode: {e})", flush=True)
    path = spectrostream.__file__
    print(f"\n===== {path} lines {lo}..{hi} =====", flush=True)
    lines = open(path).read().splitlines()
    for i in range(max(0, lo - 1), min(len(lines), hi)):
        print(f"{i + 1:>4}: {lines[i]}", flush=True)


@app.local_entrypoint()
def inspect_source(lo: int = 100, hi: int = 200):
    inspect_ss.remote(lo=lo, hi=hi)


@app.function(
    image=image,
    gpu=os.environ.get("NANO_SPIKE_GPU", "L40S"),
    timeout=60 * 20,
    volumes={"/cache": cache_vol},
)
def verify_offbyone_fix(depth: int = 32) -> None:
    """Validate the codec.encode float-ceil off-by-one guard end-to-end on GPU:
    (1) the RAW SpectroStream crashes on frame-aligned boundary lengths (F*hop where
        int(ceil((T/sr)*fr)) == F+1); (2) the PATCHED codec.encode succeeds on those
        exact lengths and returns F frames after crop; (3) a non-boundary length is
        unaffected. Proves the conv-output proxy (ceil_int(T/hop)) the guard relies
        on actually holds on the real model."""
    import numpy as np
    import torch

    from model.codec import SpectroStreamCodec

    codec = SpectroStreamCodec(device="cuda", depth=depth)
    sr, ch = int(codec.SAMPLE_RATE), int(codec.N_CHANNELS)
    hop, fr = sr // int(codec.FRAME_RATE_HZ), int(codec.FRAME_RATE_HZ)

    def ss_exp(t):
        return int(np.ceil((t / float(sr)) * float(fr)))

    def stereo(nsamp):
        t = np.linspace(0, nsamp / sr, nsamp, endpoint=False, dtype=np.float32)
        return torch.from_numpy(
            np.stack([0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 277 * t)], 0)
        )

    bad_F = [F for F in range(2, 4000) if ss_exp(F * hop) != F][:8]
    print(f"=== off-by-one fix validation (depth={depth}) ===", flush=True)
    print(f"bad frame-aligned F (float-ceil = F+1): {bad_F}", flush=True)

    # (1) RAW SS crashes on the frame-aligned boundary lengths (no guard).
    n_crash = 0
    for F in bad_F:
        wav = codec._to_waveform(stereo(F * hop))  # exactly F*hop samples
        try:
            codec.model.encode(wav)
            print(f"  raw F={F:>5}: NO crash (unexpected!)", flush=True)
        except AssertionError as e:
            n_crash += 1
            print(f"  raw F={F:>5}: crashes as expected — {str(e)[:60]}", flush=True)
    print(f"(1) RAW SS crashed on {n_crash}/{len(bad_F)} boundary lengths", flush=True)

    # (2) PATCHED codec.encode succeeds on those same lengths + returns F frames.
    ok = 0
    for F in bad_F:
        try:
            codes = codec.encode(stereo(F * hop))  # through the new guard
            got = int(codes.shape[1])
            good = got == F
            ok += int(good)
            print(f"  fixed F={F:>5}: frames={got} want={F} {'OK' if good else 'BAD'}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  fixed F={F:>5}: STILL FAILS — {type(e).__name__}: {str(e)[:60]}", flush=True)
    print(f"(2) PATCHED encode OK on {ok}/{len(bad_F)}", flush=True)

    # (3) a non-boundary length is unaffected (guard must not fire).
    good_F = next(F for F in range(100, 4000) if ss_exp(F * hop) == F)
    c = codec.encode(stereo(good_F * hop))
    gf = int(c.shape[1])
    print(f"(3) good F={good_F}: frames={gf} want={good_F} {'OK' if gf == good_F else 'BAD'}",
          flush=True)
    verdict = "PASS" if (n_crash == len(bad_F) and ok == len(bad_F) and gf == good_F) else "CHECK"
    print(f"=== VERDICT: {verdict} ===", flush=True)


@app.local_entrypoint()
def verify_fix(depth: int = 32):
    print(f"validating off-by-one fix on {os.environ.get('NANO_SPIKE_GPU', 'L40S')}...")
    verify_offbyone_fix.remote(depth=depth)


@app.local_entrypoint()
def bench(n_songs: int = 16):
    gpu = os.environ.get("NANO_SPIKE_GPU", "A100-40GB")
    print(f"fixed-file GPU encode bench on {gpu} ({n_songs} songs)...")
    bench_gpu.remote(n_songs=n_songs, gpu_label=gpu)


@app.local_entrypoint()
def check_bucket(depth: int = 32, bucket_frames: int = 256):
    # Verify NANO_SS_BUCKET_FRAMES is byte-identical (the gate) + measure the win.
    # NANO_SPIKE_GPU also picks the GPU, so this doubles as the GPU A/B: run it on
    # A100-40GB / L40S / H100 and compare the per-length encode timing + $/s.
    print(f"checking SS length-bucketing (bucket_frames={bucket_frames}, depth={depth}) "
          f"on {os.environ.get('NANO_SPIKE_GPU', 'A100-40GB')}...")
    bucket_check.remote(depth=depth, bucket_frames=bucket_frames)


@app.local_entrypoint()
def measure(start_s: int = 180, stop_s: int = 480, step_s: int = 10, depth: int = 32):
    # .spawn() (NOT .remote()) so the sweep runs fully server-side and survives the
    # local client exiting — a blocking .remote() gets cancelled mid-encode the
    # moment the (backgrounded) launcher disconnects. Pair with `modal run --detach`.
    print(f"measuring SS int32 overflow length (sweep {start_s}..{stop_s}s step {step_s}s, "
          f"depth={depth}) on one A100 — it will SIGABRT at the crossover...")
    fc = measure_overflow.spawn(start_s=start_s, stop_s=stop_s, step_s=step_s, depth=depth)
    print(f"spawned (detached) — function call id: {fc.object_id}")
    print("read: modal app logs nano-ss-spike   (find last 'OK secs=' before the F0000 abort)")


@app.local_entrypoint()
def main(n_songs: int = 6):
    print("launching SpectroStream codec spike (single A100 container)...")
    spike.remote(n_songs=n_songs)
    print("done — A/B the WAVs:")
    print("  modal volume get nano-output /ss_spike/ ./ss_spike/")


@app.local_entrypoint()
def verify(depth: int = 16):
    print(f"validating SpectroStreamCodec wrapper at depth={depth}...")
    verify_wrapper.remote(depth=depth)
