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
