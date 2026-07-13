"""MLX (Apple Silicon) backend for the SpectroStream codec.

magenta-rt 2.x dropped the TF/JAX-only top-level ``magenta_rt.spectrostream``
module (the v1 stack that tokenizes the corpus on Modal) and instead ships a
native MLX port of the SAME codec under ``magenta_rt.mlx.spectrostream`` —
token-parity with v1 was verified empirically 2026-07-13 (identical input:
cb0 100%, cb0-23 97.8% match; cross-stack decode corr 0.994), so v9 corpus
tokens decode correctly through this backend.

``MLXSpectroStream`` duck-types the two methods ``model.codec.SpectroStreamCodec``
calls on its ``self.model`` (the v1 ``spectrostream.SpectroStream`` object):

  - ``encode(waveform) -> np.ndarray[T, depth] int32``  (frame-major)
  - ``decode(codes [T,K] | [B,T,K]) -> Waveform | list[Waveform]``

so the facade's padding/bucketing/crop logic runs unchanged. Import is lazy from
``SpectroStreamCodec.__init__`` — this module (and ``mlx``) only load when the v1
package is absent (i.e. local Apple-Silicon serving).

Weights are the codec-only safetensors from the HF repo
``google/magenta-realtime-2`` (``resources/spectrostream/``, ~300 MB total:
decoder + encoder + quantizer). Search order: ``NANO_SS_WEIGHTS_DIR``, then
magenta-rt's standard ``paths.spectrostream_dir()``, else auto-download to the
latter via huggingface_hub.

Quirk handled here: the MLX decoder is causal with a 1-frame lookahead, so a
T-frame ``codes_to_waveform`` yields (T-1)*1920 samples. ``decode`` appends one
repeated last frame and trims to T*1920, restoring v1's exact output length
(tail-frame corr vs v1 ≈ 0.975).
"""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np

_WEIGHT_FILES = ("decoder.safetensors", "encoder.safetensors", "quantizer.safetensors")
_HF_REPO = "google/magenta-realtime-2"
_HF_PREFIX = "resources/spectrostream"


def _resolve_weights_dir() -> Path:
    env = os.environ.get("NANO_SS_WEIGHTS_DIR")
    if env:
        d = Path(env)
        if not all((d / f).exists() for f in _WEIGHT_FILES):
            raise FileNotFoundError(
                f"NANO_SS_WEIGHTS_DIR={env} is missing one of {_WEIGHT_FILES}"
            )
        return d
    from magenta_rt import paths

    d = paths.spectrostream_dir()
    if all((d / f).exists() for f in _WEIGHT_FILES):
        return d
    # Auto-fetch the codec-only weights (~300 MB once).
    from huggingface_hub import hf_hub_download

    print(f"[spectrostream-mlx] fetching codec weights from {_HF_REPO} -> {d}", flush=True)
    # hf_hub_download re-creates the repo-relative path under local_dir, so
    # local_dir must be the root above the resources/ prefix.
    local_root = d.parent.parent
    for f in _WEIGHT_FILES:
        hf_hub_download(
            repo_id=_HF_REPO, filename=f"{_HF_PREFIX}/{f}", local_dir=str(local_root)
        )
    return d


class MLXSpectroStream:
    """v1-``SpectroStream``-shaped facade over the magenta-rt 2.x MLX codec."""

    def __init__(self, max_rvq_depth: int):
        from magenta_rt import audio
        from magenta_rt.mlx import spectrostream as ss_mod
        from magenta_rt.mlx.spectrostream.load_weights import (
            _load_jax_params,
            load_spectrostream_weights,
        )

        import sequence_layers.mlx as sl  # vendored; importable after magenta_rt

        self._audio = audio
        self._sl = sl
        self.max_rvq_depth = int(max_rvq_depth)

        # The full 64-quantizer codec is built regardless of depth (weights are
        # for 64); encode slices the RVQ prefix, decode accepts any K <= 64.
        cfg = ss_mod.stft_spectrostream_40ms_generic_48khz_stereo_config(
            use_unique_codes=False
        )
        self._model = ss_mod.SpectroStream(cfg)
        d = _resolve_weights_dir()
        q = _load_jax_params(str(d / "quantizer.safetensors"))["params"]["soundstream"]
        dec = _load_jax_params(str(d / "decoder.safetensors"))["params"]
        load_spectrostream_weights(
            self._model,
            str(d / "decoder.safetensors"),  # dirname also locates encoder.safetensors
            soundstream_params={"quantizer": q["quantizer"], "decoder": dec["decoder"]},
        )
        self._hop = 1920  # 48000 / 25
        self._prime_kernels()

    def _prime_kernels(self) -> None:
        """Run a tiny encode + decode NOW, on the construction (main) thread.

        MLX initializes each distinct GPU op lazily on the thread that first runs
        it, and that must be the main thread — the server drives decode from
        per-request worker threads, which crash with "no Stream(gpu, 0)" on any
        op not already primed (same convention as MLXNanoAudioGPT._prime_kernels).
        Best-effort: a hiccup here only forfeits priming."""
        try:
            silence = self._audio.Waveform(
                np.zeros((2 * self._hop, 2), dtype=np.float32), 48000
            )
            self.decode(self.encode(silence))
        except Exception as e:  # noqa: BLE001 — priming is best-effort
            print(f"[spectrostream-mlx] kernel prime skipped: {type(e).__name__}: {e}",
                  flush=True)

    # -- the two methods SpectroStreamCodec calls on self.model ----------------

    def encode(self, waveform) -> np.ndarray:
        """waveform: magenta_rt Waveform (stereo, 48 kHz, frame-aligned by the
        facade). Returns [T, max_rvq_depth] int32, frame-major (v1 layout)."""
        samples = np.asarray(waveform.samples, dtype=np.float32)  # [N, 2]
        x = self._sl.Sequence(
            mx.array(samples[None]),
            mx.ones((1, samples.shape[0]), dtype=mx.bool_),
        )
        codes = self._model.waveform_to_codes_layer.layer(x)
        out = np.array(codes.values)[0]  # [T, 64]
        return np.ascontiguousarray(out[:, : self.max_rvq_depth]).astype(np.int32)

    def _decode_one(self, codes: np.ndarray):
        T = codes.shape[0]
        # Causal decoder w/ 1-frame lookahead yields (T-1) frames of samples;
        # pad one repeated frame and trim back to exactly T*hop.
        padded = np.concatenate([codes, codes[-1:]], axis=0).astype(np.uint32)
        seq = self._sl.Sequence(
            mx.array(padded[None]),
            mx.ones((1, padded.shape[0]), dtype=mx.bool_),
        )
        out = self._model.codes_to_waveform_layer.layer(seq)
        samples = np.array(out.values)[0][: T * self._hop]  # [T*1920, 2]
        return self._audio.Waveform(np.ascontiguousarray(samples), 48000)

    def decode(self, codes: np.ndarray):
        """codes: [T, K] -> Waveform, or [B, T, K] -> list[Waveform] (v1 API)."""
        arr = np.asarray(codes)
        if arr.ndim == 2:
            return self._decode_one(arr)
        return [self._decode_one(a) for a in arr]
