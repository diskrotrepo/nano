"""Inference: mp3 in → mp3 (or wav) out via the trained nano audio GPT."""
from __future__ import annotations

import contextlib
import functools
import io
import os
import platform
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from diskrot.audio_io import decode_pcm
from model.codec import DACodec, get_codec
from model.nano_audio_gpt import GPTConfig, NanoAudioGPT
from model.text_encoder import CLAPTextEncoder
from server.prompt_sweetener import PromptSweetener


def _mlx_available() -> bool:
    """True only on Apple-Silicon macOS with the `mlx` package installed.

    MLX provides a much faster autoregressive decode path than PyTorch-MPS; it
    only exists on arm64 macOS, and importing mlx.core confirms that."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def _quantize_torch_linears(model: torch.nn.Module, bits: int, group_size: int = 64) -> None:
    """Weight-only int4/int8 quantization of the model's nn.Linear layers via
    torchao — the CUDA counterpart to the MLX backend's `mnn.quantize`.

    Mirrors that path's policy (model/nano_audio_gpt_mlx.py:_quantize): quantize
    the big Linears (attention qkv/proj, MLP, output heads) but keep all
    embeddings and the small, phonetically sensitive lyric encoder in full
    precision. Done in-place via tensor subclasses, so it composes with
    torch.compile (apply this *before* compiling).

    int8 is per-output-channel and runs in fp16. int4 is group-quantized
    (group_size=64, matching the MLX path) and requires bf16 activations — the
    caller casts the model to bf16 in that case. Linears whose in_features aren't
    divisible by group_size are left unquantized (the int4 tinygemm kernel
    requires it); none of nano's Linears hit that today (d_model=2048, d_ff=8192,
    heads in=2048 are all divisible by 64), but the guard mirrors MLX."""
    from torchao.quantization import quantize_

    def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        if fqn.startswith("lyric_encoder"):
            return False
        if bits == 4 and module.in_features % group_size != 0:
            return False
        return True

    # torchao moved the weight-only quant API from functions
    # (int4_weight_only/int8_weight_only) to config objects
    # (Int4WeightOnlyConfig/Int8WeightOnlyConfig). Prefer the new API; fall back
    # to the old one for older torchao.
    try:
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
        config = (Int4WeightOnlyConfig(group_size=group_size) if bits == 4
                  else Int8WeightOnlyConfig())
    except ImportError:
        from torchao.quantization import int4_weight_only, int8_weight_only
        config = (int4_weight_only(group_size=group_size) if bits == 4
                  else int8_weight_only())
    quantize_(model, config, filter_fn=filter_fn)


# ---- single-GPU generation gate -------------------------------------------
# The Apple GPU watchdog aborts the whole process ("[METAL] Command buffer
# execution failed: ... GPU Hang Error", with the other in-flight buffers
# "Discarded (victim of GPU error/recovery)") when several long MLX generations
# run on the one local GPU at once — e.g. the webapp's multi-"take" UI firing N
# concurrent /generate_stream. It's an uncaught C++ abort, not a catchable Python
# exception, so the only defense is to PREVENT the overcommit: serialize GPU
# generation. Only the MLX (local Apple-Silicon) path needs this — CUDA/Modal
# handle concurrency via batching + container scale-out, so the gate stays a
# no-op there (enabled only once an MLX engine loads). Tune with
# NANO_MAX_CONCURRENT_GEN (default 1 = strict serialize).
_GEN_GATE = threading.BoundedSemaphore(
    max(1, int(os.environ.get("NANO_MAX_CONCURRENT_GEN", "1")))
)
_GEN_GATE_ON = False  # flipped True the first time an MLX engine is built


@contextlib.contextmanager
def _gpu_gen_gate():
    """Serialize GPU generation when gating is active (MLX); no-op otherwise."""
    if not _GEN_GATE_ON:
        yield
        return
    _GEN_GATE.acquire()
    try:
        yield
    finally:
        _GEN_GATE.release()


def _gated(fn):
    """Hold the GPU gate for a whole (non-streaming) generate call. Streaming
    methods are generators — they gate inside `_stream_mp3` instead, so the gate
    spans the actual generation rather than just the generator's creation."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _gpu_gen_gate():
            return fn(*args, **kwargs)
    return wrapper


class InferenceEngine:
    def __init__(self, ckpt_path: str | None = None, device: str | None = None):
        self.device = device or os.environ.get("NANO_DEVICE") or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        # NANO_CODEC selects the codec: "dac" (default, mono) or "spectrostream"
        # (v9, joint stereo). decode() then returns [samples] (mono) or [2,samples].
        self.codec = get_codec(device=self.device)
        self.n_channels = getattr(self.codec, "N_CHANNELS", 1)
        if self.n_channels == 2:
            # CC-BY-4.0: SpectroStream weights are Google Magenta RealTime.
            print("[inference] SpectroStream codec (stereo) — audio decode uses "
                  "SpectroStream (CC-BY-4.0, Google Magenta RealTime)", flush=True)
        self.text_encoder: CLAPTextEncoder | None = None
        self.sweetener: PromptSweetener | None = None  # lazy — only built on first use
        self._demucs = None  # (model, apply_fn) — lazy, only built on first /stem call

        ckpt_path = ckpt_path or os.environ.get("NANO_CKPT", "./checkpoints/latest.pt")
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = GPTConfig(**ckpt["cfg"])
            state = ckpt["model"]
            # strip torch.compile's _orig_mod. prefix if present
            if any(k.startswith("_orig_mod.") for k in state):
                state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

            # Backend on Apple Silicon: MLX by default (int8, ~1.9x faster than
            # torch-MPS on the v10 DAC shape and now numerically correct — the CFG
            # stage-batching that used to collapse DAC to noise is run unbatched,
            # and activations/head default to fp32). Opt back out to the
            # parity-tested torch-MPS path with NANO_MLX=0. Elsewhere: PyTorch.
            self.backend = (
                "mlx"
                if os.environ.get("NANO_MLX", "1") != "0"
                and self.device == "mps"
                and _mlx_available()
                else "torch"
            )
            if self.backend == "mlx":
                import mlx.core as mx

                from model.nano_audio_gpt_mlx import MLXNanoAudioGPT

                bits = int(os.environ.get("NANO_MLX_BITS", "8"))
                # Activation dtype. bf16 is the training dtype, BUT on the v10 DAC
                # shape MLX's bf16 matmul accumulation drifts ~2% argmax vs torch
                # and the rollout compounds it into noise (torch-bf16 itself is
                # 97.6% vs fp32 and stays coherent; MLX bf16-act is only ~95.8%).
                # fp32 activations (int8 weights kept) recover ~97.8% ≈ the torch
                # coherence bar. NANO_MLX_ACT_DTYPE=fp32|bf16 (default fp32 for DAC
                # correctness). The head is always fp32 (MLXNanoAudioGPT fp32_head).
                act = os.environ.get("NANO_MLX_ACT_DTYPE", "fp32").lower()
                act_dtype = mx.float32 if act == "fp32" else mx.bfloat16
                self.model = MLXNanoAudioGPT(
                    cfg, state, dtype=act_dtype, bits=bits if bits in (4, 8) else None
                )
                print(f"[inference] backend: mlx (Apple Silicon), weights="
                      f"{'bf16' if self.model._bits == 16 else f'int{self.model._bits}'}"
                      f", act={act}, fp32_head={self.model._fp32_head}")
                # One local GPU — serialize generation so concurrent requests
                # can't overcommit it into a watchdog GPU hang (see _gpu_gen_gate).
                global _GEN_GATE_ON
                _GEN_GATE_ON = True
            else:
                self.model = NanoAudioGPT(cfg).to(self.device)
                self.model.load_state_dict(state)
                self.model.eval()

                # Weight quantization (CUDA only), mirroring NANO_MLX_BITS on the
                # MLX path. NANO_BITS=8|4 → int8/int4 weight-only via torchao;
                # anything else (default) stays fp16. int4's tinygemm kernel needs
                # bf16 activations, so the compute dtype follows the bit-width.
                bits = int(os.environ.get("NANO_BITS", "16"))
                quantized = self.device == "cuda" and bits in (4, 8)
                # Compute dtype. The model TRAINS in bf16 (8-bit exponent). On
                # Apple Silicon, MPS accumulates fp16 matmuls in fp16 (CUDA uses
                # fp32), so fp16's narrow exponent range overflows this model's
                # activations and the autoregressive rollout collapses to a
                # single-pitch drone (verified: fp16 top-token-frac 0.90 vs bf16
                # 0.78 vs fp32 0.57 at step 17k). Use bf16 on MPS — training
                # dtype, same memory as fp16, full fp32 exponent range. CUDA fp16
                # is fine and faster, so keep it. int4 needs bf16 activations.
                dtype_override = {
                    "fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16,
                }.get(os.environ.get("NANO_DTYPE", "").lower())
                if dtype_override is not None:
                    self._torch_dtype = dtype_override
                elif quantized and bits == 4:
                    self._torch_dtype = torch.bfloat16
                elif self.device == "cuda":
                    self._torch_dtype = torch.float16
                else:  # mps (and any non-cuda accelerator)
                    self._torch_dtype = torch.bfloat16
                if self.device != "cpu":
                    self.model = self.model.to(self._torch_dtype)
                if quantized:
                    try:
                        _quantize_torch_linears(self.model, bits)
                    except Exception as e:  # never crash startup over quant
                        print(f"[inference] int{bits} quantization failed "
                              f"({e!r}); falling back to fp16 weights")
                        quantized = False
                self.model._bits = bits if quantized else 16
                # NANO_COMPILE: "off"/"0" (default) = eager decode — the reliable
                # path; "default" = plain torch.compile (fusion, no graphs);
                # "graphs" = reduce-overhead/CUDA graphs. Graphs give a ~3x decode
                # speedup (30s clip ~16s vs ~50s) BUT capture intermittently NaNs
                # on this model (cudagraph + in-place KV cache) and a device-side
                # assert poisons the container — so it's opt-in/experimental until
                # debugged with local CUDA. Default stays eager for reliability.
                compile_mode = os.environ.get("NANO_COMPILE", "off").lower()
                if self.device == "cuda" and compile_mode not in ("0", "off", "none"):
                    # Route the per-step DECODE forward through torch.compile with
                    # CUDA graphs. The old `self.model = torch.compile(self.model)`
                    # was a NO-OP for generation: model.generate() runs the eager
                    # self.forward, never the compiled wrapper, so every token was
                    # uncompiled (~13ms/forward, overhead-bound). Compiling forward
                    # and attaching it where _generate_stream's decode loop looks
                    # for it (model._compiled_forward) is what actually removes the
                    # per-step launch overhead; the decode path uses a tensor
                    # position + fixed-shape masked attention so one graph is
                    # captured and replayed for every step.
                    tc_mode = "reduce-overhead" if compile_mode == "graphs" else "default"
                    if tc_mode == "reduce-overhead":
                        # The KV cache is a persistent, in-place-mutated input to
                        # the compiled decode step; cudagraphs must be told to
                        # support that (with mark_static_address on the buffers) or
                        # it mishandles the mutation and yields NaN logits.
                        try:
                            import torch._inductor.config as _ind
                            _ind.triton.cudagraph_support_input_mutation = True
                        except Exception as e:
                            print(f"[inference] cudagraph mutation flag skipped: {e}")
                    self.model._compiled_forward = torch.compile(
                        self.model.forward, mode=tc_mode,
                    )
                    print(f"[inference] decode compile: {tc_mode}")
                wdesc = {
                    torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32",
                }[self._torch_dtype]
                if self.model._bits != 16:
                    wdesc = f"int{self.model._bits} weight-only ({wdesc} compute)"
                print(f"[inference] backend: torch ({self.device}), weights={wdesc}")
            self.ckpt_step = ckpt.get("step", -1)
            self.ckpt_path = ckpt_path
            print(f"[inference] loaded ckpt {ckpt_path} (step {self.ckpt_step})")
            print(f"[inference] config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, "
                  f"n_heads={cfg.n_heads}, max_seq_len={cfg.max_seq_len}, "
                  f"vocab_per_codebook={cfg.vocab_per_codebook}")
            print(f"[inference] model: {self.model.num_params()/1e6:.2f}M params on {self.device}")

            # From-scratch bootstrap seed: ~1s of ENCODED-SILENCE tokens instead of
            # _resolve_prompt's uniform-random column. The random seed frame sits
            # far off the training manifold; at K=24/25Hz it poisons the whole
            # rollout (from-scratch beat ~0.15) and the poison spans the entire
            # 23-frame delay ramp. A full-ramp silence runway restores structure
            # (beat ~0.48, matching real-context /extend at 0.52 — 2026-07-13 A/B;
            # a 1-frame silence seed does NOT help, the ramp must be covered).
            # v8-era history: a DAC silence seed was removed for collapsing
            # high-energy prompts to silence — the v9 A/B showed no collapse
            # (techno cfg5: RMS 0.087, 0% silent frames), so it returns for v9.
            # Torch backend only; failure degrades to the legacy random seed.
            # SPECTROSTREAM-ONLY: on DAC the silence runway re-created the exact
            # v8-era collapse (2026-07-22, v10_dac_2b: 100% silence at every
            # cfg 1-10, both modes, steps 58k/84k/87k; the same checkpoint
            # /extend-s real audio fine and makes bursty audio from the legacy
            # random seed). DAC's silence attractor is too strong — never seed
            # a DAC rollout with encoded silence.
            self._silence_seed: torch.Tensor | None = None
            _is_ss = type(self.codec).__name__.startswith("SpectroStream")
            if self.backend == "torch" and _is_ss:
                try:
                    n_ch = getattr(self.codec, "N_CHANNELS", 1)
                    n_seed = self.model.cfg.n_codebooks + 1  # cover the delay ramp
                    sil = torch.zeros(n_ch, int(self.codec.SAMPLE_RATE * 1.5))
                    seed = self.codec.encode(sil)
                    if seed.dim() == 3:
                        seed = seed[0]
                    self._silence_seed = (
                        seed[: self.model.cfg.n_codebooks, :n_seed].cpu().long()
                    )
                    print(f"[inference] bootstrap seed: encoded silence "
                          f"({self._silence_seed.shape[1]} frames)")
                except Exception as e:  # noqa: BLE001 — seed is an enhancement, not load-bearing
                    print(f"[inference] bootstrap seed unavailable ({type(e).__name__}: {e}); "
                          "falling back to random seed frame")
            elif self.backend == "torch":
                print("[inference] bootstrap seed: random (DAC — silence seed "
                      "collapses DAC rollouts, SpectroStream-only)")

            # load text encoder if model was trained with text conditioning
            if cfg.use_text_conditioning:
                self.text_encoder = CLAPTextEncoder(d_out=cfg.d_model, device=self.device)
                self.text_encoder.to(self.device)
                text_proj_loaded = False
                if "text_proj" in ckpt:
                    self.text_encoder.proj.load_state_dict(ckpt["text_proj"])
                    text_proj_loaded = True
                self.text_encoder.eval()
                print(f"[inference] text conditioning: on (text_proj loaded: {text_proj_loaded})")
            else:
                print("[inference] text conditioning: off")
        else:
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path} — train a model first or set NANO_CKPT"
            )

    def _gen_metadata(
        self,
        mode: str,
        text: str | None = None,
        negative_text: str | None = None,
        temperature: float | list[float] | None = None,
        top_k: int | None | list[int | None] = None,
        top_p: float | None | list[float | None] = None,
        cfg_scale: float | None = None,
        style_weight: float | None = None,
        **extra: object,
    ) -> dict[str, str]:
        """Build the `nano_*` metadata dict embedded into generated mp3s as ID3
        TXXX frames. Captures model identity + the conditioning + all sampling
        params so a generation is reproducible from the file alone. `None`/empty
        values are skipped; lists (per-codebook params) are stringified as-is."""
        cfg = self.model.cfg
        fields: dict[str, object | None] = {
            "nano_model": "nano",
            "nano_ckpt": os.path.basename(self.ckpt_path),
            "nano_step": self.ckpt_step,
            "nano_params": self.model.num_params(),
            "nano_d_model": cfg.d_model,
            "nano_n_layers": cfg.n_layers,
            "nano_n_heads": cfg.n_heads,
            "nano_max_seq_len": cfg.max_seq_len,
            "nano_mode": mode,
            "nano_prompt": text,
            "nano_negative": negative_text,
            "nano_temperature": temperature,
            "nano_top_k": top_k,
            "nano_top_p": top_p,
            "nano_cfg_scale": cfg_scale,
            "nano_style_weight": style_weight,
            **{f"nano_{k}": v for k, v in extra.items()},
            "nano_generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        return {k: str(v) for k, v in fields.items() if v is not None and v != ""}

    def sweeten_prompt(self, text: str) -> str:
        """Rewrite a terse user prompt into LP-MusicCaps caption style via a
        small local LLM, to strengthen CLAP conditioning. Lazy-loads the
        sweetener on first use; returns ``text`` unchanged on any failure."""
        if self.sweetener is None:
            self.sweetener = PromptSweetener(device=self.device)
        return self.sweetener.sweeten(text)

    def _build_conditioning(
        self,
        text: str | None = None,
        lyrics: str | None = None,
        style_audio_bytes: bytes | None = None,
        style_weight: float = 0.5,
        gender: str | None = None,
        bpm: float | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Build conditioning from tags (``text``), lyrics, style audio, or a mix.

        Tags and lyrics travel as SEPARATE fields — there is no ". " splitting, so
        a long prose / multi-sentence tag description is never shattered into the
        lyrics slot. ``text`` is the tag description: it's split into <=77-token
        chunks and each chunk pooled by CLAP, so the decoder cross-attends to the
        WHOLE description (``encode_chunked``) instead of one truncated vector.
        ``lyrics`` is phonemized (g2p) for the LyricEncoder cross-attention. Style
        audio (when given) is appended as an extra CLAP position. ``gender`` /
        ``bpm`` ride the lyric stream as leading markers (see below). Returns
        ``(tag_emb [1,N,D] | None, lyric_ids [1,L] | None, lyric_mask [1,L] |
        None)``. The tag cross-attn mask is None for a single item (no padding —
        the model attends to all N); batching builds it in ``_stack_conditioning``.
        """
        tags_str = (text or "").strip()
        lyrics_str = (lyrics or "").strip()

        # A selected vocal gender / tempo rides the lyric stream as a leading
        # [male]/[female] / [NNNbpm] bracket — the same dense header markers the
        # dataset injects at train time (every stream opens BOS <gender> <tempo>
        # <key> <vocals> <section>, words or not). Prepending them here makes the
        # stream non-empty even with no lyrics, so an instrumental generation can
        # still steer gender and tempo. (Key and vocal presence have no dedicated
        # request params — type [a minor] / [instrumental] into the lyrics box and
        # the parser routes them to their header slots.)
        # text_with_markers_to_phoneme_ids consumes only the first
        # gender/tempo prefix (in any order), so these override a stray marker the
        # user typed into the lyrics box. Omitting a slot leaves it at its unknown
        # marker — exactly the train-time fallback.
        if gender in ("male", "female"):
            lyrics_str = f"[{gender}] {lyrics_str}".rstrip()
        if bpm is not None and bpm > 0:
            lyrics_str = f"[{bpm:g}bpm] {lyrics_str}".rstrip()

        # --- Tags (chunked CLAP sequence) + optional style-audio blend ---
        tag_emb = None
        if self.text_encoder is not None:
            if tags_str:
                tag_emb, _ = self.text_encoder.encode_chunked([tags_str])
                tag_emb = tag_emb.to(self.device)  # [1,N,D]
            if style_audio_bytes:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(style_audio_bytes)
                    style_path = f.name
                try:
                    audio_emb = self.text_encoder.encode_audio([style_path]).to(self.device)
                finally:
                    os.unlink(style_path)
                # Style ref shares CLAP's joint text/audio space, so it's just one
                # more cross-attn position: scale the tag chunks by (1-w), append
                # the style vector scaled by w. (Style-only -> the style vector.)
                tag_emb = audio_emb if tag_emb is None else torch.cat(
                    [tag_emb * (1 - style_weight), audio_emb * style_weight], dim=1
                )
            if tag_emb is not None:
                tag_emb = tag_emb.to(self._cond_dtype())

        # --- Lyrics (phoneme + structure-marker sequence for the LyricEncoder) ---
        # Parses inline section tags like "[verse] ... [chorus] ..." into section
        # markers, reproducing the dataset's train-time stream exactly (prefix +
        # inline markers). Brackets never reach g2p; unknown labels fold to
        # <no_section>. Plain lyrics (no brackets) get a <no_section> prefix.
        lyric_ids = lyric_mask = None
        # `lyrics is None` marks the CFG negative/baseline call — keep it a null
        # lyric (matches training's lyric-drop). A POSITIVE call (lyrics is a
        # string, even "") with no sung words means the user wants an
        # INSTRUMENTAL: inject an <instrumental> marker so the model suppresses
        # vocals. Without this, empty lyrics sent NO lyric stream at all and the
        # model sang gibberish; and a bare header defaults to <unknown_vocals>,
        # which doesn't suppress vocals either — it needs an explicit
        # <instrumental> (unless the user already put [vocals]/[instrumental]).
        if lyrics is not None and getattr(self.model.cfg, "use_lyric_conditioning", False):
            import re

            from model.lyric_encoder import (
                PAD_PHONEME_ID, text_with_markers_to_phoneme_ids,
            )

            has_words = bool(re.sub(r"\[[^\]]*\]", "", lyrics_str).strip())
            has_vocal_marker = bool(
                re.search(r"\[(instrumental|vocals?|no[ _]vocals)\]", lyrics_str, re.I)
            )
            src = lyrics_str
            if not has_words and not has_vocal_marker:
                src = f"[instrumental] {lyrics_str}".strip()
            ids = text_with_markers_to_phoneme_ids(
                src, max_len=self.model.cfg.max_lyric_len
            )
            if ids:
                lyric_ids = torch.tensor(ids, dtype=torch.long, device=self.device)[None]
                lyric_mask = lyric_ids != PAD_PHONEME_ID

        return tag_emb, lyric_ids, lyric_mask

    def _stack_conditioning(
        self,
        per_item: list[tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Stack a list of per-item ``_build_conditioning`` results into one batch.

        Input is ``B`` tuples of ``(tag_emb [1,N_i,D] | None, lyric_ids [1,L] |
        None, lyric_mask [1,L] | None)``; output is the batched
        ``(tag_emb [B,Nmax,D] | None, tag_kv_mask [B,1,1,Nmax] | None, lyric_ids
        [B,Lmax] | None, lyric_mask [B,Lmax] | None)`` the model's batch-general
        decode loop consumes.

        Two NaN/equivalence subtleties make this non-trivial:
        - **Tags:** each item is now a chunked SEQUENCE whose length N_i differs
          (different description lengths), so they're padded to ``Nmax`` and an
          additive mask -inf's the pad of each row (no softmax dilution). A row
          without tags is a single un-masked ZERO chunk, which (bias-free
          cross-attn) is *exactly* "skip text conditioning" for that row — so a
          mixed present/absent batch is well-defined and never fully masked. If no
          row has tags the whole axis is ``None`` (the model skips it entirely).
        - **Lyrics:** ``encode_lyrics`` masks padded positions with ``-inf``, and a
          *fully* padded row would NaN the cross-attn softmax. So when the batch has
          any lyrics, rows without their own stream are given the minimal BOS+header
          stream (``text_with_markers_to_phoneme_ids("")`` — the same wordless
          header the model trains on, never fully padded). If no row has lyrics the
          axis is ``None``."""
        B = len(per_item)
        tags = [t for (t, _, _) in per_item]
        lyr = [l for (_, l, _) in per_item]

        if all(t is None for t in tags):
            tag_emb = tag_kv_mask = None
        else:
            ref = next(t for t in tags if t is not None)
            D = ref.shape[-1]
            n_max = max((t.shape[1] if t is not None else 1) for t in tags)
            tag_emb = torch.zeros(B, n_max, D, dtype=ref.dtype, device=ref.device)
            keep = torch.zeros(B, n_max, dtype=torch.bool, device=ref.device)
            for i, t in enumerate(tags):
                if t is None:
                    keep[i, 0] = True  # un-masked zero chunk == "no tags"
                    continue
                n = t.shape[1]
                tag_emb[i, :n] = t[0]
                keep[i, :n] = True
            tag_kv_mask = CLAPTextEncoder.additive_kv_mask(keep, tag_emb.dtype)

        if all(l is None for l in lyr):
            lyric_ids = lyric_mask = None
        else:
            from model.lyric_encoder import (
                BOS_PHONEME_ID, PAD_PHONEME_ID, text_with_markers_to_phoneme_ids,
            )

            rows: list[torch.Tensor] = []
            for l in lyr:
                if l is not None:
                    rows.append(l[0])  # [L_i]
                else:  # synthesize the wordless BOS+header stream (never fully padded)
                    ids = text_with_markers_to_phoneme_ids(
                        "", max_len=self.model.cfg.max_lyric_len
                    ) or [BOS_PHONEME_ID]
                    rows.append(torch.tensor(ids, dtype=torch.long, device=self.device))
            Lmax = max(r.shape[0] for r in rows)
            lyric_ids = torch.full(
                (B, Lmax), PAD_PHONEME_ID, dtype=torch.long, device=self.device
            )
            for i, r in enumerate(rows):
                lyric_ids[i, : r.shape[0]] = r
            lyric_mask = lyric_ids != PAD_PHONEME_ID
        return tag_emb, tag_kv_mask, lyric_ids, lyric_mask

    def _build_melody(self, melody_audio_bytes: bytes) -> "torch.Tensor":
        """Chroma for the uploaded hum -> melody tensor [1, T, 12] on device.

        Uses the SHARED ``diskrot.melody.extract_chroma`` (the same code the
        training pack runs) so the inference chroma is byte-identical to what the
        model trained on. The hum's length defines T (the cover length); chroma
        frame count = ceil(samples/512), the DAC convention.
        """
        from diskrot.melody import extract_chroma

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(melody_audio_bytes)
            mel_path = f.name
        try:
            chroma = extract_chroma(mel_path)  # [12, T] float32
        finally:
            os.unlink(mel_path)
        mel = torch.from_numpy(chroma).transpose(0, 1).contiguous()  # [T, 12]
        return mel.to(self.device).to(self._cond_dtype())[None]  # [1, T, 12]

    def _cond_dtype(self) -> torch.dtype:
        """dtype for the conditioning tensor, to match the model's compute dtype.
        fp32 only for the torch-on-CPU path; the torch accelerated paths follow
        `self._torch_dtype` (fp16, or bf16 when running int4-quantized); MLX is
        always fp16."""
        if self.backend == "torch":
            return torch.float32 if self.device == "cpu" else self._torch_dtype
        return torch.float16

    @_gated
    @torch.no_grad()
    def extend_audio(
        self,
        full_audio_bytes: bytes,
        add_seconds: float = 20.0,
        overlap_seconds: float = 8.0,
        from_seconds: float | None = None,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        lyrics: str | None = None,
        negative_text: str | None = None,
        style_audio_bytes: bytes | None = None,
        style_weight: float = 0.5,
        lyric_cfg_scale: float | None = None,
        gender: str | None = None,
        bpm: float | None = None,
    ) -> tuple[bytes, str]:
        """Continue a clip forward from a point in time. Returns [original 0→T | new].

        ``from_seconds`` (T) is the cut point: the original is kept verbatim up to T
        and the model generates forward from there, *discarding* whatever the clip
        had after T. The seed is the ``overlap_seconds`` immediately before T. When
        ``from_seconds is None`` (default) T is the clip's end, so nothing is dropped
        and you get ``[full original | new]`` — the historical behavior.

        The kept prefix is the raw uploaded waveform (never re-decoded through DAC),
        and only the small seed window counts against the context budget, so this
        chains a clip past the model's single-shot cap; each call adds ~add_seconds.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(full_audio_bytes)
            in_path = f.name
        try:
            full_wav = self._load_wav(in_path)        # [C, samples] (C matches codec)
            full_tokens = self._encode_prompt_tokens(full_wav)
        finally:
            os.unlink(in_path)

        # Cut point T (in frames): the end of the original we keep, and where the
        # model picks up. None = the clip's tail (keep everything, append to end).
        if from_seconds is None:
            cut_frame = full_tokens.shape[1]
        else:
            cut_frame = max(1, min(
                int(from_seconds * self.codec.FRAME_RATE_HZ),
                full_tokens.shape[1],
            ))
        # Seed = the overlap_seconds immediately before T.
        overlap_frames = max(1, int(overlap_seconds * self.codec.FRAME_RATE_HZ))
        seed_start = max(0, cut_frame - overlap_frames)
        prompt_tokens = full_tokens[:, seed_start:cut_frame]
        if prompt_tokens.shape[1] == 0:
            raise ValueError("Selected seed window is empty; move the point later or widen overlap_seconds.")

        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1
        new_frames = int(add_seconds * self.codec.FRAME_RATE_HZ)
        if prompt_tokens.shape[1] + new_frames > max_total:
            new_frames = max(0, max_total - prompt_tokens.shape[1])
        if new_frames == 0:
            raise ValueError("Overlap window is already at model context limit; reduce overlap_seconds.")

        prompt_dev = prompt_tokens.to(self.device)
        cond_emb, cond_lids, cond_lmask = self._build_conditioning(
            text, lyrics=lyrics, style_audio_bytes=style_audio_bytes,
            style_weight=style_weight, gender=gender, bpm=bpm)
        neg_emb, neg_lids, neg_lmask = self._build_conditioning(negative_text)
        out_tokens = self.model.generate(
            prompt_dev, num_new_frames=new_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=cond_emb, text_emb_neg=neg_emb,
            lyric_ids=cond_lids, lyric_mask=cond_lmask,
            lyric_ids_neg=neg_lids, lyric_mask_neg=neg_lmask,
            cfg_scale=cfg_scale if (cond_emb is not None or neg_emb is not None
                                    or cond_lids is not None or neg_lids is not None) else 1.0,
            lyric_cfg_scale=lyric_cfg_scale,
        )
        new_tokens = out_tokens[:, prompt_tokens.shape[1]:]  # [K, new_frames]

        new_wav = self.codec.decode(new_tokens.cpu())
        if new_wav.dim() == 1:
            new_wav = new_wav.unsqueeze(0)

        # Keep the raw original up to T, then append the continuation. The tail
        # default keeps the whole clip exactly (no frame→sample rounding loss).
        if from_seconds is None:
            keep_samples = full_wav.shape[1]
        else:
            keep_samples = min(
                int(round(cut_frame / self.codec.FRAME_RATE_HZ * self.codec.SAMPLE_RATE)),
                full_wav.shape[1],
            )
        full = torch.cat([full_wav[:, :keep_samples], new_wav], dim=1)
        meta = self._gen_metadata(
            "extend", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
            style_weight=style_weight if style_audio_bytes else None,
            add_seconds=add_seconds, overlap_seconds=overlap_seconds,
            from_seconds=from_seconds,
        )
        return _encode_audio(full, self.codec.SAMPLE_RATE, meta)


    @_gated
    @torch.no_grad()
    def cover_audio(
        self,
        melody_audio_bytes: bytes,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        lyrics: str | None = None,
        negative_text: str | None = None,
        melody_cfg_scale: float | None = None,
        lyric_cfg_scale: float | None = None,
        gender: str | None = None,
        bpm: float | None = None,
    ) -> tuple[bytes, str]:
        """Cover a hummed/uploaded melody in the prompt's timbre. Returns (bytes, mime).

        The melody audio is converted to a chromagram (NOT used as an audio prompt
        — its tokens never appear in the output); generation starts from scratch
        (random seed) and the chroma drives the contour while ``text`` (tags +
        lyrics) drives timbre/instrumentation and words. The hum's length sets the
        output length (clamped to the model's context).
        """
        if not getattr(self.model.cfg, "use_melody_conditioning", False):
            raise RuntimeError(
                "This checkpoint was trained without melody conditioning — /cover "
                "needs a model with use_melody_conditioning=True."
            )
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1

        melody = self._build_melody(melody_audio_bytes)  # [1, T, 12]
        # The seed runway occupies T_prompt frames, so the melody (placed at the
        # new-frame positions) can be at most max_total - seed_frames frames.
        seed_tokens, seed_frames = self._bootstrap()
        new_frames = min(melody.shape[1], max_total - seed_frames)
        if new_frames <= 0:
            raise ValueError("Melody audio is too short or context limit too small.")
        melody = melody[:, :new_frames, :]

        cond_emb, cond_lids, cond_lmask = self._build_conditioning(
            text, lyrics=lyrics, gender=gender, bpm=bpm)
        neg_emb, neg_lids, neg_lmask = self._build_conditioning(negative_text)
        out_tokens = self.model.generate(
            prompt=seed_tokens, num_new_frames=new_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=cond_emb, text_emb_neg=neg_emb,
            lyric_ids=cond_lids, lyric_mask=cond_lmask,
            lyric_ids_neg=neg_lids, lyric_mask_neg=neg_lmask,
            cfg_scale=cfg_scale,
            lyric_cfg_scale=lyric_cfg_scale,
            melody=melody, melody_cfg_scale=melody_cfg_scale,
        )  # [K, seed_frames + new_frames]

        out_tokens = out_tokens[:, seed_frames:]  # strip the seed runway
        wav = self.codec.decode(out_tokens.cpu())
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        meta = self._gen_metadata(
            "cover", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
            melody_cfg_scale=melody_cfg_scale, seconds=new_frames / self.codec.FRAME_RATE_HZ,
        )
        return _encode_audio(wav, self.codec.SAMPLE_RATE, meta)

    @_gated
    @torch.no_grad()
    def infill_audio(
        self,
        before_audio_bytes: bytes,
        after_audio_bytes: bytes,
        gap_seconds: float = 10.0,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        negative_text: str | None = None,
        melody_audio_bytes: bytes | None = None,
        melody_cfg_scale: float | None = None,
    ) -> tuple[bytes, str]:
        """Fill the gap between two clips. Returns ``[before | middle | after]``.

        The model is shown the ``before`` clip (prefix) and the ``after`` clip
        (suffix) in the FIM layout ``prefix <SUF> suffix <MID>`` and generates a
        ``gap_seconds`` bridge that flows out of ``before`` and into ``after``.
        ``text`` drives timbre/instrumentation (tags); lyrics are NOT used (the
        model never trained FIM with the sung-lyric stream). An optional
        ``melody_audio`` hum guides the gap's contour (clamped to gap length).

        The before/after waveforms are kept as the raw uploads (never re-decoded
        through DAC); only the generated middle is decoded, and the three are
        joined with a short equal-power crossfade at each seam to avoid clicks.
        """
        if not getattr(self.model.cfg, "use_fim", False):
            raise RuntimeError(
                "This checkpoint was trained without FIM — /infill needs a model "
                "with use_fim=True (a v8+ checkpoint trained with fim_prob>0)."
            )
        from model.fim import build_fim_prompt

        before_wav, before_codes = self._load_and_encode(before_audio_bytes)
        after_wav, after_codes = self._load_and_encode(after_audio_bytes)

        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1
        gap_frames = int(gap_seconds * self.codec.FRAME_RATE_HZ)
        if gap_frames <= 0:
            raise ValueError("gap_seconds must be positive.")
        if gap_frames > max_total - 2:
            raise ValueError("gap_seconds exceeds the model context limit.")

        # Budget: T_prompt + gap_frames <= max_total, with T_prompt = Tp + Ts + 2
        # (two sentinel frames). Keep the frames ADJACENT to the gap — the tail of
        # `before` and the head of `after` — since those carry the bridge context.
        Tp, Ts = before_codes.shape[1], after_codes.shape[1]
        budget = max_total - gap_frames - 2
        if budget < 2:
            raise ValueError("gap_seconds leaves no room for context; shorten the gap.")
        if Tp + Ts > budget:
            keep = budget // 2
            Tp, Ts = min(Tp, budget - min(Ts, keep)), min(Ts, keep)
            before_codes = before_codes[:, -Tp:]
            after_codes = after_codes[:, :Ts]

        prompt = build_fim_prompt(
            before_codes, after_codes, self.model.cfg.suf_id, self.model.cfg.mid_id,
        ).to(self.device)
        T_prompt = prompt.shape[1]

        # Optional melody for the gap (placed at the new-frame positions by
        # generate's encode_melody_delayed(offset=T_prompt)).
        melody = None
        if melody_audio_bytes and getattr(self.model.cfg, "use_melody_conditioning", False):
            melody = self._build_melody(melody_audio_bytes)[:, :gap_frames, :]

        cond_emb, _, _ = self._build_conditioning(text)
        neg_emb, _, _ = self._build_conditioning(negative_text)
        out_tokens = self.model.generate(
            prompt, num_new_frames=gap_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=cond_emb, text_emb_neg=neg_emb,
            cfg_scale=cfg_scale,
            melody=melody, melody_cfg_scale=melody_cfg_scale,
        )  # [K, T_prompt + gap_frames]
        middle_tokens = out_tokens[:, T_prompt:]  # the bridged gap

        middle_wav = self.codec.decode(middle_tokens.cpu())
        if middle_wav.dim() == 1:
            middle_wav = middle_wav.unsqueeze(0)

        full = _crossfade_concat(
            [before_wav, middle_wav, after_wav], self.codec.SAMPLE_RATE,
        )
        meta = self._gen_metadata(
            "infill", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
            melody_cfg_scale=melody_cfg_scale if melody is not None else None,
            gap_seconds=gap_frames / self.codec.FRAME_RATE_HZ,
        )
        return _encode_audio(full, self.codec.SAMPLE_RATE, meta)

    def _load_and_encode(self, audio_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode mp3 bytes to a raw mono [1, samples] waveform and its DAC codes
        [K, T]. The raw waveform is kept verbatim for stitching (never re-decoded);
        the codes feed the model as FIM prefix/suffix context."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name
        try:
            wav = self._load_wav(in_path)             # [C, samples] (C matches codec)
            codes = self._encode_prompt_tokens(wav)
        finally:
            os.unlink(in_path)
        return wav, codes

    @_gated
    @torch.no_grad()
    def generate_audio(
        self,
        seconds: float = 30.0,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        lyrics: str | None = None,
        negative_text: str | None = None,
        style_audio_bytes: bytes | None = None,
        style_weight: float = 0.5,
        lyric_cfg_scale: float | None = None,
        gender: str | None = None,
        bpm: float | None = None,
        score_clap: bool = False,
    ):
        """Generate audio from scratch (no audio prompt). Returns (audio_bytes, mime_type).

        When ``score_clap`` is True (and text conditioning is present), also
        computes the CLAP text<->audio similarity of the generated clip against
        ``text`` and returns it as a third element: (audio_bytes, mime, clap).
        Default stays a 2-tuple so existing callers are unaffected.

        The autoregressive loop is bootstrapped from a ~1s runway of encoded-
        silence tokens (see ``_bootstrap``): the legacy single random seed column
        sits far off the training manifold and at K=24/25Hz poisons the whole
        rollout (2026-07-13 A/B: beat ~0.15 random vs ~0.48 silence-seeded). The
        v8-era silence seed was removed for collapsing high-energy prompts to
        silence; the v9 A/B showed no such collapse (techno cfg5 RMS 0.087), but
        watch for it on quiet prompts.
        """
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1  # T_total such that T_total + K - 1 <= max_seq_len

        seed_tokens, seed_frames = self._bootstrap()

        new_frames = int(seconds * self.codec.FRAME_RATE_HZ)
        if seed_frames + new_frames > max_total:
            new_frames = max(0, max_total - seed_frames)
        if new_frames == 0:
            raise ValueError("Requested duration exceeds model context limit.")

        cond_emb, cond_lids, cond_lmask = self._build_conditioning(
            text, lyrics=lyrics, style_audio_bytes=style_audio_bytes,
            style_weight=style_weight, gender=gender, bpm=bpm)
        neg_emb, neg_lids, neg_lmask = self._build_conditioning(negative_text)
        out_tokens = self.model.generate(
            prompt=seed_tokens, num_new_frames=new_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=cond_emb, text_emb_neg=neg_emb,
            lyric_ids=cond_lids, lyric_mask=cond_lmask,
            lyric_ids_neg=neg_lids, lyric_mask_neg=neg_lmask,
            cfg_scale=cfg_scale if (cond_emb is not None or neg_emb is not None
                                    or cond_lids is not None or neg_lids is not None) else 1.0,
            lyric_cfg_scale=lyric_cfg_scale,
        )  # [K, seed_frames + new_frames]

        # strip the seed frame(s) before decoding
        out_tokens = out_tokens[:, seed_frames:]

        wav = self.codec.decode(out_tokens.cpu())
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        meta = self._gen_metadata(
            "generate", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
            style_weight=style_weight if style_audio_bytes else None,
            seconds=seconds,
        )
        body, mime = _encode_audio(wav, self.codec.SAMPLE_RATE, meta)
        if not score_clap:
            return body, mime

        clap = None
        if cond_emb is not None and text and text.strip():
            # Score against the prompt in raw CLAP space. Write a wav (the
            # proven path; mirrors scripts/eval_checkpoint.py:_compute_clap) so
            # CLAP's loader never has to decode mp3.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
                # CLAP is mono — downmix a stereo render before scoring.
                clap_wav = (wav.mean(dim=0) if wav.dim() == 2 and wav.shape[0] > 1
                            else wav.squeeze())
                sf.write(tmp, clap_wav.contiguous().cpu().numpy(), self.codec.SAMPLE_RATE)
            try:
                clap = self.text_encoder.audio_text_similarity(tmp, text)
            finally:
                os.unlink(tmp)
        return body, mime, clap

    @_gated
    @torch.no_grad()
    def generate_audio_batch(
        self,
        requests: list[dict],
        seconds: float = 30.0,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 7.0,
        lyric_cfg_scale: float | None = None,
        on_item=None,
    ) -> list[tuple[bytes, str]]:
        """Generate B clips from scratch in ONE batched forward — the throughput
        path behind POST /generate_batch ("generate 8 at once" on a single GPU).

        Because autoregressive decode is memory-bandwidth-bound (each step streams
        all ~2B weights from HBM regardless of batch), B clips cost ≈ the wall-clock
        of one. The model's decode loop is already batch-general; this method only
        builds the batched prompt/conditioning and decodes each row to its own mp3.

        ``requests``: one dict per clip with keys ``text`` / ``negative_text`` /
        ``gender`` / ``bpm`` (all optional). ``seconds``, the sampling ladder,
        ``cfg_scale`` and ``lyric_cfg_scale`` are SHARED across the batch — the
        decode loop applies one set batch-wide (per-item sampling/length would mean
        un-batching the sampler; not worth it). ``on_item(i, body, mime)`` is called
        as each clip finishes encoding, so the caller can persist/emit eagerly.

        Returns ``[(body, mime), ...]`` in request order."""
        if not requests:
            return []
        B = len(requests)
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1
        seed_tokens, seed_frames = self._bootstrap()
        new_frames = int(seconds * self.codec.FRAME_RATE_HZ)
        if seed_frames + new_frames > max_total:
            new_frames = max(0, max_total - seed_frames)
        if new_frames == 0:
            raise ValueError("Requested duration exceeds model context limit.")

        pos = [
            # `or ""` so a positive item with no lyrics still builds an
            # <instrumental> header (None is reserved for the CFG baseline below).
            self._build_conditioning(r.get("text"), lyrics=r.get("lyrics") or "",
                                     gender=r.get("gender"), bpm=r.get("bpm"))
            for r in requests
        ]
        neg = [self._build_conditioning(r.get("negative_text")) for r in requests]
        cond_emb, cond_tkv, cond_lids, cond_lmask = self._stack_conditioning(pos)
        neg_emb, neg_tkv, neg_lids, neg_lmask = self._stack_conditioning(neg)

        if seed_tokens is not None:
            prompt = seed_tokens.unsqueeze(0).expand(B, -1, -1).contiguous()
        else:
            prompt, _ = self.model._resolve_prompt(None, batch_size=B)  # [B, K, 1]
        has_cond = any(
            x is not None for x in (cond_emb, neg_emb, cond_lids, neg_lids)
        )
        out = self.model.generate(
            prompt=prompt, num_new_frames=new_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=cond_emb, text_emb_neg=neg_emb,
            text_kv_mask=cond_tkv, text_kv_mask_neg=neg_tkv,
            lyric_ids=cond_lids, lyric_mask=cond_lmask,
            lyric_ids_neg=neg_lids, lyric_mask_neg=neg_lmask,
            cfg_scale=cfg_scale if has_cond else 1.0,
            lyric_cfg_scale=lyric_cfg_scale,
        )  # [B, K, seed_frames + new_frames]
        out = out[:, :, seed_frames:].cpu()  # strip seed -> [B, K, new_frames]

        results: list[tuple[bytes, str]] = []
        for i in range(B):
            wav = self.codec.decode(out[i])  # [samples]
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            meta = self._gen_metadata(
                "generate", text=requests[i].get("text"),
                negative_text=requests[i].get("negative_text"),
                temperature=temperature, top_k=top_k, top_p=top_p,
                cfg_scale=cfg_scale, seconds=seconds,
            )
            body, mime = _encode_audio(wav, self.codec.SAMPLE_RATE, meta)
            if on_item is not None:
                on_item(i, body, mime)
            results.append((body, mime))
        return results

    def _encode_prompt_tokens(self, wav: torch.Tensor) -> torch.Tensor:
        """codec.encode sliced to the model's codebook count. The codec may store
        a deeper RVQ stack than the model consumes (SpectroStream stores 32, the
        model trains on the K=24 prefix — the same slice TokenDataset applies to
        the packed corpus), so every audio-prompt path must take the prefix or
        generate()'s K assertion trips."""
        return self.codec.encode(wav)[: self.model.cfg.n_codebooks]

    def _bootstrap(self) -> tuple[torch.Tensor | None, int]:
        """From-scratch seed: (silence tokens [K, T] on device, T), or (None, 1)
        when the silence seed is unavailable (model.generate then falls back to
        its legacy random column). The runway covers the full K-1 delay ramp —
        a 1-frame seed measurably does NOT restore bootstrap coherence."""
        if self._silence_seed is None:
            return None, 1
        return self._silence_seed.to(self.device), int(self._silence_seed.shape[1])

    def _load_wav(self, path: str) -> torch.Tensor:
        """Load an uploaded clip to ``[C, samples]`` matching the codec's channel
        count (mono for DAC, stereo for SpectroStream; a mono source is duplicated
        to L=R) so prompt-encode and the kept-prefix stitch stay channel-consistent
        with the decoded output."""
        # Shared ffmpeg decoder — the SAME one tokenize uses, so prompt-encode
        # tokens stay on-distribution with the trained corpus. Returns [C, N]
        # (mono source upmixed to L=R for a stereo codec).
        y = decode_pcm(path, self.codec.SAMPLE_RATE, self.n_channels)
        return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def _decode_chunk(
        self, all_new: torch.Tensor, f0: int, f1: int, ctx: int, hop: int
    ) -> np.ndarray:
        """Decode new frames [f0, f1) to mono f32 PCM, gapless against neighbours.

        DAC's decoder is convolutional, so decoding a chunk in isolation clicks at
        the seams. We decode the chunk WITH `ctx` frames of real left context
        (`all_new` is the cumulative un-delayed token buffer) and drop the
        `ctx*hop` warmup samples — the kept span was decoded with proper left
        context. The right edge needs no trim: the next chunk re-decodes the
        boundary region with full left context and only keeps its own span.
        """
        start = max(0, f0 - ctx)
        window = all_new[:, start:f1]              # [K, w] long, cpu
        wav = self.codec.decode(window)            # [samples] (mono) or [2,samples]
        drop = (f0 - start) * hop
        if wav.dim() == 1:
            return wav[drop:].contiguous().numpy().astype("float32")  # mono [kept]
        # stereo [2, samples] -> interleaved [kept*2] f32 (L0,R0,L1,R1,...) so the
        # raw-PCM ffmpeg pipe in _stream_mp3 reads it with -ac 2.
        kept = wav[:, drop:].transpose(0, 1).contiguous()  # [kept, 2]
        return kept.reshape(-1).numpy().astype("float32")

    def _stream_mp3(self, producer_fn, *, meta, on_complete, cancel=None):
        """Run ONE persistent ffmpeg (raw f32 mono PCM -> mp3), yielding mp3 bytes
        as they're produced. ``producer_fn(append_pcm, cancel)`` runs on a worker
        thread and calls ``append_pcm(np.float32 mono)`` for each PCM chunk (the GPU
        loop + decode live there; CUDA releases the GIL so this generator can read
        ffmpeg stdout concurrently — no pipe deadlock). On normal completion the
        full PCM is re-encoded with ``meta`` and handed to ``on_complete(body,
        mime)`` (the canonical save); on client disconnect (GeneratorExit) the
        producer is cancelled, ffmpeg killed, and nothing is saved. Shared by the
        generate / extend / cover stream paths."""
        sr = self.codec.SAMPLE_RATE
        n_ch = getattr(self, "n_channels", 1)  # mono unless a stereo codec is loaded
        if cancel is None:
            cancel = threading.Event()
        try:
            proc = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-f", "f32le", "-ar", str(sr), "-ac", str(n_ch), "-i", "pipe:0",
                 "-codec:a", "libmp3lame", "-b:a", "192k",
                 "-f", "mp3", "-id3v2_version", "0", "pipe:1"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg is required for streaming generation") from e

        pcm_parts: list[np.ndarray] = []
        err_box: dict[str, BaseException] = {}

        def _append(pcm: np.ndarray) -> None:
            pcm_parts.append(pcm)
            proc.stdin.write(pcm.astype("<f4").tobytes())
            proc.stdin.flush()

        def _produce():
            try:
                # Hold the single-GPU gate for the whole generation (incl. the
                # interleaved DAC decode the consumer drives between chunks), so
                # concurrent stream requests serialize instead of hanging the GPU.
                with _gpu_gen_gate():
                    producer_fn(_append, cancel)
            except BaseException as e:  # BrokenPipe on disconnect, OOM, etc.
                err_box["err"] = e
                cancel.set()
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        producer = threading.Thread(target=_produce, daemon=True)
        producer.start()

        completed = False
        try:
            while True:
                block = proc.stdout.read1(65536)
                if not block:
                    break
                yield block
            producer.join()
            if "err" in err_box:
                raise err_box["err"]
            completed = not cancel.is_set()
        finally:
            if not completed:
                cancel.set()
                try:
                    proc.kill()
                except Exception:
                    pass
            producer.join(timeout=5)
            try:
                proc.stdout.close()
            except Exception:
                pass

        if completed and pcm_parts and on_complete is not None:
            flat = np.concatenate(pcm_parts)  # interleaved f32
            if n_ch == 2:
                full = torch.from_numpy(flat.reshape(-1, 2).T.copy())  # [2, samples]
            else:
                full = torch.from_numpy(flat)[None]  # [1, samples]
            body, mime = _encode_audio(full, sr, meta)
            on_complete(body, mime)

    def _decoded_chunks(
        self, append_pcm, cancel, *, prompt, num_new_frames,
        temperature, top_k, top_p, cfg_scale,
        cond_emb=None, neg_emb=None, cond_lids=None, cond_lmask=None,
        neg_lids=None, neg_lmask=None, lyric_cfg_scale=None,
        melody=None, melody_cfg_scale=None,
        decode_ctx=16, emit_every=256, first_emit=128,
    ):
        """Stream the model's NEW frames, calling ``append_pcm`` with each
        left-context-decoded chunk. Shared decode core for all stream paths;
        ``prompt`` is an already-resolved batched [B, K, T] tensor. ``cfg_scale`` is
        passed through unguarded — ``_generate_stream`` decides whether to run CFG
        based on which conditioning (text / lyrics / melody) is actually present."""
        hop = self.codec.SAMPLE_RATE // self.codec.FRAME_RATE_HZ  # 512
        gen = self.model._generate_stream(
            prompt, num_new_frames=num_new_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=cond_emb, text_emb_neg=neg_emb,
            lyric_ids=cond_lids, lyric_mask=cond_lmask,
            lyric_ids_neg=neg_lids, lyric_mask_neg=neg_lmask,
            cfg_scale=cfg_scale, lyric_cfg_scale=lyric_cfg_scale,
            melody=melody, melody_cfg_scale=melody_cfg_scale,
            emit_every=emit_every, first_emit=first_emit,
        )
        all_new = None  # cumulative un-delayed tokens [K, frames], cpu long
        produced = 0
        try:
            for chunk in gen:
                if cancel.is_set():
                    break
                c = chunk.squeeze(0).cpu()  # [K, n]
                f0, f1 = produced, produced + c.shape[-1]
                all_new = c if all_new is None else torch.cat([all_new, c], dim=-1)
                append_pcm(self._decode_chunk(all_new, f0, f1, decode_ctx, hop))
                produced = f1
        finally:
            gen.close()

    def generate_audio_stream(
        self,
        seconds: float = 30.0,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        lyrics: str | None = None,
        negative_text: str | None = None,
        lyric_cfg_scale: float | None = None,
        gender: str | None = None,
        bpm: float | None = None,
        emit_every: int = 256,
        first_emit: int = 128,
        decode_ctx: int = 16,
        cancel: threading.Event | None = None,
        on_complete=None,
    ):
        """Streaming counterpart to ``generate_audio`` (from scratch). Token order
        is bit-identical to ``generate``; see ``_stream_mp3`` for the topology."""
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1
        seed_tokens, seed_frames = self._bootstrap()
        new_frames = int(seconds * self.codec.FRAME_RATE_HZ)
        if seed_frames + new_frames > max_total:
            new_frames = max(0, max_total - seed_frames)
        if new_frames == 0:
            raise ValueError("Requested duration exceeds model context limit.")

        cond_emb, cond_lids, cond_lmask = self._build_conditioning(text, lyrics=lyrics, gender=gender, bpm=bpm)
        neg_emb, neg_lids, neg_lmask = self._build_conditioning(negative_text)
        prompt = (seed_tokens.unsqueeze(0) if seed_tokens is not None
                  else self.model._resolve_prompt(None)[0])  # [1, K, T_seed]
        meta = self._gen_metadata(
            "generate", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p,
            cfg_scale=cfg_scale, seconds=seconds,
        )

        def producer(append_pcm, cancel):
            self._decoded_chunks(
                append_pcm, cancel, prompt=prompt, num_new_frames=new_frames,
                temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
                cond_emb=cond_emb, neg_emb=neg_emb, cond_lids=cond_lids, cond_lmask=cond_lmask,
                neg_lids=neg_lids, neg_lmask=neg_lmask, lyric_cfg_scale=lyric_cfg_scale,
                decode_ctx=decode_ctx, emit_every=emit_every, first_emit=first_emit,
            )

        yield from self._stream_mp3(producer, meta=meta, on_complete=on_complete, cancel=cancel)

    def extend_audio_stream(
        self,
        full_audio_bytes: bytes,
        add_seconds: float = 20.0,
        overlap_seconds: float = 8.0,
        from_seconds: float | None = None,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        lyrics: str | None = None,
        negative_text: str | None = None,
        lyric_cfg_scale: float | None = None,
        gender: str | None = None,
        bpm: float | None = None,
        emit_every: int = 256,
        first_emit: int = 128,
        decode_ctx: int = 16,
        cancel: threading.Event | None = None,
        on_complete=None,
    ):
        """Streaming counterpart to ``extend_audio``: yields the kept original
        verbatim FIRST (instant audio), then streams the generated continuation."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(full_audio_bytes)
            in_path = f.name
        try:
            full_wav = self._load_wav(in_path)        # [C, samples] (C matches codec)
            full_tokens = self._encode_prompt_tokens(full_wav)
        finally:
            os.unlink(in_path)

        if from_seconds is None:
            cut_frame = full_tokens.shape[1]
        else:
            cut_frame = max(1, min(
                int(from_seconds * self.codec.FRAME_RATE_HZ), full_tokens.shape[1]))
        overlap_frames = max(1, int(overlap_seconds * self.codec.FRAME_RATE_HZ))
        seed_start = max(0, cut_frame - overlap_frames)
        prompt_tokens = full_tokens[:, seed_start:cut_frame]
        if prompt_tokens.shape[1] == 0:
            raise ValueError("Selected seed window is empty; move the point later or widen overlap_seconds.")
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1
        new_frames = int(add_seconds * self.codec.FRAME_RATE_HZ)
        if prompt_tokens.shape[1] + new_frames > max_total:
            new_frames = max(0, max_total - prompt_tokens.shape[1])
        if new_frames == 0:
            raise ValueError("Overlap window is already at model context limit; reduce overlap_seconds.")
        if from_seconds is None:
            keep_samples = full_wav.shape[1]
        else:
            keep_samples = min(
                int(round(cut_frame / self.codec.FRAME_RATE_HZ * self.codec.SAMPLE_RATE)),
                full_wav.shape[1])
        prefix_pcm = full_wav[:, :keep_samples].squeeze(0).contiguous().numpy().astype("float32")

        prompt, _ = self.model._resolve_prompt(prompt_tokens.to(self.device))
        cond_emb, cond_lids, cond_lmask = self._build_conditioning(text, lyrics=lyrics, gender=gender, bpm=bpm)
        neg_emb, neg_lids, neg_lmask = self._build_conditioning(negative_text)
        meta = self._gen_metadata(
            "extend", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
            add_seconds=add_seconds, overlap_seconds=overlap_seconds, from_seconds=from_seconds,
        )

        def producer(append_pcm, cancel):
            append_pcm(prefix_pcm)  # the kept original, immediately
            self._decoded_chunks(
                append_pcm, cancel, prompt=prompt, num_new_frames=new_frames,
                temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
                cond_emb=cond_emb, neg_emb=neg_emb, cond_lids=cond_lids, cond_lmask=cond_lmask,
                neg_lids=neg_lids, neg_lmask=neg_lmask, lyric_cfg_scale=lyric_cfg_scale,
                decode_ctx=decode_ctx, emit_every=emit_every, first_emit=first_emit,
            )

        yield from self._stream_mp3(producer, meta=meta, on_complete=on_complete, cancel=cancel)

    def cover_audio_stream(
        self,
        melody_audio_bytes: bytes,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        lyrics: str | None = None,
        negative_text: str | None = None,
        melody_cfg_scale: float | None = None,
        lyric_cfg_scale: float | None = None,
        gender: str | None = None,
        bpm: float | None = None,
        emit_every: int = 256,
        first_emit: int = 128,
        decode_ctx: int = 16,
        cancel: threading.Event | None = None,
        on_complete=None,
    ):
        """Streaming counterpart to ``cover_audio``: re-render the hum's melody in
        the prompt's timbre, streaming the result as it generates."""
        if not getattr(self.model.cfg, "use_melody_conditioning", False):
            raise RuntimeError(
                "This checkpoint was trained without melody conditioning — /cover "
                "needs a model with use_melody_conditioning=True.")
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1
        melody = self._build_melody(melody_audio_bytes)  # [1, T, 12]
        seed_tokens, seed_frames = self._bootstrap()
        new_frames = min(melody.shape[1], max_total - seed_frames)
        if new_frames <= 0:
            raise ValueError("Melody audio is too short or context limit too small.")
        melody = melody[:, :new_frames, :]

        cond_emb, cond_lids, cond_lmask = self._build_conditioning(text, lyrics=lyrics, gender=gender, bpm=bpm)
        neg_emb, neg_lids, neg_lmask = self._build_conditioning(negative_text)
        prompt = (seed_tokens.unsqueeze(0) if seed_tokens is not None
                  else self.model._resolve_prompt(None)[0])  # seed runway; chroma drives contour
        meta = self._gen_metadata(
            "cover", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
            melody_cfg_scale=melody_cfg_scale, seconds=new_frames / self.codec.FRAME_RATE_HZ,
        )

        def producer(append_pcm, cancel):
            self._decoded_chunks(
                append_pcm, cancel, prompt=prompt, num_new_frames=new_frames,
                temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
                cond_emb=cond_emb, neg_emb=neg_emb, cond_lids=cond_lids, cond_lmask=cond_lmask,
                neg_lids=neg_lids, neg_lmask=neg_lmask, lyric_cfg_scale=lyric_cfg_scale,
                melody=melody, melody_cfg_scale=melody_cfg_scale,
                decode_ctx=decode_ctx, emit_every=emit_every, first_emit=first_emit,
            )

        yield from self._stream_mp3(producer, meta=meta, on_complete=on_complete, cancel=cancel)

    def _ensure_demucs(self):
        """Lazy-load Demucs (htdemucs) for stem separation; cached after first use.

        Reuses the same loader the transcribe pipeline uses, so the server and
        data-prep agree on the model. Demucs is a data-prep dependency and may be
        absent in a slim serving image — surface that as a clear error rather than
        a bare ImportError deep in the request."""
        if self._demucs is None:
            try:
                from diskrot.transcribe_lyrics import _load_demucs
                self._demucs = _load_demucs(self.device)
            except ImportError as e:
                raise RuntimeError(
                    "stem separation needs the `demucs` package, which isn't "
                    "installed in this environment (pip install demucs)."
                ) from e
        return self._demucs

    @_gated
    @torch.no_grad()
    def separate_stems(self, audio_bytes: bytes, keep: list[str]) -> tuple[bytes, str]:
        """Separate the upload with Demucs and remix only the ``keep`` stems.

        Pure source separation — the nano model is NOT involved, so this works
        with any checkpoint. Demucs (``htdemucs``) splits the audio into
        drums / bass / other / vocals; the stems named in ``keep`` are summed back
        into a single mixdown and re-encoded. ``keep=[drums,bass,other]`` yields an
        instrumental, ``keep=[vocals]`` an a-cappella. Output is mono (the rest of
        the server is mono), 44.1 kHz, mp3.
        """
        model, apply_fn = self._ensure_demucs()
        names = list(model.sources)  # ['drums', 'bass', 'other', 'vocals']
        keep_idx = [i for i, n in enumerate(names) if n in keep]
        if not keep_idx:
            raise ValueError(f"none of {keep} are Demucs stems ({names})")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name
        try:
            # Demucs is a stereo model — decode stereo (mono upmixed to L=R by
            # the shared ffmpeg decoder, matching the old librosa mono->stereo).
            audio = decode_pcm(in_path, self.codec.SAMPLE_RATE, 2)  # [2, T] f32
        finally:
            os.unlink(in_path)
        wav = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # [1, 2, T]

        sources = apply_fn(model, wav, device=self.device)  # [1, S, 2, T]
        mixed = sources[0, keep_idx].sum(dim=0)  # [2, T] — sum kept stems
        mono = mixed.mean(dim=0).cpu().unsqueeze(0)  # [1, T]

        meta = self._gen_metadata(
            "stem",
            kept=",".join(n for n in names if n in keep),
            removed=",".join(n for n in names if n not in keep),
        )
        return _encode_audio(mono, self.codec.SAMPLE_RATE, meta)

    @_gated
    @torch.no_grad()
    def add_stem(
        self,
        audio_bytes: bytes,
        target_stem: str,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        negative_text: str | None = None,
        stem_cfg_scale: float | None = None,
        output: str = "mix",
        lyrics: str | None = None,
        lyric_cfg_scale: float | None = None,
    ) -> tuple[bytes, str]:
        """Generate a NEW isolated stem that fits an existing song (the /addstem path).

        The generative inverse of ``/stem``'s removal: the upload is Demucs-separated,
        the model conditions on the song's OTHER stems (everything except
        ``target_stem``) + ``text`` (the desired stem's vibe, e.g. "funky 70s warbly
        bassline") and generates ``target_stem`` from scratch. ``output="mix"`` (the
        default) returns the song with the new stem summed in; ``output="stem"``
        returns the isolated generated stem alone. Needs a stem-trained checkpoint
        (``use_stem_conditioning``) and the ``demucs`` package. Returns (bytes, mime).

        ``lyrics`` applies ONLY when ``target_stem='vocals'`` — the model then sings
        those words (phoneme + marker stream, like /generate) over the song's other
        stems. Ignored for drums/bass/other (instrumental stems have no words).
        """
        if not getattr(self.model.cfg, "use_stem_conditioning", False):
            raise RuntimeError(
                "This checkpoint was trained without stem conditioning — /addstem "
                "needs a model with use_stem_conditioning=True."
            )
        from diskrot.stems import extract_stem_tokens
        from model.stem_encoder import STEM_TYPES, STEM_TYPE_TO_ID

        target = (target_stem or "").lower().strip()
        if target not in STEM_TYPE_TO_ID:
            raise ValueError(
                f"unknown target_stem {target_stem!r}; valid: {', '.join(STEM_TYPES)}")
        target_id = STEM_TYPE_TO_ID[target]

        demucs = self._ensure_demucs()
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1

        # Separate + tokenize every stem of the upload via the SHARED extractor —
        # byte-identical to the training stem cache (diskrot.stems).
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            in_path = f.name
        try:
            tokens = extract_stem_tokens(in_path, self.codec, demucs, device=self.device)
            orig_wav = self._load_wav(in_path) if output == "mix" else None
        finally:
            os.unlink(in_path)

        cond_names = [n for n in STEM_TYPES if n != target]
        new_frames = min(int(tokens[target].shape[1]), max_total - 1)
        if new_frames <= 0:
            raise ValueError("Audio is too short or the context limit too small.")
        dev = self.device
        # Condition on the OTHER stems (target excluded), sliced to the model's K.
        cond_stack = torch.stack(
            [tokens[n][:K, :new_frames] for n in cond_names], dim=0
        )[None].to(dev)  # [1, S, K, new_frames] long
        cond_types = torch.tensor(
            [STEM_TYPE_TO_ID[n] for n in cond_names], device=dev, dtype=torch.long)[None]
        cond_present = torch.ones(1, len(cond_names), device=dev)
        target_type = torch.tensor([target_id], device=dev, dtype=torch.long)

        # Lyrics only apply to a vocals target (the only stem that sings words). For
        # the other stems the lyric stream stays off, matching how the model trained.
        use_lyrics = (target == "vocals" and bool((lyrics or "").strip())
                      and getattr(self.model.cfg, "use_lyric_conditioning", False))
        cond_emb, cond_lids, cond_lmask = self._build_conditioning(
            text, lyrics=lyrics if use_lyrics else None)
        neg_emb, neg_lids, neg_lmask = self._build_conditioning(negative_text)
        out_tokens = self.model.generate(
            prompt=None, num_new_frames=new_frames,
            temperature=temperature, top_k=top_k, top_p=top_p,
            text_emb=cond_emb, text_emb_neg=neg_emb, cfg_scale=cfg_scale,
            lyric_ids=cond_lids, lyric_mask=cond_lmask,
            lyric_ids_neg=neg_lids, lyric_mask_neg=neg_lmask,
            lyric_cfg_scale=lyric_cfg_scale if use_lyrics else None,
            stem_tokens=cond_stack, stem_types=cond_types, stem_present=cond_present,
            target_stem_type=target_type, stem_cfg_scale=stem_cfg_scale,
        )  # [K, 1 + new_frames]
        out_tokens = out_tokens[:, 1:]  # strip the seed frame
        stem_wav = self.codec.decode(out_tokens.cpu())  # [C, samples] or [samples]
        if stem_wav.dim() == 1:
            stem_wav = stem_wav.unsqueeze(0)

        if output == "stem":
            result = stem_wav
        else:
            ow = orig_wav.unsqueeze(0) if orig_wav.dim() == 1 else orig_wav
            # Match channel count, then sum the new stem onto the song (truncate to
            # the shorter side) and normalize if the sum clips.
            if ow.shape[0] != stem_wav.shape[0]:
                if stem_wav.shape[0] == 1:
                    stem_wav = stem_wav.repeat(ow.shape[0], 1)
                elif ow.shape[0] == 1:
                    ow = ow.repeat(stem_wav.shape[0], 1)
            n = min(ow.shape[1], stem_wav.shape[1])
            result = ow[:, :n] + stem_wav[:, :n]
            peak = result.abs().max()
            if peak > 1.0:
                result = result / peak

        meta = self._gen_metadata(
            "addstem", text=text, negative_text=negative_text,
            temperature=temperature, top_k=top_k, top_p=top_p, cfg_scale=cfg_scale,
            target_stem=target, output=output, stem_cfg_scale=stem_cfg_scale,
            lyrics=(lyrics if use_lyrics else None),
            seconds=new_frames / self.codec.FRAME_RATE_HZ,
        )
        return _encode_audio(result, self.codec.SAMPLE_RATE, meta)


def _crossfade_concat(
    segments: list[torch.Tensor], sr: int, fade_seconds: float = 0.03,
) -> torch.Tensor:
    """Concatenate mono [1, samples] segments with an equal-power crossfade at
    each seam. Used by /infill to splice ``before | middle | after`` without the
    clicks a hard cut would leave at the two boundaries.

    The fade is clamped to half the shorter side of each seam, so very short
    segments still join cleanly (degrading to a near-hard cut)."""
    out = segments[0]
    for nxt in segments[1:]:
        n = min(int(fade_seconds * sr), out.shape[1], nxt.shape[1])
        if n <= 0:
            out = torch.cat([out, nxt], dim=1)
            continue
        t = torch.linspace(0, 1, n, dtype=out.dtype, device=out.device)
        fade_out, fade_in = torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2)
        seam = out[:, -n:] * fade_out + nxt[:, :n] * fade_in
        out = torch.cat([out[:, :-n], seam, nxt[:, n:]], dim=1)
    return out


def _encode_audio(
    wav: torch.Tensor, sr: int, metadata: dict[str, str] | None = None
) -> tuple[bytes, str]:
    """Encode mono [1,samples] OR stereo [2,samples] float audio to mp3 (via
    subprocess ffmpeg) or fall back to wav. ffmpeg/libmp3lame preserve the channel
    count from the written WAV, so no -ac is needed.

    Any `metadata` dict is written into the mp3 as ID3v2 TXXX (user-defined) frames
    — keys are namespaced `nano_*`, none of which collide with standard frames, so
    ffmpeg's id3v2 muxer emits each as a TXXX frame. The wav fallback carries no
    metadata (only reached when ffmpeg is unavailable)."""
    arr = wav.detach().contiguous().cpu().numpy()
    # soundfile wants channels-LAST: mono -> [samples]; stereo [2,samples] -> [samples,2].
    if arr.ndim == 2:
        audio = arr[0] if arr.shape[0] == 1 else arr.T
    else:
        audio = arr

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sr, format="WAV", subtype="PCM_16")

    meta_args: list[str] = []
    for k, v in (metadata or {}).items():
        meta_args += ["-metadata", f"{k}={v}"]

    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0",
             "-codec:a", "libmp3lame", "-b:a", "192k",
             *meta_args, "-id3v2_version", "3", "-f", "mp3", "pipe:1"],
            input=wav_buf.getvalue(), capture_output=True, check=True,
        )
        return proc.stdout, "audio/mpeg"
    except (FileNotFoundError, subprocess.CalledProcessError):
        return wav_buf.getvalue(), "audio/wav"
