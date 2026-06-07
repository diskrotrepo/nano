"""Inference: mp3 in → mp3 (or wav) out via the trained nano audio GPT."""
from __future__ import annotations

import io
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import librosa
import soundfile as sf
import torch

from model.codec import DACodec
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
    from torchao.quantization import (
        int4_weight_only,
        int8_weight_only,
        quantize_,
    )

    def filter_fn(module: torch.nn.Module, fqn: str) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        if fqn.startswith("lyric_encoder"):
            return False
        if bits == 4 and module.in_features % group_size != 0:
            return False
        return True

    config = int4_weight_only(group_size=group_size) if bits == 4 else int8_weight_only()
    quantize_(model, config, filter_fn=filter_fn)


class InferenceEngine:
    def __init__(self, ckpt_path: str | None = None, device: str | None = None):
        self.device = device or os.environ.get("NANO_DEVICE") or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        self.codec = DACodec(device=self.device)
        self.text_encoder: CLAPTextEncoder | None = None
        self.sweetener: PromptSweetener | None = None  # lazy — only built on first use

        ckpt_path = ckpt_path or os.environ.get("NANO_CKPT", "./checkpoints/latest.pt")
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = GPTConfig(**ckpt["cfg"])
            state = ckpt["model"]
            # strip torch.compile's _orig_mod. prefix if present
            if any(k.startswith("_orig_mod.") for k in state):
                state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}

            # Backend: MLX on Apple Silicon (unless NANO_MLX=0), else PyTorch.
            self.backend = (
                "mlx"
                if os.environ.get("NANO_MLX", "auto") != "0"
                and self.device == "mps"
                and _mlx_available()
                else "torch"
            )
            if self.backend == "mlx":
                import mlx.core as mx

                from model.nano_audio_gpt_mlx import MLXNanoAudioGPT

                bits = int(os.environ.get("NANO_MLX_BITS", "8"))
                self.model = MLXNanoAudioGPT(
                    cfg, state, dtype=mx.float16, bits=bits if bits in (4, 8) else None
                )
                print(f"[inference] backend: mlx (Apple Silicon), weights="
                      f"{'fp16' if self.model._bits == 16 else f'int{self.model._bits}'}")
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
                self._torch_dtype = (
                    torch.bfloat16 if (quantized and bits == 4) else torch.float16
                )
                if self.device != "cpu":
                    self.model = self.model.to(self._torch_dtype)
                if quantized:
                    _quantize_torch_linears(self.model, bits)
                self.model._bits = bits if quantized else 16
                if self.device == "cuda":
                    self.model = torch.compile(self.model)
                wdesc = "fp16" if self._torch_dtype == torch.float16 else "bf16"
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
        style_audio_bytes: bytes | None = None,
        style_weight: float = 0.5,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Build conditioning from text (tags + lyrics), style audio, or both.

        text may contain tags and lyrics separated by ". " (combined by the
        server). Tags go through the pooled CLAP encoder; lyrics are phonemized
        (g2p) into a token sequence for the LyricEncoder cross-attention — they no
        longer go through CLAP. Style audio (when given) blends into the tag
        embedding. Returns ``(tag_emb [1,1,D] | None, lyric_ids [1,L] | None,
        lyric_mask [1,L] | None)``.
        """
        # Split tags from lyrics (server joins them as "tags. lyrics")
        tags_str = ""
        lyrics_str = ""
        if text and text.strip():
            parts = text.split(". ", 1)
            tags_str = parts[0]
            lyrics_str = parts[1] if len(parts) > 1 else ""

        # --- Tags (pooled CLAP, position 0) + optional style-audio blend ---
        tag_emb = None
        if self.text_encoder is not None:
            if tags_str:
                tag_emb = self.text_encoder.encode([tags_str]).to(self.device)  # [1,1,D]
            if style_audio_bytes:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    f.write(style_audio_bytes)
                    style_path = f.name
                try:
                    audio_emb = self.text_encoder.encode_audio([style_path]).to(self.device)
                finally:
                    os.unlink(style_path)
                tag_emb = audio_emb if tag_emb is None else (
                    tag_emb * (1 - style_weight) + audio_emb * style_weight
                )
            if tag_emb is not None:
                tag_emb = tag_emb.to(self._cond_dtype())

        # --- Lyrics (phoneme + structure-marker sequence for the LyricEncoder) ---
        # Parses inline section tags like "[verse] ... [chorus] ..." into section
        # markers, reproducing the dataset's train-time stream exactly (prefix +
        # inline markers). Brackets never reach g2p; unknown labels fold to
        # <no_section>. Plain lyrics (no brackets) get a <no_section> prefix.
        lyric_ids = lyric_mask = None
        if lyrics_str and getattr(self.model.cfg, "use_lyric_conditioning", False):
            from model.lyric_encoder import (
                PAD_PHONEME_ID, text_with_markers_to_phoneme_ids,
            )

            ids = text_with_markers_to_phoneme_ids(
                lyrics_str, max_len=self.model.cfg.max_lyric_len
            )
            if ids:
                lyric_ids = torch.tensor(ids, dtype=torch.long, device=self.device)[None]
                lyric_mask = lyric_ids != PAD_PHONEME_ID

        return tag_emb, lyric_ids, lyric_mask

    def _cond_dtype(self) -> torch.dtype:
        """dtype for the conditioning tensor, to match the model's compute dtype.
        fp32 only for the torch-on-CPU path; the torch accelerated paths follow
        `self._torch_dtype` (fp16, or bf16 when running int4-quantized); MLX is
        always fp16."""
        if self.backend == "torch":
            return torch.float32 if self.device == "cpu" else self._torch_dtype
        return torch.float16

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
        negative_text: str | None = None,
        style_audio_bytes: bytes | None = None,
        style_weight: float = 0.5,
        lyric_cfg_scale: float | None = None,
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
            y, _ = librosa.load(in_path, sr=self.codec.SAMPLE_RATE, mono=True)
            full_wav = torch.from_numpy(y).unsqueeze(0)
            full_tokens = self.codec.encode(full_wav)
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
        cond_emb, cond_lids, cond_lmask = self._build_conditioning(text, style_audio_bytes, style_weight)
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


    @torch.no_grad()
    def generate_audio(
        self,
        seconds: float = 30.0,
        temperature: float | list[float] = 0.9,
        top_k: int | None | list[int | None] = 50,
        top_p: float | None | list[float | None] = 0.95,
        cfg_scale: float = 3.0,
        text: str | None = None,
        negative_text: str | None = None,
        style_audio_bytes: bytes | None = None,
        style_weight: float = 0.5,
        lyric_cfg_scale: float | None = None,
        score_clap: bool = False,
    ):
        """Generate audio from scratch (no audio prompt). Returns (audio_bytes, mime_type).

        When ``score_clap`` is True (and text conditioning is present), also
        computes the CLAP text<->audio similarity of the generated clip against
        ``text`` and returns it as a third element: (audio_bytes, mime, clap).
        Default stays a 2-tuple so existing callers are unaffected.

        The autoregressive loop is bootstrapped from a single column of random
        DAC tokens (model.generate picks a fresh seed per call). With the current
        tight sampling + CFG this produces coherent output the prompt can steer in
        any direction. (A silence-seed mode existed once but only worked for quiet
        prompts and collapsed high-energy ones to silence, so it was removed.)
        """
        K = self.model.cfg.n_codebooks
        max_total = self.model.cfg.max_seq_len - K + 1  # T_total such that T_total + K - 1 <= max_seq_len

        # Pass None so model.generate() picks a fresh random seed per call.
        seed_tokens = None
        seed_frames = 1

        new_frames = int(seconds * self.codec.FRAME_RATE_HZ)
        if seed_frames + new_frames > max_total:
            new_frames = max(0, max_total - seed_frames)
        if new_frames == 0:
            raise ValueError("Requested duration exceeds model context limit.")

        cond_emb, cond_lids, cond_lmask = self._build_conditioning(text, style_audio_bytes, style_weight)
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
                sf.write(tmp, wav.squeeze().contiguous().cpu().numpy(), self.codec.SAMPLE_RATE)
            try:
                clap = self.text_encoder.audio_text_similarity(tmp, text)
            finally:
                os.unlink(tmp)
        return body, mime, clap


def _encode_audio(
    wav: torch.Tensor, sr: int, metadata: dict[str, str] | None = None
) -> tuple[bytes, str]:
    """Encode [1, samples] mono float audio to mp3 (via subprocess ffmpeg) or fall back to wav.

    Any `metadata` dict is written into the mp3 as ID3v2 TXXX (user-defined) frames
    — keys are namespaced `nano_*`, none of which collide with standard frames, so
    ffmpeg's id3v2 muxer emits each as a TXXX frame. The wav fallback carries no
    metadata (only reached when ffmpeg is unavailable)."""
    audio = wav.squeeze().contiguous().cpu().numpy()  # [samples]

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
