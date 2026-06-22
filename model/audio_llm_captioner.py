"""Audio-LLM music captioner — rich, long, structured descriptions.

The legacy LP-MusicCaps captioner ([model/captioner.py]) reads a single 10-second
window and emits one ~40-word sentence. That is far too little for the chunked-CLAP
tag path, which can condition the decoder on a whole multi-facet description split
into <=77-token chunks (see model/text_encoder.py:encode_chunked). This module
captions the WHOLE song with an instruction-tuned audio-language model
(Qwen2-Audio by default) and prompts it for an evocative, information-dense
description — one theme per span (genre/mood, drums, bass, instruments, vocals,
production, arc) — so each CLAP chunk pools a distinct facet.

Same ``load_captioner()`` entry-point name as the BART module. Two backends share
the same prompt, windows and greedy decoding (so captions are equivalent), picked
by ``NANO_CAPTIONER_BACKEND``: ``vllm`` (default) runs MANY songs per GPU pass via
vLLM continuous batching (``caption_many``) to saturate the GPU; ``hf`` is the
one-conversation-at-a-time transformers path (the parity oracle and CPU-capable
fallback). The captioner *family* is still chosen by ``NANO_CAPTIONER``
(``audio_llm`` default, ``bart`` legacy).

The model is heavy (~7B) and can't be exercised on CPU in CI; this module is
correct-by-construction against the Qwen2-Audio HF/vLLM APIs and is meant to be
calibrated with a small ``--limit`` Modal run before a full pass.
"""
from __future__ import annotations

import os

import numpy as np

# Qwen2-Audio's Whisper-style audio encoder caps each clip at ~30 s, so a long
# song is covered by a few evenly-spaced windows passed as separate audios in one
# conversation. 16 kHz mono is the model's expected input.
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 30
MAX_WINDOWS = 4           # up to ~120 s of audio spread across the track
MAX_NEW_TOKENS = 512      # ~380 words; trimmed to MAX_CHARS below
MAX_CHARS = 3000          # the chunked-CLAP tag path's design target

DEFAULT_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

# Stamped into each tags.json entry (``{"description": ..., "captioner": MARKER}``)
# so a re-caption pass can skip entries already at this format and only redo
# legacy/short ones. Bump the suffix if the caption format changes (new prompt,
# new model) and you want a --redo to redo everything. NOTE: diskrot/modal_auto_tag.py
# duplicates this literal (its slim orchestrator image can't import this module) —
# keep the two in sync.
CAPTIONER_MARKER = "audio_llm_v1"

_CAPTION_INSTRUCTION = (
    "You are an expert music annotator. The audio contains one or more excerpts "
    "from the SAME track, in order. Write ONE rich, information-dense description "
    "of the music as flowing prose. Cover, each in its own short sentence where "
    "audible: the overall genre and mood; the drums and percussion; the bass; the "
    "harmony and lead instruments; the vocals (or say it is instrumental); the "
    "production and mix character; and how the track evolves across its sections. "
    "Be concrete and evocative — name instruments, textures, micro-genres and "
    "production artifacts. Do NOT transcribe or invent lyrics. No markdown, no "
    "lists, no preamble — description only. Aim for 150-350 words."
)


def load_song_windows(
    path: str,
    sr: int = SAMPLE_RATE,
    window_seconds: int = WINDOW_SECONDS,
    max_windows: int = MAX_WINDOWS,
) -> list[np.ndarray]:
    """Load *path* (16 kHz mono) and return up to ``max_windows`` evenly-spaced
    ``window_seconds`` excerpts covering the whole track.

    A short song yields a single (zero-padded) window; a long song is sampled at
    evenly-spaced offsets (skipping the very intro/outro) so the caption reflects
    the whole arc, not one 10 s slice. The single train==inference contract for
    "what audio the captioner sees" — mirror it if a non-Modal path captions.
    """
    import librosa

    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)
    n = sr * window_seconds
    total = audio.shape[-1]
    if total <= n:
        pad = np.zeros(n, dtype=np.float32)
        pad[:total] = audio
        return [pad]
    # How many windows actually fit (cap at max_windows), biased off the edges.
    k = max(1, min(max_windows, total // n))
    if k == 1:
        offset = min(int(total * 0.25), total - n)
        return [audio[offset:offset + n]]
    # Evenly-spaced starts across the usable span [5%, 95%-window].
    lo = int(total * 0.05)
    hi = max(lo, total - n - int(total * 0.05))
    starts = np.linspace(lo, hi, k).astype(int)
    return [audio[s:s + n] for s in starts]


def _clean(text: str) -> str:
    """Collapse whitespace, drop wrapping quotes, trim to MAX_CHARS at a word
    boundary. (Tags travel as their own field now, so internal ". " is safe.)
    Backend-agnostic — shared by the HF and vLLM captioners."""
    text = " ".join((text or "").split())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0]
    return text.strip()


def _build_conversation(windows: list[np.ndarray]) -> list[dict]:
    """The chat conversation shared by both backends so the templated prompt is
    byte-identical: the song's audio windows (in order) then the caption
    instruction. apply_chat_template only counts the audio entries to emit
    placeholders; the array values are ignored when tokenize=False."""
    return [{
        "role": "user",
        "content": (
            [{"type": "audio", "audio": w} for w in windows]
            + [{"type": "text", "text": _CAPTION_INSTRUCTION}]
        ),
    }]


class AudioLLMCaptioner:
    """Lazy-loaded audio-LLM captioner. Mirrors the lazy-load pattern of
    CLAPTextEncoder / PromptSweetener."""

    def __init__(self, device: str = "cpu", model_name: str | None = None):
        self._device = device
        self._model_name = model_name or os.environ.get(
            "NANO_CAPTION_MODEL", DEFAULT_MODEL)
        self._processor = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self._model_name, torch_dtype=dtype,
        ).to(self._device).eval()
        print(f"[audio-llm-captioner] loaded {self._model_name} on {self._device}")

    def _clean(self, text: str) -> str:
        return _clean(text)

    def caption(self, windows: list[np.ndarray], sr: int = SAMPLE_RATE) -> str:
        """Caption ONE song from its (already-extracted) audio windows."""
        import torch

        self._ensure_model()
        text = self._processor.apply_chat_template(
            _build_conversation(windows), add_generation_prompt=True,
            tokenize=False)
        inputs = self._processor(
            text=text, audios=[w for w in windows], sampling_rate=sr,
            return_tensors="pt", padding=True,
        ).to(self._device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, num_beams=1,
            )
        new_ids = out_ids[:, inputs.input_ids.size(1):]
        caption = self._processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]
        return self._clean(caption)

    def caption_path(self, path: str) -> str:
        """Convenience: window a file off disk, then caption it."""
        return self.caption(load_song_windows(path))


class VLLMAudioCaptioner:
    """vLLM-backed captioner — same model, prompt, windows and greedy decoding as
    AudioLLMCaptioner, but vLLM's continuous batching runs MANY songs per GPU pass
    (``caption_many``) instead of one-at-a-time, saturating the GPU. The captions
    are equivalent to the HF path modulo attention-kernel drift (see the --limit
    parity check / tests). Selected by ``NANO_CAPTIONER_BACKEND=vllm`` (the
    default); ``=hf`` falls back to AudioLLMCaptioner. GPU-only (vLLM has no usable
    CPU path) — for a local CPU/MPS dry-run set the backend to ``hf``."""

    def __init__(self, device: str = "cuda", model_name: str | None = None):
        self._device = device  # vLLM auto-selects the GPU; kept for surface parity
        self._model_name = model_name or os.environ.get(
            "NANO_CAPTION_MODEL", DEFAULT_MODEL)
        self._processor = None
        self._llm = None
        self._sampling = None

    def _ensure_model(self) -> None:
        if self._llm is not None:
            return
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self._processor = AutoProcessor.from_pretrained(self._model_name)
        # max_model_len holds MAX_WINDOWS*~750 audio tokens + prompt + MAX_NEW_TOKENS
        # (4096 fits the 4-window/512-token default). gpu_memory_utilization governs
        # the KV cache — lower it / max_num_seqs / max_model_len if init OOMs.
        self._llm = LLM(
            model=self._model_name,
            dtype="float16",
            max_model_len=int(os.environ.get("NANO_VLLM_MAX_MODEL_LEN", "4096")),
            max_num_seqs=int(os.environ.get("NANO_VLLM_MAX_NUM_SEQS", "32")),
            limit_mm_per_prompt={"audio": MAX_WINDOWS},
            gpu_memory_utilization=float(
                os.environ.get("NANO_VLLM_GPU_MEM_UTIL", "0.90")),
        )
        # temperature=0 == greedy, matching the HF path's do_sample=False/num_beams=1.
        self._sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        print(f"[vllm-captioner] loaded {self._model_name}")

    def caption_many(
        self, windows_per_song: list[list[np.ndarray]], sr: int = SAMPLE_RATE
    ) -> list[str]:
        """Caption MANY songs in one vLLM batch (continuous batching). Takes one
        window-list per song; returns one cleaned caption per song, in order
        (vLLM preserves request order)."""
        self._ensure_model()
        requests = []
        for windows in windows_per_song:
            prompt = self._processor.apply_chat_template(
                _build_conversation(windows), add_generation_prompt=True,
                tokenize=False)
            requests.append({
                "prompt": prompt,
                # vLLM Qwen2-Audio wants raw (waveform, sr) tuples, not tensors.
                "multi_modal_data": {"audio": [(w, sr) for w in windows]},
            })
        outputs = self._llm.generate(requests, self._sampling)
        return [_clean(o.outputs[0].text) for o in outputs]

    def caption(self, windows: list[np.ndarray], sr: int = SAMPLE_RATE) -> str:
        """Caption ONE song (delegates to the batched path) so the single-song
        callers (diskrot/auto_tag.py, caption_path) work unchanged on either
        backend."""
        return self.caption_many([windows], sr=sr)[0]

    def caption_path(self, path: str) -> str:
        """Convenience: window a file off disk, then caption it."""
        return self.caption(load_song_windows(path))


def load_captioner(
    device: str = "cpu", model_name: str | None = None
) -> "AudioLLMCaptioner | VLLMAudioCaptioner":
    """Load the audio-LLM captioner (downloads weights on first call).

    ``NANO_CAPTIONER_BACKEND`` selects the engine: ``vllm`` (default) →
    VLLMAudioCaptioner (continuous batching, GPU-only, adds ``caption_many``);
    ``hf`` → AudioLLMCaptioner (one-song-at-a-time transformers, the parity oracle
    / CPU-capable fallback). Both expose ``.caption(windows)``."""
    backend = os.environ.get("NANO_CAPTIONER_BACKEND", "vllm").strip().lower()
    if backend == "hf":
        return AudioLLMCaptioner(device=device, model_name=model_name)
    return VLLMAudioCaptioner(device=device, model_name=model_name)
