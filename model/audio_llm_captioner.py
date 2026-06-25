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
import re

import numpy as np

# Qwen2-Audio's Whisper-style audio encoder caps each clip at ~30 s, so a long
# song is covered by a few evenly-spaced windows passed as separate audios in one
# conversation. 16 kHz mono is the model's expected input.
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 30
MAX_WINDOWS = 4           # up to ~120 s of audio spread across the track
# Headroom for the description (~350 words) + the GENDER line + the four per-stem
# lines (one sentence each). Was 512 (description+gender only); the stem block adds
# ~120 tokens, so bump it or the last STEM lines get truncated (→ empty → the train
# loop falls back to the song caption for that stem, but we want all four).
MAX_NEW_TOKENS = 700
MAX_CHARS = 3000          # the chunked-CLAP tag path's design target (description)
MAX_STEM_CHARS = 400      # per-stem caption cap (one sentence; the /addstem tag)

DEFAULT_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

# Stamped into each tags.json entry (``{"description": ..., "captioner": MARKER}``)
# so a re-caption pass can skip entries already at this format and only redo
# legacy/short ones. Bump the suffix if the caption format changes (new prompt,
# new model) and you want a --redo to redo everything. NOTE: diskrot/modal_auto_tag.py
# duplicates this literal (its slim orchestrator image can't import this module) —
# keep the two in sync.
# v4: the caption now carries a trailing vocal GENDER tag, parsed into a separate
# tags.json ``gender`` field — the audio-LLM replaces the F0-on-Demucs gender
# estimate, so transcribe no longer runs Demucs. A --redo upgrades v3 -> v4 to
# populate gender (old v3 entries fall back to the lyrics-entry gender meanwhile).
# v5: after the GENDER line the caption now emits four per-stem lines
# (DRUMS/BASS/VOCALS/OTHER), parsed into a ``stems`` dict in tags.json — the
# /addstem target-stem tag source (so "add a bassline" is steered by the song's
# own bass description). A --redo upgrades v4 -> v5; until a song is re-captioned,
# stem-add training falls back to its full-song description.
CAPTIONER_MARKER = "audio_llm_v5"

# The four per-stem caption keys, in model.stem_encoder.STEM_TYPES order (kept as a
# literal so this module needn't import torch via stem_encoder; tests guard parity).
STEM_CAPTION_KEYS = ("drums", "bass", "vocals", "other")

_CAPTION_INSTRUCTION = """You are an expert music annotator. The audio contains one or more excerpts from the same track, presented in order. Listen to all of it before writing.
Write ONE rich, information-dense description of the music as flowing prose. In separate short sentences, cover each of the following where audible:

Overall genre (or micro-genre) and mood
Drums and percussion
Bass
Harmony and lead instruments
Vocals
Production and mix character
How the track evolves across its sections

HARD RULES — follow exactly:

Commit to specifics. State what you hear as fact. Never qualify with "seems," "possibly," "likely," "perhaps," "I think," "could be," or "might be." If you are unsure of a label, describe the sound directly instead of hedging.
Vocals: if you hear singing, describe its character — voice type, range, and delivery (belted, crooned, rapped, screamed, harmonized, spoken-word). If there is no singing, write exactly one short sentence stating the track is instrumental, then move on. BANNED phrases: "vocals are sparse," "occasional vocals," "minimal vocals," "some vocals," "there may be vocals," and any similar filler.
Be concrete and evocative: name specific instruments, textures, micro-genres, and production artifacts. Describe only what you actually hear.
If the genre is unclear or unfamiliar, describe instrumentation, rhythm, texture, and mood directly. Never default to a generic label like "electronic music" or "a song."
Do NOT transcribe, quote, or invent lyrics.
Do NOT comment on recording quality unless it is an intentional production choice (e.g., lo-fi tape hiss, bitcrushing).

FORMAT — follow exactly:

Output the description, then the single GENDER tag, then the four STEM lines specified below, and nothing else. No preamble ("Here is," "Sure," "This track"), no title, no closing remark, no meta-commentary.
Plain prose only for the description. No markdown, no bullet points, no headings, no numbered lists, no bold.
Do not name the dimensions you are covering (do not write "The vocals:" or "Genre:"). Weave them into continuous prose.
Length: 150–350 words. One paragraph.

After the paragraph — and only after it — output the lead vocal's gender on its own line, exactly one of:
GENDER: male
GENDER: female
GENDER: instrumental
Judge the MOST PROMINENT sung voice; if voices of both genders trade off, pick the lead. Use "instrumental" only when there is no singing at all.

After the GENDER line — and only after it — output exactly FOUR more lines, one per instrument family, in THIS order and format (uppercase label, a colon, then ONE vivid concrete sentence describing only that family's sound — its instrument(s), tone, and movement — written as a standalone production note a musician could follow to recreate it):
DRUMS: <the drums and percussion>
BASS: <the bass line and low end>
VOCALS: <the lead and backing vocals — voice type and delivery; write exactly the single word none if there is no singing>
OTHER: <the harmony and lead melodic instruments — everything that is not drums, bass, or vocals>
Each of these four lines is one sentence under the same no-hedging, no-invented-lyrics rules as the paragraph. These four lines plus the GENDER line are the sole exceptions to "nothing else".

Begin the description now with a concrete observation about the sound."""


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


# The captioner is prompted to end with a "GENDER: male|female|instrumental" line.
# Match it (colon OR dash, case/space-insensitive) and split it off the prose. The
# colon/dash requirement keeps incidental prose like "gender-bending vocals" from
# matching.
_GENDER_TAG_RE = re.compile(
    r"\bGENDER\s*[:\-]\s*(male|female|man|woman|boy|girl|instrumental|none|unknown|m|f)\b",
    re.IGNORECASE,
)
_MALE_LABELS = frozenset({"male", "man", "boy", "m"})
_FEMALE_LABELS = frozenset({"female", "woman", "girl", "f"})


def parse_gender(text: str) -> tuple[str, str | None]:
    """Split the trailing ``GENDER: <label>`` tag off the caption.

    Returns ``(description_without_the_tag, canonical_gender)`` where
    ``canonical_gender`` is ``"male"``/``"female"`` or ``None`` (instrumental /
    unknown / absent / unrecognized -> None, so the dataset sees
    ``<unknown_gender>``). This is the audio-LLM vocal-gender source that replaces
    the F0-on-Demucs estimate; ``gender_label_to_id`` in model/lyric_encoder.py
    maps the string to the marker id at train/inference."""
    if not text:
        return "", None
    m = _GENDER_TAG_RE.search(text)
    if not m:
        return text, None
    desc = text[:m.start()].strip()
    label = m.group(1).lower()
    if label in _MALE_LABELS:
        return desc, "male"
    if label in _FEMALE_LABELS:
        return desc, "female"
    return desc, None  # instrumental / none / unknown


# The four per-stem lines the captioner emits after GENDER. Tolerant of a leading
# bullet/space and colon-OR-dash; one line each. The captured sentence is the
# /addstem target-stem tag.
_STEM_LINE_RE = re.compile(
    r"(?im)^[\s>*\-]*(DRUMS|BASS|VOCALS|OTHER)\s*[:\-]\s*(.+?)\s*$"
)
# A stem line whose value is one of these (the "no such stem" sentinel) is dropped,
# so the train loop falls back to the song description for that stem.
_STEM_EMPTY_VALUES = frozenset({
    "none", "n/a", "na", "nan", "instrumental", "absent", "silent", "silence", "-",
})


def parse_stems(text: str) -> tuple[str, dict[str, str]]:
    """Split the trailing per-stem ``DRUMS:/BASS:/VOCALS:/OTHER:`` lines off a caption.

    Returns ``(text_without_those_lines, {stem_name: caption})`` keyed by
    ``STEM_CAPTION_KEYS``. Robust to missing/extra/reordered lines and markdown
    bullets — an absent or sentinel ("none"/"instrumental"/…) value is omitted, so a
    malformed block degrades to fewer (or zero) per-stem captions rather than
    garbage. Run this BEFORE ``parse_gender`` (it strips its lines from the raw
    text first, leaving ``description … GENDER: …`` for the gender parser)."""
    if not text:
        return "", {}
    stems: dict[str, str] = {}
    spans: list[tuple[int, int]] = []
    for m in _STEM_LINE_RE.finditer(text):
        name = m.group(1).lower()
        if name not in STEM_CAPTION_KEYS or name in stems:
            continue  # first occurrence wins; ignore unknown labels
        value = " ".join(m.group(2).split())
        spans.append((m.start(), m.end()))
        if value.strip().strip(".").lower() in _STEM_EMPTY_VALUES or not value:
            continue  # sentinel / empty -> no caption for this stem
        if len(value) > MAX_STEM_CHARS:
            value = value[:MAX_STEM_CHARS].rsplit(" ", 1)[0]
        stems[name] = value.strip()
    # Remove the matched lines from the text (back-to-front to keep offsets valid).
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]
    return text, stems


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
        """Caption ONE song from its windows (description only; gender stripped)."""
        return self.caption_with_gender(windows, sr=sr)[0]

    def caption_with_gender(
        self, windows: list[np.ndarray], sr: int = SAMPLE_RATE
    ) -> tuple[str, str | None]:
        """Caption ONE song; return ``(description, vocal_gender)`` (stems dropped)."""
        d, g, _stems = self.caption_with_gender_and_stems(windows, sr=sr)
        return d, g

    def caption_with_gender_and_stems(
        self, windows: list[np.ndarray], sr: int = SAMPLE_RATE
    ) -> tuple[str, str | None, dict[str, str]]:
        """Caption ONE song; return ``(description, vocal_gender, stems)``. Gender is
        the audio-LLM's male/female/instrumental judgment (the F0-on-Demucs
        replacement); ``stems`` is the per-stem caption dict (DRUMS/BASS/VOCALS/OTHER
        → one sentence each, in STEM_CAPTION_KEYS order), the /addstem target-stem
        tag source. Both are parsed off the trailing tag lines."""
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
        # parse_stems first (strips its lines), then parse_gender on the remainder.
        rest, stems = parse_stems(caption)
        desc, gender = parse_gender(rest)
        return self._clean(desc), gender, stems

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
        # max_model_len holds MAX_WINDOWS*~750 audio tokens + prompt + MAX_NEW_TOKENS.
        # Bumped 4096 -> 5120: MAX_NEW_TOKENS grew to 700 (the v5 per-stem block), so
        # 3000 audio + ~400 prompt + 700 output no longer fits 4096. gpu_memory_utilization
        # governs the KV cache — lower it / max_num_seqs / max_model_len if init OOMs.
        self._llm = LLM(
            model=self._model_name,
            dtype="float16",
            max_model_len=int(os.environ.get("NANO_VLLM_MAX_MODEL_LEN", "5120")),
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
        """Caption MANY songs in one vLLM batch — descriptions only (gender stripped)."""
        return [d for d, _g in self.caption_many_with_gender(windows_per_song, sr=sr)]

    def caption_many_with_gender(
        self, windows_per_song: list[list[np.ndarray]], sr: int = SAMPLE_RATE
    ) -> list[tuple[str, str | None]]:
        """Caption MANY songs in one vLLM batch; ``(description, gender)`` per song
        (stems dropped). See ``caption_many_with_gender_and_stems``."""
        return [(d, g) for d, g, _s
                in self.caption_many_with_gender_and_stems(windows_per_song, sr=sr)]

    def caption_many_with_gender_and_stems(
        self, windows_per_song: list[list[np.ndarray]], sr: int = SAMPLE_RATE
    ) -> list[tuple[str, str | None, dict[str, str]]]:
        """Caption MANY songs in one vLLM batch (continuous batching). Takes one
        window-list per song; returns one ``(description, vocal_gender, stems)`` per
        song, in order (vLLM preserves request order). Gender + the per-stem caption
        dict are parsed off the trailing tag lines the captioner emits."""
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
        result: list[tuple[str, str | None, dict[str, str]]] = []
        for o in outputs:
            rest, stems = parse_stems(o.outputs[0].text)
            desc, gender = parse_gender(rest)
            result.append((_clean(desc), gender, stems))
        return result

    def caption(self, windows: list[np.ndarray], sr: int = SAMPLE_RATE) -> str:
        """Caption ONE song (delegates to the batched path) so the single-song
        callers (diskrot/auto_tag.py, caption_path) work unchanged on either
        backend."""
        return self.caption_many([windows], sr=sr)[0]

    def caption_with_gender(
        self, windows: list[np.ndarray], sr: int = SAMPLE_RATE
    ) -> tuple[str, str | None]:
        """Single-song ``(description, vocal_gender)`` — backend-parity with the
        HF captioner so the Modal worker's non-batched fallback works on either."""
        return self.caption_many_with_gender([windows], sr=sr)[0]

    def caption_with_gender_and_stems(
        self, windows: list[np.ndarray], sr: int = SAMPLE_RATE
    ) -> tuple[str, str | None, dict[str, str]]:
        """Single-song ``(description, vocal_gender, stems)`` — backend-parity with
        the HF captioner."""
        return self.caption_many_with_gender_and_stems([windows], sr=sr)[0]

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
