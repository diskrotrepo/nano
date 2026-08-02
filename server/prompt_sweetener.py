"""Local LLM prompt "sweetener" for inference conditioning.

nano's text conditioning is frozen Microsoft CLAP, and the decoder is trained on
the corpus's audio-LLM captions (model/audio_llm_captioner.py — flowing
~120-160-word multi-facet prose covering genre/mood, drums, bass, harmony/lead,
vocals, production and arc, chunked across CLAP into a SEQUENCE of pooled
vectors). A terse user prompt like "lofi beat to study to" lands far from that
distribution in CLAP space, so conditioning is weak.

PromptSweetener rewrites the raw user prompt into a caption that matches that
training style using a small local Qwen2.5-Instruct model, strengthening
conditioning without retraining anything. It is opt-in (server `sweeten` flag)
and lazy-loaded — the Qwen weights are only fetched/loaded on first use.

This MUST track the captioner: if the audio-LLM prompt/style in
model/audio_llm_captioner.py changes, re-anchor the few-shots + system prompt
below, or train (long captions) and inference (sweetened prompts) drift into
different CLAP regions and adherence drops.

Tags and lyrics travel as SEPARATE request fields (no "tags. lyrics" join, no
". " split), and tags are chunked across CLAP, so a multi-sentence paragraph
caption is exactly what the decoder expects. An already caption-length prompt
(>~120 words) is passed through verbatim — it is caption-style on its own, and
rewriting it would throw away the detail the user wrote.
"""
from __future__ import annotations

import os
import re

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Few-shots written in the audio-LLM (audio_llm_v3) caption style the decoder is
# trained on — flowing multi-facet prose, concrete instruments, no hedging, no
# "low quality recording" opener. Keep these in sync with the captioner prompt.
_FEW_SHOT = [
    (
        "make me some techno",
        "A driving peak-time techno cut with a dark, hypnotic mood. A four-on-the-"
        "floor kick anchors the groove beneath crisp closed hi-hats, a sharp clap "
        "on the backbeat and an occasional reversed cymbal swell. A rolling analog "
        "sub-bass locks to the kick, gritty and saturated. The harmony is built "
        "from a detuned synth stab and a hypnotic acid line twisting through a "
        "resonant filter, with metallic percussion loops adding texture. The track "
        "is instrumental. The production is warm and analog, with tape saturation, "
        "tight sidechain pumping and a wide, immersive stereo field. It opens "
        "stripped to kick and bass, layers in the acid line over a long build, "
        "drops into a relentless peak-time section, then strips back for the outro.",
    ),
    (
        "male vocals over a hard hit 808 and funky bassline",
        "A confident trap-soul track with a smooth, late-night mood. Crisp rolling "
        "hi-hats and snappy rimshots ride over a hard-hitting 808 kick that booms "
        "and glides through the low end. Beneath it a funky electric bassline walks "
        "with syncopated groove, trading space with the 808. Warm Rhodes chords and "
        "a muted guitar lick carry the harmony while a jazzy synth lead floats on "
        "top. A male vocal delivers the hook in a melodic, half-sung croon, doubled "
        "with subtle ad-libs. The production is punchy and clean, the low end tight, "
        "the vocal set forward with light reverb and tape warmth. It opens on keys "
        "and vocal, drops the full 808 and drums on the hook, then pulls back for a "
        "stripped-down verse before the final chorus.",
    ),
    (
        "eerie instrumental background music",
        "An eerie, cinematic dark-ambient piece steeped in tension and unease. "
        "There is no drum kit; time is marked by a slow, distant pulse and "
        "occasional metallic hits that ring out and decay. A deep droning sub-bass "
        "sits underneath, swelling and receding like breath. The harmony drifts "
        "through detuned synth pads, a lone piano figure and high glassy textures "
        "that shimmer at the edges. The track is instrumental. The production is "
        "cavernous, with long cathedral reverbs, granular textures and a faint tape "
        "hiss used as atmosphere. It begins with a single sustained drone, layers "
        "in the piano and metallic accents, builds to an unsettling swell, then "
        "dissolves back toward silence.",
    ),
]

_SYSTEM_PROMPT = """You rewrite short music prompts into a single rich caption \
that matches the style a music-generation model was trained on: flowing, \
information-dense prose describing one track.

Rules:
- Output ONE caption only, as a single paragraph of natural prose. No preamble, \
no markdown, no lists, no quotes, no lyrics, no explanation. Aim for 120-160 \
words.
- Weave together, where relevant, the overall genre or micro-genre and mood; \
the drums and percussion; the bass; the harmony and lead instruments; the \
vocals; the production and mix character; and how the track evolves across its \
sections. Do NOT label these dimensions or write a list — continuous prose only.
- Be concrete: name specific instruments, textures, micro-genres and production \
artifacts (e.g. 808, sub-bass, Rhodes, analog synth, electric guitar, tape \
saturation, sidechain, reverb, crisp hi-hats).
- Vocals: if the prompt implies singing, describe the singer's character — \
gender, range and delivery (belted, crooned, rapped, screamed, harmonized, \
spoken-word). If the track is instrumental, say so in one short sentence. Do NOT \
pad with filler like "sparse vocals" or "occasional vocals". If the user does \
not mention vocals, make it instrumental or add a male vocal (the corpus skews \
instrumental and male-vocal).
- Commit to specifics — state things as fact, never "seems", "possibly" or \
"might be".
- Keep the user's intent (genre, mood, tempo, instrumentation, vocals) and \
express it in the rich caption style above."""


class PromptSweetener:
    """Lazy-loaded Qwen rewriter. Mirrors CLAPTextEncoder's lazy-load pattern."""

    def __init__(self, device: str = "cpu", model_name: str | None = None):
        self._device = device
        self._model_name = model_name or os.environ.get("NANO_SWEETEN_MODEL", DEFAULT_MODEL)
        self._tok = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype=dtype
        ).to(self._device)
        self._model.eval()
        print(f"[sweetener] loaded {self._model_name} on {self._device}")

    def _build_messages(self, raw: str) -> list[dict]:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for user, caption in _FEW_SHOT:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": caption})
        messages.append({"role": "user", "content": raw})
        return messages

    @staticmethod
    def _sanitize(text: str) -> str:
        text = text.strip()
        # Strip only a matched WRAPPING quote pair (Qwen sometimes wraps the whole
        # caption in quotes); leave embedded/edge quotes intact (e.g. a caption
        # ending in a quoted phrase).
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        text = re.sub(r"\s+", " ", text)
        # No ". " collapse: tags/lyrics are separate fields now, so a multi-
        # sentence caption is safe as natural prose and chunked CLAP reads it all.
        # Cap at ~240 words as a runaway guard for the rewriter (a sweetened
        # caption targets ~120-160 words, matching the audio-LLM training style).
        words = text.split()
        if len(words) > 240:
            text = " ".join(words[:240])
        return text.strip()

    @torch.no_grad()
    def sweeten(self, raw: str) -> str:
        """Rewrite ``raw`` into a caption-style prompt. Falls back to ``raw`` on
        any failure — sweetening must never break a generation."""
        raw = (raw or "").strip()
        if not raw:
            return raw
        # An already caption-length prompt (~the audio-LLM training length) is
        # caption-style on its own. Rewriting it would discard the detail the user
        # wrote (and chunked CLAP conditions on all of it), so pass it verbatim.
        if len(raw.split()) > 120:
            return raw
        try:
            self._ensure_model()
            text = self._tok.apply_chat_template(
                self._build_messages(raw),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tok(text, return_tensors="pt").to(self._device)
            out = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self._tok.eos_token_id,
            )
            new_tokens = out[0][inputs["input_ids"].shape[1]:]
            caption = self._tok.decode(new_tokens, skip_special_tokens=True)
            cleaned = self._sanitize(caption)
            return cleaned or raw
        except Exception as e:  # noqa: BLE001 — never fail a generation on sweetening
            print(f"[sweetener] failed ({e!r}); using raw prompt")
            return raw
