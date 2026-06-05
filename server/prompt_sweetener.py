"""Local LLM prompt "sweetener" for inference conditioning.

nano's text conditioning is frozen Microsoft CLAP, whose projection was trained
on LP-MusicCaps prose captions (see eval/tags.json — median ~40 words, e.g.
"The low quality recording features a passionate male vocal singing over electric
guitar chords, groovy bass, punchy kick and shimmering hi hats. It sounds
energetic."). A terse user prompt like "lofi beat to study to" lands far from
that distribution in CLAP space, so conditioning is weak.

PromptSweetener rewrites the raw user prompt into a caption that looks like the
training data using a small local Qwen2.5-Instruct model, strengthening
conditioning without retraining anything. It is opt-in (server `sweeten` flag)
and lazy-loaded — the Qwen weights are only fetched/loaded on first use.

Delimiter safety: the server joins tags + lyrics as "tags. lyrics" and the
inference engine splits on the FIRST ". " to recover the two cross-attention
positions. A multi-sentence caption is full of ". ", which would mis-split and
dump half the caption into the lyrics slot. sweeten() therefore replaces internal
". " with "; " so the output is delimiter-safe while still reading as prose to
CLAP.
"""
from __future__ import annotations

import os
import re

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Real captions sampled from eval/tags.json — anchor the rewrite to the corpus.
_FEW_SHOT = [
    (
        "make me some techno",
        "The low quality recording features a techno song that consists of punchy "
        "kick and snare hits, shimmering hi hats, reversed crash cymbal and groovy "
        "synth bass. It sounds energetic, aggressive and addictive.",
    ),
    (
        "emotional rock song with vocals, cinematic",
        "The low quality recording features a passionate male vocal, alongside "
        "harmonizing background vocals, singing over claps, shimmering cymbals, "
        "groovy bass and electric guitar melody. It sounds energetic and addictive "
        "- like something you would hear in movies.",
    ),
    (
        "eerie instrumental background music",
        "The song is an instrumental. The tempo is medium with a keyboard "
        "accompaniment, various percussion hits, strong bass line and synth pad "
        "section. The song is eerie and full of tension.",
    ),
]

_SYSTEM_PROMPT = """You rewrite short music prompts into a single descriptive \
caption that matches the style of the LP-MusicCaps audio-captioning dataset, \
which a music generation model was trained on.

Rules:
- Output ONE caption only. No preamble, no markdown, no quotes, no lyrics, no \
explanation. ~40 words.
- Describe concrete instruments and production, then end with a short mood \
sentence (e.g. "It sounds energetic and danceable.").
- Prefer this vocabulary, which the model knows well: synth, groovy bass, \
electric guitar, punchy kick, shimmering hi hats, shimmering cymbals, piano, \
strings, male vocal. Genres that work: electronic, techno, house, rock, metal, \
pop, ambient, hip hop, soul.
- AVOID jazz, country, latin, blues, gospel — the model has little of this data \
and generates them as noise; map vague requests toward the genres above.
- If the user does not specify vocals, prefer "male vocal" or make it \
instrumental (the training data skews male-vocal and instrumental).
- Keep the user's intent (genre, mood, tempo) but express it in the caption \
style above."""


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
        text = text.strip().strip('"').strip("'").strip()
        text = re.sub(r"\s+", " ", text)
        # Delimiter safety: the server splits tags/lyrics on the FIRST ". ".
        # Collapse internal sentence boundaries to "; " so the whole caption
        # stays in the tags slot. A trailing "." is fine (no following space).
        text = text.replace(". ", "; ")
        # Clamp to ~50 words (CLAP truncates at 77 tokens anyway).
        words = text.split()
        if len(words) > 50:
            text = " ".join(words[:50])
        return text.strip()

    @torch.no_grad()
    def sweeten(self, raw: str) -> str:
        """Rewrite ``raw`` into a caption-style prompt. Falls back to ``raw`` on
        any failure — sweetening must never break a generation."""
        raw = (raw or "").strip()
        if not raw:
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
                max_new_tokens=80,
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
