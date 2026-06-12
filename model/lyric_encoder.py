"""Phoneme lyric encoder for singing-intelligible conditioning.

The pooled CLAP lyric vector (one 1024-d "vibe" embedding) cannot carry *which
syllable to sing when* — see model/text_encoder.py and the project plan. This
module replaces it with a trainable encoder over a **phoneme-ID sequence**, which
the audio decoder cross-attends to (soft, near-monotonic alignment learned
implicitly — no inference-time duration model needed).

Two pieces live here:

1. A frozen phoneme vocabulary + ``text_to_phoneme_ids`` (g2p via ``g2p_en``).
   This is the SINGLE source of truth for the id mapping so training and
   inference agree exactly — a train/inference g2p-vocab mismatch is the biggest
   footgun in the whole change. The ARPABET symbol list is hardcoded (not read
   from ``g2p_en`` at runtime) so a ``g2p_en`` upgrade can't silently renumber.

2. ``LyricEncoder`` — a small bidirectional Transformer producing a sequence
   ``[B, L, d_model]`` plus its pad mask. It is instantiated *inside*
   ``NanoAudioGPT`` so DDP syncs it and it saves/restores with the model
   state_dict automatically (unlike the sidecar CLAP projection).
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Phoneme vocabulary -------------------------------------------------------
# ARPABET phones as emitted by g2p_en's CMUdict path, vowels carrying lexical
# stress (0/1/2). Copied from g2p_en 2.1.0's canonical ``G2p().phonemes`` minus
# its 4 internal seq2seq specials; frozen here so the id mapping never moves.
_ARPABET: tuple[str, ...] = (
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2",
    "F", "G", "HH",
    "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH",
)

# Special tokens occupy the low ids; everything downstream keys off these names.
PAD_PHONEME = "<pad>"
BOS_PHONEME = "<bos>"
WORD_BOUNDARY_PHONEME = "<wb>"  # word/space/punctuation boundary
UNK_PHONEME = "<unk>"

# Song-structure markers (intro/verse/chorus/...). These ride in the SAME phoneme
# stream as the words: the dataset injects them by section timestamp and the
# decoder cross-attends to them, so the model learns arrangement (sing this as a
# chorus) instead of trying to sing the literal word "chorus". The label set is
# exactly what the allin1 structure analyzer emits (minus its start/end
# sentinels); ``<no_section>`` is the fallback for crops/spans in a gap, so every
# stream always carries a valid section prefix. Placed BEFORE _ARPABET so a future
# g2p/ARPABET edit can't silently renumber the structure ids (same reasoning as
# the frozen ARPABET ordering above). Adding/removing any of these changes
# PHONEME_VOCAB_SIZE and is checkpoint-incompatible.
NO_SECTION_LABEL = "no_section"
STRUCTURE_LABELS: tuple[str, ...] = (
    NO_SECTION_LABEL, "intro", "verse", "chorus", "bridge",
    "outro", "break", "inst", "solo",
)
_STRUCTURE_TOKENS: tuple[str, ...] = tuple(f"<{label}>" for label in STRUCTURE_LABELS)

# Vocal-gender markers — ride the SAME phoneme stream as the structure markers,
# for the same reason: gender is a (per-song, occasionally per-section) attribute
# of the *vocals*, so emitting it as a marker the decoder cross-attends to
# conditions vocal timbre with no new module, and it CFG-drops with the lyric
# stream. The F0 labeler (diskrot.transcribe_lyrics) writes "male"/"female" per
# song; ``<unknown_gender>`` is the fallback for instrumental / unlabeled /
# ambiguous crops, so every stream always carries a valid gender slot (the same
# dense-prefix discipline as ``<no_section>``). Placed between the structure
# markers and _ARPABET so an ARPABET edit can't silently renumber them. Adding/
# removing any of these changes PHONEME_VOCAB_SIZE and is checkpoint-incompatible.
UNKNOWN_GENDER_LABEL = "unknown_gender"
GENDER_LABELS: tuple[str, ...] = (UNKNOWN_GENDER_LABEL, "male", "female")
_GENDER_TOKENS: tuple[str, ...] = tuple(f"<{label}>" for label in GENDER_LABELS)

# Tempo (BPM) markers — ride the SAME phoneme stream as the structure/gender
# markers, for the same reason: tempo is a per-song attribute the decoder can
# cross-attend to (it conditions the beat rate chroma can't carry, since chroma is
# octave-invariant pitch, not rhythm), and it CFG-drops with the lyric stream. The
# allin1 structure pass already computes a per-song ``bpm`` (diskrot.structure),
# previously discarded; here it's bucketed into coarse ranges (perception of tempo
# is roughly categorical). ``TEMPO_BPM_EDGES`` are the inclusive lower bounds of
# buckets 1..N (bucket 0 is everything below the first edge); ``<unknown_tempo>``
# is the fallback for songs without a bpm (no structure pass / instrumental), so
# every stream always carries a valid tempo slot (the same dense-prefix discipline
# as ``<no_section>``/``<unknown_gender>``). Placed between the gender markers and
# _ARPABET so an ARPABET edit can't silently renumber them. Adding/removing any of
# these changes PHONEME_VOCAB_SIZE and is checkpoint-incompatible.
UNKNOWN_TEMPO_LABEL = "unknown_tempo"
# Bucket boundaries (BPM): bucket i = [EDGES[i-1], EDGES[i]); bucket 0 = [-inf, 60),
# last bucket = [180, +inf). 10-BPM grid -> 14 buckets total (v8: was a 20-BPM grid
# with 7 buckets; the raw allin1 bpm was already on disk, so finer buckets are
# free). Tunable — changing the count or edges changes the vocab and is
# checkpoint-incompatible.
TEMPO_BPM_EDGES: tuple[float, ...] = (
    60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0,
    130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
)
N_TEMPO_BUCKETS: int = len(TEMPO_BPM_EDGES) + 1  # 14
_TEMPO_TOKENS: tuple[str, ...] = (
    f"<{UNKNOWN_TEMPO_LABEL}>",
) + tuple(f"<tempo_{i}>" for i in range(N_TEMPO_BUCKETS))

# Key (tonality) markers — ride the SAME phoneme stream as the other header
# markers, for the same reason: key is a per-song attribute the decoder can
# cross-attend to, and it CFG-drops with the lyric stream. The chroma the melody
# pass already packs (packed_NNN.mel.bin) is octave-invariant pitch-class energy,
# so a per-song Krumhansl-Schmuckler estimate over its mean (diskrot.key_detect)
# is near-free; this marker also helps the decoder resolve the key the additive
# chroma conditioning leaves implicit. 12 pitch classes (sharps-canonical;
# diskrot.key_detect and ``key_label_to_id`` both fold flats/enharmonics) x
# major/minor = 24 keys; ``<unknown_key>`` is the fallback for songs without an
# estimate, so every stream always carries a valid key slot (the same
# dense-prefix discipline as the other header markers). Placed between the tempo
# markers and _ARPABET so an ARPABET edit can't silently renumber them. Adding/
# removing any of these changes PHONEME_VOCAB_SIZE and is checkpoint-incompatible.
UNKNOWN_KEY_LABEL = "unknown_key"
_PITCH_CLASSES: tuple[str, ...] = (
    "c", "c_sharp", "d", "d_sharp", "e", "f",
    "f_sharp", "g", "g_sharp", "a", "a_sharp", "b",
)
_KEY_MODES: tuple[str, ...] = ("major", "minor")
KEY_LABELS: tuple[str, ...] = (UNKNOWN_KEY_LABEL,) + tuple(
    f"{pc}_{mode}" for mode in _KEY_MODES for pc in _PITCH_CLASSES
)
_KEY_TOKENS: tuple[str, ...] = tuple(f"<key_{label}>" if label != UNKNOWN_KEY_LABEL
                                     else f"<{label}>" for label in KEY_LABELS)

# Vocal-presence markers — ride the SAME phoneme stream, same rationale. Derived
# at train time from the transcription pass: a song with usable transcribed words
# is <vocals>, a song the pass processed but found no words in is <instrumental>,
# and a song the pass never covered is <unknown_vocals>. This is what lets a user
# *request* no vocals (``[instrumental]``) — without it an instrumental song and
# an unconditioned one look identical to the model. (Whisper occasionally
# hallucinates words on instrumentals, so the train-time label is imperfect;
# word-confidence capture is the future fix.) Placed between the key markers and
# _ARPABET so an ARPABET edit can't silently renumber them. Adding/removing any
# of these changes PHONEME_VOCAB_SIZE and is checkpoint-incompatible.
UNKNOWN_VOCALS_LABEL = "unknown_vocals"
VOCAL_LABELS: tuple[str, ...] = (UNKNOWN_VOCALS_LABEL, "vocals", "instrumental")
_VOCAL_TOKENS: tuple[str, ...] = tuple(f"<{label}>" for label in VOCAL_LABELS)

# Frozen vocab: specials first (so PAD==0), then structure markers, then gender
# markers, then tempo markers, then key markers, then vocal-presence markers,
# then ARPABET.
PHONEME_VOCAB: tuple[str, ...] = (
    PAD_PHONEME, BOS_PHONEME, WORD_BOUNDARY_PHONEME, UNK_PHONEME,
) + _STRUCTURE_TOKENS + _GENDER_TOKENS + _TEMPO_TOKENS + _KEY_TOKENS + _VOCAL_TOKENS + _ARPABET

PHONEME_TO_ID: dict[str, int] = {p: i for i, p in enumerate(PHONEME_VOCAB)}
PAD_PHONEME_ID: int = PHONEME_TO_ID[PAD_PHONEME]
BOS_PHONEME_ID: int = PHONEME_TO_ID[BOS_PHONEME]
WORD_BOUNDARY_ID: int = PHONEME_TO_ID[WORD_BOUNDARY_PHONEME]
UNK_PHONEME_ID: int = PHONEME_TO_ID[UNK_PHONEME]
PHONEME_VOCAB_SIZE: int = len(PHONEME_VOCAB)  # 129 (4 specials + 9 structure + 3 gender + 15 tempo + 25 key + 3 vocal + 70 ARPABET)

# Structure label <-> phoneme id, the single source of truth shared by the dataset
# (train-time injection) and inference (bracket parsing) so the two agree exactly.
STRUCTURE_TOKEN_TO_ID: dict[str, int] = {
    label: PHONEME_TO_ID[f"<{label}>"] for label in STRUCTURE_LABELS
}
ID_TO_STRUCTURE: dict[int, str] = {i: label for label, i in STRUCTURE_TOKEN_TO_ID.items()}
NO_SECTION_ID: int = STRUCTURE_TOKEN_TO_ID[NO_SECTION_LABEL]
STRUCTURE_IDS: frozenset[int] = frozenset(STRUCTURE_TOKEN_TO_ID.values())


def structure_label_to_id(label: str | None) -> int:
    """Map a structure label to its marker id, falling back to <no_section>.

    Normalizes case/whitespace and folds aliases the user might type but the
    labeler never emits (e.g. ``pre-chorus``/``prechorus`` -> no_section) so an
    unknown bracket can never crash or leak a bogus id. The train and inference
    paths both route section labels through this, keeping the id mapping identical.
    """
    if not label:
        return NO_SECTION_ID
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    return STRUCTURE_TOKEN_TO_ID.get(key, NO_SECTION_ID)


# Gender label <-> phoneme id, parallel to the structure mapping above. Shared by
# the dataset (train-time prefix injection) and inference (bracket parsing) so the
# two agree exactly.
GENDER_TOKEN_TO_ID: dict[str, int] = {
    label: PHONEME_TO_ID[f"<{label}>"] for label in GENDER_LABELS
}
ID_TO_GENDER: dict[int, str] = {i: label for label, i in GENDER_TOKEN_TO_ID.items()}
UNKNOWN_GENDER_ID: int = GENDER_TOKEN_TO_ID[UNKNOWN_GENDER_LABEL]
GENDER_IDS: frozenset[int] = frozenset(GENDER_TOKEN_TO_ID.values())

# Aliases a user (or a labeler) might type for the two canonical labels. Kept
# explicit so ``gender_label_to_id`` can both map AND recognize them (an unknown
# label maps to <unknown_gender>, so mapping alone can't tell "unknown" apart
# from "unrecognized" — ``is_gender_label`` needs this set to classify brackets).
_GENDER_ALIASES: dict[str, str] = {
    "m": "male", "man": "male", "men": "male", "boy": "male", "guy": "male",
    "f": "female", "woman": "female", "women": "female", "girl": "female",
    "unknown": UNKNOWN_GENDER_LABEL, "none": UNKNOWN_GENDER_LABEL,
}
_RECOGNIZED_GENDER_KEYS: frozenset[str] = frozenset(GENDER_TOKEN_TO_ID) | frozenset(_GENDER_ALIASES)


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def gender_label_to_id(label: str | None) -> int:
    """Map a gender label to its marker id, falling back to <unknown_gender>.

    Normalizes case/whitespace and folds common aliases (``m``/``man`` -> male,
    ``f``/``woman`` -> female). The train and inference paths both route gender
    labels through this so the id mapping stays identical."""
    if not label:
        return UNKNOWN_GENDER_ID
    key = _normalize_label(label)
    key = _GENDER_ALIASES.get(key, key)
    return GENDER_TOKEN_TO_ID.get(key, UNKNOWN_GENDER_ID)


def is_gender_label(label: str | None) -> bool:
    """True if ``label`` names a gender (canonical or alias).

    Used by the inference bracket parser to route ``[male]`` to the gender prefix
    slot vs ``[chorus]`` to the section slot. Distinct from ``gender_label_to_id``
    because that folds *unrecognized* labels to <unknown_gender> too."""
    if not label:
        return False
    return _normalize_label(label) in _RECOGNIZED_GENDER_KEYS


# Tempo label/bpm <-> phoneme id, parallel to the gender mapping above. Shared by
# the dataset (train-time prefix injection) and inference (bracket parsing) so the
# two agree exactly. ``<unknown_tempo>`` is index 0 of _TEMPO_TOKENS; bucket i maps
# to ``<tempo_i>``.
UNKNOWN_TEMPO_ID: int = PHONEME_TO_ID[f"<{UNKNOWN_TEMPO_LABEL}>"]
_TEMPO_BUCKET_IDS: tuple[int, ...] = tuple(
    PHONEME_TO_ID[f"<tempo_{i}>"] for i in range(N_TEMPO_BUCKETS)
)
TEMPO_IDS: frozenset[int] = frozenset((UNKNOWN_TEMPO_ID,) + _TEMPO_BUCKET_IDS)

# Named tempo aliases a user might type for a bracket; mapped to a representative
# bpm so they route through the same bucketing as a numeric ``[120bpm]``.
_TEMPO_ALIASES: dict[str, float] = {
    "slow": 65.0, "medium": 100.0, "mid": 100.0, "moderate": 100.0, "fast": 160.0,
}
_TEMPO_NUM_RE = re.compile(r"^\s*(?:tempo\s*[:=]?\s*)?(\d+(?:\.\d+)?)\s*(?:bpm)?\s*$", re.I)


def bpm_to_id(bpm: float | None) -> int:
    """Map a BPM value to its tempo-bucket marker id, falling back to <unknown_tempo>.

    ``None`` or non-finite / non-positive bpm -> <unknown_tempo>. Otherwise bucket
    by ``TEMPO_BPM_EDGES`` (bucket i = [EDGES[i-1], EDGES[i])). The train and
    inference paths both route bpm through this, keeping the id mapping identical.
    """
    if bpm is None:
        return UNKNOWN_TEMPO_ID
    try:
        b = float(bpm)
    except (TypeError, ValueError):
        return UNKNOWN_TEMPO_ID
    if not math.isfinite(b) or b <= 0:
        return UNKNOWN_TEMPO_ID
    bucket = 0
    for edge in TEMPO_BPM_EDGES:
        if b < edge:
            break
        bucket += 1
    return _TEMPO_BUCKET_IDS[bucket]


def parse_tempo_label(label: str | None) -> float | None:
    """Parse a tempo bracket (``120``, ``120bpm``, ``tempo:120``, ``fast``) to a bpm.

    Returns ``None`` when the label isn't a tempo (so the bracket parser can route
    it elsewhere). Named aliases map to a representative bpm; numeric forms parse
    directly. Shared train/inference contract via ``bpm_to_id``."""
    if not label:
        return None
    key = label.strip().lower()
    if key in _TEMPO_ALIASES:
        return _TEMPO_ALIASES[key]
    m = _TEMPO_NUM_RE.match(key)
    if m:
        return float(m.group(1))
    return None


def is_tempo_label(label: str | None) -> bool:
    """True if ``label`` names a tempo (numeric bpm or a known alias).

    Used by the inference bracket parser to route ``[120bpm]`` / ``[fast]`` to the
    tempo prefix slot vs ``[chorus]`` to the section slot."""
    return parse_tempo_label(label) is not None


# Key label <-> phoneme id, parallel to the gender/tempo mappings above. Shared by
# the dataset (train-time prefix injection, fed by diskrot.key_detect's keys.json)
# and inference (bracket parsing) so the two agree exactly.
KEY_TOKEN_TO_ID: dict[str, int] = {
    label: PHONEME_TO_ID[token] for label, token in zip(KEY_LABELS, _KEY_TOKENS)
}
ID_TO_KEY: dict[int, str] = {i: label for label, i in KEY_TOKEN_TO_ID.items()}
UNKNOWN_KEY_ID: int = KEY_TOKEN_TO_ID[UNKNOWN_KEY_LABEL]
KEY_IDS: frozenset[int] = frozenset(KEY_TOKEN_TO_ID.values())

# Natural-letter pitch classes (semitones above C); accidentals shift +-1 mod 12,
# which also folds enharmonics (db -> c_sharp, e# -> f, cb -> b, ...).
_NATURAL_SEMITONE: dict[str, int] = {
    "c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11,
}
# Matches a pre-normalized key string: letter, optional accidental, optional mode.
_KEY_RE = re.compile(r"^([a-g])(#|b)?_?(major|minor|maj|min|m)?$")


def parse_key_label(label: str | None) -> str | None:
    """Parse a key bracket (``Am``, ``a minor``, ``F# major``, ``Bb``, ``key:Am``,
    ``c_sharp_minor``) to its canonical sharps-form label, or ``None`` when the
    string isn't a key (so the bracket parser can route it elsewhere). Flats and
    spelled-out ``sharp``/``flat`` fold to the canonical sharp pitch class; a
    missing mode means major. Shared train/inference contract via
    ``key_label_to_id``."""
    if not label:
        return None
    key = label.strip().lower()
    if key == UNKNOWN_KEY_LABEL:
        return UNKNOWN_KEY_LABEL
    key = re.sub(r"^key\s*[:=]?\s*", "", key)
    key = key.replace("♯", "#").replace("♭", "b")
    key = key.replace("-", "_").replace(" ", "_")
    key = key.replace("_sharp", "#").replace("_flat", "b")
    m = _KEY_RE.match(key)
    if not m:
        return None
    letter, accidental, mode = m.groups()
    semitone = _NATURAL_SEMITONE[letter]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    pc = _PITCH_CLASSES[semitone % 12]
    return f"{pc}_{'minor' if mode in ('minor', 'min', 'm') else 'major'}"


def key_label_to_id(label: str | None) -> int:
    """Map a key label to its marker id, falling back to <unknown_key>.

    Accepts canonical labels (``a_minor``) and the user forms ``parse_key_label``
    handles. The train and inference paths both route key labels through this so
    the id mapping stays identical."""
    canonical = parse_key_label(label)
    if canonical is None:
        return UNKNOWN_KEY_ID
    return KEY_TOKEN_TO_ID.get(canonical, UNKNOWN_KEY_ID)


def is_key_label(label: str | None) -> bool:
    """True if ``label`` names a key (canonical or a recognized user form).

    Used by the inference bracket parser to route ``[a minor]`` / ``[key:Am]`` to
    the key prefix slot vs ``[chorus]`` to the section slot. NB: bare ``[f]`` and
    ``[m]`` route to gender (checked first) — use ``[f major]`` etc. for those keys."""
    return parse_key_label(label) is not None


# Vocal-presence label <-> phoneme id, parallel to the mappings above. Shared by
# the dataset (train-time prefix injection, derived from the transcription pass)
# and inference (bracket parsing) so the two agree exactly.
VOCAL_TOKEN_TO_ID: dict[str, int] = {
    label: PHONEME_TO_ID[f"<{label}>"] for label in VOCAL_LABELS
}
ID_TO_VOCAL: dict[int, str] = {i: label for label, i in VOCAL_TOKEN_TO_ID.items()}
UNKNOWN_VOCALS_ID: int = VOCAL_TOKEN_TO_ID[UNKNOWN_VOCALS_LABEL]
VOCAL_IDS: frozenset[int] = frozenset(VOCAL_TOKEN_TO_ID.values())

# Aliases a user might type. Deliberately does NOT include ``inst`` — that's an
# allin1 *section* label, and the bracket parser checks vocal labels before
# section labels, so claiming it here would steal ``[inst]`` from the section slot.
_VOCAL_ALIASES: dict[str, str] = {
    "vocal": "vocals", "voice": "vocals", "sung": "vocals",
    "no_vocals": "instrumental", "no_vocal": "instrumental",
}
_RECOGNIZED_VOCAL_KEYS: frozenset[str] = frozenset(VOCAL_TOKEN_TO_ID) | frozenset(_VOCAL_ALIASES)


def vocal_label_to_id(label: str | None) -> int:
    """Map a vocal-presence label to its marker id, falling back to <unknown_vocals>.

    Normalizes case/whitespace and folds aliases (``no vocals`` -> instrumental).
    The train and inference paths both route vocal labels through this so the id
    mapping stays identical."""
    if not label:
        return UNKNOWN_VOCALS_ID
    key = _normalize_label(label)
    key = _VOCAL_ALIASES.get(key, key)
    return VOCAL_TOKEN_TO_ID.get(key, UNKNOWN_VOCALS_ID)


def is_vocal_label(label: str | None) -> bool:
    """True if ``label`` names vocal presence (canonical or alias).

    Used by the inference bracket parser to route ``[instrumental]`` / ``[vocals]``
    to the vocal prefix slot vs ``[chorus]`` to the section slot. Distinct from
    ``vocal_label_to_id`` because that folds *unrecognized* labels to
    <unknown_vocals> too."""
    if not label:
        return False
    return _normalize_label(label) in _RECOGNIZED_VOCAL_KEYS


# Punctuation g2p_en passes through verbatim that we fold into a word boundary
# rather than dropping (keeps phrase structure the decoder can align to).
_BOUNDARY_PUNCT = frozenset({",", ".", "!", "?", ";", ":", "-", "...", " "})

_G2P = None  # lazily constructed per process (G2p() loads nltk data + a model)


def _get_g2p():
    """Lazily build a process-local g2p_en.G2p (expensive: nltk + numpy model)."""
    global _G2P
    if _G2P is None:
        from g2p_en import G2p

        _G2P = G2p()
    return _G2P


def text_to_phoneme_ids(
    text: str, max_len: int | None = None, add_bos: bool = True,
) -> list[int]:
    """Convert a lyric string to phoneme ids using the frozen vocab.

    g2p_en emits ARPABET phones, ``' '`` between words, and raw punctuation;
    we map phones via PHONEME_TO_ID, fold spaces/sentence punctuation to a single
    WORD_BOUNDARY token, and drop anything else (rare g2p artifacts). UNK_PHONEME
    is reserved for future use but never emitted — silently dropping an artifact
    keeps the id stream identical on both the train and inference paths.
    Deterministic for a given g2p_en version + nltk data — the contract train and
    inference both rely on. ``max_len`` truncates (after the optional BOS).
    """
    if not text or not text.strip():
        return []
    g2p = _get_g2p()
    try:
        syms = list(g2p(text))
    except Exception:
        # g2p_en expands digit runs via inflect, which raises NumOutOfRangeError
        # on absurd numbers (real Whisper-transcript garbage). Retry per word so
        # a single bad token drops only itself — deterministically, on both the
        # train and inference paths.
        syms = []
        for w in text.split():
            try:
                syms.extend(g2p(w))
            except Exception:
                continue
            syms.append(" ")
    ids: list[int] = [BOS_PHONEME_ID] if add_bos else []
    prev_boundary = True  # suppress a leading boundary token
    for sym in syms:
        pid = PHONEME_TO_ID.get(sym)
        if pid is not None:
            ids.append(pid)
            prev_boundary = False
        elif sym in _BOUNDARY_PUNCT:
            if not prev_boundary:  # collapse runs of space/punct
                ids.append(WORD_BOUNDARY_ID)
                prev_boundary = True
        # else: unknown artifact — dropped (UNK_PHONEME reserved, never emitted)
    # Trim a trailing boundary.
    if ids and ids[-1] == WORD_BOUNDARY_ID:
        ids.pop()
    if max_len is not None and len(ids) > max_len:
        ids = ids[:max_len]
    return ids


def text_to_word_phoneme_groups(words: list[str]) -> list[list[int]]:
    """Phonemize a word list, returning one phoneme-id group per input word.

    Runs g2p ONCE on the joined phrase (so cross-word context / POS is preserved)
    and splits the phone stream on g2p's word-boundary spaces. This lets the
    dataset phonemize a song's lyrics once and then slice the groups by word index
    for any crop window, instead of re-running g2p per segment. If the split count
    doesn't line up with the words (stray punctuation), falls back to per-word g2p
    so the 1:1 word→group alignment the caller relies on always holds.
    """
    if not words:
        return []
    g2p = _get_g2p()
    try:
        groups: list[list[int]] = [[]]
        for sym in g2p(" ".join(words)):
            if sym == " ":
                groups.append([])
                continue
            pid = PHONEME_TO_ID.get(sym)
            if pid is not None:
                groups[-1].append(pid)
            # non-space punctuation / artifacts: dropped (don't split the word; UNK
            # is reserved, never emitted — keeps train/inference id streams identical)
        if len(groups) == len(words):
            return groups
    except Exception:
        # g2p_en's inflect number expansion raises on absurd digit runs (Whisper
        # garbage) — fall through to per-word, where the bad word yields [].
        pass
    # Alignment drift — phonemize each word independently (loses some context but
    # guarantees the per-word grouping the dataset needs).
    return [_word_phoneme_ids(g2p, w) for w in words]


def _word_phoneme_ids(g2p, word: str) -> list[int]:
    """Phonemize one word; a word g2p can't handle yields an empty group."""
    try:
        return [PHONEME_TO_ID[s] for s in g2p(word) if s in PHONEME_TO_ID]
    except Exception:
        return []


# --- Structure-marker stream assembly -----------------------------------------
# The dataset (train) and the inference parser MUST build byte-identical streams.
# Both append "units" — a section marker [id] or one word's phoneme group — under
# a single separator rule so the encoder always sees ``<chorus> <wb> <phonemes>``.

def append_unit(ids: list[int], unit: list[int]) -> None:
    """Append a unit (one marker id, or one word's phoneme group) to ``ids``.

    Inserts a WORD_BOUNDARY before the unit iff something other than the leading
    BOS is already present (``len(ids) > 1``). This is the ONE separator rule used
    by both the dataset injector and the inference parser; the BOS+prefix-marker
    pair falls out of it for free (the prefix lands at ``len(ids) == 1``, so it
    gets no boundary).
    """
    if not unit:
        return
    if len(ids) > 1:
        ids.append(WORD_BOUNDARY_ID)
    ids.extend(unit)


def append_unit_capped(ids: list[int], unit: list[int], max_len: int | None) -> bool:
    """``append_unit`` with whole-unit truncation.

    Appends ``unit`` (with its boundary) only if doing so keeps ``len(ids)`` at or
    below ``max_len``; otherwise leaves ``ids`` untouched and returns ``False`` so
    the caller stops. This avoids slicing a word's phoneme group (or a marker)
    mid-unit at the cap. ``max_len=None`` never blocks. Used by BOTH the dataset
    injector and the inference parser so their truncation stays byte-identical.
    """
    if not unit:
        return True
    if max_len is not None:
        cost = len(unit) + (1 if len(ids) > 1 else 0)  # +1 for the WORD_BOUNDARY
        if len(ids) + cost > max_len:
            return False
    append_unit(ids, unit)
    return True


_MARKER_RE = re.compile(r"\[([^\[\]]+)\]")


def text_with_markers_to_phoneme_ids(
    text: str, max_len: int | None = None, add_bos: bool = True,
) -> list[int]:
    """Inference-side lyric parser: ``[female] [120bpm] [a minor] [vocals] [verse] words [chorus] ...`` -> ids.

    Reproduces the dataset's train-time stream exactly (see ``append_unit`` and
    ``TokenDataset._get_segment_lyric_ids``):

      ``BOS  <gender>  <tempo>  <key>  <vocals>  <prefix-section>  w w  <inline-section>  w ...``

    Prefix rules — every stream carries a gender slot, a tempo slot, a key slot,
    a vocal-presence slot, AND a section slot, always, matching the dense
    train-time prefix:
    - Leading ``[label]`` markers are consumed as prefixes: one gender (``[male]``
      / ``[female]``), one tempo (``[120bpm]`` / ``[tempo:120]`` / ``[fast]``), one
      key (``[a minor]`` / ``[key:Am]`` / ``[f# major]``), one vocal presence
      (``[vocals]`` / ``[instrumental]``), and one section (``[verse]`` ...), in
      any order. A slot not given defaults to its unknown marker
      (``<unknown_gender>`` / ``<unknown_tempo>`` / ``<unknown_key>`` /
      ``<unknown_vocals>`` / ``<no_section>``) — except the vocal slot, which
      defaults to ``<vocals>`` when the text contains words (matching the
      train-time invariant that transcribed words always co-occur with
      ``<vocals>``).
    - Emitted in train-time order: gender, tempo, key, vocals, then section.
    Remaining markers after the leading run are inline section markers (a stray
    inline gender/tempo/key/vocal marker is dropped — all are prefix-only, as in
    training).
    Brackets are stripped here and never reach g2p (the biggest train/inference
    footgun); unknown labels fold to ``<no_section>`` via ``structure_label_to_id``.
    Malformed brackets are handled defensively: an empty ``[]`` and the stray
    ``[``/``]`` left behind by a nested ``[[x]]`` are scrubbed so they never reach
    g2p. Word spans are phonemized with the same per-word grouping the dataset uses.
    """
    parts = _MARKER_RE.split(text or "")
    # re.split with one capture group yields: text, label, text, label, ...
    # (even indices = text spans, odd indices = captured labels).
    events: list[tuple[str, str]] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            label = part.strip()
            if label:  # skip an empty/whitespace-only [] marker
                events.append(("marker", label))
        else:
            # Scrub stray brackets (bare [] or the [ ] orphaned by nested [[x]])
            # so a literal bracket never reaches g2p.
            span = part.replace("[", " ").replace("]", " ").strip()
            if span:
                events.append(("text", span))

    ids: list[int] = [BOS_PHONEME_ID] if add_bos else []
    # Consume the leading run of markers as prefixes: at most one gender + one tempo
    # + one key + one vocal + one section, in any order. Stop at the first text span
    # or once all slots are filled.
    gender_id = UNKNOWN_GENDER_ID
    tempo_id = UNKNOWN_TEMPO_ID
    key_id = UNKNOWN_KEY_ID
    vocal_id = UNKNOWN_VOCALS_ID
    section_id = NO_SECTION_ID
    gender_set = tempo_set = key_set = vocal_set = section_set = False
    while events and events[0][0] == "marker":
        label = events[0][1]
        if is_gender_label(label) and not gender_set:
            gender_id, gender_set = gender_label_to_id(label), True
        elif is_tempo_label(label) and not tempo_set:
            tempo_id, tempo_set = bpm_to_id(parse_tempo_label(label)), True
        elif is_key_label(label) and not key_set:
            key_id, key_set = key_label_to_id(label), True
        elif is_vocal_label(label) and not vocal_set:
            vocal_id, vocal_set = vocal_label_to_id(label), True
        elif not section_set and not is_gender_label(label) and not is_tempo_label(label) \
                and not is_key_label(label) and not is_vocal_label(label):
            section_id, section_set = structure_label_to_id(label), True
        else:
            break
        events = events[1:]
    if not vocal_set and any(kind == "text" for kind, _ in events):
        # Words but no explicit vocal bracket: default to <vocals>. At train time
        # every song with transcribed words carries <vocals>, so a lyric'd request
        # left at <unknown_vocals> would be out-of-distribution; an explicit
        # [instrumental] (contradictory but allowed) still overrides.
        vocal_id = VOCAL_TOKEN_TO_ID["vocals"]
    # Compact 5-marker header (no internal word-boundary):
    # BOS <gender> <tempo> <key> <vocals> <section>.
    append_unit(ids, [gender_id, tempo_id, key_id, vocal_id, section_id])

    for kind, val in events:
        if kind == "marker":
            if (is_gender_label(val) or is_tempo_label(val)
                    or is_key_label(val) or is_vocal_label(val)):
                continue  # gender/tempo/key/vocals are prefix-only; ignore a stray inline marker
            if not append_unit_capped(ids, [structure_label_to_id(val)], max_len):
                break
        else:
            stop = False
            for group in text_to_word_phoneme_groups(val.split()):
                if not append_unit_capped(ids, group, max_len):
                    stop = True
                    break
            if stop:
                break
    return ids


# --- Encoder ------------------------------------------------------------------
class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positions added to phoneme embeddings (order matters)."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Re-derivable from (d_model, max_len); keep out of ckpts.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].to(x.dtype)


class _EncoderLayer(nn.Module):
    """Pre-norm bidirectional self-attention + GELU MLP (matches house style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, L, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class LyricEncoder(nn.Module):
    """Encode phoneme ids -> a sequence [B, L, d_model] for cross-attention.

    Bidirectional (the lyric line is fully known), so no causal mask and no RoPE;
    a fixed sinusoidal positional encoding carries order. ``forward`` returns the
    encoded sequence and the (passed-through) padding mask so the decoder's lyric
    cross-attention can mask padded positions.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 3,
        n_heads: int = 8,
        d_ff: int = 4096,
        max_len: int = 256,
        vocab_size: int = PHONEME_VOCAB_SIZE,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_PHONEME_ID)
        self.pos = _SinusoidalPositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [_EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.RMSNorm(d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            with torch.no_grad():
                m.weight[PAD_PHONEME_ID].zero_()

    def forward(
        self, ids: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ids: [B, L] long. mask: [B, L] bool (True = real phoneme).

        Returns (lyric_emb [B, L, d_model], mask [B, L]).
        """
        # Float additive key-padding mask for SDPA: [B, 1, 1, L], 0 keep / -inf drop.
        attn_mask = None
        if mask is not None:
            attn_mask = torch.zeros(
                mask.shape[0], 1, 1, mask.shape[1], dtype=torch.float32, device=mask.device,
            ).masked_fill(~mask[:, None, None, :], float("-inf"))
        x = self.drop(self.pos(self.embed(ids)))
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.ln_final(x)
        return x, mask
