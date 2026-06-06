"""Diagnostic: can a listener make out the words? (transcribe-back WER)

This is the PRIMARY success metric for the phoneme lyric conditioner — val
cross-entropy does not reveal intelligibility. The loop closes on itself:

  1. take N held-out lyric lines (sampled from the val lyrics, so phonemes are
     in-distribution) + the song's tags,
  2. generate audio conditioned on (tags, lyrics) with the trained model,
  3. transcribe the generated audio with the SAME Demucs + faster-whisper
     pipeline used to build the training lyrics (diskrot.transcribe_lyrics),
  4. compute WER(hypothesis, input lyric) with jiwer.

A gibberish baseline (the v7 pooled-CLAP model, or an untrained lyric head)
sits near ~1.0 WER. A model that genuinely sings the words drives WER down.
Track mean WER across checkpoints — falling WER == emerging intelligibility.

Run:  modal run scripts/eval_lyric_wer.py
      modal run scripts/eval_lyric_wer.py --n-clips 40 --cfg-scale 4 --lyric-cfg-scale 6
"""
from __future__ import annotations

import modal

app = modal.App("nano-eval-lyric-wer")


def _prefetch_g2p() -> None:
    """Bake g2p_en's nltk data into the image so generation never blocks on a
    download (matches diskrot/modal_train.py)."""
    import nltk

    for res in ("averaged_perceptron_tagger_eng", "cmudict", "averaged_perceptron_tagger"):
        nltk.download(res, quiet=True)
    from g2p_en import G2p

    G2p()("warm up the cache")


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch>=2.4",
        "torchaudio>=2.4",
        "librosa>=0.10",
        "descript-audio-codec>=1.0.0",
        "numpy>=1.26",
        "tqdm>=4.66",
        "soundfile>=0.12",
        "msclap",
        "g2p_en==2.1.0",
        "demucs",
        "faster-whisper",
        "jiwer",
    )
    .run_commands("pip install 'protobuf>=4'")
    .run_function(_prefetch_g2p)
    .add_local_python_source("model", "diskrot")
)

tokens_vol = modal.Volume.from_name("nano-tokens")
ckpts_vol = modal.Volume.from_name("nano-ckpts")


def _slice_lyric(words: list[dict], max_words: int) -> str:
    """A contiguous ~max_words slice of a song's transcript — the reference line
    we condition on and then try to recover from the generated audio."""
    toks = [w["word"].strip() for w in words if w.get("word", "").strip()]
    return " ".join(toks[:max_words])


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes={"/tokens": tokens_vol, "/ckpts": ckpts_vol},
)
def eval_wer(
    ckpt_path: str = "/ckpts/v8_sing/best.pt",
    n_clips: int = 30,
    seconds: float = 12.0,
    cfg_scale: float = 3.0,
    lyric_cfg_scale: float = 0.0,  # 0 → unified guidance (no separate lyric axis)
    max_words: int = 12,
    seed: int = 42,
    whisper_size: str = "large-v3",
):
    import random
    import tempfile

    import soundfile as sf
    import torch

    from model.codec import DACodec
    from model.nano_audio_gpt import GPTConfig, NanoAudioGPT
    from model.text_encoder import CLAPTextEncoder
    from model.lyric_encoder import PAD_PHONEME_ID, text_to_phoneme_ids
    from diskrot.dataset import _load_lyrics, _load_tags
    from diskrot.transcribe_lyrics import (
        _load_demucs, _separate_vocals, _transcribe,
    )

    device = "cuda"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = GPTConfig(**ckpt["cfg"])
    if not model_cfg.use_lyric_conditioning:
        raise SystemExit(f"{ckpt_path} has no lyric conditioning — nothing to measure")

    model = NanoAudioGPT(model_cfg).to(device)
    state_dict = {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in ckpt["model"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    print(f"loaded {ckpt_path} — step={ckpt['step']} best_val_loss={ckpt['best_val_loss']:.4f}")

    text_encoder = None
    if model_cfg.use_text_conditioning:
        text_encoder = CLAPTextEncoder(d_out=model_cfg.d_model, device=device).to(device).eval()
        if "text_proj" in ckpt:
            text_encoder.proj.load_state_dict(ckpt["text_proj"])
        text_encoder._ensure_clap()

    codec = DACodec(device=device)

    # Sample lyric-bearing songs from the corpus for reference lines.
    lyrics = _load_lyrics("/tokens/lyrics", verbose=False)
    tags = _load_tags("/tokens/tags.json", verbose=False)
    names = [n for n, e in lyrics.items() if e.get("words")]
    random.Random(seed).shuffle(names)
    names = names[:n_clips]
    if not names:
        raise SystemExit("no lyric-bearing songs found under /tokens/lyrics")

    print("loading Demucs + faster-whisper for transcribe-back...")
    demucs_model, apply_fn = _load_demucs(device)
    from faster_whisper import WhisperModel

    whisper = WhisperModel(whisper_size, device=device, compute_type="float16")

    K = model_cfg.n_codebooks
    max_total = model_cfg.max_seq_len - K + 1
    new_frames = min(int(seconds * DACodec.FRAME_RATE_HZ), max_total - 1)
    cond_dtype = next(model.parameters()).dtype
    lyric_cfg = lyric_cfg_scale or None

    import jiwer

    refs, hyps, rows = [], [], []
    for i, name in enumerate(names):
        ref = _slice_lyric(lyrics[name]["words"], max_words)
        if not ref:
            continue
        tag_str = tags.get(name, "")

        tag_emb = None
        if text_encoder is not None and tag_str:
            tag_emb = text_encoder.encode([tag_str]).to(device).to(cond_dtype)
        ids = text_to_phoneme_ids(ref, max_len=model_cfg.max_lyric_len)
        lyric_ids = torch.tensor(ids, dtype=torch.long, device=device)[None]
        lyric_mask = lyric_ids != PAD_PHONEME_ID

        torch.manual_seed(seed + i)
        out = model.generate(
            prompt=None, num_new_frames=new_frames,
            temperature=[0.9, 0.9, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.3],
            top_k=50, top_p=0.95,
            text_emb=tag_emb, cfg_scale=cfg_scale,
            lyric_ids=lyric_ids, lyric_mask=lyric_mask, lyric_cfg_scale=lyric_cfg,
        )[:, 1:]  # drop the 1-frame random seed
        wav = codec.decode(out.cpu())
        if wav.dim() > 1:
            wav = wav.squeeze(0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
            sf.write(wav_path, wav.float().cpu().numpy(), codec.SAMPLE_RATE)
        vocals = _separate_vocals(demucs_model, apply_fn, wav_path, device)
        result = _transcribe(whisper, vocals)
        hyp = (result or {}).get("text", "")

        # jiwer normalizes case/punctuation/whitespace before scoring.
        norm = jiwer.Compose([
            jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
        ])
        r, h = norm(ref), norm(hyp)
        wer = jiwer.wer(r, h) if r else None
        if wer is not None:
            refs.append(r); hyps.append(h)
        rows.append((name, wer, ref, hyp))
        if i < 8:
            print(f"  [{name}] WER={wer if wer is None else f'{wer:.2f}'}")
            print(f"     ref: {ref}")
            print(f"     hyp: {hyp}")

    if not refs:
        raise SystemExit("no scorable clips (all references empty)")

    corpus_wer = jiwer.wer(refs, hyps)
    mean_wer = sum(w for _, w, _, _ in rows if w is not None) / len(refs)
    print()
    print(f"clips scored: {len(refs)}/{len(names)}  "
          f"(cfg_scale={cfg_scale}, lyric_cfg_scale={lyric_cfg_scale})")
    print(f"corpus WER (pooled) = {corpus_wer:.3f}")
    print(f"mean per-clip WER   = {mean_wer:.3f}")
    print()
    if mean_wer <= 0.5:
        print("→ words are largely intelligible. The lyric conditioner is working.")
    elif mean_wer <= 0.8:
        print("→ partial intelligibility — some words land. Try a higher lyric_cfg_scale, "
              "more training, or enable the monotonic-attention prior (Phase 2).")
    else:
        print("→ near gibberish (~baseline). Lyrics aren't being sung yet — more training "
              "or the Phase-2 monotonic-alignment prior is likely needed.")


@app.local_entrypoint()
def main(
    ckpt_path: str = "/ckpts/v8_sing/best.pt",
    n_clips: int = 30,
    cfg_scale: float = 3.0,
    lyric_cfg_scale: float = 0.0,
):
    eval_wer.remote(
        ckpt_path=ckpt_path, n_clips=n_clips,
        cfg_scale=cfg_scale, lyric_cfg_scale=lyric_cfg_scale,
    )
