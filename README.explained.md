# nano — explained for engineers who haven't done ML

This is the "what's actually happening in here" companion to [README.md](README.md). It assumes you can read Python, know what a hash table and a tree are, understand floating-point and SIMD, and have heard of gradient descent in the abstract — but it does **not** assume you've ever trained a model or know what a transformer is. Every ML term is introduced in CS analogues the first time it appears.

If you want to skip straight to running things, the main README has the commands. If you want to know *why* those commands exist and what the model is doing internally, read on.

---

## 1. The one-paragraph version

Audio is just a stream of floating-point samples — 44,100 of them per second for CD-quality mono. That's far too high-bandwidth for any practical sequence model to consume directly, so we first run it through a **learned compressor** (a pre-trained network called DAC, treated as a black box here) that turns it into 9 parallel streams of integers at 86 Hz each. Now an audio clip is a `[9, T]` integer matrix instead of `[44100 × seconds]` floats. We then train a **next-token predictor** — same family of model as GPT for text, just predicting 9 integers per step instead of 1 — to learn the distribution of "what comes next" in those integer streams. At inference we sample from that distribution one frame at a time and feed the result back into DAC to get audio out. Optionally, we condition the predictor on a text description ("ambient, slow, piano") encoded by another pre-trained model called CLAP, so the same network can be steered.

That's the whole project. Everything below is plumbing.

## 2. Mental model: audio is just integers

The single biggest hurdle when switching from CS to ML is realizing that *all of this — the model, the loss, the training loop — is operating on integer sequences, not on audio*. The audio↔integer translation is done by an off-the-shelf component we don't train, called the **codec**.

In [model/codec.py](model/codec.py) you'll find `DACodec`. Think of it as a lossy `(encoder, decoder)` pair, like libvorbis but learned from data instead of hand-engineered:

```
encode:  float[1, samples] @ 44100 Hz  →  int[9, frames] @ 86 Hz, each int in [0, 1023]
decode:  int[9, frames]                →  float[1, samples] @ 44100 Hz
```

- **9** streams (called *codebooks*). Stream 0 captures the coarsest information (think: rough envelope, fundamental frequency), stream 8 captures the finest residual detail. Each stream is a separate sequence of integers.
- **1024 distinct values** per stream (the *vocab size*). Each integer is an index into a learned codebook of 1024 short audio vectors that the DAC decoder knows how to assemble back into a waveform.
- **86 frames per second** of audio. So a 10-second clip is a `[9, 860]` integer tensor — 7,740 integers total to represent ~441,000 audio samples. That's the ~57× compression ratio that makes this whole project tractable.

Constants live at [model/codec.py:20-23](model/codec.py#L20-L23). Once you've internalized that audio = `[9, T]` int matrix, the rest of the project is "build a model that predicts the next column of that matrix."

## 3. The whole pipeline

```
MP3 corpus ──tokenize──> .pt files of [9, T] int tensors
              │
              ├─ auto_tag    ──> tags.json    ({song_name: "a mellow lo-fi beat with soft piano and vinyl crackle"})
              └─ transcribe  ──> lyrics/      (sharded lyrics_NNN.json; {song_name: {text, words:[{word, start, end}, ...]}})
                                                       │
                                                       ▼
                                                  train loop
                                                       │
                                                       ▼
                                                checkpoint .pt
                                                       │
                                                       ▼
                                                 inference server
                                                       │
                                                       ▼
                                                   audio out
```

Each arrow corresponds to one Python module under [diskrot/](diskrot/). The training loop is the only step that involves backprop / gradient descent; everything else either (a) calls a pre-trained model to convert one representation to another, or (b) shuffles files around.

## 4. Stage-by-stage walkthrough

### 4a. Tokenize ([diskrot/tokenize.py](diskrot/tokenize.py))

Walks a directory of `.mp3` files, loads each one with `librosa` (a wrapper around ffmpeg), resamples to 44.1 kHz mono, runs it through DAC, and saves the resulting `[9, T]` int16 tensor as a `.pt` file (PyTorch's own pickle format). One-to-one mapping: `song.mp3` → `song.pt`. The cache is compact (~140 KB per song on average) and is the input to training.

The recent additions (`tokenize_files_streaming`, batched `encode_batch`) are throughput optimizations — they let DAC process several files in one GPU forward pass while a background thread pre-loads the next chunk's audio from disk. Conceptually it's still "for each mp3, write an int tensor."

### 4b. Auto-tag ([diskrot/auto_tag.py](diskrot/auto_tag.py)) — optional

Runs a pre-trained audio **captioner** (LP-MusicCaps — a BART model fed log-mel spectrograms, vendored in [model/captioner.py](model/captioner.py)) over each song and writes a free-form natural-language description — e.g. "a mellow lo-fi hip hop beat with soft piano and vinyl crackle" — to `tags.json`, keyed by song name. Pure inference, no training.

> **Two different models, two different jobs — don't conflate them.** *This* stage (LP-MusicCaps) **generates the description text**. A separate model, **CLAP** (covered under *Text conditioning* below), later **encodes that text** into the vector the transformer actually conditions on. Swapping the captioner changes *what the descriptions say*; CLAP is unchanged and still does the encoding. (An earlier version used CLAP zero-shot classification against a fixed label vocabulary to produce the tags — that was replaced by LP-MusicCaps captions, but CLAP-the-encoder stayed.)

### 4c. Transcribe ([diskrot/transcribe_lyrics.py](diskrot/transcribe_lyrics.py)) — optional

Two pretrained models in a pipeline: **Demucs** separates the vocal track from the instrumental, then **Whisper** transcribes the vocals with per-word timestamps. Output is a sharded `lyrics/` dir (`lyrics_NNN.json`, 256 shards keyed by a stable hash of the song name), each entry `{text, words: [{word, start, end}, ...]}`. The timestamps are used at training time to window the lyrics to the 30s crop; only the resulting text string is then encoded by CLAP (the model has no per-word alignment mechanism).

### 4d. Train ([diskrot/train.py](diskrot/train.py))

This is the only stage where we actually run an optimizer. The structure is straight out of any deep learning textbook:

```
for step in range(0, num_steps):
    batch = sample_random_10s_clips_from_corpus()      # [B, 9, 860] ints
    inputs, targets = build_train_inputs(batch)        # teacher-forced shifts
    predictions = model(inputs)                        # [B, 9, T, 1025] floats
    loss = cross_entropy(predictions, targets)         # scalar
    loss.backward()                                    # autograd populates .grad on every param
    optimizer.step()                                   # AdamW: w -= lr * adjusted_grad
    optimizer.zero_grad()
```

A few CS-friendly notes:

- **Cross-entropy** is just negative log-likelihood of the correct class under the model's predicted distribution. If the model is predicting integer `k_true` and assigns probability `p_k` to each integer `k`, the loss is `-log(p_{k_true})`. Average over all positions in the batch. Lower is better; perfect prediction is 0; uniform random over 1024 options is `log(1024) = 6.93`. The "loss is going down" updates you see in the training log are this scalar shrinking.
- **AdamW** is a variant of gradient descent with per-parameter adaptive step sizes, similar in spirit to Newton's method but using only first-order information plus running estimates of the gradient's mean and variance. From a CS standpoint: stateful optimizer, one tensor of state per model parameter, O(model size) memory overhead.
- **Cosine LR schedule with warmup** — the learning rate (step size) is ramped linearly from 0 to a peak over the first `warmup_steps` (1,500 local / 5,000 on Modal), then decayed along a half-cosine to 0 over the remaining steps. This is a standard recipe. See `_cosine_lr` at [diskrot/train.py:160](diskrot/train.py#L160).
- **Batch** = stack many independent training examples into one big tensor and process them in parallel. SIMD-style — same operations, different data. Bigger batch = more stable gradient estimate, more GPU memory, sometimes better throughput.
- **Validation** = once every 1,000 steps, run the model over a held-out 10% of the songs (the model never trains on these) and compute the same loss. Used to detect overfitting and to decide when to stop. See `_evaluate` at [diskrot/train.py:240](diskrot/train.py#L240) and the early-stopping logic at [diskrot/train.py:441](diskrot/train.py#L441).

### 4e. Inference ([server/inference.py](server/inference.py) and [server/main.py](server/main.py))

A FastAPI HTTP server that loads a trained checkpoint and exposes endpoints for generating audio. The inference loop is:

```
tokens = []                              # start with empty context (or a prompt audio's tokens)
for step in range(num_frames):
    logits = model(tokens)               # [9, 1025] floats — one prediction per codebook
    next_frame = sample(logits)          # [9] ints — one int per codebook
    tokens.append(next_frame)
output_audio = dac.decode(tokens)        # back to a waveform
```

Sampling means drawing from the predicted distribution (with temperature / top-k tweaks to control how "creative" vs "safe" the output is — see the temperature/top_k/top_p params in [server/inference.py](server/inference.py)).

The catch: re-running the whole model on the entire history every step is O(T²) work in total. The **KV cache** (see `StaticLayerKVCache` at [model/nano_audio_gpt.py:20](model/nano_audio_gpt.py#L20)) memoizes the per-step intermediate keys and values inside the attention layers, dropping the per-step cost back to O(T) work. This is exactly the kind of optimization you'd reach for in any sequence-processing system; it's just expressed as tensor ops.

## 5. The model itself

Defined in [model/nano_audio_gpt.py](model/nano_audio_gpt.py). It's a **transformer** — a specific neural network architecture that's become the default for sequence modeling since 2017. Here's the CS-flavored breakdown:

### Layout

```
input tokens [B, 9, T]
    │
    ├── per-codebook embedding lookups  (9 separate hash-tables-of-vectors,
    │   each int → float[d_model], d_model=2048)
    │                                                       
    ├── sum the 9 vectors per position  (now [B, T, d_model])
    │
    ├── N transformer blocks, each containing:
    │   ├── causal self-attention with RoPE  (rotary positional encoding —
    │   │   instead of a learned vector added at the input, position is
    │   │   baked into Q/K via a rotation that depends on the token's
    │   │   index; lets the same weights handle longer sequences than
    │   │   anything they trained on, which is how nano generates ~95s from
    │   │   a model that only ever saw 30s segments)
    │   ├── (optional) cross-attention to text/audio conditioning
    │   └── feedforward MLP (2 linear layers with a nonlinearity, d_model → d_ff → d_model)
    │
    ├── final layer norm
    │
    └── 9 output heads (each: d_model → 1025 logits, one per codebook)
```

The exact shape comes from `GPTConfig` ([model/nano_audio_gpt.py](model/nano_audio_gpt.py)) — the model is ~1.5B params (d_model=2048, 22 layers, 16 heads, d_ff=8192, 30s segments). For comparison, GPT-2 small is 124M and modern LLMs are 100B+; nano is ~12× GPT-2 small but still small by language-model standards.

### Attention, briefly

The "attention" part is what makes a transformer a transformer. Given a sequence of vectors, attention is a *learned, content-addressable lookup* over the sequence: each position computes a "query" vector and looks up "key/value" pairs from all earlier positions, weighted by a softmax over query·key dot products. From a CS standpoint it's a soft, differentiable version of an associative array lookup. The actual math is a few `nn.Linear` projections and one `F.scaled_dot_product_attention` call — see [model/nano_audio_gpt.py:61](model/nano_audio_gpt.py#L61).

"Causal" means each position can only attend to itself and earlier positions, never later ones. This is what makes the model usable for next-token prediction — it can't cheat by peeking ahead.

### The delay pattern (the only really novel bit)

Predicting 9 integers per frame in parallel has a chicken-and-egg problem: codebook 1's value at a given frame depends on codebook 0's value at the *same* frame (since they jointly describe the same audio). A pure parallel head can't see that dependency.

The fix (borrowed from Meta's MusicGen paper) is implemented in [model/delay_pattern.py](model/delay_pattern.py): **shift codebook `k` rightward by `k` positions**, padding the empty slots with a sentinel. After the shift, when the model is predicting position `p`:

- it has already produced codebook 0 at position `p-0`
- it has already produced codebook 1 at position `p-1` (which corresponds to *original* frame `p-1`)
- ...
- so codebook `k` at original frame `t` lives at delayed position `t+k`, *after* codebooks `0..k-1` at the same original frame `t`, which sit at positions `t..t+k-1`

It's a topological-sort trick: by relabeling positions, we serialize the cross-codebook dependencies into the same left-to-right order that causal attention already respects, while still letting the 9 heads emit predictions in parallel. After generation, we undo the shift (`revert_delay`) and hand the un-delayed tokens to DAC. The cost is `K-1 = 8` extra timesteps of padding per sequence.

### Text conditioning (optional)

When enabled, CLAP encodes the tag string and lyric string for each training clip into two 1024-dim vectors, a tiny learned linear layer projects them to `d_model` (1024), and each transformer block gets an extra **cross-attention** sublayer that lets the audio positions look up information from these text vectors. CLAP itself is frozen — we never train through it; only the 1024→`d_model` projection and the cross-attention weights are learned. See `CLAPTextEncoder` in [model/text_encoder.py](model/text_encoder.py) and `CrossAttention` in [model/nano_audio_gpt.py:98](model/nano_audio_gpt.py#L98).

This is the architectural mechanism behind "give me ambient electronic" working at inference time: the tag string is encoded the same way it was during training, the cross-attention layers steer the audio prediction toward regions of the distribution where similar conditioning was observed.

## 6. Reading the training log — what's good, what's bad, when to intervene

The training loop prints one line every 25 steps. There are two flavors: regular training-step lines and validation lines (every 1,000 steps). Below is a field-by-field decoder followed by a guide to interpreting trajectories.

> The sample log lines in this section are from an earlier ~287M-era run (note the `/125000` step totals and the ~7.5M `tok/s`), kept because they're good for teaching how to *read* a line. The current 1.5B model runs to `/400000` steps and sustains ~2.7M aggregate `tok/s` on 8×H100 — so treat the throughput and step-count numbers here as illustrative, not current.

### Training-step lines

```
step  4800/125000  loss 5.9893  lr 3.00e-04  tok/s 7518.7k  cb[4.27 5.38 5.74 5.98 6.12 6.25 6.36 6.46 6.63]
```

#### `step N/total`

Current gradient-descent iteration number. The total is 125,000 steps by default locally and 400,000 on Modal. There's nothing to read into this field other than progress.

#### `loss <float>`

Total cross-entropy averaged over the last 25 training batches. The reference values to keep in your head:

| Loss | What it means |
|---|---|
| **6.93** | `log(1024)` — the uniform-random baseline. A randomly initialized model starts roughly here. |
| **6.0** | Model has learned the unigram statistics — which tokens are common in the corpus. Usually reached by step ~1k. |
| **5.5** | Has learned short-range structure. Healthy runs reach this by step ~10–20k. |
| **4.5–5.0** | Knows local audio texture — can produce something recognizably musical. Step ~50k+. |
| **<3.5** | Plateau ceiling for this model size and corpus. |

**Healthy trajectory:** smooth, monotonic decrease that's steepest in the first 5–10% of steps and slows asymptotically. Step-to-step jitter of ~0.05 nats is normal — the loss is a moving window over stochastic batches.

**Warning signs and what to do:**

| Symptom | Likely cause | What to try |
|---|---|---|
| Loss stuck at ≥6.8 for >2,000 steps | Data pipeline is broken — model is seeing pad tokens or wrong-shape input | Inspect the first batch's `tokens` shape and dtype; check the dataset is loading non-empty `.pt` files |
| Loss → `NaN` or jumps to >10 | Exploding gradients — LR too high for the batch size, or warmup skipped | Restart with `--lr` cut in half and confirm `warmup_steps` ≥ 1000 |
| Loss plateaus before step ~5k at a high value (≥6.0) | Underfitting (model too small) or corpus too small | Bigger model, or more training data |
| Loss flatlines mid-training for 200–500 steps | Often transient — recovers on its own | Wait. If it persists past a full eval cycle (1,000 steps), suspect the cosine schedule has decayed too far |
| Loss visibly drops then *rises* | Either eval-only artifact (it's a single noisy snapshot) or LR genuinely too high causing oscillation | If training loss (not val) is rising, lower the LR |

#### `lr <float>`

Current learning rate, set by the cosine schedule. With the local defaults (`warmup_steps=1500`, `steps=125000`; Modal uses `warmup_steps=5000`, `steps=400000`):

- **Steps 0 → warmup_steps**: linear ramp from ~0 to peak. The number should be growing each step.
- **After warmup**: half-cosine decay from peak to 0. The number should be shrinking on a smooth curve, very slowly at first, then faster, then slowly again as it approaches 0.

This field is mostly a sanity check — if the lr is 0 or unchanging when you expect it to be changing, the schedule was misconfigured. The peak value is what you set with `--lr` on the CLI.

#### `tok/s <float>`

Aggregate throughput across all GPUs and all 9 codebooks (`batch_size × world_size × n_codebooks × seq_len` per step, divided by step time). This is a **regression detector**, not a quality metric — it tells you whether the GPU is being kept busy.

Rough expectations for this 1.5B-parameter model (the live 8×H100 run sustains ~2.7M tok/s):

| Hardware | Expected `tok/s` |
|---|---|
| H100 single-GPU, batch=64 | ~400–500k |
| 8×H100 DDP, per-rank batch=8 | ~2.5–3M (observed) |
| RTX 5090 / M4 Max | only viable on a shrunk `GPTConfig` — the full 1.5B won't fit / is impractically slow locally |

A sustained 2× drop without a hardware or batch-size change usually means I/O contention (dataloader workers starved), thermal throttling, or another process competing for the GPU. Short dips at eval steps (every 1,000) and checkpoint steps (every 5,000) are normal.

#### `cb[v0 v1 ... v8]`

Per-codebook cross-entropy. This is the **most diagnostically useful field** on the line — it tells you *which parts of the audio* the model is and isn't learning.

**Expected shape — monotonic increase from coarse to fine:**

```
cb[3.96 4.93 5.30 5.48 5.52 5.52 5.54 5.56 5.63]
    ▲                                           ▲
    cb[0]: coarse, learns first                cb[8]: fine, learns last
```

Codebook 0 captures the rough envelope (loud/quiet, basic pitch contour) and is the easiest to predict from context. Codebook 8 captures fine residual detail (timbral nuance, high-frequency texture) and is the hardest. The "staircase" of values from low to high is the canonical healthy shape.

**Reading the shape:**

| Pattern | What it means |
|---|---|
| Wide spread (cb[0]≈3.5, cb[8]≈6.5) early in training | Normal — coarse codebooks have a head start because their distribution is more structured |
| Tight cluster across cb[3..8] in mid-training | Normal — these residual codebooks model similar-difficulty information and tend to move together |
| cb[8] still near 6.93 past step ~10k | Late codebooks aren't learning — usually a model-capacity ceiling on fine detail. Either accept lower-quality high frequencies or train a bigger model |
| cb[0] below ~3.5 with the rest stuck above 5.5 | Model has overcommitted to the easy task. Fine in moderation; if extreme, may indicate a loss-masking bug |
| Persistent non-monotonic pattern (e.g., cb[2] > cb[3]) | Rare. Small inversions (≤0.05) are noise; persistent ≥0.1-nat inversions suggest a bug in the delay pattern or loss masking |
| The whole list shifts down in lockstep eval-over-eval | The best kind of progress — every codebook is still extracting signal |
| Only cb[0..2] move; cb[3..8] frozen | Coarse heads still have capacity; fine heads have plateaued. Common in late training; not actionable unless persistent for >20k steps |

### Validation lines

Every 1,000 steps the loop runs the model on a held-out 10% of songs (the model never trains on these) and prints a `VAL` line:

```
  VAL  step 19000  loss 5.3656  cb[4.03 5.04 5.40 5.59 5.61 5.62 5.62 5.65 5.72]  ★ new best
  VAL  step 20000  loss 5.3702  cb[4.04 5.05 5.39 5.59 5.62 5.63 5.63 5.66 5.74]  (1/15 no improvement)
```

Same fields as the training-step line, plus an outcome marker.

#### `★ new best`

Val loss beat all previous evals. The model is saved to `best.pt` and the early-stopping patience counter is reset to 0. **This is the checkpoint you want for inference** — it's the snapshot that generalized best.

#### `(n/15 no improvement)`

This eval didn't beat the all-time best. The counter increments by 1. When it reaches 15 (≈15,000 steps without improvement), training stops automatically.

### Train-vs-val: the generalization read

The single most informative cross-check is **comparing the most recent training loss to the most recent val loss**.

| Train vs val gap | What it means | Action |
|---|---|---|
| Within ~0.2 nats | Healthy — model is generalizing | None |
| Val higher by 0.3–0.5 nats | Mild overfitting starting | Tolerable. Watch the patience counter |
| Val higher by ≥0.5 nats and growing | Real overfitting — model is memorizing training songs | Add more songs, or stop early and use `best.pt` |
| Val < train | Normal early in training (train is a rolling avg of recent harder batches; val is a single fresh snapshot) | None |
| Both flat for 5+ evals | Hit the model's ceiling for this corpus | Consider stopping — `best.pt` is likely close to the best you'll get |

### The patience counter as a decision tool

`n/15 no improvement` is a built-in early-stopping signal. How to act on it:

| Counter | What to do |
|---|---|
| 0–4 / 15 | Ignore. Val loss is genuinely noisy at the ±0.05 nat level — `eval_batches=26` is a finite sample |
| 5–9 / 15 | Check whether training loss is still dropping. If yes, fine. If both are flat, you've hit the data/model ceiling |
| 10–14 / 15 | Run is probably about to early-stop. `best.pt` is what you'll want |
| 15 / 15 | Training stops. `best.pt` = best generalization; `latest.pt` = state at stop time (use `best.pt` for inference) |

### A worked example

Take the line at the top of this section:

```
step  26850/125000  loss 5.2948  lr 2.71e-04  tok/s 7565.9k  cb[3.96 4.93 5.30 5.48 5.52 5.52 5.54 5.56 5.63]
```

Reading it:

- **21% through training** — plenty of runway. No early-stage concerns about underfitting yet.
- **loss 5.29** — well past the 5.5 milestone, on track for the 4.5–5.0 band by step ~60k. Healthy.
- **lr 2.71e-04** — past warmup, partway down the cosine decay. The fact that this is ~90% of the peak means we were launched with peak `lr ≈ 3e-4` (matching the current Modal default of 3.0e-4; the local default is 2.5e-4).
- **tok/s 7.6M** — healthy for the smaller model this sample is from; for the current 1.5B model the comparable aggregate is ~2.7M on 8×H100. Either way, a sustained drop with no batch/hardware change signals a perf regression.
- **cb[3.96 → 5.63]** — spread of 1.67 nats, narrowing from earlier in training. Late codebooks (cb[6..8] = 5.54–5.63) are within 0.1 nats of the middle codebooks (cb[4..5] = 5.52), so the fine heads are catching up. The cluster at cb[4..7] = 5.52–5.56 is the expected "moving as a block" pattern.

**Diagnosis: healthy mid-training, no intervention needed.** Expect val loss to keep dropping for another 10–20k steps before the patience counter starts ticking in earnest.

### A pathological example, for contrast

```
step  3200/125000  loss 6.87  lr 6.40e-04  tok/s 7400k  cb[6.85 6.88 6.91 6.92 6.92 6.93 6.93 6.93 6.93]
```

Almost everything is wrong here:
- **Loss 6.87** — barely below the 6.93 random baseline at step 3,200. By this point we should be at ~6.0 or below.
- **cb[..]** — every codebook is glued to the random baseline, including cb[0]. The model has learned essentially nothing.
- **lr 6.40e-04** is suspiciously high — past the expected peak. Implies warmup was misconfigured.

The likely cause is either a broken data pipeline (the model is being fed nothing but pad tokens) or a runaway LR with no warmup that destroyed any structure the model had begun to learn. The fix is to inspect the first training batch's contents and re-confirm the warmup schedule. The `best.pt` from this run will be useless; restart from scratch.

## 7. Inference-time knobs — tuning the sampler

Training defines *what the model knows*. Inference decides *how to draw from it*. The four parameters exposed by the HTTP endpoints all act on the per-step logits before sampling — together they're the difference between coherent output and noise even when the model itself is held fixed.

A useful mental model: at each step the model produces a categorical distribution over 1,025 possible next tokens per codebook (the 1024 DAC tokens plus a pad sentinel). The sampler's job is to draw one token from that distribution. Every knob below modifies the distribution before the draw.

### `temperature` (default `0.9`)

Logits are divided by `temperature` before the softmax. Lower temperature = sharper distribution = more deterministic; higher temperature = flatter distribution = more diverse and more error-prone.

- `0.0` — argmax (deterministic, identical output for the same prompt every time).
- `0.7–0.9` — typical for audio. Diverse but tight enough to stay on-distribution.
- `1.0` — raw model distribution, unscaled.
- `>1.0` — actively flattens the distribution, sampling more low-probability tokens. Almost always produces noise for a not-yet-fully-converged model.

**Symptom of too high:** output sounds noisy, incoherent, drifts off into garbage. **Symptom of too low:** output is repetitive, loops on the same texture.

### `top_k` (default `50`)

After temperature scaling, keep only the `top_k` most-probable tokens per step and set the rest to `-inf` (probability 0). With a 1,025-vocab codebook, `top_k=50` keeps the top ~5% — a good starting point that matches MusicGen / AudioGen.

The old default of `250` was permissive enough that ~24% of the vocab could be sampled per step, which compounds across codebooks and across hundreds of frames into noise. Set this lower (`20–80`) for more focused output, `None` to disable.

**Symptom of too high:** generic-sounding output, model drifts away from prompt. **Symptom of too low:** output is too rigid, especially for late codebooks which legitimately have broader distributions.

### `top_p` / nucleus sampling (default `0.95`)

After temperature scaling, keep the smallest set of tokens whose cumulative probability mass exceeds `top_p`, drop the rest. Applied **before** `top_k` (so the two stack: top_p first, then top_k on what remains).

The advantage over `top_k` alone is adaptive size: when the model is confident, only 2–3 tokens get through; when it's uncertain, a much wider set is allowed. This produces more natural-feeling output than a fixed top-k cutoff.

`0.9–0.95` is the sweet spot for audio. `1.0` (or unset) disables it.

### `cfg_scale` — classifier-free guidance (default `3.0` when text is provided)

The most impactful knob if you're using text conditioning, and the one most worth understanding. During training the model sees its text conditioning dropped to `None` 10% of the time ([diskrot/train.py:57](diskrot/train.py#L57)) — meaning it learns both `p(next_token | history, text)` *and* `p(next_token | history)` from the same weights. At inference we can exploit both:

```
logits_blended = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
```

That's an extrapolation: the difference `(cond - uncond)` is the direction in logit space that the text prompt is pulling the distribution; scaling it >1 amplifies that pull. The cost is one extra forward pass per step (2× compute on conditioned generation), bought via a parallel KV cache.

| `cfg_scale` | Effect |
|---|---|
| `1.0` | Disabled — single forward pass, original behavior. Use when generating without text. |
| `2.0` | Mild steering. Prompt influences output but is not dominant. |
| `3.0` | Standard. Output clearly tracks the prompt. Default for text-conditioned calls. |
| `5.0–7.5` | Aggressive. Useful for niche / under-represented prompts. |
| `>10.0` | Over-steered. Output collapses to a stereotype, loses musical coherence. |

**Symptom of too low:** the text prompt makes no audible difference between calls. **Symptom of too high:** all outputs for a given prompt sound nearly identical, like a frozen one-bar loop.

### Putting it together — recommended starting points

The new defaults baked into the HTTP endpoints are tuned for the val-loss-5.30 regime:

```
temperature=0.9   top_k=50   top_p=0.95   cfg_scale=3.0 (when text provided)
```

If output sounds noisy → drop `top_k` to `30` and `temperature` to `0.7`.
If output sounds repetitive → raise `top_k` to `100` and `temperature` to `1.0`.
If the text prompt seems ignored → raise `cfg_scale` to `5.0`.
If all outputs collapse to one texture → drop `cfg_scale` to `2.0`.

### A note on seeding from scratch

`/generate` (no audio prompt) has to bootstrap the autoregressive loop with *some* starting token per codebook. It seeds with a single column of **random DAC tokens** (a fresh draw per call); tight sampling + CFG then drive the model to a coherent trajectory regardless of seed energy.

**Why random.** This took iterating to figure out. The original code seeded with one random token and produced pure noise because the old sampling was too loose; the noise compounded across hundreds of autoregressive steps. An early fix was *silence-seeding* with ~1s (86 frames) of encoded digital silence, which worked beautifully for lo-fi prompts but produced **complete silence** for high-energy ones — a "hip-hop, intense, distorted" prompt couldn't escape because the causal self-attention path (where the audio seed lives) is structurally stronger than the cross-attention path (where the text conditioning lives), and there's essentially no training example of "absolute silence → drum hit in one frame," so the model stayed silent regardless of `cfg_scale`. Once sampling was tightened, a single random column produces coherent output across **all** prompt characters, so the silence-seed mode was removed — it only ever worked for quiet prompts and silently failed for energetic ones.

### Tuning `cfg_scale` in practice

The README table earlier in this section lists `cfg_scale=3.0` as the standard default. Empirically, at val loss 5.30 the cond/uncond distributions aren't yet very distinct, so the extrapolation needs to be more aggressive than the table suggests. **`cfg_scale=5.0` is the better starting point for this checkpoint**; revisit `3.0` once training has driven val loss lower (the more separated the distributions become, the less amplification you need to express the prompt).

## 8. Where to look next

- The model itself is ~340 lines: [model/nano_audio_gpt.py](model/nano_audio_gpt.py).
- The training loop is the most "ML-y" file: [diskrot/train.py](diskrot/train.py).
- The delay-pattern trick fits on one page: [model/delay_pattern.py](model/delay_pattern.py).
- The codec wrapper makes clear what we treat as a black box: [model/codec.py](model/codec.py).
- The inference engine, including CFG and random seeding: [server/inference.py](server/inference.py).
- The unit tests in [tests/](tests/) are good entry points if you want to see the building blocks exercised on tiny inputs — they double as worked examples of the data shapes flowing through each module.

Once you've read those files, you've read essentially the whole project; the rest is data plumbing, Modal orchestration, and the HTTP server.
