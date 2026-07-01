# Unlimited-scale wave ingestion

Add songs to the nano corpus indefinitely without hitting the Modal-volume
~500k-inode cap. Per-song intermediates (mp3 / `.pt` / `.mel.npy`) are processed
in bounded **waves** and deleted after packing, so the only thing that grows is
the inode-cheap packed shard set (~1 file per 5k songs). Raw audio lives in
object storage (no inode cap). See the design in
`.claude/plans/i-think-v8-sing4-is-prancy-glacier.md`.

The same entrypoints still work in the legacy flat layout (bucket root) when you
omit `--wave-id` — waves are purely additive.

---

## One-time setup

### 1. Object storage (Cloudflare R2)

1. Cloudflare dashboard → **R2** → create bucket **`nano-audio`**.
2. **Manage R2 API Tokens** → create a token with **Object Read & Write** → copy
   the **Access Key ID** + **Secret Access Key**.
3. Create the Modal secret (keys must be named exactly this — R2 is S3-compatible):
   ```bash
   modal secret create r2-creds \
     AWS_ACCESS_KEY_ID=<key id> \
     AWS_SECRET_ACCESS_KEY=<secret>
   ```
4. Your endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` (R2 **requires** it).

### 2. Point the code at the bucket

R2 is the only audio source — set these in every shell you run/deploy from (the
mount and its env are resolved at `modal run`/`modal deploy` time):

```bash
export NANO_AUDIO_BUCKET=nano-audio
export NANO_AUDIO_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
```

`NANO_AUDIO_ENDPOINT` is **required** for R2. (The legacy `nano-corpus` Modal
Volume and the one-time Volume→R2 migration have been retired — everything is in
R2 now.)

### 3. Deploy the stage apps (only needed for the one-command orchestrator)

`modal run` apps are ephemeral and can't be looked up by name; `modal deploy`
registers them so `modal_ingest_wave.py` can call them. Deploy **with the R2 env
vars exported** so the mount resolves:

```bash
for m in prepare audio_dedup tokenize melody stems auto_tag transcribe filter_lyrics \
         align_lyrics structure tempo pack_cache wave_cleanup phonemize key_detect; do
    modal deploy diskrot/modal_$m.py
done
```

---

## Ingest a wave (~100k songs)

Upload the wave's mp3s under its bucket prefix, then run the wave. Keep waves
~100k so the heavy GPU cold-starts amortize, and run **one wave at a time**.

```bash
# upload (rclone configured for the R2 endpoint, or the Cloudflare UI / aws s3 cp)
rclone copy ./downloads/wave_17 r2:nano-audio/waves/wave_17/
```

### Option A — one command (after the deploy step above)

```bash
modal run --detach diskrot/modal_ingest_wave.py --wave-id 17
```

Runs `prepare(+quality-gate) → audio_dedup → tokenize → melody → stems → auto_tag
→ transcribe → filter_lyrics → align_lyrics → structure → tempo → pack_append →
cleanup → phonemize → key_detect`, each blocking until done (so only one GPU stage
runs at a time — respects a 50-GPU cap automatically). Resumable: a re-run with the
same `--wave-id` skips stages already marked done in
`/tokens/waves/wave_17/status.json`, and each stage also resumes mid-stage by
skip-by-existence. Toggle streams with `--no-with-melody` / `--no-with-tags` /
`--no-with-lyrics` / `--no-with-structure` / `--no-with-tempo`. **Stems are default
OFF** (`--with-stems` to enable the GPU `/addstem` conditioning stage). Leave the raw
mp3s in R2 (no inode cap, and they're the re-derivation source) — `--drop-mp3` deletes
a wave's source audio from the bucket and is only for reclaiming space you don't need.

### Option B — per-stage (no deploy needed)

```bash
modal run --detach diskrot/modal_prepare.py    --wave-id 17 --apply
modal run --detach diskrot/modal_tokenize.py   --wave-id 17
modal run --detach diskrot/modal_melody.py     --wave-id 17
modal run --detach diskrot/modal_auto_tag.py   --wave-id 17
modal run --detach diskrot/modal_transcribe.py --wave-id 17
modal run --detach diskrot/modal_filter_lyrics.py --apply        # after transcribe
modal run --detach diskrot/modal_structure.py  --wave-id 17
modal run --detach diskrot/modal_pack_cache.py --append --wave-id 17
modal run         diskrot/modal_wave_cleanup.py --wave-id 17 --apply   # dry-run without --apply
modal run --detach diskrot/modal_phonemize.py                    # global, after transcribe
modal run --detach diskrot/modal_key_detect.py                   # global, after pack
```

Run GPU stages (tokenize/auto_tag/transcribe/structure) **one at a time** to stay
under the 50-GPU cap. `cleanup` refuses to run until a packed index exists, so it
can never discard un-folded tokens. Repeat for wave 18, 19, … — cleanup returns
the loose-file count to baseline each time, so the corpus grows without bound.

---

## Train on the grown corpus

The waves append to the same `packed/`, so training just sees more songs — no
flags. Continued-pretrain from the current model (anneals **and** learns the new
data in one run; judge by listening, not val loss — the split is a fresh
baseline):

```bash
modal run --detach diskrot/modal_train.py \
    --init-from v9_stereo/best.pt --ckpt-subdir v9_genre \
    --lr 5e-5 --warmup-steps 2000 --steps 150000 --n-gpus 4
```

(The multi-GPU path is hardwired to **4×B200** — `--n-gpus 4` is the default; `--n-gpus 8`
fails the device-count assert.)

---

## Phase-2 (only at multi-million songs)

`tags.json` / `keys.json` are single files. Below ~1M songs they're fine. When
they get large, shard them (the dataset reads either layout):

```bash
modal run diskrot/modal_shard_stores.py        # tags.json/keys.json -> tags/ keys/ shards
```

**Not yet done** (Modal-side perf, irrelevant at <1M; tracked as follow-ups):
auto_tag/key_detect emitting sharded stores *directly* (O(touched) flush — the
converter above bridges it), and a persisted CLAP embedding cache so startup
encodes only new tags.

---

## Notes

- Everything is resumable and idempotent — re-run the same `--wave-id` to resume.
- Tested locally (`tests/test_pack_append.py`, `tests/test_sharded_store.py`,
  276 passing); the Modal pipeline itself should be validated with a small
  (~500-song) smoke wave before a full 100k one.
- The append invariant (prior shards stay byte-identical; the existing pack — a
  ~427k-song snapshot at the time this was written — is never repacked) is the guard
  that makes growth cheap and safe.
