# Understanding training log output

Reference for reading the logs from `modal run --detach diskrot/modal_train.py [--n-gpus 4]`. Covers the phase ordering, what each log prefix means, healthy throughput numbers at scale, and how to diagnose silent stalls.

## Monitoring commands

```bash
# Find the live training app id
modal app list | grep -E "nano-train.*ephemeral"

# Stream logs (substitute the app id)
modal app logs ap-... -f

# One-shot snapshot of recent logs
modal app logs ap-... | tail -50

# Stop a run
modal app stop ap-... -y
```

```bash
# Peek at GPU utilization mid-run (substitute the app id)
modal container exec --no-pty \
  $(modal container list 2>&1 | awk '/ap-.../{print $2}') \
  -- nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

## Phase order (8×H100 DDP)

The DDP path ([modal_train.py](diskrot/modal_train.py)) preloads everything once in the parent process, then `mp.spawn`s 8 rank workers that share the parent's memory. Single-GPU runs skip the parent phases — they just call `train_run` directly inside the container.

The dataset is loaded one of two ways depending on whether the `packed/` mmap dir exists:

- **Sharded mmap** (`packed/packed_index.json` present — the production path at scale): `load_mmap_bundle` just opens each shard as an `np.memmap` and reads the JSON sidecars (seconds, not minutes — the heavy lifting was done by the offline pack step).
- **In-RAM bundle** (no `packed/` — only viable on a tiny smoke-test cache): `load_shared_bundle` parallel-loads every `.pt` and concatenates into one shared-memory tensor.

Timings below are for the production sharded-mmap path at scale on 8×H100.

| # | Phase | Log marker | What's happening | Typical wall-clock |
|---|---|---|---|---|
| 1 | Parent: announce | `[parent] v2 sharded layout detected ... using mmap bundle` | Orchestrator starting; picks the mmap path | < 1s |
| 2 | Parent: split | `[bundle] N train + M val .pt files in /tokens` | Listing + train/val split | < 5s |
| 3 | Parent: train load | shard index read | mmap open (no per-file `torch.load`) | < 5s |
| 4 | Parent: val load | shard index read | Same, smaller split | < 5s |
| 5 | Parent: pack | skipped (packing was done offline by the pack step) | — | 0 |
| 6 | Parent: tags/lyrics | `[tags] loaded NNNNNN entries from /tokens/tags.json` | JSON read | ~30s |
| 7 | Parent: CLAP precompute | `[clap-parent] sharding NNNNNN tags across 8 GPUs (batch=512, bf16)...` + per-GPU `[clap-gpu*]` progress | One-time CLAP text encoding, sharded across all GPUs | ~90 s (measured 2026-06-12: 241,603 tags in 89s) |
| 8 | Workers spawn | `using preloaded shared bundle` + `loaded N precomputed CLAP tag embeddings` | All 8 ranks confirm they inherited the parent's mmap | < 5s |
| 9 | Workers: setup | `segment_frames=2580`, `DDP active: world_size=8, per-rank batch=8`, `model: 1514.XX M params`, `torch.compile enabled` | NCCL init, model on GPU, torch.compile | ~2 min (large graph) |
| 10 | Training loop | `step 25/400000 loss X.XXXX lr X.XXe-XX tok/s XX.Xk cb[X.XX X.XX ...]` every 25 steps | Real training | ~30–50 h |
| 11 | Validation | `checkup at step N score X.XXXX ...` every 1000 steps | Eval pass over val_loader | ~30s per checkup |
| 12 | Checkpointing | `saved ckpt -> /ckpts/v7_1500m/step_NNNNNNN.pt` every 5000 steps | Disk write + async volume commit | ~30s |

**Total setup before first `step` line: a few minutes** (the CLAP precompute that used to dominate at ~90 min on a single GPU is now sharded across all ranks — ~90 s at full corpus scale; what remains is NCCL init + the ~2 min torch.compile). Single-GPU runs skip phases 1–8 but each rank pays the load + CLAP cost itself — single-GPU is **not recommended** for this reason (and the memory footprint is too tight regardless).

## Decoding individual log prefixes

### `[parent]`, `[bundle]`, `[train]`, `[val]`, `[tags]`, `[lyrics]`, `[clap-parent]`
Emitted by the **parent process** of the DDP path. Only visible during setup phases 1–7. After `mp.spawn`, the parent goes silent (waits for ranks to finish). See [diskrot/dataset.py](diskrot/dataset.py) and [diskrot/modal_train.py:_precompute_clap_cache](diskrot/modal_train.py).

### `DDP active: world_size=8, per-rank batch=8, global batch=64`
Proves all 8 ranks initialized NCCL and joined the process group. If this line is missing, DDP did not start — workers may be silently stuck in CUDA init or NCCL discovery. Source: [train.py](diskrot/train.py).

### `model: XX.XX M params on cuda`
Confirms model is on GPU. Param count should be ~1.5B (1514.3M), matching the `DEFAULTS` in [modal_train.py](diskrot/modal_train.py#L83-L99) (~1.14B / 1145.2M if text conditioning is off). A wildly different number means an architecture override on the CLI didn't land as intended.

### `step N/T loss L lr LR tok/s X.Xk cb[a b c d e f g h i]`
- **`loss`** — average over the last `log_every=25` steps. Watch for it to start in the 5–7 range and drop into 3.5–4.5 ("recognizably musical" per README.modal.md).
- **`lr`** — current cosine-decayed learning rate (warmup for the first `warmup_steps`, 5000 on Modal).
- **`tok/s`** — aggregate token throughput across all ranks. On 8×H100 DDP expect ~6–7× single-H100 (NCCL overhead). If `tok/s` is only ~1× a single-H100 run, DDP is broken or one rank is starving the others.
- **`cb[...]`** — per-codebook losses (9 values). Codebook 0 is usually highest (carries most signal); later codebooks should be lower. If they're all equal, something's wrong with the delay pattern.

### `checkup at step N score X.XXXX`
Validation pass output. Three flavours:
- `first checkup` — initial val loss, no comparison yet
- `NEW BEST! beat previous best (X) by Y` — val loss improved, `best.pt` saved
- `Y worse than best (X from step Z) — strike N of 20` — val loss regressed; counts toward early-stop patience

When strikes hit `cfg.patience` (20 on Modal), training stops: `early stopping at step N — val loss has not improved for 20 evals`.

### `saved ckpt -> /ckpts/v7_1500m/step_NNNNNNN.pt`
Step checkpoint written to volume. Every 5000 steps + final step. `latest.pt` is also overwritten so resumes pick up the most recent. `best.pt` is updated separately on val-loss improvements. The subdir is whatever `--ckpt-subdir` resolved to (`v7_1500m` by default).

## Healthy throughput numbers

Sharded mmap path (large corpus / 8×H100):

| Metric | Healthy | Trouble if |
|---|---|---|
| Train/val "load" | seconds (just mmap open + JSON read) | minutes (you're on the in-RAM path — confirm `[parent] v2 sharded layout detected` fired) |
| Packing | n/a (offline, see [diskrot/pack_cache.py](diskrot/pack_cache.py)) | — |
| CLAP precompute | ~2.5k tags/s per GPU, 8-way sharded (~90 s @ ~240k unique tags) | minutes-long ETAs per GPU (CLAP not on GPU?) |
| First `step` line | within ~10 min of container start | > 30 min (something is silently stuck) |
| `tok/s` on 8×H100 | ~6–7× the single-H100 rate (NCCL overhead) | ~1× (DDP broken) or far lower than expected (grad-checkpointing misconfigured or seq-len wrong) |

## Diagnosing silent runs

If `modal app logs ap-...` shows zero lines after several minutes:

1. **Confirm the container is alive**: `modal app list | grep nano-train` — should show `ephemeral (detached)` with `Tasks: 1`.
2. **Confirm there's a running task**: `modal container list | grep ap-...` — should return one `ta-...` ID.
3. **Peek at the process state**:
   ```bash
   modal container exec --no-pty <ta-...> -- bash -c \
     'for pid in $(ls /proc | grep -E "^[0-9]+$"); do
        [ -r /proc/$pid/status ] && echo "PID $pid: $(grep -E "^(State|VmRSS|Threads)" /proc/$pid/status | tr "\n" " ")";
      done'
   ```
   - `State: D` (disk sleep) = blocked on I/O (volume, network, disk) — usually progressing, just slowly
   - `State: R` = running CPU work
   - `State: S` = sleeping (waiting on a syscall or socket)
   - `VmRSS` growing as pages fault in off the mmap shards = the loader is working (it does **not** preload the whole cache on the sharded path)
   - 4 python workers under PID 2 = DDP ranks spawned correctly
4. **Check GPUs** with the nvidia-smi one-liner above. During CLAP precompute all GPUs are briefly busy (8-way shard); during training all should sit at 80–99%.

## Known-benign log noise

- **`[modal-client] ... Heartbeat attempt failed`** — your local CLI's monitor hiccupping, not the remote container. The detached training run doesn't care. Safe to ignore.
- **`terminate called without an active exception`** right after `[clap-parent] done` — a CLAP shard worker process exiting with a joinable C++ thread. The parent survives and proceeds to spawn the DDP ranks; only worry if the app's task count actually drops.
- **`Runner interrupted due to worker preemption`** (tokenize only, not training) — Modal preempted the spot instance. Modal restarts the same input automatically.
- **mpg123 / id3 warnings** (tokenize only) — malformed MP3 tags in the source files. Decode proceeds; tokens are still produced.
- **`PySoundFile failed. Trying audioread instead`** (tokenize only) — librosa fallback path. Slower but functionally identical.
- **`weights_only=False` FutureWarning** — DAC's checkpoint loader uses old-style pickle. Pinned to the version in [modal_tokenize.py:38-53](diskrot/modal_tokenize.py#L38-L53), safe.

## Common mistakes when reading logs

- **"No `step` line yet → something is broken."** Setup legitimately takes a few minutes (sharded CLAP precompute ~90 s, then NCCL init + torch.compile) before the first step. Past ~30 min, check `modal container exec` and confirm processes are progressing before killing anything.
- **"`tok/s` is the per-rank throughput."** It's aggregate across all ranks. Comparing `tok/s` across runs only makes sense if `world_size` matches.
- **"All ranks should print everything."** Only rank 0 (the main rank) prints. The other 7 are silent by design — see `if main:` guards throughout [train.py](diskrot/train.py). Their work shows up indirectly via `tok/s` and `DDP active: world_size=8`.
- **"`free -m` doesn't exist, so something's wrong."** The slim Debian image used by [modal_tokenize.py:39](diskrot/modal_tokenize.py#L39) and [modal_train.py:41](diskrot/modal_train.py#L41) ships without `procps`. Use `/proc/<pid>/status` directly (see the diagnostic snippet above).
