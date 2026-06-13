"""One-off forensic: reconstruct the gradient-norm trajectory from saved
optimizer state (AdamW exp_avg_sq is a ~20-step EMA of g^2 at betas=(0.9,0.95)),
since the live Modal logs for 55k->64k have rotated.

  sqrt(sum_over_params exp_avg_sq) ~= the logged grad norm at that step
  per-tensor sqrt(mean exp_avg_sq) = RMS gradient for that tensor (localizes it)

Run:  modal run scripts/_grad_forensics.py
"""
import modal

app = modal.App("nano-grad-forensics")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.4.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("numpy>=1.26", "g2p_en==2.1.0")
    .add_local_python_source("model", "diskrot")
)

ckpts_vol = modal.Volume.from_name("nano-ckpts")


@app.function(image=image, cpu=4.0, memory=32768, timeout=1800,
              volumes={"/ckpts": ckpts_vol})
def forensics(subdir: str = "v8_sing"):
    import gc
    import math
    import torch
    from model.nano_audio_gpt import GPTConfig, NanoAudioGPT

    names = ["step_0050000.pt", "step_0055000.pt", "step_0060000.pt", "step_0065000.pt"]

    # Build the optimizer-index -> param-name map exactly as train.py does:
    # named = model.named_parameters() + text_proj.*; then split_decay_param_groups
    # puts ndim>=2 (and not '.null') into the decay group, the rest into no_decay;
    # AdamW indexes decay-group params first (in order), then no_decay-group.
    def build_flat_names(cfg_dict, text_proj_sd):
        with torch.device("meta"):
            model = NanoAudioGPT(GPTConfig(**cfg_dict))
        named = [(n, p.ndim) for n, p in model.named_parameters() if p.requires_grad]
        named += [(f"text_proj.{k}", v.ndim) for k, v in (text_proj_sd or {}).items()]
        decay = [n for n, nd in named if not (nd < 2 or n.endswith(".null"))]
        no_decay = [n for n, nd in named if (nd < 2 or n.endswith(".null"))]
        return decay + no_decay

    flat_names = None
    rows = []  # (step, total_norm, [(name, contrib_sqrt, g_rms, numel)...])
    for nm in names:
        path = f"/ckpts/{subdir}/{nm}"
        ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if flat_names is None:
            flat_names = build_flat_names(ck["cfg"], ck.get("text_proj"))
        state = ck["optim"]["state"]
        opt_step = None
        total_sq = 0.0
        per = []
        for idx in sorted(state.keys()):
            st = state[idx]
            if "exp_avg_sq" not in st:
                continue
            v = st["exp_avg_sq"]
            if opt_step is None and "step" in st:
                s = st["step"]
                opt_step = int(s.item()) if hasattr(s, "item") else int(s)
            ssum = float(v.sum())
            total_sq += ssum
            name = flat_names[idx] if idx < len(flat_names) else f"<idx{idx}>"
            per.append((name, math.sqrt(max(ssum, 0.0)), math.sqrt(ssum / v.numel()), v.numel()))
        per.sort(key=lambda r: r[1], reverse=True)
        rows.append((nm, opt_step, math.sqrt(max(total_sq, 0.0)), per[:14]))
        del ck, state
        gc.collect()

    print("\n==================== GRAD-NORM RECONSTRUCTION (from AdamW exp_avg_sq, b2=0.95) ====================")
    print(f"{'ckpt':>16} {'opt_step':>9} {'recon_grad_norm':>16}")
    for nm, step, tot, _ in rows:
        print(f"{nm:>16} {str(step):>9} {tot:>16.2f}")

    for nm, step, tot, per in rows:
        print(f"\n--- {nm} (opt_step={step}) total_recon_norm={tot:.2f} :: top tensors by contribution ---")
        print(f"     {'contrib(sqrt sum)':>18} {'g_rms(elem)':>12} {'numel':>12}  name")
        for name, contrib, g_rms, numel in per:
            print(f"     {contrib:>18.3f} {g_rms:>12.5f} {numel:>12d}  {name}")

    # Per-tensor delta 55k -> 65k to localize the ramp.
    def as_map(per):
        return {r[0]: r for r in per}
    m55 = next((as_map(p) for nm, *_ , p in [(r[0], r[1], r[2], r[3]) for r in rows] if nm == "step_0055000.pt"), {})
    m65 = next((as_map(p) for nm, *_ , p in [(r[0], r[1], r[2], r[3]) for r in rows] if nm == "step_0065000.pt"), {})
    if m55 and m65:
        print("\n--- biggest g_rms growth 55k -> 65k (tensors in both top-14) ---")
        deltas = []
        for name in set(m55) & set(m65):
            r55, r65 = m55[name], m65[name]
            if r55[2] > 0:
                deltas.append((name, r65[2] / r55[2], r55[2], r65[2]))
        deltas.sort(key=lambda d: d[1], reverse=True)
        for name, ratio, a, b in deltas[:12]:
            print(f"     x{ratio:>8.1f}   {a:.5f} -> {b:.5f}   {name}")


@app.local_entrypoint()
def main(subdir: str = "v8_sing"):
    forensics.remote(subdir=subdir)
