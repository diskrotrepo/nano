"""Param counter for NanoAudioGPT configs. Mirrors the module structure in
model/nano_audio_gpt.py exactly so we can size a 1.5B variant before training."""

K = 9            # n_codebooks
V = 1025         # vocab_with_pad (1024 + 1 pad)


def count(d_model, n_layers, d_ff, text_cond=True, n_heads=16):
    assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
    # token embeddings: K independent nn.Embedding(V, d)
    embeds = K * V * d_model
    # output heads: K independent nn.Linear(d, V, bias=False)
    heads = K * d_model * V
    # per block
    self_attn = 4 * d_model * d_model           # qkv (3d²) + proj (d²)
    cross_attn = 4 * d_model * d_model if text_cond else 0  # q(d²)+kv(2d²)+out(d²)
    mlp = 2 * d_model * d_ff                     # fc1 + fc2
    norms = (3 if text_cond else 2) * d_model    # RMSNorm weight vectors
    per_block = self_attn + cross_attn + mlp + norms
    total = embeds + heads + n_layers * per_block + d_model  # +ln_final
    return total


def show(name, d, L, F, text=True, heads=16):
    n = count(d, L, F, text, heads)
    hd = d // heads
    print(f"{name:14s} d={d:5d} L={L:3d} heads={heads:3d} (hd={hd:3d}) "
          f"d_ff={F:6d} text={int(text)} -> {n/1e6:7.1f}M  ({n/1e9:.3f}B)")


print("=== sanity check vs known sizes ===")
# Historical: the 1024/16/4096 ~287M config was the old v6 model; config A
# (2048/22/8192) below is now the live default (v7_1500m).
show("old-v6(287M)", 1024, 16, 4096, text=True)
show("old-no-text", 1024, 16, 4096, text=False)

print("\n=== 1.5B candidate configs (text-conditioned) ===")
show("A wide-shallow*", 2048, 22, 8192, text=True, heads=16)  # * = current v7_1500m
show("B balanced",     1920, 26, 7680, text=True, heads=16)
show("C deep-narrow",  1664, 34, 6656, text=True, heads=16)
show("D gpt2xl-ish",   1600, 30, 6400, text=True, heads=25)

print("\n=== same shapes WITHOUT text conditioning ===")
show("A no-text", 2048, 22, 8192, text=False, heads=16)
show("B no-text", 1920, 26, 7680, text=False, heads=16)
