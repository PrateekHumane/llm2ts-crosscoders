"""
Combined figure: phase coherence + gradient coherence across training steps
for LangInit (FT) and RandInit (RI), with inset t-SNE snapshots at Layer 13
showing sine wave geometry at step 1 and step 512.

Usage:
    python3 reproduction/sine_evolution/plot_coherence_combined.py
"""
import json
import threading

import numpy as np
import torch
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/sine_evolution"

# ── Load phase coherence ─────────────────────────────────────────────────────

with open(f"{OUTPUT_DIR}/phase_coherence_checkpoints.json") as f:
    pc_data = json.load(f)

pc_checkpoints = pc_data["checkpoints"]
pt_pc = pc_data["PT"]

pc_families = {}
for key, val in pc_checkpoints.items():
    m = val["model"]
    if m not in pc_families:
        pc_families[m] = []
    pc_families[m].append(val)
for m in pc_families:
    pc_families[m].sort(key=lambda x: x["step"])

# ── Gradient coherence (from user) ──────────────────────────────────────────

grad_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
grad_ri =    [0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0010, 0.0042, 0.0020, -0.0011, 0.2279, 0.2533, 0.2014, 0.2263, 0.2283]
grad_ft =    [0.4598, 0.4594, 0.4551, 0.4346, 0.2410, 0.2115, 0.6195, 0.5878, 0.1222, 0.1102, 0.1651, 0.1593, 0.1170, 0.0611, 0.0593]

# ── Extract t-SNE insets ─────────────────────────────────────────────────────

T = 512
PERIOD = 64
SKIP = 5
LAYER = 13
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0

t_arr = np.arange(T, dtype=np.float32)
sine = np.sin(2 * np.pi * t_arr / PERIOD)
mean, std = float(np.mean(sine)), float(np.std(sine))
if std < 1e-8:
    std = 1.0
normed = (sine - mean) / std
clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
bins = np.clip(bins, 0, N_BINS - 1)
input_ids = torch.from_numpy(bins).unsqueeze(0)
phase = ((t_arr % PERIOD) / PERIOD)[SKIP:]

INSET_CHECKPOINTS = {
    "FT_step1": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420/checkpoint-1",
    "FT_step512": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420/checkpoint-512",
    "RI_step1": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-1",
    "RI_step512": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-512",
}

@torch.no_grad()
def extract_layer(model_path, device):
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    model = full_model.model
    model.eval()

    captured = {}
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["hs"] = o.detach().float().cpu()
    handle = model.layers[LAYER].register_forward_hook(hook)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=input_ids.to(device), use_cache=False)

    handle.remove()
    hs = captured["hs"].squeeze(0).numpy()[SKIP:]
    del full_model, model
    torch.cuda.empty_cache()
    return hs

print("Extracting t-SNE inset hidden states...", flush=True)
inset_hidden = {}
lock = threading.Lock()
gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
threads = []

def worker(name, path, device):
    hs = extract_layer(path, device)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(hs)
    with lock:
        inset_hidden[name] = proj

for (name, path), gpu in zip(INSET_CHECKPOINTS.items(), gpus):
    t = threading.Thread(target=worker, args=(name, path, gpu))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
print("  Done.", flush=True)

# ── Plot ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

input_name = "Sine p=64"
C_FT = "#D55E00"
C_RI = "#0072B2"

fig, ax1 = plt.subplots(figsize=(6.5, 3.5))

# Phase coherence (left y-axis)
ft_steps = [e["step"] for e in pc_families["FT"]]
ft_pc = [e["coherence"][input_name] for e in pc_families["FT"]]
ri_steps = [e["step"] for e in pc_families["RI"]]
ri_pc = [e["coherence"][input_name] for e in pc_families["RI"]]

l1, = ax1.plot(ft_steps, ft_pc, "-o", color=C_FT, markersize=3, linewidth=1.5,
               label="LangInit — phase coherence")
l2, = ax1.plot(ri_steps, ri_pc, "-o", color=C_RI, markersize=3, linewidth=1.5,
               label="RandInit — phase coherence")
ax1.axhline(pt_pc[input_name], color="#999999", linestyle=":", linewidth=1, alpha=0.7)
ax1.text(1.2, pt_pc[input_name] + 0.015, "PT baseline", fontsize=6.5, color="#999999")

ax1.set_xscale("log")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Phase Coherence (lower = more periodic)")
ax1.set_ylim(-0.05, 0.95)
ax1.spines["right"].set_visible(False)

# Gradient coherence (right y-axis)
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)

l3, = ax2.plot(grad_steps, grad_ft, "--s", color=C_FT, markersize=3, linewidth=1.2,
               alpha=0.7, label="LangInit — gradient coherence")
l4, = ax2.plot(grad_steps, grad_ri, "--s", color=C_RI, markersize=3, linewidth=1.2,
               alpha=0.7, label="RandInit — gradient coherence")

ax2.set_ylabel("Gradient Coherence")
ax2.set_ylim(-0.05, 0.7)

# Combined legend
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="center right", framealpha=0.9, fontsize=6.5)

# ── Inset t-SNE plots ───────────────────────────────────────────────────────

def draw_inset(ax_parent, proj, phase, rect, border_color, title):
    ax_in = fig.add_axes(rect)
    mx = np.abs(proj).max()
    if mx > 0:
        proj = proj / mx
    for j in range(len(proj) - 1):
        c = plt.cm.hsv(phase[j])
        ax_in.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c,
                   linewidth=0.5, alpha=0.85, solid_capstyle="round")
    ax_in.set_xlim(-1.3, 1.3)
    ax_in.set_ylim(-1.3, 1.3)
    ax_in.set_aspect("equal")
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_facecolor("#f8fafc")
    for spine in ax_in.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    ax_in.set_title(title, fontsize=5.5, pad=2, color=border_color, fontweight="bold")

# Inset positions: [left, bottom, width, height] in figure coords
# Step 1: top-left area (early training, high phase coherence region)
# Step 512: middle area (after the transition)
draw_inset(ax1, inset_hidden["FT_step1"], phase,
           [0.12, 0.68, 0.12, 0.22], C_FT, "LangInit s1")
draw_inset(ax1, inset_hidden["RI_step1"], phase,
           [0.25, 0.68, 0.12, 0.22], C_RI, "RandInit s1")
draw_inset(ax1, inset_hidden["FT_step512"], phase,
           [0.44, 0.15, 0.12, 0.22], C_FT, "LangInit s512")
draw_inset(ax1, inset_hidden["RI_step512"], phase,
           [0.57, 0.15, 0.12, 0.22], C_RI, "RandInit s512")

fig.patch.set_facecolor("white")
out_path = f"{OUTPUT_DIR}/fig_coherence_combined.png"
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
plt.close()
print(f"Saved {out_path}")
