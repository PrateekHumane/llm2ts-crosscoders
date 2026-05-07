"""
Three layout variants for the coherence + geometry figure.

Extracts Layer 13 t-SNE for sine wave at key checkpoints, then plots:
  A: Filmstrip rows (LangInit / RandInit) + stacked curve panels
  B: Filmstrip rows + single combined curve panel
  C: Two columns (one per model), each with filmstrip + curves

Usage:
    python3 reproduction/sine_evolution/plot_coherence_variants.py
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
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = "results/sine_evolution"

# ── Load data ────────────────────────────────────────────────────────────────

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

grad_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
grad_ri = [0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0010, 0.0042, 0.0020, -0.0011, 0.2279, 0.2533, 0.2014, 0.2263, 0.2283]
grad_ft = [0.4598, 0.4594, 0.4551, 0.4346, 0.2410, 0.2115, 0.6195, 0.5878, 0.1222, 0.1102, 0.1651, 0.1593, 0.1170, 0.0611, 0.0593]

# ── Extract t-SNE at key steps ───────────────────────────────────────────────

T = 512
PERIOD = 64
SKIP = 5
LAYER = 13
N_BINS = 1024
BIN_LOW, BIN_HIGH = -5.0, 5.0

t_arr = np.arange(T, dtype=np.float32)
sine = np.sin(2 * np.pi * t_arr / PERIOD)
mean, std = float(np.mean(sine)), float(np.std(sine))
normed = (sine - mean) / std
clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
bins = np.clip(bins, 0, N_BINS - 1)
input_ids = torch.from_numpy(bins).unsqueeze(0)
phase = ((t_arr % PERIOD) / PERIOD)[SKIP:]

SNAPSHOT_STEPS = [1, 64, 256, 1024, 8192]
CKPT_BASE = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
}

all_jobs = []
for model in ["FT", "RI"]:
    for step in SNAPSHOT_STEPS:
        path = f"{CKPT_BASE[model]}/checkpoint-{step}"
        all_jobs.append((f"{model}_s{step}", path))

@torch.no_grad()
def extract_tsne(name, model_path, device):
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
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    return tsne.fit_transform(hs)

print("Extracting t-SNE snapshots...", flush=True)
tsne_projs = {}
lock = threading.Lock()
gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

for round_idx in range(0, len(all_jobs), 4):
    batch = all_jobs[round_idx:round_idx + 4]
    threads = []
    for i, (name, path) in enumerate(batch):
        def worker(n, p, d):
            proj = extract_tsne(n, p, d)
            with lock:
                tsne_projs[n] = proj
        t = threading.Thread(target=worker, args=(name, path, gpus[i]))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    print(f"  Round {round_idx // 4 + 1} done", flush=True)

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
})

C_FT = "#D55E00"
C_RI = "#0072B2"
input_name = "Sine p=64"

ft_steps_pc = [e["step"] for e in pc_families["FT"]]
ft_pc = [e["coherence"][input_name] for e in pc_families["FT"]]
ri_steps_pc = [e["step"] for e in pc_families["RI"]]
ri_pc = [e["coherence"][input_name] for e in pc_families["RI"]]


def draw_tsne(ax, proj, phase, border_color=None):
    mx = np.abs(proj).max()
    if mx > 0:
        proj = proj / mx
    for j in range(len(proj) - 1):
        c = plt.cm.hsv(phase[j])
        ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c,
                linewidth=0.6, alpha=0.85, solid_capstyle="round")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#f8fafc")
    if border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.2)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# VERSION A: filmstrip rows + two stacked curve panels
# ══════════════════════════════════════════════════════════════════════════════

def plot_version_a():
    n_snap = len(SNAPSHOT_STEPS)
    fig = plt.figure(figsize=(5.5, 5.0))
    gs = GridSpec(4, n_snap, figure=fig,
                  height_ratios=[1, 1, 1.2, 1.2],
                  hspace=0.35, wspace=0.08)

    # Row 0: LangInit t-SNE
    for ci, step in enumerate(SNAPSHOT_STEPS):
        ax = fig.add_subplot(gs[0, ci])
        draw_tsne(ax, tsne_projs[f"FT_s{step}"], phase, C_FT)
        ax.set_title(f"s{step}", fontsize=6, pad=2)
        if ci == 0:
            ax.set_ylabel("LangInit", fontsize=7, fontweight="bold", labelpad=2)

    # Row 1: RandInit t-SNE
    for ci, step in enumerate(SNAPSHOT_STEPS):
        ax = fig.add_subplot(gs[1, ci])
        draw_tsne(ax, tsne_projs[f"RI_s{step}"], phase, C_RI)
        if ci == 0:
            ax.set_ylabel("RandInit", fontsize=7, fontweight="bold", labelpad=2)

    # Row 2: Phase coherence
    ax = fig.add_subplot(gs[2, :])
    ax.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=2.5, linewidth=1.3, label="LangInit")
    ax.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=2.5, linewidth=1.3, label="RandInit")
    ax.axhline(pt_pc[input_name], color="#999", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(1.2, pt_pc[input_name] + 0.02, "PT baseline", fontsize=5.5, color="#999")
    ax.set_xscale("log")
    ax.set_ylabel("Phase Coherence\n(lower = more periodic)", fontsize=8)
    ax.set_ylim(-0.05, 0.95)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_xticklabels([])

    # Row 3: Gradient coherence
    ax = fig.add_subplot(gs[3, :])
    ax.plot(grad_steps, grad_ft, "-s", color=C_FT, markersize=2.5, linewidth=1.3, label="LangInit")
    ax.plot(grad_steps, grad_ri, "-s", color=C_RI, markersize=2.5, linewidth=1.3, label="RandInit")
    ax.set_xscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Coherence", fontsize=8)
    ax.set_ylim(-0.05, 0.7)
    ax.legend(loc="upper right", fontsize=6.5)

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_A.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# VERSION B: filmstrip rows + single curve panel (phase only, gradient as fill)
# ══════════════════════════════════════════════════════════════════════════════

def plot_version_b():
    n_snap = len(SNAPSHOT_STEPS)
    fig = plt.figure(figsize=(5.5, 4.0))
    gs = GridSpec(3, n_snap, figure=fig,
                  height_ratios=[1, 1, 1.8],
                  hspace=0.3, wspace=0.08)

    # Row 0: LangInit
    for ci, step in enumerate(SNAPSHOT_STEPS):
        ax = fig.add_subplot(gs[0, ci])
        draw_tsne(ax, tsne_projs[f"FT_s{step}"], phase, C_FT)
        ax.set_title(f"Step {step}", fontsize=6, pad=2)
        if ci == 0:
            ax.set_ylabel("LangInit", fontsize=7, fontweight="bold", labelpad=2)

    # Row 1: RandInit
    for ci, step in enumerate(SNAPSHOT_STEPS):
        ax = fig.add_subplot(gs[1, ci])
        draw_tsne(ax, tsne_projs[f"RI_s{step}"], phase, C_RI)
        if ci == 0:
            ax.set_ylabel("RandInit", fontsize=7, fontweight="bold", labelpad=2)

    # Row 2: Combined curves
    ax1 = fig.add_subplot(gs[2, :])
    l1, = ax1.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=2.5, linewidth=1.3,
                   label="LangInit — phase coh.")
    l2, = ax1.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=2.5, linewidth=1.3,
                   label="RandInit — phase coh.")
    ax1.axhline(pt_pc[input_name], color="#999", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Phase Coherence\n(lower = more periodic)", fontsize=8)
    ax1.set_ylim(-0.05, 0.95)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    l3, = ax2.plot(grad_steps, grad_ft, "--s", color=C_FT, markersize=2, linewidth=1,
                   alpha=0.6, label="LangInit — grad. coh.")
    l4, = ax2.plot(grad_steps, grad_ri, "--s", color=C_RI, markersize=2, linewidth=1,
                   alpha=0.6, label="RandInit — grad. coh.")
    ax2.set_ylabel("Gradient Coherence", fontsize=8)
    ax2.set_ylim(-0.05, 0.7)

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right", fontsize=5.5, framealpha=0.9)

    # Vertical lines at snapshot steps
    for step in SNAPSHOT_STEPS:
        ax1.axvline(step, color="#ddd", linewidth=0.5, zorder=0)

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_B.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# VERSION C: two columns (LangInit | RandInit), each with filmstrip + curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_version_c():
    n_snap = len(SNAPSHOT_STEPS)
    fig = plt.figure(figsize=(5.5, 4.5))
    gs = GridSpec(3, 2 * n_snap, figure=fig,
                  height_ratios=[1, 1.3, 1.3],
                  hspace=0.35, wspace=0.06)

    # Row 0: filmstrip — left half LangInit, right half RandInit
    for ci, step in enumerate(SNAPSHOT_STEPS):
        ax = fig.add_subplot(gs[0, ci])
        draw_tsne(ax, tsne_projs[f"FT_s{step}"], phase, C_FT)
        ax.set_title(f"s{step}", fontsize=5, pad=2)
        if ci == 0:
            ax.set_ylabel("LangInit", fontsize=6.5, fontweight="bold", labelpad=2,
                           color=C_FT)

    for ci, step in enumerate(SNAPSHOT_STEPS):
        ax = fig.add_subplot(gs[0, n_snap + ci])
        draw_tsne(ax, tsne_projs[f"RI_s{step}"], phase, C_RI)
        ax.set_title(f"s{step}", fontsize=5, pad=2)
        if ci == 0:
            ax.set_ylabel("RandInit", fontsize=6.5, fontweight="bold", labelpad=2,
                           color=C_RI)

    # Row 1: Phase coherence (full width)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=2.5, linewidth=1.3, label="LangInit")
    ax.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=2.5, linewidth=1.3, label="RandInit")
    ax.axhline(pt_pc[input_name], color="#999", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(1.2, pt_pc[input_name] + 0.02, "PT baseline", fontsize=5.5, color="#999")
    ax.set_xscale("log")
    ax.set_ylabel("Phase Coherence\n(lower = more periodic)", fontsize=7.5)
    ax.set_ylim(-0.05, 0.95)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_xticklabels([])

    # Row 2: Gradient coherence (full width)
    ax = fig.add_subplot(gs[2, :])
    ax.plot(grad_steps, grad_ft, "-s", color=C_FT, markersize=2.5, linewidth=1.3, label="LangInit")
    ax.plot(grad_steps, grad_ri, "-s", color=C_RI, markersize=2.5, linewidth=1.3, label="RandInit")
    ax.set_xscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Coherence", fontsize=7.5)
    ax.set_ylim(-0.05, 0.7)
    ax.legend(loc="upper right", fontsize=6)

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_C.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ── Run ──────────────────────────────────────────────────────────────────────

print("\nPlotting variants...", flush=True)
plot_version_a()
plot_version_b()
plot_version_c()
print("Done!")
