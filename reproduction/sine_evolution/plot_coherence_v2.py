"""
Two refined variants of the coherence + geometry figure.

B2: Filmstrip (3 steps, aligned to x-axis) + dual-axis curve below
B3: Dual-axis curve with 2 insets in whitespace, arrows to data points

Usage:
    python3 reproduction/sine_evolution/plot_coherence_v2.py
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
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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

# ── Extract t-SNE ────────────────────────────────────────────────────────────

T = 512; PERIOD = 64; SKIP = 5; LAYER = 13
N_BINS = 1024; BIN_LOW = -5.0; BIN_HIGH = 5.0

t_arr = np.arange(T, dtype=np.float32)
sine = np.sin(2 * np.pi * t_arr / PERIOD)
mean, std = float(np.mean(sine)), float(np.std(sine))
normed = (sine - mean) / std
clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
bins = np.clip(bins, 0, N_BINS - 1)
input_ids = torch.from_numpy(bins).unsqueeze(0)
phase = ((t_arr % PERIOD) / PERIOD)[SKIP:]

SNAPSHOT_STEPS = [1, 256, 8192]
CKPT_BASE = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
}

all_jobs = []
for model in ["FT", "RI"]:
    for step in SNAPSHOT_STEPS:
        all_jobs.append((f"{model}_s{step}", f"{CKPT_BASE[model]}/checkpoint-{step}"))

@torch.no_grad()
def extract_tsne(name, model_path, device):
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device)
    model = full_model.model; model.eval()
    captured = {}
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["hs"] = o.detach().float().cpu()
    handle = model.layers[LAYER].register_forward_hook(hook)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=input_ids.to(device), use_cache=False)
    handle.remove()
    hs = captured["hs"].squeeze(0).numpy()[SKIP:]
    del full_model, model; torch.cuda.empty_cache()
    return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(hs)

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
            with lock: tsne_projs[n] = proj
        t = threading.Thread(target=worker, args=(name, path, gpus[i]))
        threads.append(t); t.start()
    for t in threads: t.join()
    print(f"  Round {round_idx // 4 + 1} done", flush=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
})

C_FT = "#D55E00"; C_RI = "#0072B2"
input_name = "Sine p=64"
ft_steps_pc = [e["step"] for e in pc_families["FT"]]
ft_pc = [e["coherence"][input_name] for e in pc_families["FT"]]
ri_steps_pc = [e["step"] for e in pc_families["RI"]]
ri_pc = [e["coherence"][input_name] for e in pc_families["RI"]]


def render_tsne_image(proj, phase, size_px=120):
    """Render a t-SNE plot to an RGBA image array for embedding."""
    fig_t, ax_t = plt.subplots(figsize=(1.2, 1.2), dpi=size_px)
    mx = np.abs(proj).max()
    if mx > 0: proj = proj / mx
    for j in range(len(proj) - 1):
        c = plt.cm.hsv(phase[j])
        ax_t.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c,
                  linewidth=0.7, alpha=0.85, solid_capstyle="round")
    ax_t.set_xlim(-1.3, 1.3); ax_t.set_ylim(-1.3, 1.3)
    ax_t.set_aspect("equal"); ax_t.set_xticks([]); ax_t.set_yticks([])
    ax_t.set_facecolor("#f8fafc")
    for spine in ax_t.spines.values(): spine.set_visible(False)
    fig_t.patch.set_facecolor("white")
    fig_t.tight_layout(pad=0)
    fig_t.canvas.draw()
    w, h = fig_t.canvas.get_width_height()
    img = np.frombuffer(fig_t.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    # ARGB -> RGBA
    img = np.roll(img, -1, axis=2)
    plt.close(fig_t)
    return img


def draw_tsne_ax(ax, proj, phase, border_color=None):
    mx = np.abs(proj).max()
    if mx > 0: proj = proj / mx
    for j in range(len(proj) - 1):
        c = plt.cm.hsv(phase[j])
        ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c,
                linewidth=0.6, alpha=0.85, solid_capstyle="round")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
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
# VARIANT B2: 3 filmstrip columns aligned to x-axis + curve below
# ══════════════════════════════════════════════════════════════════════════════

def plot_b2():
    n_snap = len(SNAPSHOT_STEPS)

    fig = plt.figure(figsize=(5.5, 4.2))

    # Use log positions to figure out column placement
    # x-axis range: ~1 to 10000 in log
    log_min, log_max = np.log10(0.7), np.log10(15000)
    log_range = log_max - log_min

    # Filmstrip row heights
    film_h = 0.22
    film_w = 0.14
    gap = 0.03  # between LangInit and RandInit rows

    for ci, step in enumerate(SNAPSHOT_STEPS):
        cx = (np.log10(step) - log_min) / log_range
        # Shift to account for figure margins
        fx = 0.10 + cx * 0.78 - film_w / 2

        # LangInit (top row)
        ax = fig.add_axes([fx, 0.72, film_w, film_h])
        draw_tsne_ax(ax, tsne_projs[f"FT_s{step}"], phase, C_FT)
        if ci == 0:
            ax.set_ylabel("LangInit", fontsize=6, fontweight="bold",
                           labelpad=2, color=C_FT)
        ax.set_title(f"Step {step}", fontsize=6, pad=2)

        # RandInit (second row)
        ax = fig.add_axes([fx, 0.72 - film_h - gap, film_w, film_h])
        draw_tsne_ax(ax, tsne_projs[f"RI_s{step}"], phase, C_RI)
        if ci == 0:
            ax.set_ylabel("RandInit", fontsize=6, fontweight="bold",
                           labelpad=2, color=C_RI)

        # Vertical connector line
        line_x = fx + film_w / 2
        fig.add_artist(plt.Line2D(
            [line_x, line_x], [0.72 - film_h - gap - 0.01, 0.42],
            transform=fig.transFigure, color="#ccc", linewidth=0.6,
            linestyle=":", zorder=0))

    # Curve panel
    ax1 = fig.add_axes([0.10, 0.08, 0.78, 0.32])
    l1, = ax1.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=2.5,
                   linewidth=1.3, label="LangInit — phase coh.")
    l2, = ax1.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=2.5,
                   linewidth=1.3, label="RandInit — phase coh.")
    ax1.axhline(pt_pc[input_name], color="#999", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.text(1.3, pt_pc[input_name] + 0.02, "PT baseline", fontsize=5.5, color="#999")
    ax1.set_xscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Phase Coherence\n(lower = more periodic)", fontsize=8)
    ax1.set_ylim(-0.05, 0.95)
    ax1.set_xlim(0.7, 15000)

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

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_B2.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT B3: Dual-axis plot with 2 insets in whitespace + arrows
# ══════════════════════════════════════════════════════════════════════════════

def plot_b3():
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))

    # Phase coherence
    l1, = ax1.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=3,
                   linewidth=1.5, label="LangInit — phase coh.")
    l2, = ax1.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=3,
                   linewidth=1.5, label="RandInit — phase coh.")
    ax1.axhline(pt_pc[input_name], color="#999", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.text(1.3, pt_pc[input_name] - 0.04, "PT baseline", fontsize=6, color="#999")
    ax1.set_xscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Phase Coherence (lower = more periodic)")
    ax1.set_ylim(-0.08, 1.05)
    ax1.set_xlim(0.7, 15000)

    # Gradient coherence
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    l3, = ax2.plot(grad_steps, grad_ft, "--s", color=C_FT, markersize=2.5,
                   linewidth=1, alpha=0.55, label="LangInit — grad. coh.")
    l4, = ax2.plot(grad_steps, grad_ri, "--s", color=C_RI, markersize=2.5,
                   linewidth=1, alpha=0.55, label="RandInit — grad. coh.")
    ax2.set_ylabel("Gradient Coherence")
    ax2.set_ylim(-0.08, 1.05 * 0.7 / 0.95)

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="upper center", fontsize=6, framealpha=0.9, ncol=2,
               bbox_to_anchor=(0.5, 0.99))

    # ── Inset pair 1: step 1 (top-left whitespace) ──
    # LangInit at step 1 — place in upper-left
    inset_size = 0.16
    ax_ft1 = fig.add_axes([0.13, 0.58, inset_size, inset_size * 6.0 / 3.8])
    draw_tsne_ax(ax_ft1, tsne_projs["FT_s1"], phase, C_FT)
    ax_ft1.set_title("LangInit", fontsize=5, color=C_FT, fontweight="bold", pad=1)

    ax_ri1 = fig.add_axes([0.13 + inset_size + 0.01, 0.58, inset_size, inset_size * 6.0 / 3.8])
    draw_tsne_ax(ax_ri1, tsne_projs["RI_s1"], phase, C_RI)
    ax_ri1.set_title("RandInit", fontsize=5, color=C_RI, fontweight="bold", pad=1)

    # Label
    fig.text(0.13 + inset_size, 0.58 + inset_size * 6.0 / 3.8 + 0.02,
             "Step 1", fontsize=6, ha="center", fontweight="bold", color="#444")

    # Arrow from insets to step=1 data point
    ax1.annotate("", xy=(1, ft_pc[0]), xytext=(3.5, 0.72),
                 arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.7, ls="--"))

    # ── Inset pair 2: step 8192 (right whitespace) ──
    ax_ft2 = fig.add_axes([0.62, 0.22, inset_size, inset_size * 6.0 / 3.8])
    draw_tsne_ax(ax_ft2, tsne_projs["FT_s8192"], phase, C_FT)

    ax_ri2 = fig.add_axes([0.62 + inset_size + 0.01, 0.22, inset_size, inset_size * 6.0 / 3.8])
    draw_tsne_ax(ax_ri2, tsne_projs["RI_s8192"], phase, C_RI)

    fig.text(0.62 + inset_size, 0.22 + inset_size * 6.0 / 3.8 + 0.02,
             "Step 8192", fontsize=6, ha="center", fontweight="bold", color="#444")

    # Arrow from insets to step=8192 data point
    ax1.annotate("", xy=(8192, ft_pc[-2]), xytext=(3000, 0.38),
                 arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.7, ls="--"))

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_B3.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ── Run ──────────────────────────────────────────────────────────────────────

print("\nPlotting variants...", flush=True)
plot_b2()
plot_b3()
print("Done!")
