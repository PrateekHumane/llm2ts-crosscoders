"""
Refined variants of B2 and B3.

B2a: Side-by-side pairs with bracket/arrow pointing down to the curve
B2b: Pairs above curve, connected by colored arrow stems to data points
B2c: Pairs below the x-axis label area, connected by drop lines from data points
B3b: Insets repositioned to avoid all curve overlap

Usage:
    python3 reproduction/sine_evolution/plot_coherence_v3.py
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

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


def draw_tsne_ax(ax, proj, phase, border_color=None):
    p = proj.copy()
    mx = np.abs(p).max()
    if mx > 0: p = p / mx
    for j in range(len(p) - 1):
        c = plt.cm.hsv(phase[j])
        ax.plot(p[j:j+2, 0], p[j:j+2, 1], color=c,
                linewidth=0.6, alpha=0.85, solid_capstyle="round")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f8fafc")
    if border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.5)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)


def get_pc_at_step(model, step):
    for e in pc_families[model]:
        if e["step"] == step:
            return e["coherence"][input_name]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# B2a: Side-by-side pairs above curve, with colored arrow stems to data points
# ══════════════════════════════════════════════════════════════════════════════

def plot_b2a():
    fig = plt.figure(figsize=(5.5, 4.5))

    # Main curve axes
    ax1 = fig.add_axes([0.11, 0.10, 0.76, 0.38])

    l1, = ax1.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=3,
                   linewidth=1.5, label="LangInit — phase coh.", zorder=5)
    l2, = ax1.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=3,
                   linewidth=1.5, label="RandInit — phase coh.", zorder=5)
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
                   alpha=0.55, label="LangInit — grad. coh.")
    l4, = ax2.plot(grad_steps, grad_ri, "--s", color=C_RI, markersize=2, linewidth=1,
                   alpha=0.55, label="RandInit — grad. coh.")
    ax2.set_ylabel("Gradient Coherence", fontsize=8)
    ax2.set_ylim(-0.05, 0.7)

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right", fontsize=5.5, framealpha=0.9)

    # Filmstrip: side-by-side pairs with arrow stems
    log_min, log_max = np.log10(0.7), np.log10(15000)
    log_range = log_max - log_min
    pair_w = 0.10  # width of each tsne box
    pair_h = 0.18
    pair_gap = 0.005
    film_y_ft = 0.73
    film_y_ri = film_y_ft - pair_h - 0.015

    for step in SNAPSHOT_STEPS:
        cx = (np.log10(step) - log_min) / log_range
        fx_center = 0.11 + cx * 0.76
        fx_ft = fx_center - pair_w - pair_gap / 2
        fx_ri = fx_center + pair_gap / 2

        # LangInit box
        ax_ft = fig.add_axes([fx_ft, film_y_ft, pair_w, pair_h])
        draw_tsne_ax(ax_ft, tsne_projs[f"FT_s{step}"], phase, C_FT)

        # RandInit box
        ax_ri = fig.add_axes([fx_ri, film_y_ft, pair_w, pair_h])
        draw_tsne_ax(ax_ri, tsne_projs[f"RI_s{step}"], phase, C_RI)

        # Step label above the pair
        fig.text(fx_center, film_y_ft + pair_h + 0.015,
                 f"Step {step}", fontsize=6.5, ha="center", fontweight="bold",
                 color="#333")

        # Arrow stem from center-bottom of pair down to curve area
        arrow_top = film_y_ft - 0.01
        arrow_bot = 0.49
        fig.add_artist(plt.Line2D(
            [fx_center, fx_center], [arrow_top, arrow_bot],
            transform=fig.transFigure, color="#aaa", linewidth=0.8,
            zorder=0))
        # Small triangle at bottom
        fig.add_artist(plt.Polygon(
            [[fx_center - 0.006, arrow_bot + 0.01],
             [fx_center + 0.006, arrow_bot + 0.01],
             [fx_center, arrow_bot]],
            transform=fig.transFigure, color="#aaa", zorder=0))

    # Model labels on the left
    fig.text(0.02, film_y_ft + pair_h / 2, "LangInit",
             fontsize=6, color=C_FT, fontweight="bold", ha="center",
             va="center", rotation=90)
    fig.text(0.02 + pair_w + pair_gap, film_y_ft + pair_h / 2, "RandInit",
             fontsize=6, color=C_RI, fontweight="bold", ha="center",
             va="center", rotation=90)

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_B2a.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# B2b: Pairs stacked vertically, with colored highlight markers on curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_b2b():
    fig = plt.figure(figsize=(5.5, 4.8))

    ax1 = fig.add_axes([0.11, 0.10, 0.76, 0.35])

    l1, = ax1.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=3,
                   linewidth=1.5, label="LangInit — phase coh.", zorder=3)
    l2, = ax1.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=3,
                   linewidth=1.5, label="RandInit — phase coh.", zorder=3)
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
                   alpha=0.55, label="LangInit — grad. coh.")
    l4, = ax2.plot(grad_steps, grad_ri, "--s", color=C_RI, markersize=2, linewidth=1,
                   alpha=0.55, label="RandInit — grad. coh.")
    ax2.set_ylabel("Gradient Coherence", fontsize=8)
    ax2.set_ylim(-0.05, 0.7)

    # Highlight the snapshot steps with larger markers
    for step in SNAPSHOT_STEPS:
        ft_val = get_pc_at_step("FT", step)
        ri_val = get_pc_at_step("RI", step)
        if ft_val is not None:
            ax1.plot(step, ft_val, "o", color=C_FT, markersize=7,
                     markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        if ri_val is not None:
            ax1.plot(step, ri_val, "o", color=C_RI, markersize=7,
                     markeredgecolor="white", markeredgewidth=1.5, zorder=6)

    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="center right", fontsize=5.5, framealpha=0.9)

    # Filmstrip above: LangInit row then RandInit row
    log_min, log_max = np.log10(0.7), np.log10(15000)
    log_range = log_max - log_min
    box_w = 0.13
    box_h = 0.17
    row_gap = 0.005

    for ci, step in enumerate(SNAPSHOT_STEPS):
        cx = (np.log10(step) - log_min) / log_range
        fx = 0.11 + cx * 0.76 - box_w / 2

        y_ft = 0.74
        y_ri = y_ft - box_h - row_gap

        ax_ft = fig.add_axes([fx, y_ft, box_w, box_h])
        draw_tsne_ax(ax_ft, tsne_projs[f"FT_s{step}"], phase, C_FT)

        ax_ri = fig.add_axes([fx, y_ri, box_w, box_h])
        draw_tsne_ax(ax_ri, tsne_projs[f"RI_s{step}"], phase, C_RI)

        # Step label
        fig.text(fx + box_w / 2, y_ft + box_h + 0.01,
                 f"Step {step}", fontsize=6.5, ha="center", fontweight="bold",
                 color="#333")

        # Thin line from bottom of RI box to top of plot area
        line_x = fx + box_w / 2
        fig.add_artist(plt.Line2D(
            [line_x, line_x], [y_ri - 0.005, 0.46],
            transform=fig.transFigure, color="#ccc", linewidth=0.6,
            linestyle="-", zorder=0))

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_B2b.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# B3b: Dual-axis with 2 insets, repositioned to avoid overlap
# ══════════════════════════════════════════════════════════════════════════════

def plot_b3b():
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))

    # Phase coherence
    l1, = ax1.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=3,
                   linewidth=1.5, label="LangInit — phase coh.", zorder=5)
    l2, = ax1.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=3,
                   linewidth=1.5, label="RandInit — phase coh.", zorder=5)
    ax1.axhline(pt_pc[input_name], color="#999", linestyle=":", linewidth=0.8, alpha=0.7)
    ax1.text(15, pt_pc[input_name] - 0.045, "PT baseline", fontsize=6, color="#999")
    ax1.set_xscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Phase Coherence (lower = more periodic)")
    ax1.set_ylim(-0.08, 1.1)
    ax1.set_xlim(0.7, 15000)

    # Gradient coherence
    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    l3, = ax2.plot(grad_steps, grad_ft, "--s", color=C_FT, markersize=2.5,
                   linewidth=1, alpha=0.5, label="LangInit — grad. coh.")
    l4, = ax2.plot(grad_steps, grad_ri, "--s", color=C_RI, markersize=2.5,
                   linewidth=1, alpha=0.5, label="RandInit — grad. coh.")
    ax2.set_ylabel("Gradient Coherence")
    ax2.set_ylim(-0.08, 1.1 * 0.7 / 0.95)

    # Legend at top center, horizontal
    lines = [l1, l2, l3, l4]
    ax1.legend(lines, [l.get_label() for l in lines],
               loc="upper center", fontsize=5.5, framealpha=0.85, ncol=2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)

    # Highlight the 2 snapshot steps
    for step in [1, 8192]:
        ft_val = get_pc_at_step("FT", step)
        ri_val = get_pc_at_step("RI", step)
        if ft_val:
            ax1.plot(step, ft_val, "o", color=C_FT, markersize=8,
                     markeredgecolor="white", markeredgewidth=2, zorder=6)
        if ri_val:
            ax1.plot(step, ri_val, "o", color=C_RI, markersize=8,
                     markeredgecolor="white", markeredgewidth=2, zorder=6)

    # ── Inset pair 1: Step 1 — upper right (above the converged flat region) ──
    inset_w = 0.13
    inset_h = 0.25

    ax_ft1 = fig.add_axes([0.62, 0.62, inset_w, inset_h])
    draw_tsne_ax(ax_ft1, tsne_projs["FT_s1"], phase, C_FT)

    ax_ri1 = fig.add_axes([0.62 + inset_w + 0.008, 0.62, inset_w, inset_h])
    draw_tsne_ax(ax_ri1, tsne_projs["RI_s1"], phase, C_RI)

    fig.text(0.62 + inset_w + 0.004, 0.62 + inset_h + 0.015,
             "Step 1", fontsize=7, ha="center", fontweight="bold", color="#444")

    # Arrow from step-1 inset group to step=1 data point on curve
    # Use ax1 data coords for the target, figure coords for the source
    ax1.annotate("",
                 xy=(1, 0.82),  # near RI step-1 point
                 xytext=(0.57, 0.78),
                 textcoords="figure fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0,
                                 connectionstyle="arc3,rad=-0.15"))

    # ── Inset pair 2: Step 8192 — lower center-left ──
    ax_ft2 = fig.add_axes([0.18, 0.17, inset_w, inset_h])
    draw_tsne_ax(ax_ft2, tsne_projs["FT_s8192"], phase, C_FT)

    ax_ri2 = fig.add_axes([0.18 + inset_w + 0.008, 0.17, inset_w, inset_h])
    draw_tsne_ax(ax_ri2, tsne_projs["RI_s8192"], phase, C_RI)

    fig.text(0.18 + inset_w + 0.004, 0.17 + inset_h + 0.015,
             "Step 8192", fontsize=7, ha="center", fontweight="bold", color="#444")

    # Arrow from step-8192 inset group to data point
    ax1.annotate("",
                 xy=(8192, 0.18),  # near FT step-8192 point
                 xytext=(0.47, 0.38),
                 textcoords="figure fraction",
                 arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.0,
                                 connectionstyle="arc3,rad=0.15"))

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/fig_coherence_B3b.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"  Saved {out}")


# ── Run ──────────────────────────────────────────────────────────────────────

print("\nPlotting variants...", flush=True)
plot_b2a()
plot_b2b()
plot_b3b()
print("Done!")
