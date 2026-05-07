"""
Generate the 3 effective-rank figures for the NeurIPS paper.

Saves directly to NeurIPS26/figures/sec5/ so you can recompile the paper
immediately after running this script.

Outputs:
  fig_erank_trajectory.png   — Fig 4: mean erank vs training step
  fig_erank_layers.png       — Fig 5a: per-layer spaghetti (all 28 layers)
  fig_erank_foldchange.png   — Fig 5b: net fold-change bars per layer

Usage:
    python reproduction/effective_rank/plot_paper.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

COLORS = {"FT": "#0072B2", "RI": "#CC79A7"}
LABELS = {"FT": "LangInit", "RI": "RandInit"}
N_LAYERS = 28
OUT_DIR = "NeurIPS26/figures/sec5"
DATA_PATH = "results/effective_rank_checkpoints/results.json"
CRPS_PATHS = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
}
MAX_STEP = 4096
PPL_PATH = "results/wikitext_perplexity.json"


def load_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    models = {}
    for key, val in data["checkpoints"].items():
        model = val["model"]
        if model not in ("FT", "RI"):
            continue
        if val["step"] > MAX_STEP:
            continue
        if model not in models:
            models[model] = []
        models[model].append({
            "step": val["step"],
            "ts_erank": val["ts_erank"],
            "text_erank": val.get("text_erank"),
        })
    for model in models:
        models[model].sort(key=lambda x: x["step"])
    return models


def load_crps():
    crps = {}
    for model, base_path in CRPS_PATHS.items():
        entries = []
        for d in os.listdir(base_path):
            if not d.startswith("checkpoint-"):
                continue
            step = int(d.split("-")[1])
            if step > MAX_STEP:
                continue
            eval_file = os.path.join(base_path, d, "eval_results.json")
            if not os.path.isfile(eval_file):
                continue
            with open(eval_file) as f:
                metrics = json.load(f)["metrics"]
            entries.append({"step": step, "crps": metrics["eval_1/crps"]})
        entries.sort(key=lambda x: x["step"])
        crps[model] = entries
    return crps


def load_perplexity():
    with open(PPL_PATH) as f:
        data = json.load(f)
    ppl = {}
    for model in ("FT",):
        entries = []
        for step_str, val in data[model].items():
            step = int(step_str)
            if step > MAX_STEP:
                continue
            entries.append({"step": step, "perplexity": val["perplexity"]})
        entries.sort(key=lambda x: x["step"])
        ppl[model] = entries
    return ppl


# ── Fig 4: Mean erank vs training step ────────────────────────────────────

def plot_trajectory(models, crps, ppl):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.6))

    # (a) Time Series Input — erank + loss overlay
    for name in ["FT", "RI"]:
        entries = models[name]
        steps = [e["step"] for e in entries]
        means = [np.mean(e["ts_erank"]) for e in entries]
        ls = "--" if name == "RI" else "-"
        ax1.plot(steps, means, ls, color=COLORS[name],
                 linewidth=1.3, label=LABELS[name])

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Mean Effective Rank")
    ax1.set_title("(a) Time Series Input", fontsize=9)

    ax1r = ax1.twinx()
    for name in ["FT", "RI"]:
        entries = crps[name]
        steps = [e["step"] for e in entries]
        vals = [e["crps"] for e in entries]
        ls = "--" if name == "RI" else "-"
        ax1r.plot(steps, vals, ls, color=COLORS[name], alpha=0.25,
                  linewidth=1.3, label=f"{LABELS[name]} (loss)")
    ax1r.set_ylabel("Loss", fontsize=8)
    ax1r.set_yscale("log")
    ax1r.spines["right"].set_visible(True)
    ax1r.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1r.yaxis.get_major_formatter().set_scientific(False)
    ax1r.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1r.tick_params(axis="y", labelsize=7)

    # (b) Text Input (Forgetting) — LangInit erank + perplexity
    entries = models["FT"]
    steps = [e["step"] for e in entries if e["text_erank"] is not None]
    means = [np.mean(e["text_erank"]) for e in entries if e["text_erank"] is not None]
    ax2.plot(steps, means, "-", color=COLORS["FT"],
             linewidth=1.3, label=LABELS["FT"])

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Training Step")
    ax2.set_title("(b) Text Input (Forgetting)", fontsize=9)

    ax2r = ax2.twinx()
    ppl_entries = ppl["FT"]
    ppl_steps = [e["step"] for e in ppl_entries]
    ppl_vals = [e["perplexity"] for e in ppl_entries]
    ax2r.plot(ppl_steps, ppl_vals, "-", color=COLORS["FT"], alpha=0.25,
              linewidth=1.3, label=f"{LABELS['FT']} (loss)")
    ax2r.set_ylabel("Perplexity", fontsize=8)
    ax2r.set_yscale("log")
    ax2r.spines["right"].set_visible(True)
    ax2r.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2r.yaxis.get_major_formatter().set_scientific(False)
    ax2r.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2r.tick_params(axis="y", labelsize=7)

    # Shared legend across both subplots
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4,
               framealpha=0.9, fontsize=6.5, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(bottom=0.26)
    plt.savefig(f"{OUT_DIR}/fig_erank_trajectory.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  fig_erank_trajectory.png")


# ── Fig 5: Per-layer erank + fold-change (combined 2×2) ──────────────────

def plot_layers_and_foldchange(models):
    model_cfgs = [("RI", LABELS["RI"]), ("FT", LABELS["FT"])]
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(5.0, 4.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.35, 1],
                          hspace=0.1, wspace=0.3)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(gs[0, 0])
    axes[0, 1] = fig.add_subplot(gs[0, 1])
    axes[1, 0] = fig.add_subplot(gs[2, 0])
    axes[1, 1] = fig.add_subplot(gs[2, 1])
    # gs[1, :] is an empty spacer row for the caption
    fig.subplots_adjust(right=0.88)

    # Compute fold-change data and shared x-limits for bottom row
    fold_data = {}
    for key, _ in model_cfgs:
        entries = [e for e in models[key] if e["step"] <= MAX_STEP]
        matrix = np.array([e["ts_erank"] for e in entries])
        baseline = matrix[0]
        baseline_safe = np.where(baseline < 1e-6, 1e-6, baseline)
        fold_data[key] = np.log2(matrix[-1] / baseline_safe)

    all_vals = [fold_data[k] for k, _ in model_cfgs]
    bar_min = min(v.min() for v in all_vals)
    bar_max = max(v.max() for v in all_vals)
    pad = (bar_max - bar_min) * 0.05
    bar_xlim = (bar_min - pad, bar_max + pad)

    for col, (key, label) in enumerate(model_cfgs):
        # Top row: spaghetti
        ax_top = axes[0, col]
        entries = [e for e in models[key] if e["step"] <= MAX_STEP]
        steps = np.array([e["step"] for e in entries])
        matrix = np.array([e["ts_erank"] for e in entries])

        for layer in range(N_LAYERS):
            ax_top.plot(steps, matrix[:, layer], "-",
                        color=cmap(layer / (N_LAYERS - 1)),
                        linewidth=0.7, alpha=0.8)

        ax_top.set_xscale("log")
        ax_top.set_yscale("log")
        if col == 0:
            ax_top.set_ylabel("Effective Rank")
        ax_top.set_title(label, fontsize=9)

        # Bottom row: fold-change bars
        ax_bot = axes[1, col]
        log2_final = fold_data[key]
        colors = ["#D55E00" if v > 0 else "#0072B2" for v in log2_final]
        ax_bot.barh(range(N_LAYERS), log2_final, height=0.8, color=colors,
                    edgecolor="white", linewidth=0.3)
        ax_bot.axvline(0, color="black", linewidth=0.5)
        ax_bot.set_yticks(np.arange(0, N_LAYERS, 7))
        ax_bot.set_ylim(-0.5, N_LAYERS - 0.5)
        ax_bot.set_xlim(bar_xlim)
        if col == 0:
            ax_bot.set_ylabel("Layer")

    # Sync y-axis across top row using the full range from both panels
    ymin = min(axes[0, c].get_ylim()[0] for c in range(2))
    ymax = max(axes[0, c].get_ylim()[1] for c in range(2))
    for c in range(2):
        axes[0, c].set_ylim(ymin, ymax)

    # Subcaptions between rows and below figure (LaTeX subfigure style)
    top_bottom = axes[0, 0].get_position().y0
    bot_top = axes[1, 0].get_position().y1
    caption_b_y = (top_bottom + bot_top) / 2
    fig.text(0.44, caption_b_y, "(b) Per-layer trajectories across training",
             fontsize=8, ha="center", va="center")
    fig.text(0.44, 0.01, "(c) Net change per layer",
             fontsize=8, ha="center", va="bottom")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.90, 0.55, 0.015, 0.35])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_LAYERS - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Layer", fontsize=7)
    cbar.set_ticks([0, 7, 14, 21, 27])
    cbar.ax.tick_params(labelsize=6)

    plt.savefig(f"{OUT_DIR}/fig_erank_layers.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  fig_erank_layers.png")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    models = load_data()
    crps = load_crps()
    ppl = load_perplexity()
    print("Generating paper figures...")
    plot_trajectory(models, crps, ppl)
    plot_layers_and_foldchange(models)
    print("Done! Recompile with:")
    print("  cd NeurIPS26 && pdflatex -interaction=nonstopmode neurips_2026.tex")
