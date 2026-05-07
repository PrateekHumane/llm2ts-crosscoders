"""
Generate NeurIPS-quality plots for effective rank and subspace alignment results.

Plots:
  1. Effective rank vs layer (line plot, all conditions)
  2. Subspace alignment vs layer (with random baseline band)
  3. Eigenvalue spectrum at selected layers (log scale)
  4. Grouped bar chart at representative mid-layer
  5. Pairwise alignment heatmap across layers

Usage:
    python reproduction/effective_rank/plot.py [--results results/effective_rank/results.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# NeurIPS style
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

# Color palette — colorblind-friendly
COLORS = {
    "PT": "#0072B2",
    "FT": "#D55E00",
    "FT_IO": "#009E73",
    "RI": "#CC79A7",
}

LABELS = {
    "PT_TS": "PT (TS)",
    "FT_TS": "FT (TS)",
    "FT_IO_TS": "FT$_{\\mathrm{IO}}$ (TS)",
    "RI_TS": "RI (TS)",
    "PT_Text": "PT (Text)",
    "FT_Text": "FT (Text)",
    "FT_IO_Text": "FT$_{\\mathrm{IO}}$ (Text)",
}

PAIR_LABELS = {
    "PT_vs_FT": "PT vs FT",
    "PT_vs_FT_IO": "PT vs FT$_{\\mathrm{IO}}$",
    "PT_vs_RI": "PT vs RI",
    "FT_vs_FT_IO": "FT vs FT$_{\\mathrm{IO}}$",
    "FT_vs_RI": "FT vs RI",
    "FT_IO_vs_RI": "FT$_{\\mathrm{IO}}$ vs RI",
}

PAIR_COLORS = {
    "PT_vs_FT": "#E69F00",
    "PT_vs_FT_IO": "#56B4E9",
    "PT_vs_RI": "#CC79A7",
    "FT_vs_FT_IO": "#009E73",
    "FT_vs_RI": "#F0E442",
    "FT_IO_vs_RI": "#999999",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def load_eigenvalues(path):
    with open(path) as f:
        return json.load(f)


# ── Plot 1: Effective Rank vs Layer ──────────────────────────────────────────

def plot_effective_rank(results, out_dir):
    erank = results["effective_rank"]
    n_layers = results["config"]["n_layers"]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ts_keys = ["PT_TS", "FT_TS", "FT_IO_TS", "RI_TS"]
    text_keys = ["PT_Text", "FT_Text", "FT_IO_Text"]

    for key in ts_keys:
        model = key.split("_")[0] if key != "FT_IO_TS" else "FT_IO"
        ax.plot(layers, erank[key], "-o", color=COLORS[model],
                markersize=3, linewidth=1.5, label=LABELS[key])

    for key in text_keys:
        model = key.split("_")[0] if key != "FT_IO_Text" else "FT_IO"
        ax.plot(layers, erank[key], "--s", color=COLORS[model],
                markersize=2.5, linewidth=1.2, alpha=0.8, label=LABELS[key])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective Rank")
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.legend(ncol=2, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig1_effective_rank.pdf")
    plt.savefig(f"{out_dir}/fig1_effective_rank.png")
    plt.close()
    print("  Plot 1: fig1_effective_rank.pdf")


# ── Plot 2: Subspace Alignment vs Layer ──────────────────────────────────────

def plot_subspace_alignment(results, out_dir):
    alignment = results["subspace_alignment"]
    baselines = results["random_baseline"]
    n_layers = results["config"]["n_layers"]
    layers = np.arange(n_layers)

    # Two panels: "vs PT" and "between finetuned models"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=True)

    # Panel A: comparisons involving PT
    pt_pairs = ["PT_vs_FT", "PT_vs_FT_IO", "PT_vs_RI"]
    for pair in pt_pairs:
        vals = [e["alignment"] for e in alignment[pair]]
        ax1.plot(layers, vals, "-o", color=PAIR_COLORS[pair],
                 markersize=3, linewidth=1.5, label=PAIR_LABELS[pair])

    # Panel B: between finetuned / RI
    other_pairs = ["FT_vs_FT_IO", "FT_vs_RI", "FT_IO_vs_RI"]
    for pair in other_pairs:
        vals = [e["alignment"] for e in alignment[pair]]
        ax2.plot(layers, vals, "-o", color=PAIR_COLORS[pair],
                 markersize=3, linewidth=1.5, label=PAIR_LABELS[pair])

    # Random baseline band on both panels
    all_ks = sorted(set(int(k) for k in baselines.keys()))
    if all_ks:
        max_baseline = max(baselines[str(k)]["mean"] + 2 * baselines[str(k)]["std"]
                          for k in all_ks)
        for ax in [ax1, ax2]:
            ax.axhspan(0, max_baseline, color="gray", alpha=0.15, label="Random baseline")

    for ax, title in [(ax1, "(a) Alignment with PT"), (ax2, "(b) Between finetuned models")]:
        ax.set_xlabel("Layer")
        ax.set_xlim(-0.5, n_layers - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    ax1.set_ylabel("Subspace Alignment")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig2_subspace_alignment.pdf")
    plt.savefig(f"{out_dir}/fig2_subspace_alignment.png")
    plt.close()
    print("  Plot 2: fig2_subspace_alignment.pdf")


# ── Plot 3: Eigenvalue Spectrum at Selected Layers ───────────────────────────

def plot_eigenvalue_spectrum(eigenvalue_data, out_dir):
    selected_layers = [0, 7, 14, 27]
    conditions = ["PT_TS", "FT_TS", "FT_IO_TS", "RI_TS", "PT_Text", "FT_Text", "FT_IO_Text"]

    fig, axes = plt.subplots(1, len(selected_layers), figsize=(7, 2.5), sharey=True)

    for ax, layer_idx in zip(axes, selected_layers):
        for key in conditions:
            if key not in eigenvalue_data:
                continue
            evals = np.array(eigenvalue_data[key][layer_idx])
            evals = evals[evals > 0]
            evals_sorted = np.sort(evals)[::-1]
            # Normalize
            evals_norm = evals_sorted / evals_sorted.sum()

            model = key.split("_")[0] if not key.startswith("FT_IO") else "FT_IO"
            is_text = key.endswith("Text")
            style = "--" if is_text else "-"
            alpha = 0.7 if is_text else 1.0

            ax.plot(evals_norm, style, color=COLORS[model], linewidth=1.0,
                    alpha=alpha, label=LABELS.get(key, key))

        ax.set_yscale("log")
        ax.set_xlim(0, 200)
        ax.set_title(f"Layer {layer_idx}", fontsize=9)
        ax.set_xlabel("Component")

    axes[0].set_ylabel("Normalized eigenvalue")

    # Single legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6.5,
              bbox_to_anchor=(0.5, 1.12), framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig3_eigenvalue_spectrum.pdf", bbox_inches="tight")
    plt.savefig(f"{out_dir}/fig3_eigenvalue_spectrum.png", bbox_inches="tight")
    plt.close()
    print("  Plot 3: fig3_eigenvalue_spectrum.pdf")


# ── Plot 4: Grouped Bar Chart at Representative Layer ────────────────────────

def plot_bar_comparison(results, out_dir):
    erank = results["effective_rank"]

    # Find layer with maximum divergence between PT and FT on TS
    pt_ts = np.array(erank["PT_TS"])
    ft_ts = np.array(erank["FT_TS"])
    mid_layer = int(np.argmax(np.abs(pt_ts - ft_ts)))

    models = ["PT", "FT", "FT_IO", "RI"]
    model_labels = ["PT", "FT", "FT$_{\\mathrm{IO}}$", "RI"]

    ts_vals = [erank[f"{m}_TS"][mid_layer] for m in models]
    text_vals = [erank.get(f"{m}_Text", [None]*28)[mid_layer] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 3))

    bars_ts = ax.bar(x - width/2, ts_vals, width, color=[COLORS[m] for m in models],
                     edgecolor="white", linewidth=0.5, label="Time Series")

    text_colors = [COLORS[m] for m in models]
    text_plot_vals = [v if v is not None else 0 for v in text_vals]
    bars_text = ax.bar(x + width/2, text_plot_vals, width,
                       color=text_colors, edgecolor="white", linewidth=0.5,
                       alpha=0.5, hatch="//", label="Text")

    # Hide RI text bar (no data)
    if text_vals[-1] is None:
        bars_text[-1].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Effective Rank")
    ax.set_title(f"Effective Rank at Layer {mid_layer}", fontsize=10)

    legend_elements = [
        Patch(facecolor="gray", edgecolor="white", label="TS"),
        Patch(facecolor="gray", edgecolor="white", alpha=0.5, hatch="//", label="Text"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig4_bar_comparison_layer{mid_layer}.pdf")
    plt.savefig(f"{out_dir}/fig4_bar_comparison_layer{mid_layer}.png")
    plt.close()
    print(f"  Plot 4: fig4_bar_comparison_layer{mid_layer}.pdf")


# ── Plot 5: Pairwise Alignment Heatmap ──────────────────────────────────────

def plot_alignment_heatmap(results, out_dir):
    alignment = results["subspace_alignment"]
    n_layers = results["config"]["n_layers"]

    pair_order = ["PT_vs_FT", "PT_vs_FT_IO", "FT_vs_FT_IO",
                  "PT_vs_RI", "FT_vs_RI", "FT_IO_vs_RI"]
    pair_labels_short = [
        "PT–FT", "PT–FT$_{\\mathrm{IO}}$", "FT–FT$_{\\mathrm{IO}}$",
        "PT–RI", "FT–RI", "FT$_{\\mathrm{IO}}$–RI",
    ]

    matrix = np.zeros((len(pair_order), n_layers))
    for i, pair in enumerate(pair_order):
        for entry in alignment[pair]:
            matrix[i, entry["layer"]] = entry["alignment"]

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_yticks(range(len(pair_order)))
    ax.set_yticklabels(pair_labels_short)
    ax.set_xlabel("Layer")
    ax.set_xticks(np.arange(0, n_layers, 4))

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Alignment", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig5_alignment_heatmap.pdf")
    plt.savefig(f"{out_dir}/fig5_alignment_heatmap.png")
    plt.close()
    print("  Plot 5: fig5_alignment_heatmap.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/effective_rank/results.json")
    parser.add_argument("--eigenvalues", default="results/effective_rank/eigenvalues.json")
    parser.add_argument("--out_dir", default="reproduction/effective_rank/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = load_results(args.results)
    print(f"Loaded results: {list(results['effective_rank'].keys())}")

    eigenvalue_data = None
    if os.path.exists(args.eigenvalues):
        eigenvalue_data = load_eigenvalues(args.eigenvalues)

    print("Generating plots...")
    plot_effective_rank(results, args.out_dir)
    plot_subspace_alignment(results, args.out_dir)
    if eigenvalue_data:
        plot_eigenvalue_spectrum(eigenvalue_data, args.out_dir)
    plot_bar_comparison(results, args.out_dir)
    plot_alignment_heatmap(results, args.out_dir)
    print("Done!")


if __name__ == "__main__":
    main()
