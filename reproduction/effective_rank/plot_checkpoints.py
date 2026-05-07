"""
NeurIPS-quality plots for effective rank across training checkpoints.

Plots:
  1. Training trajectory: mean erank vs step (TS and Text, all 3 models)
  2. Per-layer heatmaps: erank(layer, step) for each model
  3. Forgetting ratio: text_erank / PT_text_erank across training
  4. Layer-resolved trajectories: selected layers across training steps
  5. RI collapse: dramatic rank reduction from random init
  6. Combined 2-panel: FT vs FT_IO text forgetting side-by-side

Usage:
    python reproduction/effective_rank/plot_checkpoints.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

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

COLORS = {"FT": "#D55E00", "FT_IO": "#009E73", "RI": "#CC79A7", "PT": "#0072B2"}
N_LAYERS = 28
OUT_DIR = "reproduction/effective_rank/plots"


def load_data():
    with open("results/effective_rank_checkpoints/results.json") as f:
        data = json.load(f)

    pt = data["PT"]
    models = {}  # model_name -> sorted list of (step, ts_erank[28], text_erank[28] or None)
    for key, val in data["checkpoints"].items():
        model = val["model"]
        if model not in models:
            models[model] = []
        models[model].append({
            "step": val["step"],
            "ts_erank": val["ts_erank"],
            "text_erank": val.get("text_erank"),
        })

    for model in models:
        models[model].sort(key=lambda x: x["step"])

    return pt, models


# ── Plot 1: Mean erank vs training step ──────────────────────────────────────

def plot_training_trajectory(pt, models, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.2))

    for model_name in ["FT", "FT_IO", "RI"]:
        entries = models[model_name]
        steps = [e["step"] for e in entries]
        means = [np.mean(e["ts_erank"]) for e in entries]
        ax1.plot(steps, means, "-o", color=COLORS[model_name], markersize=2,
                 linewidth=1.3, label=model_name.replace("FT_IO", "FT$_{\\mathrm{IO}}$"))

    ax1.axhline(np.mean(pt["ts_erank"]), color=COLORS["PT"], linestyle="--",
                linewidth=1, alpha=0.7, label="PT baseline")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Mean Effective Rank")
    ax1.set_title("(a) Time Series Input", fontsize=9)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=6.5)

    for model_name in ["FT", "FT_IO"]:
        entries = models[model_name]
        steps = [e["step"] for e in entries if e["text_erank"] is not None]
        means = [np.mean(e["text_erank"]) for e in entries if e["text_erank"] is not None]
        ax2.plot(steps, means, "-o", color=COLORS[model_name], markersize=2,
                 linewidth=1.3, label=model_name.replace("FT_IO", "FT$_{\\mathrm{IO}}$"))

    ax2.axhline(np.mean(pt["text_erank"]), color=COLORS["PT"], linestyle="--",
                linewidth=1, alpha=0.7, label="PT baseline")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Training Step")
    ax2.set_title("(b) Text Input (Forgetting)", fontsize=9)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=6.5)

    plt.tight_layout(pad=0.4)
    plt.savefig(f"{out_dir}/fig6_training_trajectory.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  Plot 1: fig6_training_trajectory")


# ── Plot 2: Per-layer heatmaps ───────────────────────────────────────────────

def plot_layer_heatmaps(pt, models, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5))

    configs = [
        ("FT", "ts_erank", "FT — TS"),
        ("FT", "text_erank", "FT — Text"),
        ("RI", "ts_erank", "RI — TS"),
        ("FT_IO", "ts_erank", "FT$_{\\mathrm{IO}}$ — TS"),
        ("FT_IO", "text_erank", "FT$_{\\mathrm{IO}}$ — Text"),
    ]

    for idx, (model_name, erank_key, title) in enumerate(configs):
        ax = axes.flat[idx]
        entries = [e for e in models[model_name] if e.get(erank_key) is not None]
        steps = [e["step"] for e in entries]
        matrix = np.array([e[erank_key] for e in entries])  # (n_steps, n_layers)

        im = ax.imshow(matrix.T, aspect="auto", cmap="viridis",
                       norm=mcolors.LogNorm(vmin=1, vmax=500),
                       interpolation="nearest", origin="lower")
        ax.set_xticks(range(0, len(steps), max(1, len(steps) // 5)))
        ax.set_xticklabels([str(steps[i]) for i in range(0, len(steps), max(1, len(steps) // 5))],
                           rotation=45, fontsize=6)
        ax.set_ylabel("Layer" if idx % 3 == 0 else "")
        ax.set_xlabel("Step")
        ax.set_title(title, fontsize=9)

    # Hide unused subplot
    axes.flat[5].axis("off")

    cbar = fig.colorbar(im, ax=axes.flat[5], shrink=0.8, location="left")
    cbar.set_label("Effective Rank", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig7_layer_heatmaps.png")
    plt.close()
    print("  Plot 2: fig7_layer_heatmaps")


# ── Plot 3: Forgetting ratio ────────────────────────────────────────────────

def plot_forgetting_ratio(pt, models, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    pt_text_mean = np.mean(pt["text_erank"])

    for model_name in ["FT", "FT_IO"]:
        entries = [e for e in models[model_name] if e["text_erank"] is not None]
        steps = [e["step"] for e in entries]
        ratios = [np.mean(e["text_erank"]) / pt_text_mean for e in entries]
        label = model_name.replace("FT_IO", "FT$_{\\mathrm{IO}}$")
        ax.plot(steps, ratios, "-o", color=COLORS[model_name], markersize=3,
                linewidth=1.5, label=label)

    ax.axhline(1.0, color=COLORS["PT"], linestyle="--", linewidth=1, alpha=0.7, label="PT (no forgetting)")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Text Erank / PT Text Erank")
    ax.set_title("Text Representation Retention")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig8_forgetting_ratio.png")
    plt.close()
    print("  Plot 3: fig8_forgetting_ratio")


# ── Plot 4: Layer-resolved trajectories ──────────────────────────────────────

def plot_layer_trajectories(pt, models, out_dir):
    selected_layers = [0, 7, 14, 21, 27]
    cmap = plt.cm.plasma
    layer_colors = {l: cmap(i / (len(selected_layers) - 1)) for i, l in enumerate(selected_layers)}

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8), sharey=True)
    model_names = ["FT", "FT_IO", "RI"]
    titles = ["FT", "FT$_{\\mathrm{IO}}$", "RI"]

    for ax, model_name, title in zip(axes, model_names, titles):
        entries = models[model_name]
        steps = [e["step"] for e in entries]

        for layer in selected_layers:
            vals = [e["ts_erank"][layer] for e in entries]
            ax.plot(steps, vals, "-", color=layer_colors[layer], linewidth=1.2,
                    alpha=0.9, label=f"L{layer}")
            # PT baseline for this layer
            ax.axhline(pt["ts_erank"][layer], color=layer_colors[layer],
                       linestyle=":", linewidth=0.5, alpha=0.4)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training Step")
        ax.set_title(title)

    axes[0].set_ylabel("Effective Rank")
    axes[0].legend(fontsize=6, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig9_layer_trajectories.png")
    plt.close()
    print("  Plot 4: fig9_layer_trajectories")


# ── Plot 5: RI collapse ─────────────────────────────────────────────────────

def plot_ri_collapse(pt, models, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.0))

    entries = models["RI"]
    steps = [e["step"] for e in entries]
    means = [np.mean(e["ts_erank"]) for e in entries]

    # Panel A: mean erank trajectory
    ax1.plot(steps, means, "-o", color=COLORS["RI"], markersize=4, linewidth=1.5)
    ax1.axhline(np.mean(pt["ts_erank"]), color=COLORS["PT"], linestyle="--",
                linewidth=1, alpha=0.7, label="PT baseline")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Mean Effective Rank")
    ax1.set_title("(a) RI Mean Erank Collapse")
    ax1.legend(framealpha=0.9)

    # Panel B: per-layer at selected steps
    step_indices = [0, 4, 7, 9, 13]  # steps 1, 16, 128, 512, 8192
    cmap = plt.cm.coolwarm_r
    for i, si in enumerate(step_indices):
        if si < len(entries):
            e = entries[si]
            color = cmap(i / (len(step_indices) - 1))
            ax2.plot(range(N_LAYERS), e["ts_erank"], "-", color=color,
                     linewidth=1.2, label=f"Step {e['step']}")

    ax2.plot(range(N_LAYERS), pt["ts_erank"], "--", color=COLORS["PT"],
             linewidth=1, alpha=0.7, label="PT")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effective Rank")
    ax2.set_yscale("log")
    ax2.set_title("(b) RI Per-Layer Profile")
    ax2.legend(fontsize=6, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig10_ri_collapse.png")
    plt.close()
    print("  Plot 5: fig10_ri_collapse")


# ── Plot 6: FT vs FT_IO text forgetting side-by-side ────────────────────────

def plot_forgetting_comparison(pt, models, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=True)

    pt_text = np.array(pt["text_erank"])

    # Panel A: FT text erank per layer across steps
    entries_ft = [e for e in models["FT"] if e["text_erank"] is not None]
    steps_ft = [e["step"] for e in entries_ft]
    # Select ~6 evenly spaced steps on log scale
    log_steps = np.logspace(0, np.log10(max(steps_ft)), 6).astype(int)
    selected_ft = []
    for target in log_steps:
        closest = min(entries_ft, key=lambda e: abs(e["step"] - target))
        if closest not in selected_ft:
            selected_ft.append(closest)

    cmap = plt.cm.Reds
    for i, e in enumerate(selected_ft):
        color = cmap(0.3 + 0.7 * i / (len(selected_ft) - 1))
        ax1.plot(range(N_LAYERS), e["text_erank"], "-", color=color,
                 linewidth=1.2, label=f"Step {e['step']}")

    ax1.plot(range(N_LAYERS), pt_text, "--", color=COLORS["PT"],
             linewidth=1, alpha=0.7, label="PT")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Effective Rank")
    ax1.set_yscale("log")
    ax1.set_title("(a) FT — Text Erank")
    ax1.legend(fontsize=5.5, loc="upper right", framealpha=0.9, ncol=2)

    # Panel B: FT_IO text erank per layer across steps
    entries_io = [e for e in models["FT_IO"] if e["text_erank"] is not None]
    steps_io = [e["step"] for e in entries_io]
    log_steps_io = np.logspace(0, np.log10(max(steps_io)), 6).astype(int)
    selected_io = []
    for target in log_steps_io:
        closest = min(entries_io, key=lambda e: abs(e["step"] - target))
        if closest not in selected_io:
            selected_io.append(closest)

    cmap = plt.cm.Greens
    for i, e in enumerate(selected_io):
        color = cmap(0.3 + 0.7 * i / (len(selected_io) - 1))
        ax2.plot(range(N_LAYERS), e["text_erank"], "-", color=color,
                 linewidth=1.2, label=f"Step {e['step']}")

    ax2.plot(range(N_LAYERS), pt_text, "--", color=COLORS["PT"],
             linewidth=1, alpha=0.7, label="PT")
    ax2.set_xlabel("Layer")
    ax2.set_yscale("log")
    ax2.set_title("(b) FT$_{\\mathrm{IO}}$ — Text Erank")
    ax2.legend(fontsize=5.5, loc="upper right", framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig11_forgetting_comparison.png")
    plt.close()
    print("  Plot 6: fig11_forgetting_comparison")


# ── Plot 7: TS vs Text erank scatter across training ─────────────────────────

def plot_ts_vs_text_scatter(pt, models, out_dir):
    fig, ax = plt.subplots(figsize=(4, 3.5))

    for model_name in ["FT", "FT_IO"]:
        entries = [e for e in models[model_name] if e["text_erank"] is not None]
        ts_means = [np.mean(e["ts_erank"]) for e in entries]
        text_means = [np.mean(e["text_erank"]) for e in entries]
        steps = [e["step"] for e in entries]

        scatter = ax.scatter(ts_means, text_means, c=np.log10(np.array(steps) + 1),
                            cmap="plasma" if model_name == "FT" else "winter",
                            s=25, alpha=0.8, edgecolors="white", linewidths=0.3,
                            zorder=3)

        # Connect with line
        label = model_name.replace("FT_IO", "FT$_{\\mathrm{IO}}$")
        ax.plot(ts_means, text_means, "-", color=COLORS[model_name],
                linewidth=0.8, alpha=0.5, label=label)

        # Annotate start and end
        ax.annotate(f"Step {steps[0]}", (ts_means[0], text_means[0]),
                   fontsize=5, color=COLORS[model_name], alpha=0.7,
                   xytext=(5, 5), textcoords="offset points")
        ax.annotate(f"Step {steps[-1]}", (ts_means[-1], text_means[-1]),
                   fontsize=5, color=COLORS[model_name], alpha=0.7,
                   xytext=(5, -8), textcoords="offset points")

    # PT reference point
    ax.scatter([np.mean(pt["ts_erank"])], [np.mean(pt["text_erank"])],
              marker="*", s=100, color=COLORS["PT"], zorder=5, label="PT")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean TS Effective Rank")
    ax.set_ylabel("Mean Text Effective Rank")
    ax.set_title("TS vs Text Rank Across Training")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig12_ts_vs_text_scatter.png")
    plt.close()
    print("  Plot 7: fig12_ts_vs_text_scatter")


# ── Plot 8: RI uniform collapse vs FT divergent layer dynamics ─────────────

def _get_fold_change(entries):
    """Returns (steps, log2_fc matrix, log2_final_per_layer)."""
    steps = np.array([e["step"] for e in entries])
    matrix = np.array([e["ts_erank"] for e in entries])
    baseline = matrix[0]
    baseline_safe = np.where(baseline < 1e-6, 1e-6, baseline)
    log2_fc = np.log2(matrix / baseline_safe[None, :])
    log2_final = np.log2(matrix[-1] / baseline_safe)
    return steps, log2_fc, log2_final


def plot_layer_dynamics_comparison(pt, models, out_dir):
    model_cfgs = [
        ("RI", "RI"),
        ("FT", "FT"),
        ("FT_IO", "FT$_{\\mathrm{IO}}$"),
    ]

    # Compute fold changes and find global vmax for shared color scale
    fold_data = {}
    global_vmax = 0
    for model_key, _ in model_cfgs:
        steps, log2_fc, log2_final = _get_fold_change(models[model_key])
        fold_data[model_key] = (steps, log2_fc, log2_final)
        global_vmax = max(global_vmax, np.max(np.abs(log2_fc)))

    # ── Version A: Side-by-side fold-change heatmaps + net-change bars ────
    fig, axes = plt.subplots(2, 3, figsize=(7, 4.5),
                              gridspec_kw={"height_ratios": [3, 1.2]})

    for col, (model_key, label) in enumerate(model_cfgs):
        steps, log2_fc, log2_final = fold_data[model_key]
        ax_heat = axes[0, col]
        ax_bar = axes[1, col]

        im = ax_heat.imshow(log2_fc.T, aspect="auto", cmap="RdBu_r",
                            vmin=-global_vmax, vmax=global_vmax,
                            interpolation="nearest", origin="lower")
        ax_heat.set_yticks(np.arange(0, N_LAYERS, 4))
        n_ticks = min(6, len(steps))
        tick_idx = np.linspace(0, len(steps) - 1, n_ticks, dtype=int)
        ax_heat.set_xticks(tick_idx)
        ax_heat.set_xticklabels([str(steps[i]) for i in tick_idx],
                                 rotation=45, fontsize=5.5)
        ax_heat.set_title(label, fontsize=10)
        if col == 0:
            ax_heat.set_ylabel("Layer")
        else:
            ax_heat.set_yticklabels([])
        ax_heat.set_xlabel("")

        bar_colors = ["#D55E00" if v > 0 else "#0072B2" for v in log2_final]
        ax_bar.barh(range(N_LAYERS), log2_final, color=bar_colors,
                    edgecolor="white", linewidth=0.3)
        ax_bar.axvline(0, color="black", linewidth=0.5)
        ax_bar.set_yticks(np.arange(0, N_LAYERS, 4))
        ax_bar.set_ylim(-0.5, N_LAYERS - 0.5)
        ax_bar.set_xlabel("log$_2$ fold change")
        if col == 0:
            ax_bar.set_ylabel("Layer")
        else:
            ax_bar.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.7, pad=0.02,
                        location="right")
    cbar.set_label("log$_2$(erank / erank$_0$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig13a_foldchange_comparison.png")
    plt.close()
    print("  Plot 8a: fig13a_foldchange_comparison")

    # ── Version B: Normalized spaghetti, side by side ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(7, 3.0), sharey=True)
    cmap = plt.cm.viridis

    for ax, (model_key, label) in zip(axes, model_cfgs):
        entries = models[model_key]
        steps = np.array([e["step"] for e in entries])
        matrix = np.array([e["ts_erank"] for e in entries])
        baseline = matrix[0]
        baseline_safe = np.where(baseline < 1e-6, 1e-6, baseline)

        for layer in range(N_LAYERS):
            normed = matrix[:, layer] / baseline_safe[layer]
            color = cmap(layer / (N_LAYERS - 1))
            ax.plot(steps, normed, "-", color=color, linewidth=0.8, alpha=0.8)

        ax.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training Step")
        ax.set_title(label, fontsize=10)

    axes[0].set_ylabel("Erank / Erank$_0$")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_LAYERS - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Layer", fontsize=8)
    cbar.set_ticks([0, 7, 14, 21, 27])

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig13b_normalized_spaghetti.png")
    plt.close()
    print("  Plot 8b: fig13b_normalized_spaghetti")

    # ── Version B2: Raw erank spaghetti, side by side (compact) ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), sharey=True)
    fig.subplots_adjust(right=0.87, wspace=0.12)
    cmap = plt.cm.viridis

    for ax, (model_key, label) in zip(axes, model_cfgs):
        entries = models[model_key]
        steps = np.array([e["step"] for e in entries])
        matrix = np.array([e["ts_erank"] for e in entries])

        for layer in range(N_LAYERS):
            color = cmap(layer / (N_LAYERS - 1))
            ax.plot(steps, matrix[:, layer], "-", color=color,
                    linewidth=0.7, alpha=0.8)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training Step")
        ax.set_title(label, fontsize=9)

    axes[0].set_ylabel("Effective Rank")

    cbar_ax = fig.add_axes([0.89, 0.18, 0.015, 0.65])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, N_LAYERS - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Layer", fontsize=7)
    cbar.set_ticks([0, 7, 14, 21, 27])
    cbar.ax.tick_params(labelsize=6)

    plt.savefig(f"{out_dir}/fig13b2_raw_spaghetti.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  Plot 8b2: fig13b2_raw_spaghetti")

    # ── Version B3: Net fold-change bars, stacked vertically (compact) ─────
    all_log2_finals = [fold_data[mk][2] for mk, _ in model_cfgs]
    bar_min = min(v.min() for v in all_log2_finals)
    bar_max = max(v.max() for v in all_log2_finals)
    bar_pad = (bar_max - bar_min) * 0.05
    bar_xlim = (bar_min - bar_pad, bar_max + bar_pad)

    fig, axes = plt.subplots(3, 1, figsize=(2.5, 4.0), sharex=True)

    for ax, (model_key, label) in zip(axes, model_cfgs):
        _, _, log2_final = fold_data[model_key]
        bar_colors = ["#D55E00" if v > 0 else "#0072B2" for v in log2_final]
        ax.barh(range(N_LAYERS), log2_final, color=bar_colors,
                edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_yticks(np.arange(0, N_LAYERS, 7))
        ax.set_ylim(-0.5, N_LAYERS - 0.5)
        ax.set_xlim(bar_xlim)
        ax.set_ylabel("Layer")
        ax.set_title(label, fontsize=9)

    axes[-1].set_xlabel("log$_2$ fold change")

    plt.tight_layout(pad=0.3)
    plt.savefig(f"{out_dir}/fig13b3_foldchange_bars.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    print("  Plot 8b3: fig13b3_foldchange_bars")

    # ── Version C: Net change profile overlay ─────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    for model_key, label in model_cfgs:
        _, _, log2_final = fold_data[model_key]
        ax.plot(range(N_LAYERS), log2_final, "-o", color=COLORS[model_key],
                markersize=3, linewidth=1.5, label=label)

    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.fill_between(range(N_LAYERS), 0, 0, alpha=0)  # dummy for axis
    ax.set_xlabel("Layer")
    ax.set_ylabel("log$_2$(erank$_{\\mathrm{final}}$ / erank$_0$)")
    ax.set_title("Net Effective Rank Change per Layer", fontsize=10)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/fig13c_net_change_profile.png")
    plt.close()
    print("  Plot 8c: fig13c_net_change_profile")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pt, models = load_data()
    print("Generating checkpoint plots...")

    plot_training_trajectory(pt, models, OUT_DIR)
    plot_layer_heatmaps(pt, models, OUT_DIR)
    plot_forgetting_ratio(pt, models, OUT_DIR)
    plot_layer_trajectories(pt, models, OUT_DIR)
    plot_ri_collapse(pt, models, OUT_DIR)
    plot_forgetting_comparison(pt, models, OUT_DIR)
    plot_ts_vs_text_scatter(pt, models, OUT_DIR)
    plot_layer_dynamics_comparison(pt, models, OUT_DIR)

    print("Done!")


if __name__ == "__main__":
    main()
