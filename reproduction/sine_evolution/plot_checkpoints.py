"""
Exploration plots for sine wave representations across training checkpoints.

For each model family (FT, FT_IO, RI), generates:
  1. A grid: rows = checkpoints (time), columns = layers
  2. A PCA variance heatmap: step × layer

Usage:
    python3 reproduction/sine_evolution/plot_checkpoints.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/sine_evolution_checkpoints"
DATA_PATH = "results/sine_evolution_checkpoints/results.json"

print("Loading data...", flush=True)
with open(DATA_PATH) as f:
    data = json.load(f)

phase = np.array(data["phase"])
pt_data = data["PT"]
checkpoints = data["checkpoints"]
N_LAYERS = data["config"]["n_layers"]

# Group by model family
families = {}
for key, val in checkpoints.items():
    model = val["model"]
    if model not in families:
        families[model] = []
    families[model].append(val)

for model in families:
    families[model].sort(key=lambda x: x["step"])

LAYERS_TO_SHOW = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 27]


def plot_trajectory_grid(model_name):
    """Grid: rows=checkpoints, cols=selected layers. Shows PCA trajectory evolution."""
    entries = families[model_name]
    n_ckpts = len(entries)
    n_cols = len(LAYERS_TO_SHOW)

    fig, axes = plt.subplots(n_ckpts + 1, n_cols,
                              figsize=(2.0 * n_cols, 1.8 * (n_ckpts + 1)))

    # First row: PT baseline
    for ci, li in enumerate(LAYERS_TO_SHOW):
        ax = axes[0, ci]
        proj = np.array(pt_data[str(li)]["pca"])
        var = pt_data[str(li)]["pca_var"]
        _plot_trajectory(ax, proj, phase, var)
        if ci == 0:
            ax.set_ylabel("PT", fontsize=8, fontweight="bold", labelpad=3)
        ax.set_title(f"L{li}", fontsize=7, pad=2)

    # Checkpoint rows
    for ri, entry in enumerate(entries):
        step = entry["step"]
        for ci, li in enumerate(LAYERS_TO_SHOW):
            ax = axes[ri + 1, ci]
            proj = np.array(entry["layers"][str(li)]["pca"])
            var = entry["layers"][str(li)]["pca_var"]
            _plot_trajectory(ax, proj, phase, var)
            if ci == 0:
                ax.set_ylabel(f"s{step}", fontsize=6, fontweight="bold",
                              labelpad=3, rotation=0, ha="right", va="center")

    fig.suptitle(f"{model_name} — Sine wave PCA across training",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0.03, 0, 1, 0.99])

    out = f"{OUTPUT_DIR}/{model_name}_evolution_grid.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {out}")


def _plot_trajectory(ax, proj, phase, var):
    mx = np.abs(proj).max()
    if mx > 0:
        proj = proj / mx
    for j in range(len(proj) - 1):
        c = plt.cm.hsv(phase[j])
        ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c,
                linewidth=0.5, alpha=0.85, solid_capstyle="round")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f8fafc")
    ax.text(0.97, 0.03, f"{var:.0f}%", transform=ax.transAxes,
            fontsize=4, color="#64748b", ha="right", va="bottom",
            fontfamily="monospace")


def plot_variance_heatmap(model_name):
    """Heatmap: x=layer, y=step, color=PCA explained variance."""
    entries = families[model_name]
    steps = [e["step"] for e in entries]
    matrix = np.zeros((len(entries), N_LAYERS))

    for ri, entry in enumerate(entries):
        for li in range(N_LAYERS):
            matrix[ri, li] = entry["layers"][str(li)]["pca_var"]

    fig, ax = plt.subplots(figsize=(8, max(3, len(entries) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=100)

    ax.set_xticks(range(0, N_LAYERS, 2))
    ax.set_xticklabels(range(0, N_LAYERS, 2), fontsize=6)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([str(s) for s in steps], fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Training Step")
    ax.set_title(f"{model_name} — PCA Explained Variance (%) per layer × step",
                 fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% variance in 2 PCs", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Add PT baseline as a separate row annotation
    pt_vars = [pt_data[str(li)]["pca_var"] for li in range(N_LAYERS)]
    ax.text(-0.02, 1.02, f"PT baseline mean: {np.mean(pt_vars):.0f}%",
            transform=ax.transAxes, fontsize=7, color="#666", va="bottom")

    fig.patch.set_facecolor("white")
    out = f"{OUTPUT_DIR}/{model_name}_variance_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {out}")


# ── Run ──────────────────────────────────────────────────────────────────────

print("Plotting...", flush=True)
for model_name in ["FT", "FT_IO", "RI"]:
    plot_trajectory_grid(model_name)
    plot_variance_heatmap(model_name)

print("Done!")
