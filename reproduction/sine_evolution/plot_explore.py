"""
Exploration plots: 4x7 grid per model for PCA and t-SNE.
Reads from results/sine_evolution/results.json.

Usage:
    python3 reproduction/sine_evolution/plot_explore.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_LAYERS = 28
OUTPUT_DIR = "results/sine_evolution"
DATA_PATH = "results/sine_evolution/results.json"

with open(DATA_PATH) as f:
    data = json.load(f)

phase = data["phase"]
models = data["models"]


def plot_model_grid(model_name, method="pca"):
    nrows, ncols = 4, 7
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))

    for li in range(N_LAYERS):
        row, col = li // ncols, li % ncols
        ax = axes[row, col]

        proj = np.array(models[model_name][str(li)][method])
        mx = np.abs(proj).max()
        if mx > 0:
            proj = proj / mx

        for j in range(len(proj) - 1):
            c = plt.cm.hsv(phase[j])
            ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c,
                    linewidth=0.8, alpha=0.85, solid_capstyle="round")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#f8fafc")

        title = f"L{li}"
        if method == "pca":
            var = models[model_name][str(li)]["pca_var"]
            title += f" ({var:.0f}%)"
        ax.set_title(title, fontsize=8, pad=3)

    fig.suptitle(f"{model_name} — {method.upper()} of sine wave (period=64)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("white")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = f"{OUTPUT_DIR}/{model_name}_{method}_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  {out}")


for name in ["PT", "FT", "RI"]:
    for method in ["pca", "tsne"]:
        plot_model_grid(name, method)

print("Done!")
