"""
Sine wave PCA evolution across 5 layers: LangInit (top) vs RandInit (bottom).
Reads from results/sine_evolution/results.json (all 28 layers, PT/FT/RI).

Usage:
    python3 reproduction/sine_evolution/plot_sine_evolution.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "results/sine_evolution/results.json"
OUTPUT_DIR = "results/sine_evolution"

LAYERS = [8, 13, 18]
MODEL_MAP = {
    "LangInit": "FT",
    "RandInit": "RI",
}

with open(DATA_PATH) as f:
    data = json.load(f)

phase = np.array(data["phase"])
models = data["models"]

n_layers = len(LAYERS)
fig, axes = plt.subplots(2, n_layers, figsize=(3.2, 2.4),
                          gridspec_kw={"hspace": 0.12, "wspace": 0.08})

for mi, (label, key) in enumerate(MODEL_MAP.items()):
    for ci, li in enumerate(LAYERS):
        ax = axes[mi, ci]
        proj = np.array(models[key][str(li)]["pca"])
        var_exp = models[key][str(li)]["pca_var"]

        mx = np.abs(proj).max()
        if mx > 0:
            proj = proj / mx

        for j in range(len(proj) - 1):
            c = plt.cm.hsv(phase[j])
            ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c,
                    linewidth=0.8, alpha=0.85, solid_capstyle="round")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#f8fafc")
        ax.patch.set_alpha(0.5)

        ax.text(0.97, 0.03, f"{var_exp:.0f}%", transform=ax.transAxes,
                fontsize=4.5, color="#64748b", ha="right", va="bottom",
                fontfamily="monospace")

        if mi == 0:
            ax.set_title(f"Layer {li}", fontsize=6.5, fontweight="600",
                          pad=4, color="#1e293b")
        if ci == 0:
            ax.set_ylabel(label, fontsize=8, fontweight="bold",
                           labelpad=3, color="#1e293b")

fig.patch.set_facecolor("white")
out_path = f"{OUTPUT_DIR}/fig_sine_evolution.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"Saved {out_path}")
