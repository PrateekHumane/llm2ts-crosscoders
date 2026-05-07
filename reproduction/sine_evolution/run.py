"""
Sine wave hidden-state analysis across all 28 layers for PT, FT, RI.

Generates a sine wave (period 64, length 512), tokenizes it,
passes through each model, extracts hidden states at all layers,
then saves 2D PCA and t-SNE projections colored by input phase.

Outputs all-layer grids for exploration, then user picks final layers.

Usage:
    python reproduction/sine_evolution/run.py
"""
import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────

T = 512
PERIOD = 64
SKIP = 5  # skip first positions (attention sink)
N_LAYERS = 28
HIDDEN_SIZE = 1024
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0

MODEL_PATHS = {
    "PT": "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420/checkpoint-8192",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-8192",
}

OUTPUT_DIR = "results/sine_evolution"

# ── Tokenization ─────────────────────────────────────────────────────────────

def make_sine_tokens():
    t_arr = np.arange(T, dtype=np.float32)
    sine = np.sin(2 * np.pi * t_arr / PERIOD)

    mean = float(np.mean(sine))
    std = float(np.std(sine))
    if std < 1e-8:
        std = 1.0
    normed = (sine - mean) / std

    clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
    bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
    bins = np.clip(bins, 0, N_BINS - 1)

    phase = (t_arr % PERIOD) / PERIOD
    return torch.from_numpy(bins).unsqueeze(0), phase


# ── Extract hidden states ────────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model_name, model_path, input_ids, device):
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model = full_model.model
    model.eval()

    captured = {}
    handles = []
    for li in range(N_LAYERS):
        def make_hook(l):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[l] = o.detach().float().cpu()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=input_ids.to(device), use_cache=False)

    for h in handles:
        h.remove()

    hidden_states = {}
    for li in range(N_LAYERS):
        hidden_states[li] = captured[li].squeeze(0).numpy()  # (T, 1024)

    del full_model, model
    torch.cuda.empty_cache()

    print(f"  {model_name}: extracted {N_LAYERS} layers", flush=True)
    return hidden_states


# ── PCA + t-SNE projections ──────────────────────────────────────────────────

def compute_projections(hidden_states, phase):
    phase_clean = phase[SKIP:]
    results = {}

    for li in range(N_LAYERS):
        hs = hidden_states[li][SKIP:]  # (T-SKIP, 1024)

        pca = PCA(n_components=2)
        pca_proj = pca.fit_transform(hs)
        pca_var = pca.explained_variance_ratio_.sum() * 100

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_proj = tsne.fit_transform(hs)

        results[li] = {
            "pca": pca_proj.tolist(),
            "pca_var": float(pca_var),
            "tsne": tsne_proj.tolist(),
        }

    return results, phase_clean.tolist()


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_grid(all_results, phase, method="pca", suffix=""):
    """Plot all 28 layers × N models grid."""
    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(n_models, N_LAYERS, figsize=(42, 3 * n_models),
                              gridspec_kw={"hspace": 0.15, "wspace": 0.08})
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for mi, name in enumerate(model_names):
        for li in range(N_LAYERS):
            ax = axes[mi, li]
            proj = np.array(all_results[name][str(li)][method])

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
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("#f8fafc")

            if method == "pca":
                var = all_results[name][str(li)]["pca_var"]
                ax.text(0.97, 0.03, f"{var:.0f}%", transform=ax.transAxes,
                        fontsize=5, color="#64748b", ha="right", va="bottom",
                        fontfamily="monospace")

            if mi == 0:
                ax.set_title(f"L{li}", fontsize=6, pad=2)
            if li == 0:
                ax.set_ylabel(name, fontsize=9, fontweight="bold", labelpad=3)

    method_label = "PCA" if method == "pca" else "t-SNE"
    fig.suptitle(f"Sine wave (period={PERIOD}) — {method_label} projections, all layers",
                 fontsize=12, y=1.02)
    fig.patch.set_facecolor("white")

    out_path = f"{OUTPUT_DIR}/grid_{method}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_ids, phase = make_sine_tokens()
    print(f"Sine wave: T={T}, period={PERIOD}, skip={SKIP}", flush=True)

    # Run models in parallel on separate GPUs
    all_hidden = {}
    gpus = ["cuda:0", "cuda:1", "cuda:2"]
    threads = []
    lock = threading.Lock()

    def worker(name, path, device):
        hs = extract_hidden_states(name, path, input_ids, device)
        with lock:
            all_hidden[name] = hs

    for (name, path), gpu in zip(MODEL_PATHS.items(), gpus):
        t = threading.Thread(target=worker, args=(name, path, gpu))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Compute projections
    print("\nComputing projections...", flush=True)
    all_results = {}
    phase_clean = None
    for name in MODEL_PATHS:
        results, pc = compute_projections(all_hidden[name], phase)
        all_results[name] = {str(k): v for k, v in results.items()}
        phase_clean = pc

    # Save raw results
    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump({"phase": phase_clean, "models": all_results}, f)
    print(f"Results saved to {out_path}", flush=True)

    # Plot grids
    print("\nPlotting grids...", flush=True)
    plot_grid(all_results, phase_clean, method="pca")
    plot_grid(all_results, phase_clean, method="tsne")

    print("\nDone! Check results/sine_evolution/ for exploration grids.")


if __name__ == "__main__":
    main()
