"""
Hidden-state trajectories for 5 synthetic inputs through PT, FT, RI.

Extracts hidden states at multiple layers, computes PCA and t-SNE projections.
Generates one figure per (layer, method) combination.

Usage:
    python3 reproduction/sine_evolution/run_synthetic.py
"""
import json
import os
import threading

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
SKIP = 5
LAYERS = [8, 13]
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0

MODEL_PATHS = {
    "Random": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-1",
    "Base": "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "LangInit": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420/checkpoint-8192",
    "RandInit": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-8192",
}

OUTPUT_DIR = "results/sine_evolution"

# ── Synthetic inputs ─────────────────────────────────────────────────────────

t_arr = np.arange(T, dtype=np.float32)

def make_sine():
    return np.sin(2 * np.pi * t_arr / PERIOD)

def make_square():
    return np.sign(np.sin(2 * np.pi * t_arr / PERIOD))

def make_sawtooth():
    return 2 * ((t_arr % PERIOD) / PERIOD) - 1

def make_two_freq():
    return np.sin(2 * np.pi * t_arr / 64) + 0.5 * np.sin(2 * np.pi * t_arr / 17)

def make_trend():
    return t_arr / T + 0.3 * np.sin(2 * np.pi * t_arr / 80)

INPUTS = [
    ("Sine", make_sine()),
    ("Square wave", make_square()),
    ("Sawtooth", make_sawtooth()),
    ("Two frequencies", make_two_freq()),
    ("Trend + oscillation", make_trend()),
]

# ── Tokenization ─────────────────────────────────────────────────────────────

def tokenize(signal):
    mean, std = float(np.mean(signal)), float(np.std(signal))
    if std < 1e-8:
        std = 1.0
    normed = (signal - mean) / std
    clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
    bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
    bins = np.clip(bins, 0, N_BINS - 1)
    return torch.from_numpy(bins).unsqueeze(0), normed


# ── Extract hidden states at target layers ───────────────────────────────────

@torch.no_grad()
def extract_layers(model_name, model_path, all_input_ids, device):
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    model = full_model.model
    model.eval()

    captured = {}
    handles = []
    for li in LAYERS:
        def make_hook(l):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[l] = o.detach().float().cpu()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))

    results = {}
    for name, input_ids in all_input_ids.items():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=input_ids.to(device), use_cache=False)
        results[name] = {li: captured[li].squeeze(0).numpy() for li in LAYERS}

    for h in handles:
        h.remove()
    del full_model, model
    torch.cuda.empty_cache()
    print(f"  {model_name}: extracted layers {LAYERS} for {len(results)} inputs", flush=True)
    return results


# ── Plotting helper ──────────────────────────────────────────────────────────

def plot_figure(all_proj, all_normed, phase, layer, method, tag=""):
    model_names = list(MODEL_PATHS.keys())
    n_inputs = len(INPUTS)
    n_models = len(model_names)

    fig, axes = plt.subplots(n_inputs, n_models + 1, figsize=(5.5, 5.5),
                              gridspec_kw={"hspace": 0.15, "wspace": 0.08,
                                           "width_ratios": [0.8] + [1] * n_models})

    # Input column
    for ri, (name, _) in enumerate(INPUTS):
        ax = axes[ri, 0]
        normed = all_normed[name]
        for j in range(SKIP, T - 1):
            c = plt.cm.hsv(phase[j - SKIP])
            ax.plot([t_arr[j], t_arr[j+1]], [normed[j], normed[j+1]],
                    color=c, linewidth=0.8, solid_capstyle="round")
        ax.set_xlim(0, T)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#f8fafc")
        ax.patch.set_alpha(0.5)
        ax.set_ylabel(name, fontsize=5.5, fontweight="600", rotation=90,
                       labelpad=3, color="#1e293b")
        if ri == 0:
            ax.set_title("Input", fontsize=6.5, fontweight="600", pad=4, color="#1e293b")

    # Projection columns
    method_upper = method.upper().replace("TSNE", "t-SNE")
    for mi, model_name in enumerate(model_names):
        for ri, (input_name, _) in enumerate(INPUTS):
            ax = axes[ri, mi + 1]
            entry = all_proj[model_name][input_name][layer][method]
            proj = np.array(entry["proj"])
            var_text = ""
            if method == "pca":
                var_text = f'{entry["var"]:.0f}%'

            mx = np.abs(proj).max()
            if mx > 0:
                proj = proj / mx

            for j in range(len(proj) - 1):
                c = plt.cm.hsv(phase[j])
                ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c, linewidth=0.7,
                        alpha=0.85, solid_capstyle="round")

            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("#f8fafc")
            ax.patch.set_alpha(0.5)
            if var_text:
                ax.text(0.97, 0.03, var_text, transform=ax.transAxes,
                        fontsize=4.5, color="#64748b", ha="right", va="bottom",
                        fontfamily="monospace")
            if ri == 0:
                ax.set_title(model_name, fontsize=6.5, fontweight="600",
                              pad=4, color="#1e293b")

    fig.patch.set_facecolor("white")
    out_path = f"{OUTPUT_DIR}/fig_synthetic_L{layer}_{method}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    phase = ((t_arr % PERIOD) / PERIOD)[SKIP:]

    # Tokenize
    all_input_ids = {}
    all_normed = {}
    for name, signal in INPUTS:
        ids, normed = tokenize(signal)
        all_input_ids[name] = ids
        all_normed[name] = normed

    # Extract hidden states (3 models in parallel)
    print("Extracting hidden states...", flush=True)
    all_hidden = {}
    lock = threading.Lock()
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    threads = []

    def worker(model_name, model_path, device):
        hs = extract_layers(model_name, model_path, all_input_ids, device)
        with lock:
            all_hidden[model_name] = hs

    for (model_name, model_path), gpu in zip(MODEL_PATHS.items(), gpus):
        t = threading.Thread(target=worker, args=(model_name, model_path, gpu))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Compute projections (PCA + t-SNE) for each layer
    print("\nComputing projections...", flush=True)
    all_proj = {}
    for model_name in MODEL_PATHS:
        all_proj[model_name] = {}
        for input_name, _ in INPUTS:
            all_proj[model_name][input_name] = {}
            for li in LAYERS:
                hs = all_hidden[model_name][input_name][li][SKIP:]

                pca = PCA(n_components=2)
                pca_proj = pca.fit_transform(hs)
                pca_var = pca.explained_variance_ratio_.sum() * 100

                tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                tsne_proj = tsne.fit_transform(hs)

                all_proj[model_name][input_name][li] = {
                    "pca": {"proj": pca_proj.tolist(), "var": float(pca_var)},
                    "tsne": {"proj": tsne_proj.tolist()},
                }

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "synthetic_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "config": {"T": T, "period": PERIOD, "skip": SKIP, "layers": LAYERS, "n_bins": N_BINS},
            "phase": phase.tolist(),
            "projections": all_proj,
        }, f)
    print(f"Results saved to {results_path}", flush=True)

    # Plot: one figure per (layer, method)
    print("\nPlotting...", flush=True)
    for li in LAYERS:
        for method in ["pca", "tsne"]:
            plot_figure(all_proj, all_normed, phase, li, method)

    print("Done!")


if __name__ == "__main__":
    main()
