"""
Follow-up geometry analysis: WHERE on the 1D manifold do different inputs land?

The main analysis showed text@PT and rand@PT both collapse to ~1D trajectories.
But text produces diverse TS while random tokens collapse. The difference must be
in the SPREAD of inputs along the manifold, not the manifold shape itself.

This script:
1. Projects many sequences onto their shared PC1 at each layer
2. Measures the variance/spread of PC1 projections across sequences
3. Compares: do text inputs spread more than random inputs along PC1?
4. Visualizes the PC1 trajectories for text vs random

Usage:
    /usr/bin/python3 scripts/geometry_followup.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.wikitext import load_wikitext_sequences

T = 512
N_LAYERS = 28
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/geometry"
N_SEQS = 100  # more sequences for better spread statistics
SEED = 42


def extract_single(model, token_ids, device):
    captured = {}; handles = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook(m, i, o):
                captured[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=ids, use_cache=False)
    # Skip position 0 (BOS outlier with 200x norm)
    layers = [captured[i].squeeze(0)[1:, :].float().cpu() for i in range(N_LAYERS)]
    for h in handles: h.remove()
    return layers


def main():
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # Load data
    print("Loading data...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=N_SEQS, seq_len=T, hf_token=hf_token)
    rng = np.random.default_rng(SEED)
    rand_seqs = [{"input_ids": rng.integers(0, 151936, size=T).astype(np.int64)} for _ in range(N_SEQS)]
    print(f"  WikiText: {len(wiki_seqs)}, Random: {len(rand_seqs)}")

    # Load PT model
    print("Loading PT model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16).model.to(DEVICE).eval()

    # For select layers, collect PC1 projections across all sequences
    analysis_layers = [0, 2, 4, 8, 12, 16, 20, 24, 27]

    # First pass: compute shared PCA basis from a few sequences
    print("Computing PCA basis...", flush=True)
    pca_basis = {}
    basis_seqs = wiki_seqs[:10] + rand_seqs[:10]  # mix for shared basis
    for li in analysis_layers:
        all_h = []
        for seq in basis_seqs:
            layers = extract_single(model, seq["input_ids"], DEVICE)
            all_h.append(layers[li])
        stacked = torch.cat(all_h, dim=0).numpy()  # (20*T, D)
        stacked_c = stacked - stacked.mean(axis=0)
        pca = PCA(n_components=3)
        pca.fit(stacked_c)
        pca_basis[li] = pca
        print(f"  L{li}: PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%")

    # Second pass: project all sequences
    print("\nProjecting text sequences...", flush=True)
    text_projections = {li: [] for li in analysis_layers}
    for si, seq in enumerate(wiki_seqs):
        if si % 20 == 0:
            print(f"  text {si}/{N_SEQS}", flush=True)
        layers = extract_single(model, seq["input_ids"], DEVICE)
        for li in analysis_layers:
            h = layers[li].numpy()
            h_c = h - pca_basis[li].mean_
            proj = h_c @ pca_basis[li].components_[0]  # project onto PC1: (T,)
            text_projections[li].append(proj)

    print("Projecting random sequences...", flush=True)
    rand_projections = {li: [] for li in analysis_layers}
    for si, seq in enumerate(rand_seqs):
        if si % 20 == 0:
            print(f"  rand {si}/{N_SEQS}", flush=True)
        layers = extract_single(model, seq["input_ids"], DEVICE)
        for li in analysis_layers:
            h = layers[li].numpy()
            h_c = h - pca_basis[li].mean_
            proj = h_c @ pca_basis[li].components_[0]
            rand_projections[li].append(proj)

    del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ Analysis ═══
    print("\nAnalyzing spread...", flush=True)

    results = {}
    for li in analysis_layers:
        text_projs = np.stack(text_projections[li])  # (N, T)
        rand_projs = np.stack(rand_projections[li])  # (N, T)

        # Inter-sequence spread: for each timestep, how much do different sequences vary?
        text_inter_var = text_projs.var(axis=0).mean()  # mean over timesteps of cross-seq variance
        rand_inter_var = rand_projs.var(axis=0).mean()

        # Intra-sequence spread: for each sequence, how much does PC1 vary over time?
        text_intra_var = text_projs.var(axis=1).mean()  # mean over sequences of within-seq variance
        rand_intra_var = rand_projs.var(axis=1).mean()

        # Range of PC1 values across all sequences
        text_range = float(text_projs.max() - text_projs.min())
        rand_range = float(rand_projs.max() - rand_projs.min())

        # Correlation between sequences: mean pairwise correlation of PC1 trajectories
        text_corrs = np.corrcoef(text_projs[:50])  # 50x50 corr matrix
        rand_corrs = np.corrcoef(rand_projs[:50])
        # Mean off-diagonal correlation
        mask = ~np.eye(50, dtype=bool)
        text_mean_corr = float(text_corrs[mask].mean())
        rand_mean_corr = float(rand_corrs[mask].mean())

        results[li] = {
            "text_inter_var": float(text_inter_var),
            "rand_inter_var": float(rand_inter_var),
            "text_intra_var": float(text_intra_var),
            "rand_intra_var": float(rand_intra_var),
            "text_range": text_range,
            "rand_range": rand_range,
            "text_mean_corr": text_mean_corr,
            "rand_mean_corr": rand_mean_corr,
            "inter_var_ratio": float(text_inter_var / rand_inter_var) if rand_inter_var > 0 else 0,
        }

        print(f"  L{li}: text_inter={text_inter_var:.2f} rand_inter={rand_inter_var:.2f} "
              f"ratio={text_inter_var/rand_inter_var:.2f}x  "
              f"text_corr={text_mean_corr:.3f} rand_corr={rand_mean_corr:.3f}")

    # ═══ Plots ═══
    print("\nGenerating plots...", flush=True)

    # Plot 1: PC1 trajectories — 20 text vs 20 random, at key layers
    plot_layers = [2, 8, 16, 27]
    fig, axes = plt.subplots(2, len(plot_layers), figsize=(5*len(plot_layers), 8))
    fig.suptitle("PC1 Projections: Text (top) vs Random Tokens (bottom) through PT\n"
                 "Same 1D manifold, but text spreads more — different inputs reach different positions",
                 fontsize=13, fontweight='bold')

    for col, li in enumerate(plot_layers):
        text_projs = np.stack(text_projections[li])
        rand_projs = np.stack(rand_projections[li])

        for i in range(min(20, N_SEQS)):
            axes[0, col].plot(text_projs[i], alpha=0.15, linewidth=0.5, color='#2196F3')
        axes[0, col].set_title(f"Layer {li}", fontsize=11, fontweight='bold')
        axes[0, col].set_ylabel("PC1 value") if col == 0 else None

        for i in range(min(20, N_SEQS)):
            axes[1, col].plot(rand_projs[i], alpha=0.15, linewidth=0.5, color='#FF9800')
        axes[1, col].set_xlabel("Timestep")
        axes[1, col].set_ylabel("PC1 value") if col == 0 else None

        # Shared y-limits
        ymin = min(text_projs[:20].min(), rand_projs[:20].min())
        ymax = max(text_projs[:20].max(), rand_projs[:20].max())
        axes[0, col].set_ylim(ymin, ymax)
        axes[1, col].set_ylim(ymin, ymax)

    axes[0, 0].set_ylabel("Text@PT\n(PC1 value)", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Rand@PT\n(PC1 value)", fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/pc1_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Inter-sequence variance across layers
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Spread Along the 1D Manifold: Text vs Random Tokens", fontsize=14, fontweight='bold')

    lx = analysis_layers
    axes[0].plot(lx, [results[li]["text_inter_var"] for li in lx], 'o-', color='#2196F3', label="Text@PT")
    axes[0].plot(lx, [results[li]["rand_inter_var"] for li in lx], 'o-', color='#FF9800', label="Rand@PT")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Inter-sequence variance on PC1")
    axes[0].set_title("How much do different inputs spread\nalong PC1? (higher = more diverse)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(lx, [results[li]["text_intra_var"] for li in lx], 'o-', color='#2196F3', label="Text@PT")
    axes[1].plot(lx, [results[li]["rand_intra_var"] for li in lx], 'o-', color='#FF9800', label="Rand@PT")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Intra-sequence variance on PC1")
    axes[1].set_title("How much does PC1 vary within\na single sequence? (temporal variation)")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(lx, [results[li]["text_mean_corr"] for li in lx], 'o-', color='#2196F3', label="Text@PT")
    axes[2].plot(lx, [results[li]["rand_mean_corr"] for li in lx], 'o-', color='#FF9800', label="Rand@PT")
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Mean pairwise correlation")
    axes[2].set_title("Are PC1 trajectories correlated across\nsequences? (lower = more diverse)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/pc1_spread.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Variance ratio (text/rand) across layers
    fig, ax = plt.subplots(figsize=(10, 5))
    ratios = [results[li]["inter_var_ratio"] for li in lx]
    ax.bar(range(len(lx)), ratios, color=['#9C27B0']*len(lx), tick_label=[f"L{li}" for li in lx])
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel("Variance ratio (text / random)")
    ax.set_title("Text produces Nx more spread along PC1 than random tokens\n"
                 "(>1 means text is more diverse, <1 means random is more diverse)",
                 fontsize=12, fontweight='bold')
    for i, v in enumerate(ratios):
        ax.text(i, v + 0.05, f"{v:.1f}x", ha='center', fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/variance_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save results
    with open(f"{OUT_DIR}/data/followup_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("FOLLOW-UP: PC1 SPREAD ANALYSIS")
    print(f"{'='*70}")
    print(f"\n{'Layer':>5} {'Text Inter':>12} {'Rand Inter':>12} {'Ratio':>8} "
          f"{'Text Corr':>10} {'Rand Corr':>10}")
    print("-" * 60)
    for li in analysis_layers:
        r = results[li]
        print(f"  L{li:<3} {r['text_inter_var']:>12.2f} {r['rand_inter_var']:>12.2f} "
              f"{r['inter_var_ratio']:>7.1f}x {r['text_mean_corr']:>10.3f} {r['rand_mean_corr']:>10.3f}")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
