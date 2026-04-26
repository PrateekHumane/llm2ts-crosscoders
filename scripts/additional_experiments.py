"""
Additional NeurIPS experiments:
  1. Train-Free Random Projection Baseline
  2. Hidden State Diversity Analysis (PCA, effective rank, participation ratio)
  3. Spectral Alignment Analysis (PSD comparison with real TS)

Processes each ablation condition sequentially to manage memory.
Saves tables (CSV + JSON), plots (PNG), and summaries.

Usage:
    /usr/bin/python3 scripts/additional_experiments.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, time, gc, csv

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from transformers import AutoModelForCausalLM, AutoConfig
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences
from scripts.mapping_experiment import load_ts_windows, compute_acf

# ── Constants ──
D_CONCAT = 28 * 1024
T = 512
N_SEQS = 500        # sequences per condition (enough for stats, manageable in RAM)
N_RAND_PROJ = 50    # random projections for Exp 1
DEVICE = torch.device("cuda:0")
SEED = 42

ABLATION_DIR = "mapping_results/ablation"
OUT_DIR = "mapping_results/additional_experiments"

MODELS = {
    "text_PT":       {"model_type": "pt",          "tokens": "text"},
    "text_RandInit": {"model_type": "random_init",  "tokens": "text"},
    "rand_PT":       {"model_type": "pt",          "tokens": "random"},
    "rand_RandInit": {"model_type": "random_init",  "tokens": "random"},
}

COLORS = {
    "text_PT": "#2196F3",
    "text_RandInit": "#FF9800",
    "rand_PT": "#4CAF50",
    "rand_RandInit": "#E91E63",
    "real_TS": "#000000",
}

LABELS = {
    "text_PT": "Text + PT",
    "text_RandInit": "Text + RandInit",
    "rand_PT": "Random + PT",
    "rand_RandInit": "Random + RandInit",
    "real_TS": "Real TS",
}


# ═══════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════

def extract_all_layers(model, sequences, device, desc=""):
    """Extract all 28 layers concatenated. Returns (N, T, D_CONCAT) float16."""
    captured = {}
    handles = []
    for li in range(28):
        def make_hook(idx):
            def hook(m, i, o):
                captured[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))
    all_hs = []
    N = len(sequences)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, 8):
            batch = sequences[start:start + 8]
            ids = torch.tensor(np.stack([s["input_ids"] for s in batch]),
                               dtype=torch.long, device=device)
            model(input_ids=ids, use_cache=False)
            layers = [captured[i].float().cpu()[:, :T, :] for i in range(28)]
            all_hs.append(torch.cat(layers, dim=-1).half())
            captured.clear()
            if start % 80 == 0 and start > 0:
                print(f"    {desc} {start}/{N} ({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    result = torch.cat(all_hs, dim=0)
    print(f"    {desc} Done: {result.shape} ({time.time()-t0:.0f}s)", flush=True)
    return result


def predict_with_mapper(mapper, h_data, device):
    """Generate normalized predictions from hidden states."""
    mapper.eval()
    all_pred = []
    with torch.no_grad():
        for start in range(0, h_data.shape[0], 32):
            h_b = h_data[start:start + 32].float().to(device)
            yr = mapper(h_b).squeeze(-1)
            ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
            all_pred.append(((yr - yr.mean(-1, keepdim=True)) / ys).cpu())
    return torch.cat(all_pred, dim=0)


def compute_nn_metrics(pred, ts_bank, ts_labels):
    """Compute NN distance, unique matches, cluster coverage, entropy.
    Uses batched GPU computation for speed."""
    N = pred.shape[0]
    ts_gpu = ts_bank.to(DEVICE)
    nn_indices = []; nn_dists = []
    for start in range(0, N, 64):
        batch = pred[start:start + 64].to(DEVICE)  # (B, T)
        # (B, 1, T) - (1, N_ts, T) -> (B, N_ts, T) -> (B, N_ts)
        d2 = ((batch.unsqueeze(1) - ts_gpu.unsqueeze(0)) ** 2).mean(-1)
        nn_dists.append(d2.min(dim=1).values.cpu())
        nn_indices.append(d2.argmin(dim=1).cpu())
    nn_dists = torch.cat(nn_dists).numpy()
    nn_indices = torch.cat(nn_indices).numpy()

    unique_set = set(nn_indices.tolist())
    matched_clusters = set(ts_labels[list(unique_set)])

    # Entropy of match distribution
    counts = np.bincount(nn_indices, minlength=ts_bank.shape[0])
    probs = counts[counts > 0] / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(N)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "nn_mean": float(nn_dists.mean()),
        "nn_median": float(np.median(nn_dists)),
        "n_unique": len(unique_set),
        "clusters": int(len(matched_clusters)),
        "entropy": float(norm_entropy),
    }


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Random Projection Baseline
# ═══════════════════════════════════════════════════════════════

def exp1_random_projections(h_data, ts_bank, ts_labels, name, rng):
    """Run N_RAND_PROJ random projections and evaluate each."""
    N, T_len, D = h_data.shape
    results = []

    for proj_i in range(N_RAND_PROJ):
        # Sample random W ~ N(0, 1/D) for stable variance
        W = torch.from_numpy(rng.standard_normal((1, D)).astype(np.float32)) / np.sqrt(D)

        # Project: (N, T, D) @ (D, 1) -> (N, T, 1) -> (N, T)
        all_pred = []
        for start in range(0, N, 32):
            h_b = h_data[start:start + 32].float()
            yr = (h_b @ W.T).squeeze(-1)  # (batch, T)
            ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
            all_pred.append((yr - yr.mean(-1, keepdim=True)) / ys)
        pred = torch.cat(all_pred, dim=0)

        metrics = compute_nn_metrics(pred, ts_bank, ts_labels)
        results.append(metrics)

        if proj_i % 10 == 0:
            print(f"    [{name}] proj {proj_i}/{N_RAND_PROJ}: "
                  f"NN={metrics['nn_mean']:.4f} unique={metrics['n_unique']}", flush=True)

    # Aggregate
    agg = {}
    for key in results[0]:
        vals = [r[key] for r in results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
        agg[f"{key}_min"] = float(np.min(vals))
        agg[f"{key}_max"] = float(np.max(vals))

    return agg, results


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Hidden State Diversity Analysis
# ═══════════════════════════════════════════════════════════════

def exp2_diversity_analysis(h_data, name, n_pca_samples=50000):
    """PCA spectrum, effective rank, participation ratio."""
    N, T_len, D = h_data.shape

    # Subsample for PCA (N*T can be ~256K, use subset)
    h_flat = h_data.reshape(-1, D).float()  # (N*T, D)
    n_total = h_flat.shape[0]
    if n_total > n_pca_samples:
        idx = torch.randperm(n_total)[:n_pca_samples]
        h_sub = h_flat[idx].numpy()
    else:
        h_sub = h_flat.numpy()

    print(f"    [{name}] PCA on {h_sub.shape} ...", flush=True)
    t0 = time.time()

    # Per-dimension variance
    var_per_dim = np.var(h_sub, axis=0)
    mean_var = float(var_per_dim.mean())
    median_var = float(np.median(var_per_dim))

    # PCA (use randomized for speed)
    n_components = min(500, h_sub.shape[0], h_sub.shape[1])
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=SEED)
    pca.fit(h_sub)
    eigenvalues = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_

    # Total trace = sum of ALL eigenvalues (not just top-K)
    total_var = float(var_per_dim.sum())  # sum of per-dim variances = trace of covariance

    # Effective rank (using total trace as denominator)
    # Top-K eigenvalues
    p_top = eigenvalues / total_var
    h_top = -np.sum(p_top[p_top > 1e-15] * np.log(p_top[p_top > 1e-15]))
    # Tail: remaining variance spread across remaining dimensions
    tail_var = total_var - eigenvalues.sum()
    n_tail = h_sub.shape[1] - n_components
    if tail_var > 0 and n_tail > 0:
        p_tail_each = tail_var / n_tail / total_var
        h_tail = -n_tail * p_tail_each * np.log(p_tail_each) if p_tail_each > 1e-15 else 0
    else:
        h_tail = 0
    h_entropy = h_top + h_tail
    eff_rank = float(np.exp(h_entropy))

    # Participation ratio (using total trace)
    tail_sq_sum = n_tail * (tail_var / n_tail) ** 2 if n_tail > 0 and tail_var > 0 else 0
    pr = float(total_var ** 2 / ((eigenvalues ** 2).sum() + tail_sq_sum))

    # Cumulative variance explained
    cum_var_90 = int(np.searchsorted(np.cumsum(explained_ratio), 0.90) + 1)
    cum_var_95 = int(np.searchsorted(np.cumsum(explained_ratio), 0.95) + 1)
    cum_var_99 = int(np.searchsorted(np.cumsum(explained_ratio), 0.99) + 1)

    print(f"    [{name}] Done ({time.time()-t0:.0f}s): eff_rank={eff_rank:.1f} PR={pr:.1f}", flush=True)

    return {
        "mean_var": mean_var,
        "median_var": median_var,
        "eff_rank": eff_rank,
        "participation_ratio": pr,
        "cum_var_90": cum_var_90,
        "cum_var_95": cum_var_95,
        "cum_var_99": cum_var_99,
        "top_100_eigenvalues": eigenvalues[:100].tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "explained_ratio": explained_ratio.tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# Experiment 3: Spectral Alignment Analysis
# ═══════════════════════════════════════════════════════════════

def compute_normalized_psd(sequences):
    """Compute normalized PSD for a batch of sequences. Returns (N, F) where F = T//2+1."""
    fft = torch.fft.rfft(sequences, dim=-1)
    psd = (fft.abs() ** 2)
    # Normalize each sequence's PSD to sum to 1
    psd_norm = psd / psd.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return psd_norm


def exp3_spectral_analysis(pred, ts_bank, name):
    """PSD comparison between predictions and real TS."""
    # Compute PSDs
    pred_psd = compute_normalized_psd(pred)   # (N_pred, F)
    real_psd = compute_normalized_psd(ts_bank)  # (N_ts, F)

    # Mean PSDs
    mean_pred_psd = pred_psd.mean(dim=0)  # (F,)
    mean_real_psd = real_psd.mean(dim=0)  # (F,)

    # L2 distance between mean PSDs
    l2_dist = float(((mean_pred_psd - mean_real_psd) ** 2).sum().sqrt())

    # KL divergence: KL(pred || real)
    eps = 1e-12
    kl_div = float((mean_pred_psd * torch.log((mean_pred_psd + eps) / (mean_real_psd + eps))).sum())

    # Frequency band analysis
    F = mean_pred_psd.shape[0]
    low_end = int(F * 0.10)
    mid_end = int(F * 0.50)

    pred_low = float(pred_psd[:, :low_end].sum(dim=-1).mean())
    pred_mid = float(pred_psd[:, low_end:mid_end].sum(dim=-1).mean())
    pred_high = float(pred_psd[:, mid_end:].sum(dim=-1).mean())

    real_low = float(real_psd[:, :low_end].sum(dim=-1).mean())
    real_mid = float(real_psd[:, low_end:mid_end].sum(dim=-1).mean())
    real_high = float(real_psd[:, mid_end:].sum(dim=-1).mean())

    return {
        "psd_l2": l2_dist,
        "kl_div": kl_div,
        "pred_low": pred_low,
        "pred_mid": pred_mid,
        "pred_high": pred_high,
        "real_low": real_low,
        "real_mid": real_mid,
        "real_high": real_high,
        "mean_pred_psd": mean_pred_psd.numpy().tolist(),
        "mean_real_psd": mean_real_psd.numpy().tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for sub in ["exp1_random_proj", "exp2_diversity", "exp3_spectral"]:
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)

    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")
    rng = np.random.default_rng(SEED)

    # ── Load shared data ──
    print("Loading WikiText...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=2000, seq_len=T, hf_token=hf_token)
    plot_wiki = wiki_seqs[:N_SEQS]

    print("Loading TS bank...", flush=True)
    ts = load_ts_windows(cfg, 10000, hf_token)

    from sklearn.cluster import KMeans
    ts_labels = KMeans(n_clusters=20, n_init=5, random_state=42).fit_predict(
        compute_acf(ts, max_lag=32).numpy())

    # Random token sequences
    vocab_size = 151936
    rand_seqs = [{"input_ids": rng.integers(0, vocab_size, size=T).astype(np.int64)}
                 for _ in range(N_SEQS)]

    # Storage for results
    exp1_results = {}
    exp2_results = {}
    exp3_results = {}
    exp3_preds = {}  # save predictions for plotting

    # ── Process each condition ──
    for name, info in MODELS.items():
        print(f"\n{'='*60}")
        print(f"CONDITION: {name}")
        print(f"{'='*60}")

        # Load model
        if info["model_type"] == "pt":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
            ).model.to(DEVICE).eval()
        else:
            config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).model.to(DEVICE).eval()

        seqs = plot_wiki if info["tokens"] == "text" else rand_seqs

        # Extract hidden states
        print(f"  Extracting hidden states...", flush=True)
        h = extract_all_layers(model, seqs, DEVICE, desc=name)
        del model; gc.collect(); torch.cuda.empty_cache()

        # ── Experiment 1: Random Projections ──
        print(f"\n  [EXP 1] Random projections...", flush=True)
        exp1_agg, exp1_raw = exp1_random_projections(h, ts, ts_labels, name, rng)
        exp1_results[name] = exp1_agg
        print(f"  [EXP 1] {name}: NN={exp1_agg['nn_mean_mean']:.4f}+/-{exp1_agg['nn_mean_std']:.4f} "
              f"unique={exp1_agg['n_unique_mean']:.1f}+/-{exp1_agg['n_unique_std']:.1f}")

        # ── Experiment 2: Diversity Analysis ──
        print(f"\n  [EXP 2] Diversity analysis...", flush=True)
        exp2_results[name] = exp2_diversity_analysis(h, name)
        print(f"  [EXP 2] {name}: eff_rank={exp2_results[name]['eff_rank']:.1f} "
              f"PR={exp2_results[name]['participation_ratio']:.1f}")

        # ── Experiment 3: Generate predictions for spectral analysis ──
        print(f"\n  [EXP 3] Generating predictions...", flush=True)
        mapper = nn.Linear(D_CONCAT, 1)
        mapper_path = f"{ABLATION_DIR}/mapper_{name}.pt"
        mapper.load_state_dict(torch.load(mapper_path, map_location="cpu", weights_only=True))
        mapper = mapper.to(DEVICE).eval()
        pred = predict_with_mapper(mapper, h, DEVICE)
        exp3_preds[name] = pred

        exp3_results[name] = exp3_spectral_analysis(pred, ts, name)
        print(f"  [EXP 3] {name}: PSD_L2={exp3_results[name]['psd_l2']:.6f} "
              f"KL={exp3_results[name]['kl_div']:.6f}")

        # Save predictions for reuse
        torch.save(pred, f"{OUT_DIR}/pred_{name}.pt")

        del h, mapper, pred; gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # Save results + generate plots
    # ═══════════════════════════════════════════════════════════

    # ── Experiment 1: Save + Plot ──
    print(f"\n{'='*60}")
    print("SAVING EXPERIMENT 1 RESULTS")
    print(f"{'='*60}")

    with open(f"{OUT_DIR}/exp1_random_proj/results.json", "w") as f:
        json.dump(exp1_results, f, indent=2)

    # CSV
    with open(f"{OUT_DIR}/exp1_random_proj/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Condition", "NN Dist (mean)", "NN Dist (std)",
                     "Unique (mean)", "Unique (std)", "Clusters (mean)", "Clusters (std)",
                     "Entropy (mean)", "Entropy (std)"])
        for name in MODELS:
            r = exp1_results[name]
            w.writerow([name,
                        f"{r['nn_mean_mean']:.4f}", f"{r['nn_mean_std']:.4f}",
                        f"{r['n_unique_mean']:.1f}", f"{r['n_unique_std']:.1f}",
                        f"{r['clusters_mean']:.1f}", f"{r['clusters_std']:.1f}",
                        f"{r['entropy_mean']:.4f}", f"{r['entropy_std']:.4f}"])

    # Summary table
    print(f"\n{'Condition':<20} {'NN Dist':>16} {'Unique':>14} {'Clusters':>14} {'Entropy':>14}")
    print("-" * 80)
    for name in MODELS:
        r = exp1_results[name]
        print(f"{name:<20} {r['nn_mean_mean']:>7.4f}+/-{r['nn_mean_std']:.4f}"
              f" {r['n_unique_mean']:>6.1f}+/-{r['n_unique_std']:.1f}"
              f" {r['clusters_mean']:>6.1f}+/-{r['clusters_std']:.1f}"
              f" {r['entropy_mean']:>6.4f}+/-{r['entropy_std']:.4f}")

    # ── Experiment 2: Save + Plot ──
    print(f"\n{'='*60}")
    print("SAVING EXPERIMENT 2 RESULTS")
    print(f"{'='*60}")

    # Save (without large eigenvalue arrays for readability)
    exp2_summary = {}
    for name in MODELS:
        r = exp2_results[name]
        exp2_summary[name] = {k: v for k, v in r.items()
                              if k not in ("eigenvalues", "explained_ratio", "top_100_eigenvalues")}
    with open(f"{OUT_DIR}/exp2_diversity/results.json", "w") as f:
        json.dump(exp2_summary, f, indent=2)
    # Full eigenvalues saved separately
    with open(f"{OUT_DIR}/exp2_diversity/eigenvalues.json", "w") as f:
        json.dump({name: r["eigenvalues"] for name, r in exp2_results.items()}, f)

    # CSV
    with open(f"{OUT_DIR}/exp2_diversity/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Condition", "Mean Var", "Median Var", "Effective Rank",
                     "Participation Ratio", "PCs for 90%", "PCs for 95%", "PCs for 99%"])
        for name in MODELS:
            r = exp2_results[name]
            w.writerow([name,
                        f"{r['mean_var']:.6f}", f"{r['median_var']:.6f}",
                        f"{r['eff_rank']:.1f}", f"{r['participation_ratio']:.1f}",
                        r['cum_var_90'], r['cum_var_95'], r['cum_var_99']])

    # Summary table
    print(f"\n{'Condition':<20} {'Mean Var':>10} {'Eff Rank':>10} {'PR':>10} {'90%':>6} {'95%':>6} {'99%':>6}")
    print("-" * 70)
    for name in MODELS:
        r = exp2_results[name]
        print(f"{name:<20} {r['mean_var']:>10.6f} {r['eff_rank']:>10.1f} "
              f"{r['participation_ratio']:>10.1f} {r['cum_var_90']:>6} {r['cum_var_95']:>6} {r['cum_var_99']:>6}")

    # Plot: PCA spectrum (all 4 overlaid, log scale)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("PCA Spectrum of Hidden States (All 4 Ablation Conditions)", fontsize=14, fontweight='bold')

    for name in MODELS:
        eigs = np.array(exp2_results[name]["eigenvalues"])
        axes[0].plot(eigs[:100], color=COLORS[name], linewidth=1.5, label=LABELS[name])
        axes[1].plot(eigs[:100], color=COLORS[name], linewidth=1.5, label=LABELS[name])

    axes[0].set_xlabel("Component"); axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Linear scale (top 100)"); axes[0].legend(fontsize=10)
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].set_xlabel("Component"); axes[1].set_ylabel("Eigenvalue (log)")
    axes[1].set_title("Log scale (top 100)"); axes[1].set_yscale("log"); axes[1].legend(fontsize=10)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp2_diversity/pca_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot: Cumulative variance explained
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MODELS:
        cum = np.cumsum(exp2_results[name]["explained_ratio"])
        ax.plot(cum[:200], color=COLORS[name], linewidth=1.5, label=LABELS[name])
    ax.axhline(0.90, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='95%')
    ax.set_xlabel("Number of Components"); ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("Cumulative Variance Explained (top 200 components)")
    ax.legend(fontsize=10); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp2_diversity/cumulative_variance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot: Effective rank + PR bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    names = list(MODELS.keys())
    labels = [LABELS[n] for n in names]
    colors = [COLORS[n] for n in names]

    eff_ranks = [exp2_results[n]["eff_rank"] for n in names]
    axes[0].bar(labels, eff_ranks, color=colors)
    axes[0].set_ylabel("Effective Rank"); axes[0].set_title("Effective Rank (exp of eigenvalue entropy)")
    for j, v in enumerate(eff_ranks):
        axes[0].text(j, v + 0.5, f"{v:.0f}", ha='center', fontsize=11, fontweight='bold')

    prs = [exp2_results[n]["participation_ratio"] for n in names]
    axes[1].bar(labels, prs, color=colors)
    axes[1].set_ylabel("Participation Ratio"); axes[1].set_title("Participation Ratio ((sum lambda)^2 / sum lambda^2)")
    for j, v in enumerate(prs):
        axes[1].text(j, v + 0.3, f"{v:.0f}", ha='center', fontsize=11, fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp2_diversity/rank_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Experiment 3: Save + Plot ──
    print(f"\n{'='*60}")
    print("SAVING EXPERIMENT 3 RESULTS")
    print(f"{'='*60}")

    # Save (without large PSD arrays)
    exp3_summary = {}
    for name in MODELS:
        r = exp3_results[name]
        exp3_summary[name] = {k: v for k, v in r.items()
                              if k not in ("mean_pred_psd", "mean_real_psd")}
    with open(f"{OUT_DIR}/exp3_spectral/results.json", "w") as f:
        json.dump(exp3_summary, f, indent=2)

    # CSV
    with open(f"{OUT_DIR}/exp3_spectral/results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Condition", "PSD L2", "KL Div",
                     "Pred Low", "Pred Mid", "Pred High",
                     "Real Low", "Real Mid", "Real High"])
        for name in MODELS:
            r = exp3_results[name]
            w.writerow([name,
                        f"{r['psd_l2']:.6f}", f"{r['kl_div']:.6f}",
                        f"{r['pred_low']:.4f}", f"{r['pred_mid']:.4f}", f"{r['pred_high']:.4f}",
                        f"{r['real_low']:.4f}", f"{r['real_mid']:.4f}", f"{r['real_high']:.4f}"])

    # Summary table
    print(f"\n{'Condition':<20} {'PSD L2':>10} {'KL Div':>10} {'Low':>8} {'Mid':>8} {'High':>8}")
    print("-" * 66)
    for name in MODELS:
        r = exp3_results[name]
        print(f"{name:<20} {r['psd_l2']:>10.6f} {r['kl_div']:>10.6f} "
              f"{r['pred_low']:>8.4f} {r['pred_mid']:>8.4f} {r['pred_high']:>8.4f}")
    print(f"{'Real TS':<20} {'':>10} {'':>10} "
          f"{exp3_results['text_PT']['real_low']:>8.4f} "
          f"{exp3_results['text_PT']['real_mid']:>8.4f} "
          f"{exp3_results['text_PT']['real_high']:>8.4f}")

    # Plot: Mean PSD curves
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Mean Power Spectral Density: Predictions vs Real TS", fontsize=14, fontweight='bold')

    real_psd = np.array(exp3_results["text_PT"]["mean_real_psd"])
    F = len(real_psd)
    freqs = np.arange(F) / (2 * F)  # normalized frequency [0, 0.5]

    for name in MODELS:
        pred_psd = np.array(exp3_results[name]["mean_pred_psd"])
        axes[0].plot(freqs, pred_psd, color=COLORS[name], linewidth=1.2,
                     alpha=0.8, label=LABELS[name])
        axes[1].plot(freqs, pred_psd, color=COLORS[name], linewidth=1.2,
                     alpha=0.8, label=LABELS[name])
    axes[0].plot(freqs, real_psd, color='black', linewidth=2, alpha=0.8, label="Real TS")
    axes[1].plot(freqs, real_psd, color='black', linewidth=2, alpha=0.8, label="Real TS")

    axes[0].set_xlabel("Normalized Frequency"); axes[0].set_ylabel("PSD")
    axes[0].set_title("Linear scale"); axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Normalized Frequency"); axes[1].set_ylabel("PSD (log)")
    axes[1].set_title("Log scale"); axes[1].set_yscale("log"); axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp3_spectral/psd_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot: Frequency band bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Energy Distribution Across Frequency Bands", fontsize=14, fontweight='bold')

    band_names = ["Low (0-10%)", "Mid (10-50%)", "High (50-100%)"]
    band_keys = [("pred_low", "real_low"), ("pred_mid", "real_mid"), ("pred_high", "real_high")]

    for bi, (bname, (pk, rk)) in enumerate(zip(band_names, band_keys)):
        ax = axes[bi]
        cond_labels = [LABELS[n] for n in MODELS] + ["Real TS"]
        cond_colors = [COLORS[n] for n in MODELS] + ["#000000"]
        vals = [exp3_results[n][pk] for n in MODELS] + [exp3_results["text_PT"][rk]]
        ax.bar(cond_labels, vals, color=cond_colors)
        ax.set_title(bname); ax.set_ylabel("Fraction of energy")
        for j, v in enumerate(vals):
            ax.text(j, v + 0.005, f"{v:.3f}", ha='center', fontsize=8, fontweight='bold')
        ax.tick_params(axis='x', rotation=20)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/exp3_spectral/frequency_bands.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to {OUT_DIR}/")
    for sub in ["exp1_random_proj", "exp2_diversity", "exp3_spectral"]:
        files = os.listdir(f"{OUT_DIR}/{sub}")
        print(f"  {sub}/: {', '.join(sorted(files))}")


if __name__ == "__main__":
    main()
