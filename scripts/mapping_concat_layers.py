"""
Large-scale mapping experiment: concatenate ALL 28 layers of hidden states
and train a linear map to produce diverse time series.

Key changes from previous experiments:
- All 28 layers concatenated → 28,672-dim input per timestep
- More WikiText data (up to 10K sequences)
- PSD diversity penalty
- Comprehensive cluster-based evaluation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, time, gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences
from scripts.mapping_experiment import load_ts_windows, compute_acf, compute_psd


# ---------------------------------------------------------------------------
# Extract ALL layers at once
# ---------------------------------------------------------------------------

def extract_all_layers(model_source, wiki_sequences, device, n_layers=28):
    """
    Extract hidden states from ALL layers for each WikiText sequence.
    Returns (N, T, n_layers * d) tensor.

    model_source: "pt", "random_init", or a model path
    """
    if model_source == "pt":
        print("  Loading PT model...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
        ).model.to(device).eval()
    elif model_source == "random_init":
        print("  Loading random-init model...")
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).model.to(device).eval()
    else:
        raise ValueError(f"Unknown model source: {model_source}")

    captured = {}
    handles = []
    for layer_idx in range(n_layers):
        def make_hook(idx):
            def hook(module, input, output):
                hs = output[0] if isinstance(output, tuple) else output
                captured[idx] = hs.detach()
            return hook
        h = model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        handles.append(h)

    N = len(wiki_sequences)
    T = 512
    d = 1024
    BS = 8  # small batch to fit all layer activations in memory

    all_concat = []
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, N, BS):
            batch = wiki_sequences[start:start + BS]
            input_ids = torch.tensor(
                np.stack([s["input_ids"] for s in batch]),
                dtype=torch.long, device=device
            )
            model(input_ids=input_ids, use_cache=False)

            # Concatenate all layers: (B, T, n_layers * d)
            layer_hs = [captured[i].float().cpu()[:, :T, :] for i in range(n_layers)]
            concat = torch.cat(layer_hs, dim=-1)  # (B, T, 28*1024)
            all_concat.append(concat)

            captured.clear()

            if start % (BS * 10) == 0:
                elapsed = time.time() - t0
                print(f"    {start}/{N} ({elapsed:.0f}s)", flush=True)

    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    result = torch.cat(all_concat, dim=0)
    print(f"  Extracted: {result.shape} in {(time.time()-t0)/60:.1f}m")
    return result


# ---------------------------------------------------------------------------
# Diversity penalties
# ---------------------------------------------------------------------------

def psd_diversity_penalty(y_pred):
    psd = compute_psd(y_pred)
    psd_norm = F.normalize(psd, dim=-1)
    sim = psd_norm @ psd_norm.T
    mask = 1 - torch.eye(sim.shape[0], device=sim.device)
    return (sim * mask).sum() / mask.sum()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_concat_mapper(
    h_data, ts_data, device,
    n_epochs=80, batch_size=64, lr=1e-3,
    lambda_div=0.5, log_freq=20,
):
    """Train linear map from concatenated hidden states to TS."""
    d_concat = h_data.shape[-1]  # 28 * 1024 = 28672
    mapper = nn.Linear(d_concat, 1).to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=lr)

    N = h_data.shape[0]
    N_ts = ts_data.shape[0]
    losses = []

    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            h_b = h_data[idx].to(device)
            ts_b = ts_data[torch.randint(0, N_ts, (256,))].to(device)

            y_raw = mapper(h_b).squeeze(-1)
            y_std = y_raw.std(dim=-1, keepdim=True).clamp(min=1e-4)
            y = (y_raw - y_raw.mean(dim=-1, keepdim=True)) / y_std

            # E-step
            with torch.no_grad():
                d2 = ((y.unsqueeze(1) - ts_b.unsqueeze(0)) ** 2).mean(-1)
                tgt = ts_b[d2.argmin(1)]

            loss = F.mse_loss(y, tgt)
            if lambda_div > 0 and y.shape[0] > 1:
                loss = loss + lambda_div * psd_diversity_penalty(y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % log_freq == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}", flush=True)

    mapper.eval()
    with torch.no_grad():
        # Predict in chunks to avoid OOM
        all_pred = []
        for start in range(0, N, batch_size):
            h_b = h_data[start:start + batch_size].to(device)
            yr = mapper(h_b).squeeze(-1)
            ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
            yp = (yr - yr.mean(-1, keepdim=True)) / ys
            all_pred.append(yp.cpu())
        pred = torch.cat(all_pred, dim=0)

    return pred, mapper, losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(pred, ts_bank, ts_labels, n_clusters):
    N_pred = pred.shape[0]
    N_ts = ts_bank.shape[0]

    # Batched NN
    nn_indices = []
    nn_dists = []
    chunk = 50
    for i in range(0, N_pred, chunk):
        p_chunk = pred[i:i + chunk]
        ts_sub_idx = np.random.choice(N_ts, min(2000, N_ts), replace=False)
        ts_sub = ts_bank[ts_sub_idx]
        d2 = ((p_chunk.unsqueeze(1) - ts_sub.unsqueeze(0)) ** 2).mean(-1)
        mins = d2.min(dim=1)
        for j in range(p_chunk.shape[0]):
            nn_indices.append(ts_sub_idx[mins.indices[j].item()])
            nn_dists.append(mins.values[j].item())
    nn_indices = np.array(nn_indices)
    nn_dists = np.array(nn_dists)

    unique = len(set(nn_indices))
    matched_clusters = set(ts_labels[nn_indices])
    coverage = len(matched_clusters) / n_clusters

    # Per-cluster distance
    cluster_dists = []
    for c in range(n_clusters):
        c_indices = np.where(ts_labels == c)[0]
        if len(c_indices) == 0:
            continue
        c_sub = c_indices[:200]
        p_sub = pred[:min(300, N_pred)]
        d2 = ((p_sub.unsqueeze(1) - ts_bank[c_sub].unsqueeze(0)) ** 2).mean(-1)
        cluster_dists.append(d2.min().item())
    mean_cluster_dist = np.mean(cluster_dists)

    # Entropy
    from collections import Counter
    counts = Counter(nn_indices.tolist())
    probs = np.array(list(counts.values()), dtype=float) / N_pred
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    norm_entropy = entropy / np.log(N_pred)

    return {
        "nn_dist": float(nn_dists.mean()),
        "unique_matches": int(unique),
        "clusters_covered": int(len(matched_clusters)),
        "cluster_coverage": float(coverage),
        "mean_cluster_dist": float(mean_cluster_dist),
        "norm_entropy": float(norm_entropy),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(all_results, pred_dict, ts, ts_labels, n_clusters, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Summary bars
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    names = list(all_results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    for ax_idx, (title, key) in enumerate([
        ("NN Distance ↓", "nn_dist"),
        ("Unique Matches ↑", "unique_matches"),
        ("Clusters Covered ↑", "clusters_covered"),
        ("Entropy ↑", "norm_entropy"),
    ]):
        vals = [all_results[n][key] for n in names]
        bars = axes[ax_idx].bar(range(len(names)), vals, color=colors, edgecolor="white")
        axes[ax_idx].set_title(title, fontsize=11, fontweight='bold')
        axes[ax_idx].set_xticks(range(len(names)))
        axes[ax_idx].set_xticklabels(names, fontsize=7, rotation=30, ha='right')
        for bar, val in zip(bars, vals):
            axes[ax_idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                              f"{val:.2f}" if isinstance(val, float) else str(val),
                              ha='center', fontsize=7, va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Prediction overlay for each model
    fig, axes = plt.subplots(1, len(pred_dict), figsize=(5 * len(pred_dict), 5))
    if len(pred_dict) == 1:
        axes = [axes]
    fig.suptitle("50 predictions overlaid per model", fontsize=13)
    for col, (name, pred) in enumerate(pred_dict.items()):
        sample = np.random.choice(pred.shape[0], min(50, pred.shape[0]), replace=False)
        for i in sample:
            axes[col].plot(pred[i].numpy(), alpha=0.15, linewidth=0.5, color="blue")
        axes[col].set_title(name, fontsize=10, fontweight='bold')
        axes[col].set_ylim(-4, 4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlay.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Cluster histogram
    fig, axes = plt.subplots(1, len(pred_dict), figsize=(5 * len(pred_dict), 4))
    if len(pred_dict) == 1:
        axes = [axes]
    fig.suptitle("NN match cluster distribution", fontsize=13)
    for col, (name, pred) in enumerate(pred_dict.items()):
        nn_clusters = []
        ts_sub_idx = np.random.choice(len(ts), min(2000, len(ts)), replace=False)
        for i in range(min(pred.shape[0], 500)):
            d2 = ((pred[i:i + 1] - ts[ts_sub_idx]) ** 2).mean(-1).squeeze(0)
            nn_clusters.append(ts_labels[ts_sub_idx[d2.argmin().item()]])
        axes[col].hist(nn_clusters, bins=np.arange(-0.5, n_clusters + 0.5, 1),
                       color="steelblue", edgecolor="white")
        axes[col].set_title(name, fontsize=10, fontweight='bold')
        axes[col].set_xlim(-0.5, n_clusters - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_hist.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Per-cluster best matches for top model
    top_name = min(all_results, key=lambda n: all_results[n]["mean_cluster_dist"]
                   if "Random" not in n or "Init" in n else 999)
    pred_top = pred_dict[top_name]
    n_show = min(n_clusters, 10)
    fig, axes = plt.subplots(n_show, 2, figsize=(16, 3 * n_show))
    fig.suptitle(f"Best prediction per cluster — {top_name}", fontsize=13, fontweight='bold')
    axes[0, 0].set_title("Predicted", fontsize=11)
    axes[0, 1].set_title("Nearest Real TS", fontsize=11)

    for row in range(n_show):
        c = row
        c_indices = np.where(ts_labels == c)[0]
        if len(c_indices) == 0:
            continue
        best_dist = float('inf')
        best_pi = 0
        best_ti = c_indices[0]
        for pi in range(0, pred_top.shape[0], 5):
            d2 = ((pred_top[pi:pi + 1] - ts[c_indices[:100]]) ** 2).mean(-1).squeeze(0)
            md = d2.min().item()
            if md < best_dist:
                best_dist = md
                best_pi = pi
                best_ti = c_indices[d2.argmin().item()]

        axes[row, 0].plot(pred_top[best_pi].numpy(), color="blue", linewidth=1)
        axes[row, 0].set_ylim(-4, 4)
        axes[row, 0].set_ylabel(f"C{c} d={best_dist:.2f}", fontsize=8)
        axes[row, 1].plot(ts[best_ti].numpy(), color="green", linewidth=1)
        axes[row, 1].set_ylim(-4, 4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_cluster_best.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plots saved to {output_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda:0")
    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")
    n_clusters = 20
    output_dir = "mapping_results/concat_layers"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading WikiText (2000 sequences)...")
    wiki_seqs = load_wikitext_sequences(max_sequences=2000, seq_len=512, hf_token=hf_token)
    print(f"  Loaded {len(wiki_seqs)} sequences")

    print("Loading time series...")
    ts = load_ts_windows(cfg, 10000, hf_token)
    print(f"  Loaded {ts.shape}")

    # Cluster TS
    print("Clustering TS...")
    ts_acf = compute_acf(ts, max_lag=32).numpy()
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    ts_labels = km.fit_predict(ts_acf)
    counts = np.bincount(ts_labels, minlength=n_clusters)
    print(f"  {n_clusters} clusters: min={counts.min()} max={counts.max()}")

    all_results = {}
    pred_dict = {}

    # ─── PT: all layers concatenated ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PT — ALL LAYERS CONCATENATED")
    print("=" * 60)
    h_pt = extract_all_layers("pt", wiki_seqs, device)

    # Try key diversity lambdas (0.3 and 1.0 skipped — 0.5 is the sweet spot)
    for lam in [0.0, 0.5]:
        name = f"PT_concat_div{lam}"
        print(f"\n--- {name} ---")
        pred, mapper, losses = train_concat_mapper(
            h_pt, ts, device, n_epochs=80, lr=1e-3, lambda_div=lam,
        )
        results = evaluate(pred, ts, ts_labels, n_clusters)
        all_results[name] = results
        pred_dict[name] = pred
        print(f"  NN={results['nn_dist']:.4f} Unique={results['unique_matches']} "
              f"Clusters={results['clusters_covered']}/{n_clusters} "
              f"ClustDist={results['mean_cluster_dist']:.4f} "
              f"Entropy={results['norm_entropy']:.4f}")

        # Save weights for best model
        if lam == 0.5:
            torch.save(mapper.state_dict(), os.path.join(output_dir, "mapper_pt_concat.pt"))
            # Analyze which layers the linear map uses most
            w = mapper.weight.data.cpu().reshape(28, 1024)  # (28, 1024)
            layer_norms = w.norm(dim=1).numpy()
            print(f"  Layer weight norms: {', '.join(f'L{i}={layer_norms[i]:.4f}' for i in range(28))}")
            top_layers = np.argsort(layer_norms)[::-1][:5]
            print(f"  Top 5 most used layers: {top_layers.tolist()}")

    del h_pt; gc.collect(); torch.cuda.empty_cache()

    # ─── RandomInit: all layers concatenated ──────────────────────────
    print("\n" + "=" * 60)
    print("RANDOM-INIT — ALL LAYERS CONCATENATED")
    print("=" * 60)
    h_ri = extract_all_layers("random_init", wiki_seqs, device)

    for lam in [0.0, 0.5]:
        name = f"RandInit_concat_div{lam}"
        print(f"\n--- {name} ---")
        pred, mapper, losses = train_concat_mapper(
            h_ri, ts, device, n_epochs=80, lr=1e-3, lambda_div=lam,
        )
        results = evaluate(pred, ts, ts_labels, n_clusters)
        all_results[name] = results
        pred_dict[name] = pred
        print(f"  NN={results['nn_dist']:.4f} Unique={results['unique_matches']} "
              f"Clusters={results['clusters_covered']}/{n_clusters} "
              f"ClustDist={results['mean_cluster_dist']:.4f} "
              f"Entropy={results['norm_entropy']:.4f}")

    del h_ri; gc.collect(); torch.cuda.empty_cache()

    # ─── Random vectors (same dim) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("RANDOM VECTORS")
    print("=" * 60)
    h_rand = torch.randn(len(wiki_seqs), 512, 28 * 1024)
    pred_rand, _, _ = train_concat_mapper(
        h_rand, ts, device, n_epochs=200, lr=1e-3, lambda_div=0,
    )
    results_rand = evaluate(pred_rand, ts, ts_labels, n_clusters)
    all_results["Random_concat"] = results_rand
    pred_dict["Random_concat"] = pred_rand
    print(f"  NN={results_rand['nn_dist']:.4f} Unique={results_rand['unique_matches']} "
          f"Clusters={results_rand['clusters_covered']}/{n_clusters}")
    del h_rand; gc.collect()

    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL RESULTS — ALL LAYERS CONCATENATED")
    print("=" * 70)
    print(f"{'Method':<28} {'NN Dist':>8} {'Unique':>7} {'Clusters':>9} "
          f"{'ClustDist':>10} {'Entropy':>8}")
    print("-" * 75)
    for name, r in all_results.items():
        print(f"{name:<28} {r['nn_dist']:>8.4f} {r['unique_matches']:>7} "
              f"{r['clusters_covered']:>5}/{n_clusters:<3} "
              f"{r['mean_cluster_dist']:>10.4f} {r['norm_entropy']:>8.4f}")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Plots
    print("\nGenerating plots...")
    plot_results(all_results, pred_dict, ts, ts_labels, n_clusters, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
