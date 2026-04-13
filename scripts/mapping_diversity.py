"""
Mapping experiment with diversity penalties + cluster evaluation.

Diversity penalties (shift-invariant):
  - PSD: penalize similar frequency spectra across predictions
  - ACF: penalize similar autocorrelation across predictions

Evaluation:
  - Cluster real TS into groups by ACF profile
  - Per-cluster distance, coverage, reverse coverage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, time, gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences
from scripts.mapping_experiment import (
    extract_pt_hidden_states, load_ts_windows, compute_psd, compute_acf
)


def psd_diversity_penalty(y_pred):
    """Penalize predictions with similar frequency spectra."""
    psd = compute_psd(y_pred)  # (B, F)
    psd_norm = F.normalize(psd, dim=-1)  # unit normalize
    sim = psd_norm @ psd_norm.T  # (B, B) cosine sim
    # Zero diagonal, take mean of off-diagonal
    mask = 1 - torch.eye(sim.shape[0], device=sim.device)
    return (sim * mask).sum() / mask.sum()


def acf_diversity_penalty(y_pred, max_lag=32):
    """Penalize predictions with similar autocorrelation."""
    acf = compute_acf(y_pred, max_lag=max_lag)  # (B, max_lag)
    acf_norm = F.normalize(acf, dim=-1)
    sim = acf_norm @ acf_norm.T
    mask = 1 - torch.eye(sim.shape[0], device=sim.device)
    return (sim * mask).sum() / mask.sum()


def train_with_diversity(
    h_data, ts_data, device, d=1024,
    n_epochs=150, batch_size=64, lr=1e-3,
    div_type="none", lambda_div=0.5,
):
    """Train linear mapper with optional diversity penalty."""
    mapper = nn.Linear(d, 1).to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=lr)
    N, N_ts = h_data.shape[0], ts_data.shape[0]

    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            h_b = h_data[idx].to(device)
            ts_b = ts_data[torch.randint(0, N_ts, (256,))].to(device)

            y_raw = mapper(h_b).squeeze(-1)
            y_std = y_raw.std(dim=-1, keepdim=True).clamp(min=1e-4)
            y = (y_raw - y_raw.mean(dim=-1, keepdim=True)) / y_std

            # E-step: find nearest match
            with torch.no_grad():
                d2 = ((y.unsqueeze(1) - ts_b.unsqueeze(0)) ** 2).mean(-1)
                tgt = ts_b[d2.argmin(1)]

            # M-step: MSE + diversity
            loss = F.mse_loss(y, tgt)

            if div_type == "psd" and y.shape[0] > 1:
                loss = loss + lambda_div * psd_diversity_penalty(y)
            elif div_type == "acf" and y.shape[0] > 1:
                loss = loss + lambda_div * acf_diversity_penalty(y)

            opt.zero_grad()
            loss.backward()
            opt.step()

    mapper.eval()
    with torch.no_grad():
        h_e = h_data.to(device)
        yr = mapper(h_e).squeeze(-1)
        ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
        yp = ((yr - yr.mean(-1, keepdim=True)) / ys).cpu()
    return yp, mapper


def cluster_ts(ts_data, n_clusters=20):
    """Cluster TS by ACF profile."""
    acf = compute_acf(ts_data, max_lag=32).numpy()
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    labels = km.fit_predict(acf)
    return labels, km


def evaluate_with_clusters(pred, ts_bank, ts_labels, n_clusters):
    """Diversity-aware evaluation — batched for speed."""
    N_pred = pred.shape[0]
    N_ts = ts_bank.shape[0]

    # Batched NN computation: process predictions in chunks
    nn_indices = []
    nn_dists = []
    chunk = 50
    for i in range(0, N_pred, chunk):
        p_chunk = pred[i:i + chunk]  # (chunk, T)
        # Compute against a subsample of TS for speed
        ts_sub_idx = np.random.choice(N_ts, min(2000, N_ts), replace=False)
        ts_sub = ts_bank[ts_sub_idx]
        d2 = ((p_chunk.unsqueeze(1) - ts_sub.unsqueeze(0)) ** 2).mean(-1)  # (chunk, sub)
        mins = d2.min(dim=1)
        for j in range(p_chunk.shape[0]):
            nn_indices.append(ts_sub_idx[mins.indices[j].item()])
            nn_dists.append(mins.values[j].item())
    nn_indices = np.array(nn_indices)
    nn_dists = np.array(nn_dists)

    # Unique matches
    unique = len(set(nn_indices))

    # Cluster coverage
    matched_clusters = set(ts_labels[nn_indices])
    coverage = len(matched_clusters) / n_clusters

    # Per-cluster best distance (batched)
    cluster_dists = []
    for c in range(n_clusters):
        c_indices = np.where(ts_labels == c)[0]
        if len(c_indices) == 0:
            continue
        # Subsample cluster members and predictions for speed
        c_sub = c_indices[:200]
        p_sub = pred[:200]
        d2 = ((p_sub.unsqueeze(1) - ts_bank[c_sub].unsqueeze(0)) ** 2).mean(-1)
        cluster_dists.append(d2.min().item())
    mean_cluster_dist = np.mean(cluster_dists)

    # Reverse coverage (batched): fraction of real TS with a nearby prediction
    n_sample = min(500, N_ts)
    ts_sample = ts_bank[:n_sample]
    threshold = np.median(nn_dists)
    reverse_covered = 0
    for j in range(0, n_sample, chunk):
        ts_chunk = ts_sample[j:j + chunk]
        d2 = ((ts_chunk.unsqueeze(1) - pred[:500].unsqueeze(0)) ** 2).mean(-1)
        mins = d2.min(dim=1).values
        reverse_covered += (mins.numpy() < threshold).sum()
    reverse_coverage = float(reverse_covered) / n_sample

    # Match entropy
    from collections import Counter
    counts = Counter(nn_indices.tolist())
    probs = np.array(list(counts.values()), dtype=float) / N_pred
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(N_pred)
    norm_entropy = entropy / max_entropy

    return {
        "nn_dist": float(nn_dists.mean()),
        "unique_matches": int(unique),
        "cluster_coverage": float(coverage),
        "clusters_covered": int(len(matched_clusters)),
        "mean_cluster_dist": float(mean_cluster_dist),
        "reverse_coverage": float(reverse_coverage),
        "norm_entropy": float(norm_entropy),
    }


def extract_random_init_hs(layer_idx, wiki_sequences, device):
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).model.to(device).eval()
    captured = {}
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()
    handle = model.layers[layer_idx].register_forward_hook(hook)
    model.layers = model.layers[:layer_idx + 1]
    all_hs = []
    with torch.no_grad():
        for start in range(0, len(wiki_sequences), 32):
            batch = wiki_sequences[start:start + 32]
            ids = torch.tensor(np.stack([s["input_ids"] for s in batch]),
                               dtype=torch.long, device=device)
            model(input_ids=ids, use_cache=False)
            all_hs.append(captured["hs"].float().cpu()[:, :512, :])
    handle.remove()
    del model; torch.cuda.empty_cache()
    return torch.cat(all_hs, dim=0)


def main():
    device = torch.device("cuda:0")
    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")
    d = 1024
    layer = 8
    n_clusters = 20

    os.makedirs("mapping_results/diversity", exist_ok=True)

    # Load data
    wiki_seqs = load_wikitext_sequences(max_sequences=1000, seq_len=512, hf_token=hf_token)
    ts = load_ts_windows(cfg, 5000, hf_token)

    # Cluster real TS
    print("Clustering real TS...")
    ts_labels, _ = cluster_ts(ts, n_clusters)
    counts = np.bincount(ts_labels, minlength=n_clusters)
    print(f"  {n_clusters} clusters: min={counts.min()} max={counts.max()} mean={counts.mean():.0f}")

    # Extract hidden states
    print("\nExtracting PT hidden states...")
    h_pt = extract_pt_hidden_states(layer, wiki_seqs, cfg, device)

    print("Extracting RandomInit hidden states...")
    h_rinit = extract_random_init_hs(layer, wiki_seqs, device)

    h_rand = torch.randn(1000, 512, d)

    # Run experiments
    configs = [
        ("PT", h_pt, "none", 0),
        ("PT+PSD_div_0.1", h_pt, "psd", 0.1),
        ("PT+PSD_div_0.5", h_pt, "psd", 0.5),
        ("PT+PSD_div_1.0", h_pt, "psd", 1.0),
        ("PT+ACF_div_0.1", h_pt, "acf", 0.1),
        ("PT+ACF_div_0.5", h_pt, "acf", 0.5),
        ("PT+ACF_div_1.0", h_pt, "acf", 1.0),
        ("RandomInit", h_rinit, "none", 0),
        ("RandomInit+PSD_0.5", h_rinit, "psd", 0.5),
        ("RandomInit+ACF_0.5", h_rinit, "acf", 0.5),
        ("Random", h_rand, "none", 0),
    ]

    all_results = {}
    for name, h_data, div_type, lambda_div in configs:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        t0 = time.time()
        pred, mapper = train_with_diversity(
            h_data, ts, device, d=d,
            div_type=div_type, lambda_div=lambda_div,
        )
        train_time = time.time() - t0

        results = evaluate_with_clusters(pred, ts, ts_labels, n_clusters)
        results["train_time"] = train_time
        all_results[name] = results

        print(f"  NN dist:         {results['nn_dist']:.4f}")
        print(f"  Unique matches:  {results['unique_matches']}")
        print(f"  Clusters covered:{results['clusters_covered']}/{n_clusters}")
        print(f"  Mean cluster dist:{results['mean_cluster_dist']:.4f}")
        print(f"  Reverse coverage:{results['reverse_coverage']:.3f}")
        print(f"  Entropy (norm):  {results['norm_entropy']:.4f}")
        print(f"  Time: {train_time:.0f}s")

    # Free memory
    del h_pt, h_rinit, h_rand
    gc.collect(); torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 100)
    print("FULL RESULTS")
    print("=" * 100)
    print(f"{'Method':<22} {'NN Dist':>8} {'Unique':>7} {'Clusters':>9} "
          f"{'ClustDist':>10} {'RevCov':>8} {'Entropy':>8}")
    print("-" * 78)
    for name, r in all_results.items():
        print(f"{name:<22} {r['nn_dist']:>8.4f} {r['unique_matches']:>7} "
              f"{r['clusters_covered']:>5}/{n_clusters:<3} "
              f"{r['mean_cluster_dist']:>10.4f} {r['reverse_coverage']:>8.3f} "
              f"{r['norm_entropy']:>8.4f}")

    with open("mapping_results/diversity/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to mapping_results/diversity/results.json")


if __name__ == "__main__":
    main()
