"""
Mapping Experiment: Decode time series from PT hidden state sequences.

Learns a linear map W ∈ R^{1×d} applied per-timestep:
    ŷ_{i,t} = W @ h_{i,t} + b

Trained so that predicted sequences match SOME real time series (no pairing).

Two variants:
  1. Soft retrieval (contrastive-like loss)
  2. EM-style hard matching

Usage:
    /usr/bin/python3 scripts/mapping_experiment.py --layer 8
"""
import argparse
import os
import sys
import json
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_pt_hidden_states(layer_idx: int, wiki_sequences: list, cfg: Config,
                              device: torch.device) -> torch.Tensor:
    """Extract PT hidden states from WikiText at a specific layer.
    Returns (N, T, d) float32 tensor."""
    from transformers import AutoModelForCausalLM

    print(f"Loading PT model for layer {layer_idx} extraction...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_pt, dtype=torch.bfloat16
    ).model.to(device).eval()

    captured = {}
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()
    handle = model.layers[layer_idx].register_forward_hook(hook)
    original_layers = model.layers
    model.layers = original_layers[:layer_idx + 1]

    all_hs = []
    BS = 32
    N = len(wiki_sequences)
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, N, BS):
            batch = wiki_sequences[start:start + BS]
            input_ids = torch.tensor(
                np.stack([s["input_ids"] for s in batch]),
                dtype=torch.long, device=device
            )
            model(input_ids=input_ids, use_cache=False)
            hs = captured["hs"].float().cpu()
            all_hs.append(hs[:, :512, :])  # Ensure T=512

            if start % (BS * 10) == 0:
                elapsed = time.time() - t0
                print(f"  Wiki extraction: {start}/{N} ({elapsed:.0f}s)", flush=True)

    model.layers = original_layers
    handle.remove()
    del model
    torch.cuda.empty_cache()

    result = torch.cat(all_hs, dim=0)  # (N, 512, 1024)
    print(f"  Extracted: {result.shape}, took {(time.time()-t0)/60:.1f}m")
    return result


def extract_ri_hidden_states(layer_idx: int, wiki_sequences: list, cfg: Config,
                              device: torch.device) -> torch.Tensor:
    """Extract RI hidden states from WikiText (baseline)."""
    from transformers import AutoModelForCausalLM

    print(f"Loading RI model for layer {layer_idx} extraction...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_ri, dtype=torch.bfloat16
    ).model.to(device).eval()

    captured = {}
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()
    handle = model.layers[layer_idx].register_forward_hook(hook)
    original_layers = model.layers
    model.layers = original_layers[:layer_idx + 1]

    all_hs = []
    BS = 32
    N = len(wiki_sequences)

    with torch.no_grad():
        for start in range(0, N, BS):
            batch = wiki_sequences[start:start + BS]
            input_ids = torch.tensor(
                np.stack([s["input_ids"] for s in batch]),
                dtype=torch.long, device=device
            )
            model(input_ids=input_ids, use_cache=False)
            hs = captured["hs"].float().cpu()
            all_hs.append(hs[:, :512, :])

    model.layers = original_layers
    handle.remove()
    del model
    torch.cuda.empty_cache()

    return torch.cat(all_hs, dim=0)


def load_ts_windows(cfg: Config, n_windows: int, hf_token: str | None) -> torch.Tensor:
    """Load time series windows as z-scored continuous values.
    Returns (N, T) float32 tensor."""
    from src.data.dataset import load_gifteval_series, temporal_split, WindowDataset

    print("Loading GiftEval time series...")
    series_list = load_gifteval_series(hf_token)
    val_splits = []
    for s in series_list:
        _, va, _ = temporal_split(s, cfg.train_frac, cfg.val_frac)
        if len(va) >= cfg.context_length:
            val_splits.append(va)

    val_ds = WindowDataset(val_splits, cfg.context_length, stride=cfg.context_length)
    n_use = min(n_windows, len(val_ds))
    print(f"  Available: {len(val_ds)}, using: {n_use}")

    windows = []
    for i in range(n_use):
        w = val_ds[i]
        values = w["values"]
        # Z-score normalize each window
        mean = float(values.mean())
        std = float(values.std())
        if std < 1e-6:
            continue  # Skip constant windows
        normalized = (values - mean) / std
        t = torch.tensor(normalized, dtype=torch.float32)
        if torch.isnan(t).any() or torch.isinf(t).any():
            continue  # Skip bad windows
        windows.append(t)
        if len(windows) >= n_windows:
            break

    result = torch.stack(windows)
    print(f"  Clean windows: {result.shape[0]} (skipped {n_use - result.shape[0]} bad)")
    return result


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def compute_psd(x: torch.Tensor) -> torch.Tensor:
    """Power spectral density via FFT magnitude. x: (..., T)"""
    fft = torch.fft.rfft(x, dim=-1)
    return fft.abs()


def compute_acf(x: torch.Tensor, max_lag: int = 64) -> torch.Tensor:
    """Autocorrelation function. x: (..., T)"""
    T = x.shape[-1]
    x_centered = x - x.mean(dim=-1, keepdim=True)
    var = (x_centered ** 2).sum(dim=-1, keepdim=True).clamp(min=1e-8)

    acfs = []
    for lag in range(max_lag):
        if lag == 0:
            acfs.append(torch.ones(*x.shape[:-1], 1, device=x.device))
        else:
            c = (x_centered[..., :T-lag] * x_centered[..., lag:]).sum(dim=-1, keepdim=True)
            acfs.append(c / var)
    return torch.cat(acfs, dim=-1)


def similarity(y_pred: torch.Tensor, y_real: torch.Tensor,
               lam_mse=1.0, lam_psd=1.0, lam_acf=1.0) -> torch.Tensor:
    """Compute similarity between predicted and real time series.
    y_pred: (B1, T), y_real: (B2, T) → returns (B1, B2) similarity matrix."""
    B1, T = y_pred.shape
    B2 = y_real.shape[0]

    # MSE: (B1, B2)
    mse = -((y_pred.unsqueeze(1) - y_real.unsqueeze(0)) ** 2).mean(dim=-1)

    # PSD distance: (B1, B2)
    psd_pred = compute_psd(y_pred)  # (B1, F)
    psd_real = compute_psd(y_real)  # (B2, F)
    psd_dist = -((psd_pred.unsqueeze(1) - psd_real.unsqueeze(0)) ** 2).mean(dim=-1)

    # ACF distance: (B1, B2)
    acf_pred = compute_acf(y_pred)  # (B1, max_lag)
    acf_real = compute_acf(y_real)  # (B2, max_lag)
    acf_dist = -((acf_pred.unsqueeze(1) - acf_real.unsqueeze(0)) ** 2).mean(dim=-1)

    # Normalize each component to similar scale
    mse_scale = mse.abs().mean().clamp(min=1e-8)
    psd_scale = psd_dist.abs().mean().clamp(min=1e-8)
    acf_scale = acf_dist.abs().mean().clamp(min=1e-8)

    return lam_mse * mse / mse_scale + lam_psd * psd_dist / psd_scale + lam_acf * acf_dist / acf_scale


# ---------------------------------------------------------------------------
# Linear mapping model
# ---------------------------------------------------------------------------

class LinearTimestepMapper(nn.Module):
    """Linear map applied per-timestep: y_t = W @ h_t + b"""
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d) → (B, T)"""
        return self.linear(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training: Variant 1 (Soft Retrieval)
# ---------------------------------------------------------------------------

def train_soft_retrieval(
    mapper: LinearTimestepMapper,
    h_text: torch.Tensor,      # (N_text, T, d)
    ts_windows: torch.Tensor,  # (N_ts, T)
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 128,
    ts_batch_size: int = 256,
    lr: float = 1e-3,
    log_freq: int = 10,
) -> list:
    """Train with soft retrieval loss."""
    mapper = mapper.to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=lr)

    N_text = h_text.shape[0]
    N_ts = ts_windows.shape[0]
    losses = []

    for epoch in range(n_epochs):
        perm = torch.randperm(N_text)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, N_text, batch_size):
            idx = perm[start:start + batch_size]
            h_batch = h_text[idx].to(device)

            # Random sample of TS windows for comparison
            ts_idx = torch.randint(0, N_ts, (ts_batch_size,))
            ts_batch = ts_windows[ts_idx].to(device)

            # Forward
            y_pred_raw = mapper(h_batch)  # (B, T)

            # Safe normalization
            pred_std = y_pred_raw.std(dim=-1, keepdim=True).clamp(min=1e-4)
            y_pred = (y_pred_raw - y_pred_raw.mean(dim=-1, keepdim=True)) / pred_std

            # Similarity matrix (B_text, B_ts)
            sim = similarity(y_pred, ts_batch)

            # Soft retrieval loss: encourage each prediction to match its best TS
            max_sim = sim.max(dim=1).values  # (B_text,)
            logsumexp = torch.logsumexp(sim, dim=1)  # (B_text,)
            loss = (logsumexp - max_sim).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % log_freq == 0 or epoch == n_epochs - 1:
            print(f"  [Soft] Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}", flush=True)

    return losses


# ---------------------------------------------------------------------------
# Training: Variant 2 (EM-Style Hard Matching)
# ---------------------------------------------------------------------------

def train_em_matching(
    mapper: LinearTimestepMapper,
    h_text: torch.Tensor,
    ts_windows: torch.Tensor,
    device: torch.device,
    n_epochs: int = 100,
    batch_size: int = 128,
    ts_batch_size: int = 512,
    lr: float = 1e-3,
    log_freq: int = 10,
) -> list:
    """Train with EM-style hard matching."""
    mapper = mapper.to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=lr)

    N_text = h_text.shape[0]
    N_ts = ts_windows.shape[0]
    losses = []

    for epoch in range(n_epochs):
        perm = torch.randperm(N_text)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, N_text, batch_size):
            idx = perm[start:start + batch_size]
            h_batch = h_text[idx].to(device)

            # Random TS batch
            ts_idx = torch.randint(0, N_ts, (ts_batch_size,))
            ts_batch = ts_windows[ts_idx].to(device)

            # Forward
            y_pred_raw = mapper(h_batch)  # (B, T)
            # Safe normalization
            pred_std = y_pred_raw.std(dim=-1, keepdim=True).clamp(min=1e-4)
            y_pred = (y_pred_raw - y_pred_raw.mean(dim=-1, keepdim=True)) / pred_std

            # E-step: find best match for each prediction
            with torch.no_grad():
                best_idx = []
                chunk_size = 32
                for ci in range(0, y_pred.shape[0], chunk_size):
                    pred_chunk = y_pred[ci:ci+chunk_size]
                    dists = ((pred_chunk.unsqueeze(1) - ts_batch.unsqueeze(0)) ** 2).mean(dim=-1)
                    best_idx.append(dists.argmin(dim=1))
                best_idx = torch.cat(best_idx)
                targets = ts_batch[best_idx]

            # M-step: minimize MSE to matched targets
            loss = F.mse_loss(y_pred, targets)

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if epoch % log_freq == 0 or epoch == n_epochs - 1:
            print(f"  [EM] Epoch {epoch:3d}/{n_epochs}  loss={avg_loss:.4f}", flush=True)

    return losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    mapper: LinearTimestepMapper,
    h_text: torch.Tensor,
    ts_windows: torch.Tensor,
    device: torch.device,
    name: str = "",
    n_eval: int = 1000,
    n_ts_compare: int = 5000,
) -> dict:
    """Evaluate the mapping: nearest neighbor retrieval + distributional stats."""
    mapper.eval()

    # Predict from a subset of text
    idx = torch.randperm(h_text.shape[0])[:n_eval]
    h_eval = h_text[idx].to(device)

    with torch.no_grad():
        y_pred_raw = mapper(h_eval)
        pred_std = y_pred_raw.std(dim=-1, keepdim=True).clamp(min=1e-4)
        y_pred = ((y_pred_raw - y_pred_raw.mean(dim=-1, keepdim=True)) / pred_std).cpu()

    # TS comparison set
    ts_idx = torch.randperm(ts_windows.shape[0])[:n_ts_compare]
    ts_eval = ts_windows[ts_idx]

    # Nearest neighbor distances
    nn_dists = []
    BS = 100
    for start in range(0, n_eval, BS):
        end = min(start + BS, n_eval)
        pred_batch = y_pred[start:end]
        dists = ((pred_batch.unsqueeze(1) - ts_eval.unsqueeze(0)) ** 2).mean(dim=-1)
        nn_dist = dists.min(dim=1).values
        nn_dists.append(nn_dist)
    nn_dists = torch.cat(nn_dists)

    # Distributional stats
    pred_var = y_pred.var(dim=-1).mean().item()
    real_var = ts_eval.var(dim=-1).mean().item()

    pred_psd = compute_psd(y_pred).mean(dim=0)
    real_psd = compute_psd(ts_eval).mean(dim=0)
    psd_dist = ((pred_psd - real_psd) ** 2).mean().item()

    pred_acf = compute_acf(y_pred).mean(dim=0)
    real_acf = compute_acf(ts_eval).mean(dim=0)
    acf_dist = ((pred_acf - real_acf) ** 2).mean().item()

    results = {
        "name": name,
        "nn_dist_mean": nn_dists.mean().item(),
        "nn_dist_median": nn_dists.median().item(),
        "nn_dist_std": nn_dists.std().item(),
        "pred_var": pred_var,
        "real_var": real_var,
        "psd_dist": psd_dist,
        "acf_dist": acf_dist,
    }

    print(f"  [{name}] NN dist: mean={results['nn_dist_mean']:.4f} "
          f"median={results['nn_dist_median']:.4f} | "
          f"PSD dist={psd_dist:.4f} ACF dist={acf_dist:.4f} | "
          f"Var: pred={pred_var:.3f} real={real_var:.3f}")

    return results, y_pred, ts_eval


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_examples(y_pred, ts_windows, output_path, n_examples=10):
    """Plot predicted sequences and their nearest real matches."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(16, 2.5 * n_examples))

    for i in range(n_examples):
        pred = y_pred[i].numpy()
        dists = ((y_pred[i:i+1] - ts_windows) ** 2).mean(dim=-1).squeeze(0)
        nn_idx = dists.argmin().item()
        real = ts_windows[nn_idx].numpy()
        nn_dist = dists[nn_idx].item()

        axes[i, 0].plot(pred, color="blue", linewidth=0.8)
        axes[i, 0].set_title(f"Predicted #{i}" if i > 0 else "Predicted (from text)")
        axes[i, 0].set_ylim(-4, 4)

        axes[i, 1].plot(real, color="green", linewidth=0.8)
        axes[i, 1].set_title(f"Nearest real TS (dist={nn_dist:.3f})")
        axes[i, 1].set_ylim(-4, 4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_distributions(y_pred, ts_windows, output_path):
    """Plot PSD and ACF distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PSD
    pred_psd = compute_psd(y_pred).mean(dim=0).numpy()
    real_psd = compute_psd(ts_windows[:1000]).mean(dim=0).numpy()
    axes[0, 0].plot(pred_psd[:100], label="Predicted", alpha=0.8)
    axes[0, 0].plot(real_psd[:100], label="Real TS", alpha=0.8)
    axes[0, 0].set_title("Mean PSD (first 100 freq bins)")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")

    # ACF
    pred_acf = compute_acf(y_pred).mean(dim=0).numpy()
    real_acf = compute_acf(ts_windows[:1000]).mean(dim=0).numpy()
    axes[0, 1].plot(pred_acf, label="Predicted", alpha=0.8)
    axes[0, 1].plot(real_acf, label="Real TS", alpha=0.8)
    axes[0, 1].set_title("Mean ACF")
    axes[0, 1].legend()

    # Variance distribution
    pred_vars = y_pred.var(dim=-1).numpy()
    real_vars = ts_windows[:1000].var(dim=-1).numpy()

    def safe_bins(data, max_bins=50):
        n_unique = len(np.unique(np.round(data, 4)))
        return min(max_bins, max(n_unique, 2))

    axes[1, 0].hist(pred_vars, bins=safe_bins(pred_vars), alpha=0.5, label="Predicted", density=True)
    axes[1, 0].hist(real_vars, bins=safe_bins(real_vars), alpha=0.5, label="Real TS", density=True)
    axes[1, 0].set_title("Variance Distribution")
    axes[1, 0].legend()

    # NN distance distribution
    nn_dists = []
    for i in range(min(500, y_pred.shape[0])):
        dists = ((y_pred[i:i+1] - ts_windows[:5000]) ** 2).mean(dim=-1)
        nn_dists.append(dists.min().item())
    nn_dists = np.array(nn_dists)
    axes[1, 1].hist(nn_dists, bins=safe_bins(nn_dists), alpha=0.7, color="purple")
    axes[1, 1].set_title("Nearest Neighbor Distance Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--n_wiki", type=int, default=10000)
    parser.add_argument("--n_ts", type=int, default=50000)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="mapping_results")
    args = parser.parse_args()

    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")
    device = torch.device("cuda:0")
    os.makedirs(args.output_dir, exist_ok=True)

    layer_dir = os.path.join(args.output_dir, f"layer_{args.layer}")
    os.makedirs(layer_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"MAPPING EXPERIMENT — Layer {args.layer}")
    print("=" * 60)

    # WikiText
    from src.data.wikitext import load_wikitext_sequences
    wiki_sequences = load_wikitext_sequences(
        max_sequences=args.n_wiki, seq_len=512, hf_token=hf_token
    )
    print(f"WikiText sequences: {len(wiki_sequences)}")

    # Time series
    ts_windows = load_ts_windows(cfg, args.n_ts, hf_token)
    print(f"TS windows: {ts_windows.shape}")

    # ── Extract hidden states ─────────────────────────────────────────────
    # PT hidden states
    pt_hs_path = os.path.join(layer_dir, "pt_hidden_states.pt")
    if os.path.exists(pt_hs_path):
        print("Loading cached PT hidden states...")
        h_pt = torch.load(pt_hs_path, weights_only=True)
    else:
        h_pt = extract_pt_hidden_states(args.layer, wiki_sequences, cfg, device)
        torch.save(h_pt, pt_hs_path)

    # Split into train/eval
    n_train = int(0.8 * h_pt.shape[0])
    h_train = h_pt[:n_train]
    h_eval = h_pt[n_train:]
    ts_train = ts_windows[:int(0.8 * ts_windows.shape[0])]
    ts_eval_set = ts_windows[int(0.8 * ts_windows.shape[0]):]

    print(f"Train: {h_train.shape[0]} text, {ts_train.shape[0]} TS")
    print(f"Eval:  {h_eval.shape[0]} text, {ts_eval_set.shape[0]} TS")

    d = h_pt.shape[-1]

    # ── Variant 2: EM-style ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VARIANT 2: EM-Style Hard Matching")
    print("=" * 60)

    mapper_em = LinearTimestepMapper(d)
    losses_em = train_em_matching(
        mapper_em, h_train, ts_train, device,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, log_freq=20,
    )

    results_em, pred_em, ts_for_eval = evaluate(
        mapper_em, h_eval, ts_eval_set, device, name="EM-PT"
    )

    plot_examples(pred_em, ts_for_eval,
                  os.path.join(layer_dir, "em_examples.png"))
    plot_distributions(pred_em, ts_for_eval,
                       os.path.join(layer_dir, "em_distributions.png"))

    # ── Variant 1: Soft Retrieval ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VARIANT 1: Soft Retrieval")
    print("=" * 60)

    mapper_soft = LinearTimestepMapper(d)
    losses_soft = train_soft_retrieval(
        mapper_soft, h_train, ts_train, device,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, log_freq=20,
    )

    results_soft, pred_soft, _ = evaluate(
        mapper_soft, h_eval, ts_eval_set, device, name="Soft-PT"
    )

    plot_examples(pred_soft, ts_for_eval,
                  os.path.join(layer_dir, "soft_examples.png"))
    plot_distributions(pred_soft, ts_for_eval,
                       os.path.join(layer_dir, "soft_distributions.png"))

    # ── Baselines ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINES")
    print("=" * 60)

    # Baseline 1: Random hidden states
    print("\n--- Baseline: Random hidden states ---")
    h_random = torch.randn_like(h_train)
    mapper_rand = LinearTimestepMapper(d)
    losses_rand = train_em_matching(
        mapper_rand, h_random, ts_train, device,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, log_freq=50,
    )
    h_random_eval = torch.randn_like(h_eval)
    results_rand, pred_rand, _ = evaluate(
        mapper_rand, h_random_eval, ts_eval_set, device, name="Random"
    )
    plot_examples(pred_rand, ts_for_eval,
                  os.path.join(layer_dir, "random_examples.png"))

    # Baseline 2: RI model hidden states
    print("\n--- Baseline: RI hidden states ---")
    ri_hs_path = os.path.join(layer_dir, "ri_hidden_states.pt")
    if os.path.exists(ri_hs_path):
        print("Loading cached RI hidden states...")
        h_ri = torch.load(ri_hs_path, weights_only=True)
    else:
        h_ri = extract_ri_hidden_states(args.layer, wiki_sequences, cfg, device)
        torch.save(h_ri, ri_hs_path)

    h_ri_train = h_ri[:n_train]
    h_ri_eval = h_ri[n_train:]
    mapper_ri = LinearTimestepMapper(d)
    losses_ri = train_em_matching(
        mapper_ri, h_ri_train, ts_train, device,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, log_freq=50,
    )
    results_ri, pred_ri, _ = evaluate(
        mapper_ri, h_ri_eval, ts_eval_set, device, name="RI"
    )
    plot_examples(pred_ri, ts_for_eval,
                  os.path.join(layer_dir, "ri_examples.png"))

    # Baseline 3: Shuffled time dimension
    print("\n--- Baseline: Shuffled time dimension ---")
    h_shuffled = h_train.clone()
    for i in range(h_shuffled.shape[0]):
        perm = torch.randperm(512)
        h_shuffled[i] = h_shuffled[i, perm]
    mapper_shuf = LinearTimestepMapper(d)
    losses_shuf = train_em_matching(
        mapper_shuf, h_shuffled, ts_train, device,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, log_freq=50,
    )
    h_eval_shuf = h_eval.clone()
    for i in range(h_eval_shuf.shape[0]):
        perm = torch.randperm(512)
        h_eval_shuf[i] = h_eval_shuf[i, perm]
    results_shuf, pred_shuf, _ = evaluate(
        mapper_shuf, h_eval_shuf, ts_eval_set, device, name="Shuffled-Time"
    )
    plot_examples(pred_shuf, ts_for_eval,
                  os.path.join(layer_dir, "shuffled_examples.png"))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_results = {
        "layer": args.layer,
        "n_wiki": len(wiki_sequences),
        "n_ts": ts_windows.shape[0],
        "n_epochs": args.n_epochs,
        "results": {
            "EM_PT": results_em,
            "Soft_PT": results_soft,
            "Random": results_rand,
            "RI": results_ri,
            "Shuffled_Time": results_shuf,
        },
        "losses": {
            "EM_PT": losses_em,
            "Soft_PT": losses_soft,
            "Random": losses_rand,
            "RI": losses_ri,
            "Shuffled_Time": losses_shuf,
        },
    }

    with open(os.path.join(layer_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'Method':<20} {'NN Dist':>10} {'PSD Dist':>10} {'ACF Dist':>10} {'Pred Var':>10}")
    print("-" * 65)
    for name, res in all_results["results"].items():
        print(f"{name:<20} {res['nn_dist_mean']:>10.4f} {res['psd_dist']:>10.4f} "
              f"{res['acf_dist']:>10.4f} {res['pred_var']:>10.3f}")

    print(f"\nResults saved to {layer_dir}/")

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses_em, label="EM (PT)", linewidth=2)
    ax.plot(losses_soft, label="Soft (PT)", linewidth=2)
    ax.plot(losses_rand, label="Random baseline", linewidth=1, linestyle="--")
    ax.plot(losses_ri, label="RI baseline", linewidth=1, linestyle="--")
    ax.plot(losses_shuf, label="Shuffled time", linewidth=1, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — Layer {args.layer}")
    ax.legend()
    ax.set_yscale("log")
    plt.savefig(os.path.join(layer_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
