"""
Forecasting test: can the linear mapping predict missing TS timesteps?

Variation A: Text → predict masked TS points
  1. Project 100K WikiText hidden states through mapper W → predictions
  2. For each prediction, find best-matching real TS
  3. Mask 32 points (end / middle / random)
  4. Score: MSE of W @ h_t at masked positions vs real TS values

Variation B: TS → retrieve best text projection on visible points → forecast masked
  1. Take real TS, mask 32 points
  2. Find WikiText projection that best matches on VISIBLE points
  3. Use that projection's values at masked positions as forecast
  4. Score: MSE at masked positions

Baselines: last-value, linear interpolation, mean of visible.

Usage:
    /usr/bin/python3 scripts/forecasting_test.py
"""
import torch
import torch.nn as nn
import numpy as np
import json, os, sys, gc, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences
from scripts.mapping_experiment import load_ts_windows

T = 512
MASK_LEN = 32
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/forecasting"
SEED = 42


def load_predictions(n_seqs):
    """Load or verify saved predictions from the 100K run."""
    pred_path = "mapping_results/reverse_mapping_100k/pred_all_100000.pt"
    if os.path.exists(pred_path):
        pred = torch.load(pred_path, map_location="cpu", weights_only=True)
        return pred[:n_seqs]
    else:
        print(f"  Predictions not found at {pred_path}. Run reverse_mapping_full.py first.")
        sys.exit(1)


def create_masks(T, mask_len, mask_type):
    """Create boolean mask (True = visible, False = masked)."""
    mask = np.ones(T, dtype=bool)
    if mask_type == "end":
        mask[T - mask_len:] = False
    elif mask_type == "middle":
        start = (T - mask_len) // 2
        mask[start:start + mask_len] = False
    elif mask_type == "random":
        rng = np.random.default_rng(SEED)
        idx = rng.choice(T, mask_len, replace=False)
        mask[idx] = False
    return mask


def baseline_last_value(ts_visible, mask):
    """Predict masked points by carrying forward the last visible value before each gap."""
    pred = ts_visible.copy()
    masked_idx = np.where(~mask)[0]
    for i in masked_idx:
        # Find last visible index before i
        prev_visible = np.where(mask[:i])[0]
        if len(prev_visible) > 0:
            pred[i] = ts_visible[prev_visible[-1]]
        else:
            # No visible point before — use first visible after
            next_visible = np.where(mask[i:])[0]
            if len(next_visible) > 0:
                pred[i] = ts_visible[i + next_visible[0]]
            else:
                pred[i] = 0.0
    return pred


def baseline_linear_interp(ts_visible, mask):
    """Linear interpolation between visible points."""
    visible_idx = np.where(mask)[0]
    visible_vals = ts_visible[visible_idx]
    all_idx = np.arange(len(ts_visible))
    return np.interp(all_idx, visible_idx, visible_vals)


def baseline_mean(ts_visible, mask):
    """Predict masked points with mean of visible points."""
    pred = ts_visible.copy()
    mean_val = ts_visible[mask].mean()
    pred[~mask] = mean_val
    return pred


def main():
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # Load predictions and TS bank
    print("Loading predictions...", flush=True)
    N_PRED = 10000  # use first 10K for speed
    pred_all = load_predictions(N_PRED)
    print(f"  Predictions: {pred_all.shape}")

    print("Loading TS bank...", flush=True)
    cfg = Config()
    ts_bank = load_ts_windows(cfg, 10000, hf_token)
    print(f"  TS bank: {ts_bank.shape}")

    # ═══ Find best matches (full sequence) ═══
    print("\nFinding best matches for each prediction...", flush=True)
    # For each prediction, find nearest TS in bank
    N_EVAL = 500  # evaluate on 500 predictions
    nn_indices = []
    nn_dists = []
    t0 = time.time()
    ts_gpu = ts_bank.to(DEVICE)
    for i in range(N_EVAL):
        d2 = ((pred_all[i:i+1].to(DEVICE) - ts_gpu) ** 2).mean(-1).squeeze(0)
        nn_indices.append(d2.argmin().item())
        nn_dists.append(d2.min().item())
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{N_EVAL} ({time.time()-t0:.0f}s)", flush=True)
    nn_indices = np.array(nn_indices)
    nn_dists = np.array(nn_dists)
    print(f"  Mean NN dist: {nn_dists.mean():.4f}, Unique: {len(set(nn_indices))}")

    # ═══ Mask types ═══
    mask_types = ["end", "middle", "random"]

    # ═══ Variation A: use linear map predictions at masked positions ═══
    print(f"\n{'='*60}")
    print("VARIATION A: Linear map predicts masked points")
    print(f"{'='*60}")

    results_a = {}
    for mask_type in mask_types:
        mask = create_masks(T, MASK_LEN, mask_type)
        masked_idx = np.where(~mask)[0]

        mse_mapper = []
        mse_last = []
        mse_interp = []
        mse_mean = []

        for i in range(N_EVAL):
            ti = nn_indices[i]
            real_ts = ts_bank[ti].numpy()
            pred_ts = pred_all[i].numpy()

            # Ground truth at masked positions
            gt = real_ts[masked_idx]

            # Mapper prediction
            mapper_pred = pred_ts[masked_idx]
            mse_mapper.append(((mapper_pred - gt) ** 2).mean())

            # Baselines use the REAL TS values at visible positions
            real_visible = real_ts.copy()
            real_visible[~mask] = np.nan  # mark masked

            last_pred = baseline_last_value(real_ts, mask)
            mse_last.append(((last_pred[masked_idx] - gt) ** 2).mean())

            interp_pred = baseline_linear_interp(real_ts, mask)
            mse_interp.append(((interp_pred[masked_idx] - gt) ** 2).mean())

            mean_pred = baseline_mean(real_ts, mask)
            mse_mean.append(((mean_pred[masked_idx] - gt) ** 2).mean())

        results_a[mask_type] = {
            "mapper": float(np.mean(mse_mapper)),
            "last_value": float(np.mean(mse_last)),
            "linear_interp": float(np.mean(mse_interp)),
            "mean": float(np.mean(mse_mean)),
        }
        print(f"\n  Mask: {mask_type} ({MASK_LEN} points)")
        print(f"    Mapper:       MSE = {results_a[mask_type]['mapper']:.4f}")
        print(f"    Last value:   MSE = {results_a[mask_type]['last_value']:.4f}")
        print(f"    Linear interp:MSE = {results_a[mask_type]['linear_interp']:.4f}")
        print(f"    Mean:         MSE = {results_a[mask_type]['mean']:.4f}")

    # ═══ Variation B: retrieve best match on visible points, forecast masked ═══
    print(f"\n{'='*60}")
    print("VARIATION B: Retrieve on visible, forecast masked")
    print(f"{'='*60}")

    # Use a subset of TS as "queries" and all predictions as the "database"
    N_QUERY = 200
    query_indices = np.random.choice(len(ts_bank), N_QUERY, replace=False)

    results_b = {}
    for mask_type in mask_types:
        mask = create_masks(T, MASK_LEN, mask_type)
        visible_idx = np.where(mask)[0]
        masked_idx = np.where(~mask)[0]

        mse_retrieval = []
        mse_last = []
        mse_interp = []
        mse_mean = []

        t0 = time.time()
        for qi, ts_idx in enumerate(query_indices):
            real_ts = ts_bank[ts_idx].numpy()
            real_visible = real_ts[visible_idx]  # (T - mask_len,)
            gt = real_ts[masked_idx]

            # Find best matching prediction on VISIBLE points only
            pred_visible = pred_all[:N_PRED, visible_idx]  # (N_PRED, T-mask_len)
            d2 = ((pred_visible - torch.tensor(real_visible).unsqueeze(0)) ** 2).mean(-1)
            best_pred_idx = d2.argmin().item()

            # Use that prediction's values at masked positions
            retrieval_pred = pred_all[best_pred_idx].numpy()[masked_idx]
            mse_retrieval.append(((retrieval_pred - gt) ** 2).mean())

            # Baselines
            last_pred = baseline_last_value(real_ts, mask)
            mse_last.append(((last_pred[masked_idx] - gt) ** 2).mean())

            interp_pred = baseline_linear_interp(real_ts, mask)
            mse_interp.append(((interp_pred[masked_idx] - gt) ** 2).mean())

            mean_pred = baseline_mean(real_ts, mask)
            mse_mean.append(((mean_pred[masked_idx] - gt) ** 2).mean())

            if (qi+1) % 50 == 0:
                print(f"    query {qi+1}/{N_QUERY} ({time.time()-t0:.0f}s)", flush=True)

        results_b[mask_type] = {
            "retrieval": float(np.mean(mse_retrieval)),
            "last_value": float(np.mean(mse_last)),
            "linear_interp": float(np.mean(mse_interp)),
            "mean": float(np.mean(mse_mean)),
        }
        print(f"\n  Mask: {mask_type} ({MASK_LEN} points)")
        print(f"    Retrieval:    MSE = {results_b[mask_type]['retrieval']:.4f}")
        print(f"    Last value:   MSE = {results_b[mask_type]['last_value']:.4f}")
        print(f"    Linear interp:MSE = {results_b[mask_type]['linear_interp']:.4f}")
        print(f"    Mean:         MSE = {results_b[mask_type]['mean']:.4f}")

    # ═══ Save ═══
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump({"variation_a": results_a, "variation_b": results_b,
                    "mask_len": MASK_LEN, "n_eval_a": N_EVAL, "n_query_b": N_QUERY,
                    "n_pred": N_PRED}, f, indent=2)

    # ═══ Plots ═══
    print("\nGenerating plots...", flush=True)

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Forecasting Test: Can Linear Map Predict {MASK_LEN} Missing Timesteps?",
                 fontsize=14, fontweight='bold')

    # Variation A
    ax = axes[0]
    x = np.arange(len(mask_types))
    w = 0.2
    methods_a = ["mapper", "last_value", "linear_interp", "mean"]
    colors_a = ["#9C27B0", "#4CAF50", "#2196F3", "#FF9800"]
    labels_a = ["LM mapper", "Last value", "Linear interp", "Mean"]
    for mi, (method, color, label) in enumerate(zip(methods_a, colors_a, labels_a)):
        vals = [results_a[mt][method] for mt in mask_types]
        ax.bar(x + mi*w, vals, w, color=color, label=label)
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels([f"Mask {mt}" for mt in mask_types])
    ax.set_ylabel("MSE at masked positions"); ax.set_title("Var. A: Map predicts masked points")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Variation B
    ax = axes[1]
    methods_b = ["retrieval", "last_value", "linear_interp", "mean"]
    labels_b = ["Retrieval", "Last value", "Linear interp", "Mean"]
    for mi, (method, color, label) in enumerate(zip(methods_b, colors_a, labels_b)):
        vals = [results_b[mt][method] for mt in mask_types]
        ax.bar(x + mi*w, vals, w, color=color, label=label)
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels([f"Mask {mt}" for mt in mask_types])
    ax.set_ylabel("MSE at masked positions"); ax.set_title("Var. B: Retrieve on visible, forecast masked")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/forecasting_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Example plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f"Forecasting Examples (Variation B: retrieve on visible, predict masked)\n"
                 f"Green = real TS, purple = retrieved prediction, gray = masked region",
                 fontsize=13, fontweight='bold')

    for col, mask_type in enumerate(mask_types):
        mask = create_masks(T, MASK_LEN, mask_type)
        visible_idx = np.where(mask)[0]
        masked_idx = np.where(~mask)[0]

        for row in range(3):
            ax = axes[row, col]
            ts_idx = query_indices[row]
            real_ts = ts_bank[ts_idx].numpy()

            # Retrieve
            pred_visible = pred_all[:N_PRED, visible_idx]
            d2 = ((pred_visible - torch.tensor(real_ts[visible_idx]).unsqueeze(0)) ** 2).mean(-1)
            best_idx = d2.argmin().item()
            best_pred = pred_all[best_idx].numpy()

            # Plot
            ax.plot(real_ts, color='green', linewidth=1, alpha=0.7, label='Real TS')
            ax.plot(best_pred, color='#9C27B0', linewidth=1, alpha=0.7, label='Retrieved pred')

            # Highlight masked region
            for i in masked_idx:
                ax.axvspan(i-0.5, i+0.5, color='gray', alpha=0.15)

            mse = ((best_pred[masked_idx] - real_ts[masked_idx]) ** 2).mean()
            ax.set_ylim(-4, 4)
            ax.set_title(f"Mask: {mask_type}, MSE={mse:.3f}" if row == 0 else f"MSE={mse:.3f}",
                         fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/forecasting_examples.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("FORECASTING RESULTS")
    print(f"{'='*60}")
    print(f"\nVariation A (mapper predicts masked, {N_EVAL} samples):")
    print(f"{'Mask':<10} {'Mapper':>8} {'Last':>8} {'Interp':>8} {'Mean':>8}")
    for mt in mask_types:
        r = results_a[mt]
        print(f"{mt:<10} {r['mapper']:>8.4f} {r['last_value']:>8.4f} {r['linear_interp']:>8.4f} {r['mean']:>8.4f}")

    print(f"\nVariation B (retrieve on visible, {N_QUERY} queries from {N_PRED} predictions):")
    print(f"{'Mask':<10} {'Retrieve':>8} {'Last':>8} {'Interp':>8} {'Mean':>8}")
    for mt in mask_types:
        r = results_b[mt]
        print(f"{mt:<10} {r['retrieval']:>8.4f} {r['last_value']:>8.4f} {r['linear_interp']:>8.4f} {r['mean']:>8.4f}")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
