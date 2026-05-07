"""
Re-extract top features using window-level SUM instead of peak activation.
Precomputes activations on-the-fly (single GPU) since disk copies were deleted.

Usage:
    python3 scripts/extract_window_sum.py --layer 13 --gpu 0 --category PT_FT_RI
"""
import argparse
import os
import sys
import json
import time
import heapq

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.crosscoder.model import Crosscoder
from src.models.extractor import ModelExtractor
from src.data.dataset import (
    build_datasets, WindowDataset, load_gifteval_series, temporal_split,
)

N_VAL_WINDOWS = 50_000
TOP_FEATURES = 30
TOP_WINDOWS = 10
FIRING_THRESHOLD = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--category", type=str, default="PT_FT_RI")
    args = parser.parse_args()

    cfg = Config()
    cfg.linear_crosscoder = True
    cfg.latent_dim = 4096
    cfg.top_k = 64
    cfg.checkpoint_dir = "checkpoints/linear_d4096"

    device = torch.device(f"cuda:{args.gpu}")
    hf_token = os.environ.get("HF_TOKEN")
    T, H = cfg.context_length, cfg.hidden_size

    # Load crosscoder
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"layer_{args.layer}", "crosscoder.pt")
    cc = Crosscoder(cfg)
    cc.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    cc = cc.to(device).eval()
    print(f"Loaded crosscoder from {ckpt_path}", flush=True)

    # Load val dataset
    print("Loading val dataset...", flush=True)
    series_list = load_gifteval_series(hf_token)
    val_splits = []
    for s in series_list:
        _, va, _ = temporal_split(s, cfg.train_frac, cfg.val_frac)
        if len(va) >= cfg.context_length:
            val_splits.append(va)
    val_ds = WindowDataset(val_splits, cfg.context_length, stride=cfg.context_length)
    n_use = min(N_VAL_WINDOWS, len(val_ds))
    print(f"  Val windows: {n_use:,}", flush=True)

    # Load extractor
    print("Loading 3 models for extraction...", flush=True)
    extractor = ModelExtractor(cfg, device, hf_token=hf_token, pt_sub_batch=16)

    # Load existing categorization to get feature IDs for this category
    cat_path = f"analysis/layer_{args.layer}/categorization.json"
    cat_data = json.load(open(cat_path))
    feature_stats = cat_data["feature_stats"]

    cat_features = [
        int(fid) for fid, stats in feature_stats.items()
        if stats["category"] == args.category
    ]
    # Rank by mean activation to pick top features (same as original)
    def sort_key(j):
        s = feature_stats[str(j)]
        vals = []
        for domain in ["pt", "ft", "ri"]:
            if domain.upper() in args.category or args.category == "PT_FT_RI":
                vals.append(s[f"mean_act_{domain}"])
        return sum(vals) / len(vals) if vals else 0

    ranked_features = sorted(cat_features, key=sort_key, reverse=True)
    top_feats = ranked_features[:TOP_FEATURES]
    print(f"  {args.category}: {len(cat_features)} features, extracting top {len(top_feats)}", flush=True)

    # Scan all val windows, compute window-level SUM for each top feature
    # Track top-K windows by sum
    sum_topk = {j: [] for j in top_feats}
    batch_size = 16
    t0 = time.time()

    print(f"Scanning {n_use} windows (window-sum ranking)...", flush=True)
    with torch.no_grad():
        for start in range(0, n_use, batch_size):
            end = min(start + batch_size, n_use)
            B = end - start
            windows = [val_ds[i] for i in range(start, end)]

            pt_acts, ft_acts, ri_acts = extractor.extract_all(windows, [args.layer])

            x_pt = pt_acts[args.layer].float()  # (B*T, H)
            x_ft = ft_acts[args.layer].float()
            x_ri = ri_acts[args.layer].float()

            z_pt, _, _ = cc.encode_single(x_pt, "PT")
            z_ft, _, _ = cc.encode_single(x_ft, "FT")
            z_ri, _, _ = cc.encode_single(x_ri, "RI")

            z_pt = z_pt.reshape(B, T, -1)
            z_ft = z_ft.reshape(B, T, -1)
            z_ri = z_ri.reshape(B, T, -1)

            for j in top_feats:
                z_combined = z_pt[:, :, j] + z_ft[:, :, j] + z_ri[:, :, j]
                window_sums = z_combined.sum(dim=1)  # (B,)
                frac_active = (z_combined > 0).float().mean(dim=1)  # (B,)

                for b_idx in range(B):
                    s = window_sums[b_idx].item()
                    frac = frac_active[b_idx].item()
                    if s <= 0:
                        continue
                    win_idx = start + b_idx
                    peak_t = z_combined[b_idx].argmax().item()
                    entry = (s, win_idx, peak_t, frac)
                    if len(sum_topk[j]) < TOP_WINDOWS:
                        heapq.heappush(sum_topk[j], entry)
                    elif s > sum_topk[j][0][0]:
                        heapq.heapreplace(sum_topk[j], entry)

            if (start // batch_size) % 50 == 0 and start > 0:
                elapsed = time.time() - t0
                pct = start / n_use * 100
                rate = start / elapsed
                eta = (n_use - start) / rate / 60 if rate > 0 else 0
                print(f"  {start}/{n_use} ({pct:.0f}%) {rate:.0f} win/s ETA {eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"Scan done in {elapsed/60:.1f}m", flush=True)

    # Free extractor
    del extractor
    torch.cuda.empty_cache()

    # Collect results and save
    out_dir = f"analysis/layer_{args.layer}/{args.category}_window_sum"
    os.makedirs(out_dir, exist_ok=True)

    all_features = []
    for j in top_feats:
        top_wins = sorted(sum_topk[j], key=lambda x: -x[0])
        feat_data = {
            "feature_id": j,
            "stats": feature_stats[str(j)],
            "top_windows": [],
        }

        for s, win_idx, peak_t, frac in top_wins:
            window = val_ds[win_idx]
            raw_values = window["values"].tolist()
            feat_data["top_windows"].append({
                "window_sum": s,
                "frac_active": frac,
                "window_idx": win_idx,
                "peak_timestep": peak_t,
                "series_idx": window["series_idx"],
                "offset": window["offset"],
                "raw_values": raw_values,
            })

        all_features.append(feat_data)

        # Save per-feature data
        feat_dir = os.path.join(out_dir, f"feature_{j}")
        os.makedirs(feat_dir, exist_ok=True)
        with open(os.path.join(feat_dir, "windows.json"), "w") as f:
            json.dump(feat_data["top_windows"], f)

    # Save ranking
    ranking = []
    for feat_data in all_features:
        j = feat_data["feature_id"]
        s = feat_data["stats"]
        best_sum = feat_data["top_windows"][0]["window_sum"] if feat_data["top_windows"] else 0
        best_frac = feat_data["top_windows"][0]["frac_active"] if feat_data["top_windows"] else 0
        ranking.append({
            "feature_id": j,
            "category": s["category"],
            "rate_pt": s["rate_pt"],
            "rate_ft": s["rate_ft"],
            "rate_ri": s["rate_ri"],
            "best_window_sum": best_sum,
            "best_frac_active": best_frac,
        })

    with open(os.path.join(out_dir, "ranking.json"), "w") as f:
        json.dump(ranking, f, indent=2)

    # Plot top windows
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection

    for feat_data in all_features:
        j = feat_data["feature_id"]
        windows = feat_data["top_windows"][:TOP_WINDOWS]
        if not windows:
            continue

        n_plots = len(windows)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), squeeze=False)

        for i, win in enumerate(windows):
            ax = axes[i, 0]
            values = np.array(win["raw_values"], dtype=np.float64)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

            x = np.arange(len(values))
            ax.plot(x, values, color="darkorange", linewidth=1.0, alpha=0.8)

            v_min, v_max = float(values.min()), float(values.max())
            v_range = max(abs(v_max - v_min), 1e-6)
            ax.set_xlim(0, len(values))
            ax.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)

            ax.set_ylabel(f"#{i+1}\nsum={win['window_sum']:.1f}\n{win['frac_active']:.0%} active",
                         fontsize=7)
            if i == 0:
                ax.set_title(f"Feature {j} — Top {n_plots} by Window Sum "
                            f"({feat_data['stats']['category']})", fontsize=10)
            if i < n_plots - 1:
                ax.set_xticklabels([])

        axes[-1, 0].set_xlabel("Timestep")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"feature_{j}", "plot_window_sum.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved to {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
