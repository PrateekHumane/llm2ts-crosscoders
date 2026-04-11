"""
Per-layer feature analysis pipeline.

For a single layer:
  A. Precompute TS activations (50k val windows, 4 GPUs)
  B. Precompute WikiText activations (30k sequences, PT only)
  C. Run crosscoder, categorize features
  D. Rank & extract top features for PT_FT_RI, PT_FT, FT_RI
  E. Plot top features
  F. Cleanup raw activations

Usage:
    /usr/bin/python3 scripts/analyze_layer.py --layer 13
    /usr/bin/python3 scripts/analyze_layer.py --all   # run all layers in priority order
"""
import argparse
import os
import sys
import json
import shutil
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import build_datasets
from src.data.wikitext import load_wikitext_sequences
from src.crosscoder.precompute import precompute
from src.crosscoder.model import Crosscoder

LAYER_ORDER = [13, 0, 27, 6, 20, 3, 10, 16, 24, 1, 4, 8, 11, 14, 18, 22, 26,
               2, 5, 7, 9, 12, 15, 17, 19, 21, 23, 25]
SKIP_LAYERS = {27}  # NaN during training

ANALYSIS_DIR = "analysis"
N_TS_WINDOWS = 50_000
N_WIKI_SEQS = 30_000
WIKI_SEQ_LEN = 512
TOP_FEATURES_PER_CAT = 30
TOP_WINDOWS_PER_FEAT = 10
TOP_WIKI_PER_FEAT = 10
WIKI_CONTEXT_TOKENS = 50  # ±50 tokens around peak activation
FIRING_THRESHOLD = 0.01   # 1% of timesteps

FOCUS_CATEGORIES = ["PT_FT_RI", "PT_FT", "FT_RI"]
ALL_CATEGORIES = ["PT_FT_RI", "PT_FT", "FT_RI", "PT_RI",
                  "PT_only", "FT_only", "RI_only", "None"]


# ---------------------------------------------------------------------------
# Precompute WikiText activations for one layer
# ---------------------------------------------------------------------------

def precompute_wiki_activations(
    layer_idx: int,
    wiki_sequences: list[dict],
    cfg: Config,
    output_dir: str,
    hf_token: str | None = None,
):
    """
    Run PT model on WikiText sequences, extract one layer's hidden states.
    Saves float16 memmap of shape (N_seqs, seq_len, hidden_size).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0")
    N = len(wiki_sequences)
    T = WIKI_SEQ_LEN
    H = cfg.hidden_size

    out_path = os.path.join(output_dir, "wiki_pt.bin")
    meta_path = os.path.join(output_dir, "wiki_pt.json")
    expected = N * T * H * 2

    if os.path.exists(out_path) and os.path.getsize(out_path) == expected:
        print(f"  WikiText activations already on disk.", flush=True)
        return

    print(f"  Loading PT model for WikiText extraction...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_pt, dtype=torch.bfloat16, token=hf_token
    ).model.to(device).eval()

    shape = (N, T, H)
    mm = np.memmap(out_path, dtype="float16", mode="w+", shape=shape)
    with open(meta_path, "w") as f:
        json.dump({"shape": list(shape), "dtype": "float16"}, f)

    # Hook to capture layer output
    captured = {}
    def hook(module, input, output):
        # output is (B, T, H) tensor for Qwen3 layers
        hs = output[0] if isinstance(output, tuple) else output
        captured["hs"] = hs.detach()
    handle = model.layers[layer_idx].register_forward_hook(hook)

    # Truncate model to only run through needed layers
    original_layers = model.layers
    model.layers = original_layers[:layer_idx + 1]

    batch_size = 32
    t0 = time.time()
    cursor = 0

    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = wiki_sequences[start:start + batch_size]
            input_ids = torch.tensor(
                np.stack([s["input_ids"] for s in batch]),
                dtype=torch.long, device=device
            )
            model(input_ids=input_ids, use_cache=False)
            hs = captured["hs"].float().cpu().half().numpy()  # (B, T, H)
            B = hs.shape[0]
            mm[cursor:cursor + B] = hs[:, :T, :]
            cursor += B

            if (start // batch_size) % 20 == 0:
                done = cursor
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (N - done) / rate if rate > 0 else 0
                print(f"    Wiki: {done}/{N} ({done/N*100:.0f}%) "
                      f"{rate:.0f} seq/s ETA {eta:.0f}s", flush=True)

    mm.flush()
    model.layers = original_layers
    handle.remove()
    del model
    torch.cuda.empty_cache()

    print(f"  WikiText extraction done in {(time.time()-t0)/60:.1f}m", flush=True)


# ---------------------------------------------------------------------------
# Wiki normalization
# ---------------------------------------------------------------------------

def compute_wiki_norm_stats(wiki_dir: str, n_wiki_seqs: int, cfg: Config):
    """Compute mean/std of WikiText hidden states for proper normalization."""
    H = cfg.hidden_size
    wiki_shape = (n_wiki_seqs, WIKI_SEQ_LEN, H)
    wiki_mm = np.memmap(os.path.join(wiki_dir, "wiki_pt.bin"), dtype="float16",
                        mode="r", shape=wiki_shape)

    # Streaming mean/variance to avoid loading all into memory
    mean_acc = np.zeros(H, dtype=np.float64)
    sq_acc = np.zeros(H, dtype=np.float64)
    n = 0
    BS = 50
    for start in range(0, n_wiki_seqs, BS):
        end = min(start + BS, n_wiki_seqs)
        chunk = wiki_mm[start:end].astype(np.float32).reshape(-1, H)
        mean_acc += chunk.sum(axis=0)
        sq_acc += (chunk ** 2).sum(axis=0)
        n += chunk.shape[0]

    wiki_mean = torch.tensor(mean_acc / n, dtype=torch.float32)
    wiki_std = torch.tensor(np.sqrt(sq_acc / n - (mean_acc / n) ** 2), dtype=torch.float32).clamp(min=1e-6)
    return wiki_mean, wiki_std


def encode_wiki_with_norm(x: torch.Tensor, encoder, wiki_mean: torch.Tensor,
                          wiki_std: torch.Tensor) -> tuple:
    """Encode WikiText hidden states using wiki-specific normalization."""
    device = x.device
    n = (x - wiki_mean.to(device)) / (wiki_std.to(device) + 1e-8)
    z, pre = encoder(n)
    return z, pre, n


# ---------------------------------------------------------------------------
# Categorize features
# ---------------------------------------------------------------------------

def categorize_features(
    layer_idx: int,
    cfg: Config,
    precompute_dir: str,
    wiki_dir: str,
    n_ts_windows: int,
    n_wiki_seqs: int,
) -> dict:
    """
    Run crosscoder on precomputed activations, compute per-domain firing rates,
    categorize all features.

    Returns dict with:
      - categories: {category_name: [feature_ids]}
      - feature_stats: {feature_id: {domain: firing_rate, ...}}
      - wiki_stats: {feature_id: firing_rate}  (PT wiki only)
    """
    device = torch.device("cuda:0")
    T, H = cfg.context_length, cfg.hidden_size

    # Load crosscoder
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}", "crosscoder.pt")
    cc = Crosscoder(cfg)
    cc.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    cc = cc.to(device).eval()

    # Load TS memmaps
    ts_dir = os.path.join(precompute_dir, f"layer_{layer_idx}")
    N_ts = n_ts_windows
    ts_shape = (N_ts, T, H)
    pt_mm = np.memmap(os.path.join(ts_dir, "pt.bin"), dtype="float16", mode="r", shape=ts_shape)
    ft_mm = np.memmap(os.path.join(ts_dir, "ft.bin"), dtype="float16", mode="r", shape=ts_shape)
    ri_mm = np.memmap(os.path.join(ts_dir, "ri.bin"), dtype="float16", mode="r", shape=ts_shape)

    # Load Wiki memmap
    N_wiki = n_wiki_seqs
    wiki_shape = (N_wiki, WIKI_SEQ_LEN, H)
    wiki_mm = np.memmap(os.path.join(wiki_dir, "wiki_pt.bin"), dtype="float16",
                        mode="r", shape=wiki_shape)

    # Accumulate firing counts
    fire_pt = torch.zeros(cfg.latent_dim, device=device)
    fire_ft = torch.zeros(cfg.latent_dim, device=device)
    fire_ri = torch.zeros(cfg.latent_dim, device=device)
    fire_wiki = torch.zeros(cfg.latent_dim, device=device)

    # Also accumulate activation sums for mean activation
    act_sum_pt = torch.zeros(cfg.latent_dim, device=device)
    act_sum_ft = torch.zeros(cfg.latent_dim, device=device)
    act_sum_ri = torch.zeros(cfg.latent_dim, device=device)

    n_ts_samples = 0
    n_wiki_samples = 0

    BS = 20  # windows per batch

    print(f"  Running crosscoder on TS activations ({N_ts:,} windows)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N_ts, BS):
            end = min(start + BS, N_ts)
            x_pt = torch.from_numpy(pt_mm[start:end].copy()).float().reshape(-1, H).to(device)
            x_ft = torch.from_numpy(ft_mm[start:end].copy()).float().reshape(-1, H).to(device)
            x_ri = torch.from_numpy(ri_mm[start:end].copy()).float().reshape(-1, H).to(device)

            z_pt, _, _ = cc.encode_single(x_pt, "PT")
            z_ft, _, _ = cc.encode_single(x_ft, "FT")
            z_ri, _, _ = cc.encode_single(x_ri, "RI")

            fire_pt += (z_pt > 0).float().sum(dim=0)
            fire_ft += (z_ft > 0).float().sum(dim=0)
            fire_ri += (z_ri > 0).float().sum(dim=0)
            act_sum_pt += z_pt.sum(dim=0)
            act_sum_ft += z_ft.sum(dim=0)
            act_sum_ri += z_ri.sum(dim=0)
            n_ts_samples += x_pt.shape[0]

            if (start // BS) % 200 == 0 and start > 0:
                elapsed = time.time() - t0
                pct = start / N_ts * 100
                print(f"    TS: {start}/{N_ts} ({pct:.0f}%) "
                      f"t={elapsed/60:.1f}m", flush=True)

    # Compute wiki-specific normalization stats
    print(f"  Computing WikiText normalization stats...", flush=True)
    wiki_mean, wiki_std = compute_wiki_norm_stats(wiki_dir, N_wiki, cfg)

    print(f"  Running crosscoder on WikiText ({N_wiki:,} seqs, wiki_norm)...", flush=True)
    with torch.no_grad():
        for start in range(0, N_wiki, BS):
            end = min(start + BS, N_wiki)
            x_wiki = torch.from_numpy(wiki_mm[start:end].copy()).float().reshape(-1, H).to(device)
            z_wiki, _, _ = encode_wiki_with_norm(x_wiki, cc.encoder,
                                                  wiki_mean, wiki_std)
            fire_wiki += (z_wiki > 0).float().sum(dim=0)
            n_wiki_samples += x_wiki.shape[0]

    elapsed = time.time() - t0
    print(f"  Categorization done in {elapsed/60:.1f}m", flush=True)

    # Compute rates
    rate_pt = (fire_pt / n_ts_samples).cpu()
    rate_ft = (fire_ft / n_ts_samples).cpu()
    rate_ri = (fire_ri / n_ts_samples).cpu()
    rate_wiki = (fire_wiki / n_wiki_samples).cpu()
    mean_act_pt = (act_sum_pt / n_ts_samples).cpu()
    mean_act_ft = (act_sum_ft / n_ts_samples).cpu()
    mean_act_ri = (act_sum_ri / n_ts_samples).cpu()

    # Categorize
    active_pt = rate_pt > FIRING_THRESHOLD
    active_ft = rate_ft > FIRING_THRESHOLD
    active_ri = rate_ri > FIRING_THRESHOLD

    categories = {cat: [] for cat in ALL_CATEGORIES}
    feature_stats = {}

    for j in range(cfg.latent_dim):
        pt_on = active_pt[j].item()
        ft_on = active_ft[j].item()
        ri_on = active_ri[j].item()

        if pt_on and ft_on and ri_on:
            cat = "PT_FT_RI"
        elif pt_on and ft_on:
            cat = "PT_FT"
        elif ft_on and ri_on:
            cat = "FT_RI"
        elif pt_on and ri_on:
            cat = "PT_RI"
        elif pt_on:
            cat = "PT_only"
        elif ft_on:
            cat = "FT_only"
        elif ri_on:
            cat = "RI_only"
        else:
            cat = "None"

        categories[cat].append(j)
        feature_stats[j] = {
            "category": cat,
            "rate_pt": rate_pt[j].item(),
            "rate_ft": rate_ft[j].item(),
            "rate_ri": rate_ri[j].item(),
            "rate_wiki": rate_wiki[j].item(),
            "mean_act_pt": mean_act_pt[j].item(),
            "mean_act_ft": mean_act_ft[j].item(),
            "mean_act_ri": mean_act_ri[j].item(),
        }

    del cc
    torch.cuda.empty_cache()

    return {
        "categories": {k: v for k, v in categories.items()},
        "feature_stats": feature_stats,
        "n_ts_samples": n_ts_samples,
        "n_wiki_samples": n_wiki_samples,
        "wiki_mean": wiki_mean,
        "wiki_std": wiki_std,
    }


# ---------------------------------------------------------------------------
# Extract top features
# ---------------------------------------------------------------------------

def extract_top_features(
    layer_idx: int,
    cfg: Config,
    cat_result: dict,
    precompute_dir: str,
    wiki_dir: str,
    val_ds,
    wiki_sequences: list[dict],
    n_ts_windows: int,
    n_wiki_seqs: int,
    wiki_mean: torch.Tensor = None,
    wiki_std: torch.Tensor = None,
) -> dict:
    """
    For PT_FT_RI, PT_FT, FT_RI: get top 30 features, find top activating
    windows and WikiText spans.
    """
    device = torch.device("cuda:0")
    T, H = cfg.context_length, cfg.hidden_size

    # Load crosscoder
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}", "crosscoder.pt")
    cc = Crosscoder(cfg)
    cc.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    cc = cc.to(device).eval()

    # Load memmaps
    ts_dir = os.path.join(precompute_dir, f"layer_{layer_idx}")
    N_ts = n_ts_windows
    ts_shape = (N_ts, T, H)
    pt_mm = np.memmap(os.path.join(ts_dir, "pt.bin"), dtype="float16", mode="r", shape=ts_shape)
    ft_mm = np.memmap(os.path.join(ts_dir, "ft.bin"), dtype="float16", mode="r", shape=ts_shape)
    ri_mm = np.memmap(os.path.join(ts_dir, "ri.bin"), dtype="float16", mode="r", shape=ts_shape)

    N_wiki = n_wiki_seqs
    wiki_shape = (N_wiki, WIKI_SEQ_LEN, H)
    wiki_mm = np.memmap(os.path.join(wiki_dir, "wiki_pt.bin"), dtype="float16",
                        mode="r", shape=wiki_shape)

    feature_stats = cat_result["feature_stats"]
    results = {}

    for cat in FOCUS_CATEGORIES:
        feat_ids = cat_result["categories"].get(cat, [])
        if not feat_ids:
            results[cat] = {"top_features": []}
            continue

        # Rank by mean activation across active domains
        def sort_key(j):
            s = feature_stats[j]
            vals = []
            if "PT" in cat or cat == "PT_FT_RI":
                vals.append(s["mean_act_pt"])
            if "FT" in cat or cat == "PT_FT_RI":
                vals.append(s["mean_act_ft"])
            if "RI" in cat or cat == "PT_FT_RI":
                vals.append(s["mean_act_ri"])
            return sum(vals) / len(vals) if vals else 0

        ranked = sorted(feat_ids, key=sort_key, reverse=True)
        top_feats = ranked[:TOP_FEATURES_PER_CAT]

        print(f"  Extracting top {len(top_feats)} features for {cat}...", flush=True)

        # For each top feature, scan all windows to find top activating timesteps
        # We'll process in batches and maintain a top-K heap per feature

        # Initialize per-feature top-K trackers
        # Store (activation_value, window_idx, timestep_idx) tuples
        import heapq
        ts_topk = {j: [] for j in top_feats}  # min-heaps of size TOP_WINDOWS_PER_FEAT
        wiki_topk = {j: [] for j in top_feats}

        BS = 20
        print(f"    Scanning TS windows...", flush=True)
        t0 = time.time()
        with torch.no_grad():
            for start in range(0, N_ts, BS):
                end = min(start + BS, N_ts)
                B = end - start

                x_pt = torch.from_numpy(pt_mm[start:end].copy()).float().to(device)
                x_ft = torch.from_numpy(ft_mm[start:end].copy()).float().to(device)
                x_ri = torch.from_numpy(ri_mm[start:end].copy()).float().to(device)

                # Get per-domain activations (B, T, H) → (B*T, H) → encode → (B*T, latent)
                z_pt, _, _ = cc.encode_single(x_pt.reshape(-1, H), "PT")
                z_ft, _, _ = cc.encode_single(x_ft.reshape(-1, H), "FT")
                z_ri, _, _ = cc.encode_single(x_ri.reshape(-1, H), "RI")

                # Reshape back to (B, T, latent)
                z_pt = z_pt.reshape(B, T, -1)
                z_ft = z_ft.reshape(B, T, -1)
                z_ri = z_ri.reshape(B, T, -1)

                # Combined activation = sum across active domains
                for j in top_feats:
                    # Max across domains per timestep
                    z_combined = torch.zeros(B, T, device=device)
                    if "PT" in cat or cat == "PT_FT_RI":
                        z_combined += z_pt[:, :, j]
                    if "FT" in cat or cat == "PT_FT_RI":
                        z_combined += z_ft[:, :, j]
                    if "RI" in cat or cat == "PT_FT_RI":
                        z_combined += z_ri[:, :, j]

                    # Find max activation per window
                    max_vals, max_ts = z_combined.max(dim=1)  # (B,)
                    for b_idx in range(B):
                        val = max_vals[b_idx].item()
                        if val <= 0:
                            continue
                        win_idx = start + b_idx
                        ts_idx = max_ts[b_idx].item()

                        # Also get per-domain activation arrays for this window
                        entry = (val, win_idx, ts_idx)
                        if len(ts_topk[j]) < TOP_WINDOWS_PER_FEAT:
                            heapq.heappush(ts_topk[j], entry)
                        elif val > ts_topk[j][0][0]:
                            heapq.heapreplace(ts_topk[j], entry)

                if (start // BS) % 500 == 0 and start > 0:
                    elapsed = time.time() - t0
                    print(f"      TS: {start}/{N_ts} ({start/N_ts*100:.0f}%) "
                          f"t={elapsed/60:.1f}m", flush=True)

        # Scan WikiText for PT-involving categories
        pt_in_cat = "PT" in cat or cat == "PT_FT_RI"
        if pt_in_cat and wiki_mean is not None:
            print(f"    Scanning WikiText (wiki_norm)...", flush=True)
            with torch.no_grad():
                for start in range(0, N_wiki, BS):
                    end = min(start + BS, N_wiki)
                    B = end - start
                    x_wiki = torch.from_numpy(wiki_mm[start:end].copy()).float().to(device)
                    z_wiki, _, _ = encode_wiki_with_norm(
                        x_wiki.reshape(-1, H), cc.encoder, wiki_mean, wiki_std)
                    z_wiki = z_wiki.reshape(B, WIKI_SEQ_LEN, -1)

                    for j in top_feats:
                        z_feat = z_wiki[:, :, j]  # (B, T)
                        max_vals, max_ts = z_feat.max(dim=1)
                        for b_idx in range(B):
                            val = max_vals[b_idx].item()
                            if val <= 0:
                                continue
                            seq_idx = start + b_idx
                            tok_idx = max_ts[b_idx].item()
                            entry = (val, seq_idx, tok_idx)
                            if len(wiki_topk[j]) < TOP_WIKI_PER_FEAT:
                                heapq.heappush(wiki_topk[j], entry)
                            elif val > wiki_topk[j][0][0]:
                                heapq.heapreplace(wiki_topk[j], entry)

        # Now collect detailed data for each top feature
        cat_results = []
        for j in top_feats:
            feat_data = {
                "feature_id": j,
                "stats": feature_stats[j],
                "top_windows": [],
                "top_wiki_spans": [],
            }

            # Get top windows sorted descending
            top_wins = sorted(ts_topk[j], key=lambda x: -x[0])
            for val, win_idx, ts_idx in top_wins:
                # Get the raw time series values for this window
                window = val_ds[win_idx]
                raw_values = window["values"].tolist()

                # Get per-domain activations for this window
                x_pt_w = torch.from_numpy(pt_mm[win_idx].copy()).float().to(device)
                x_ft_w = torch.from_numpy(ft_mm[win_idx].copy()).float().to(device)
                x_ri_w = torch.from_numpy(ri_mm[win_idx].copy()).float().to(device)

                with torch.no_grad():
                    z_pt_w, _, _ = cc.encode_single(x_pt_w, "PT")
                    z_ft_w, _, _ = cc.encode_single(x_ft_w, "FT")
                    z_ri_w, _, _ = cc.encode_single(x_ri_w, "RI")

                feat_data["top_windows"].append({
                    "activation_value": val,
                    "window_idx": win_idx,
                    "peak_timestep": ts_idx,
                    "series_idx": window["series_idx"],
                    "offset": window["offset"],
                    "raw_values": raw_values,
                    "activations_pt": z_pt_w[:, j].cpu().tolist(),
                    "activations_ft": z_ft_w[:, j].cpu().tolist(),
                    "activations_ri": z_ri_w[:, j].cpu().tolist(),
                })

            # Get top WikiText spans
            if pt_in_cat:
                top_wiki = sorted(wiki_topk[j], key=lambda x: -x[0])
                for val, seq_idx, tok_idx in top_wiki:
                    seq = wiki_sequences[seq_idx]
                    text = seq["text"]
                    input_ids = seq["input_ids"]

                    # Get activation for full sequence (wiki normalization)
                    x_w = torch.from_numpy(wiki_mm[seq_idx].copy()).float().to(device)
                    with torch.no_grad():
                        z_w, _, _ = encode_wiki_with_norm(
                            x_w, cc.encoder, wiki_mean, wiki_std)
                    activations = z_w[:, j].cpu().tolist()

                    # Context window around peak
                    ctx_start = max(0, tok_idx - WIKI_CONTEXT_TOKENS)
                    ctx_end = min(WIKI_SEQ_LEN, tok_idx + WIKI_CONTEXT_TOKENS + 1)

                    feat_data["top_wiki_spans"].append({
                        "activation_value": val,
                        "seq_idx": seq_idx,
                        "peak_token_idx": tok_idx,
                        "full_text": text,
                        "context_start": ctx_start,
                        "context_end": ctx_end,
                        "activations": activations[ctx_start:ctx_end],
                    })

            cat_results.append(feat_data)

        results[cat] = {"top_features": cat_results}

    del cc
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_top_windows(feat_data: dict, output_path: str):
    """Plot top 10 activating windows with color gradient showing activation."""
    windows = feat_data["top_windows"][:TOP_WINDOWS_PER_FEAT]
    if not windows:
        return

    n_plots = len(windows)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), squeeze=False)

    for i, win in enumerate(windows):
        ax = axes[i, 0]
        values = np.array(win["raw_values"], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        # Sum activations across domains for color
        acts_pt = np.array(win["activations_pt"])
        acts_ft = np.array(win["activations_ft"])
        acts_ri = np.array(win["activations_ri"])
        acts = acts_pt + acts_ft + acts_ri

        # Create colored line segments
        x = np.arange(len(values))
        points = np.array([x, values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize activations for color mapping
        act_colors = (acts[:-1] + acts[1:]) / 2  # midpoint activation per segment
        act_max = np.nanmax(act_colors) if len(act_colors) > 0 else 0
        if act_max > 0:
            norm = mcolors.Normalize(vmin=0, vmax=act_max)
        else:
            norm = mcolors.Normalize(vmin=0, vmax=1)

        lc = LineCollection(segments, cmap="YlOrRd", norm=norm)
        lc.set_array(act_colors)
        lc.set_linewidth(1.5)
        ax.add_collection(lc)

        v_min, v_max = float(values.min()), float(values.max())
        v_range = max(abs(v_max - v_min), 1e-6)
        ax.set_xlim(0, len(values))
        ax.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)

        # Mark peak timestep
        peak = win["peak_timestep"]
        ax.axvline(x=peak, color="red", alpha=0.5, linestyle="--", linewidth=0.8)

        ax.set_ylabel(f"#{i+1}\nact={win['activation_value']:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(f"Feature {feat_data['feature_id']} — "
                        f"Top {n_plots} Activating Windows", fontsize=10)
        if i < n_plots - 1:
            ax.set_xticklabels([])

    axes[-1, 0].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_layer_analysis(
    layer_idx: int,
    cat_result: dict,
    top_result: dict,
    val_ds,
    wiki_sequences: list[dict],
):
    """Save categorization, rankings, plots, and data files."""
    layer_dir = os.path.join(ANALYSIS_DIR, f"layer_{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)

    # Save full categorization
    cat_summary = {
        "layer": layer_idx,
        "n_ts_samples": cat_result["n_ts_samples"],
        "n_wiki_samples": cat_result["n_wiki_samples"],
        "category_counts": {k: len(v) for k, v in cat_result["categories"].items()},
        "feature_stats": {str(k): v for k, v in cat_result["feature_stats"].items()},
    }
    with open(os.path.join(layer_dir, "categorization.json"), "w") as f:
        json.dump(cat_summary, f, indent=2)

    # Save per-category results
    for cat in FOCUS_CATEGORIES:
        cat_dir = os.path.join(layer_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)

        cat_data = top_result.get(cat, {})
        top_features = cat_data.get("top_features", [])

        # Save ranking
        ranking = [
            {
                "rank": i + 1,
                "feature_id": f["feature_id"],
                "category": f["stats"]["category"],
                "rate_pt": f["stats"]["rate_pt"],
                "rate_ft": f["stats"]["rate_ft"],
                "rate_ri": f["stats"]["rate_ri"],
                "mean_act_pt": f["stats"]["mean_act_pt"],
                "mean_act_ft": f["stats"]["mean_act_ft"],
                "mean_act_ri": f["stats"]["mean_act_ri"],
            }
            for i, f in enumerate(top_features)
        ]
        with open(os.path.join(cat_dir, "ranking.json"), "w") as f:
            json.dump(ranking, f, indent=2)

        # Save per-feature data and plots
        for feat_data in top_features:
            fid = feat_data["feature_id"]
            feat_dir = os.path.join(cat_dir, f"feature_{fid}")
            os.makedirs(feat_dir, exist_ok=True)

            # Save window data
            with open(os.path.join(feat_dir, "windows.json"), "w") as f:
                json.dump(feat_data["top_windows"], f)

            # Save wiki spans
            if feat_data["top_wiki_spans"]:
                with open(os.path.join(feat_dir, "wiki_spans.json"), "w") as f:
                    json.dump(feat_data["top_wiki_spans"], f)

            # Save info (stats, placeholder for score)
            info = {
                "feature_id": fid,
                "layer": layer_idx,
                "category": feat_data["stats"]["category"],
                "stats": feat_data["stats"],
                "qualitative_score": None,
                "interpretation": None,
            }
            with open(os.path.join(feat_dir, "info.json"), "w") as f:
                json.dump(info, f, indent=2)

            # Plot
            plot_path = os.path.join(feat_dir, "plot.png")
            plot_top_windows(feat_data, plot_path)

    print(f"  Analysis saved to {layer_dir}/", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_single_layer(
    layer_idx: int,
    cfg: Config,
    val_ds,
    wiki_sequences: list[dict],
    hf_token: str | None,
):
    """Full analysis pipeline for one layer."""
    print(f"\n{'='*60}")
    print(f"ANALYZING LAYER {layer_idx}")
    print(f"{'='*60}")
    t_start = time.time()

    precompute_dir = cfg.precompute_dir
    wiki_dir = os.path.join(precompute_dir, f"layer_{layer_idx}")

    # Step A: Precompute TS activations
    print(f"  Step A: Precomputing TS activations ({N_TS_WINDOWS:,} windows)...")
    n_use = min(N_TS_WINDOWS, len(val_ds))
    n_use = (n_use // cfg.num_gpus) * cfg.num_gpus
    cfg.n_precompute_windows = n_use
    if not _layer_acts_exist(layer_idx, cfg):
        from src.crosscoder.precompute import precompute_layer
        precompute_layer(layer_idx, cfg, val_ds, hf_token)
    else:
        print(f"    Already on disk.", flush=True)

    # Step B: Precompute WikiText activations
    print(f"  Step B: Precomputing WikiText activations ({len(wiki_sequences):,} seqs)...")
    precompute_wiki_activations(layer_idx, wiki_sequences, cfg, wiki_dir, hf_token)

    # Step C: Categorize
    print(f"  Step C: Categorizing features...")
    cat_result = categorize_features(
        layer_idx, cfg, precompute_dir, wiki_dir, n_use, len(wiki_sequences)
    )
    for cat_name in ALL_CATEGORIES:
        n = len(cat_result["categories"].get(cat_name, []))
        print(f"    {cat_name}: {n}", flush=True)

    # Step D: Extract top features
    print(f"  Step D: Extracting top features...")
    top_result = extract_top_features(
        layer_idx, cfg, cat_result, precompute_dir, wiki_dir,
        val_ds, wiki_sequences, n_use, len(wiki_sequences),
        wiki_mean=cat_result["wiki_mean"],
        wiki_std=cat_result["wiki_std"],
    )

    # Step E: Plot & save
    print(f"  Step E: Saving analysis & plots...")
    save_layer_analysis(layer_idx, cat_result, top_result, val_ds, wiki_sequences)

    # Step G: Cleanup
    print(f"  Step G: Cleaning up raw activations...")
    layer_act_dir = os.path.join(precompute_dir, f"layer_{layer_idx}")
    if os.path.isdir(layer_act_dir):
        shutil.rmtree(layer_act_dir)
        print(f"    Deleted {layer_act_dir}", flush=True)

    elapsed = time.time() - t_start
    print(f"  Layer {layer_idx} analysis complete in {elapsed/60:.1f}m", flush=True)

    return cat_result


def _layer_acts_exist(layer_idx, cfg):
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")
    N, T, H = cfg.n_precompute_windows, cfg.context_length, cfg.hidden_size
    expected = N * T * H * 2
    return all(
        os.path.exists(os.path.join(layer_dir, f"{d}.bin"))
        and os.path.getsize(os.path.join(layer_dir, f"{d}.bin")) == expected
        for d in ("pt", "ft", "ri")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(cfg.precompute_dir, exist_ok=True)

    # Load val dataset (non-overlapping windows)
    print("Loading GiftEval validation set (non-overlapping)...")
    _, val_ds, _ = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=hf_token
    )
    # Rebuild with non-overlapping stride for cleaner analysis
    from src.data.dataset import WindowDataset, load_gifteval_series, temporal_split
    series_list = load_gifteval_series(hf_token)
    val_splits = []
    for s in series_list:
        _, va, _ = temporal_split(s, cfg.train_frac, cfg.val_frac)
        if len(va) >= cfg.context_length:
            val_splits.append(va)
    val_ds = WindowDataset(val_splits, cfg.context_length, stride=cfg.context_length)
    print(f"  Val windows (non-overlapping): {len(val_ds):,}")

    # Load WikiText
    print(f"Loading WikiText ({N_WIKI_SEQS:,} sequences)...")
    wiki_sequences = load_wikitext_sequences(
        max_sequences=N_WIKI_SEQS, seq_len=WIKI_SEQ_LEN, hf_token=hf_token
    )
    print(f"  WikiText sequences: {len(wiki_sequences):,}")

    if args.layer is not None:
        analyze_single_layer(args.layer, cfg, val_ds, wiki_sequences, hf_token)
    elif args.all:
        for layer_idx in LAYER_ORDER:
            if layer_idx in SKIP_LAYERS:
                print(f"\nSkipping layer {layer_idx} (broken)")
                continue
            # Check if already analyzed
            analysis_path = os.path.join(ANALYSIS_DIR, f"layer_{layer_idx}",
                                          "categorization.json")
            if os.path.exists(analysis_path):
                print(f"\nLayer {layer_idx}: already analyzed, skipping.")
                continue
            analyze_single_layer(layer_idx, cfg, val_ds, wiki_sequences, hf_token)
    else:
        parser.error("Specify --layer N or --all")


if __name__ == "__main__":
    main()
