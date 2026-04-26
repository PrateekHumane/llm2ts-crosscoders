"""
Geometry comparison v2: same TS input through PT, FT, and RI.
Also text through all 3 models (6 conditions total).

Fixes from critic reviews:
  - PCA cap raised to 500 (was 100, saturated for text@PT)
  - Tortuosity uses np.nanmean, filters inf
  - Spectral energy uses np.nanmean, handles zero-sum PSDs
  - Subspace alignment computed over ALL sequences (not just 1 representative)
  - PH adapts PCA dims to condition's effective rank
  - Random baseline for principal angles
  - Error bars (std) on all metrics
  - Added text@FT and text@RI conditions

Usage:
    /usr/bin/python3 scripts/geometry_three_models_v2.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, time, glob
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ripser import ripser
from sklearn.decomposition import PCA

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.wikitext import load_wikitext_sequences

T = 512
N_LAYERS = 28
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/geometry_v2"
N_SEQS = 20
N_PH_POINTS = 300
SEED = 42

# 6 conditions: TS and text through all 3 models
CONDITIONS = ["ts@PT", "ts@FT", "ts@RI", "text@PT", "text@FT", "text@RI"]

COLORS = {
    "ts@PT": "#2196F3", "ts@FT": "#4CAF50", "ts@RI": "#E91E63",
    "text@PT": "#FF9800", "text@FT": "#8BC34A", "text@RI": "#9C27B0",
}
LABELS = {
    "ts@PT": "TS→PT", "ts@FT": "TS→FT", "ts@RI": "TS→RI",
    "text@PT": "Text→PT", "text@FT": "Text→FT", "text@RI": "Text→RI",
}
MODEL_PATHS = {"PT": "Qwen/Qwen3-0.6B", "FT": "models/ft", "RI": "models/ri"}


def load_ts_windows(hf_token, n_windows=20):
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    arrow_files = sorted(glob.glob(os.path.join(path, "**/*.arrow"), recursive=True))
    windows = []; seen = set()
    for f in arrow_files:
        domain = os.path.relpath(f, path).split("/")[0]
        if domain in seen: continue
        with open(f, "rb") as fp:
            tbl = ipc.open_stream(fp).read_all()
        for t in tbl['target'].to_pylist():
            arr = np.array(t, dtype=np.float32)
            if len(arr) >= T:
                start = int(len(arr) * 0.7)
                if start + T > len(arr): start = len(arr) - T
                w = arr[start:start+T]
                mu, sigma = w.mean(), w.std()
                if sigma < 1e-6: continue
                w = (w - mu) / sigma
                bins = ((np.clip(w, -5, 5) + 5) / 10 * 512).astype(np.int64).clip(0, 511)
                windows.append({"input_ids": bins, "values": w})
                seen.add(domain); break
        if len(windows) >= n_windows: break
    return windows


def extract_per_layer(model, token_ids, device):
    captured = {}; handles = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook(m, i, o): captured[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=ids, use_cache=False)
    # Skip position 0 (has extreme norm outlier in pretrained models)
    layers = [captured[i].squeeze(0)[1:, :].float().cpu() for i in range(N_LAYERS)]
    for h in handles: h.remove()
    captured.clear()
    return layers


# ═══ Analysis functions (fixed) ═══

def compute_pca_stats(H):
    H_np = H.numpy()
    H_c = H_np - H_np.mean(axis=0)
    # FIX: cap at 500, not 100
    n_comp = min(500, H_np.shape[0] - 1, H_np.shape[1])
    pca = PCA(n_components=n_comp, svd_solver='full').fit(H_c)
    evals = pca.explained_variance_
    total_var = H_c.var(axis=0, ddof=1).sum()
    if total_var < 1e-15:
        return {"eff_rank": 1.0, "pr": 1.0, "pcs_90": 1, "top1_frac": 1.0,
                "top5_frac": 1.0, "eigenvalues": []}
    p = evals / total_var; p = p[p > 1e-15]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    pr = float(evals.sum()**2 / (evals**2).sum()) if (evals**2).sum() > 0 else 1.0
    cum = np.cumsum(pca.explained_variance_ratio_)
    pcs_90 = int(np.searchsorted(cum, 0.90) + 1)
    return {
        "eff_rank": eff_rank, "pr": pr, "pcs_90": pcs_90,
        "top1_frac": float(evals[0] / total_var),
        "top5_frac": float(evals[:5].sum() / total_var),
        "eigenvalues": evals[:20].tolist(),
    }


def compute_smoothness(H):
    diffs = H[1:] - H[:-1]
    step_sizes = diffs.norm(dim=1).numpy()
    d_norm = F.normalize(diffs, dim=1)
    cos_angles = (d_norm[:-1] * d_norm[1:]).sum(dim=1).clamp(-1, 1).numpy()
    angles = np.arccos(cos_angles) * 180 / np.pi
    path_length = float(step_sizes.sum())
    e2e = float((H[-1] - H[0]).norm())
    # FIX: use nan instead of inf for degenerate cases
    tort = path_length / e2e if e2e > 1e-6 else np.nan
    return {
        "speed_mean": float(step_sizes.mean()),
        "speed_std": float(step_sizes.std()),
        "angle_mean": float(angles.mean()),
        "path_length": path_length,
        "tortuosity": tort,
    }


def compute_spectrum(H):
    H_np = H.numpy()
    H_c = H_np - H_np.mean(axis=0)
    fft = np.fft.rfft(H_c, axis=0)
    psd = np.abs(fft)**2
    mean_psd = psd.mean(axis=1)
    total = mean_psd.sum()
    # FIX: handle zero-sum PSD
    if total < 1e-15:
        return {"low_energy": np.nan, "mid_energy": np.nan, "high_energy": np.nan}
    mean_psd_norm = mean_psd / total
    F_len = len(mean_psd)
    low_end = int(F_len * 0.10); mid_end = int(F_len * 0.50)
    return {
        "low_energy": float(mean_psd_norm[:low_end].sum()),
        "mid_energy": float(mean_psd_norm[low_end:mid_end].sum()),
        "high_energy": float(mean_psd_norm[mid_end:].sum()),
    }


def compute_ph(H, n_points=N_PH_POINTS):
    T_len = H.shape[0]
    H_np = H.numpy() if T_len <= n_points else H[np.linspace(0, T_len-1, n_points).astype(int)].numpy()
    H_np = H_np - H_np.mean(axis=0)
    # FIX: adapt PCA dims to data — use enough to capture 95% variance
    n_max = min(50, H_np.shape[0]-1, H_np.shape[1])
    pca = PCA(n_components=n_max).fit(H_np)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_pca = min(int(np.searchsorted(cum, 0.95) + 1), n_max)
    n_pca = max(n_pca, 3)  # minimum 3 dims for meaningful topology
    H_proj = pca.transform(H_np)[:, :n_pca]
    std = H_proj.std()
    if std > 1e-8: H_proj = H_proj / std
    try:
        result = ripser(H_proj, maxdim=2, thresh=2.0)
        betti = []
        for dim in range(3):
            if dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                p = dgm[:, 1] - dgm[:, 0]
                betti.append(int((p[np.isfinite(p)] > 0.1).sum()))
            else: betti.append(0)
        return {"betti": betti, "ph_dims": n_pca}
    except:
        return {"betti": [0, 0, 0], "ph_dims": n_pca}


def compute_principal_angles(H1, H2, k=10):
    """Principal angles between top-k subspaces of two trajectories."""
    H1_c = H1.numpy() - H1.numpy().mean(axis=0)
    H2_c = H2.numpy() - H2.numpy().mean(axis=0)
    k1 = min(k, H1_c.shape[0]-1, H1_c.shape[1])
    k2 = min(k, H2_c.shape[0]-1, H2_c.shape[1])
    k_use = min(k1, k2)
    U1 = PCA(n_components=k_use).fit(H1_c).components_.T
    U2 = PCA(n_components=k_use).fit(H2_c).components_.T
    _, s, _ = np.linalg.svd(U1.T @ U2)
    angles = np.arccos(np.clip(s, -1, 1)) * 180 / np.pi
    return {"mean_angle": float(angles.mean()), "min_angle": float(angles.min()),
            "angles": angles.tolist()}


def compute_random_baseline_angles(d=1024, k=10, n_trials=500):
    """Null distribution: principal angles between random k-dim subspaces in R^d."""
    angles = []
    for _ in range(n_trials):
        U1 = np.linalg.qr(np.random.randn(d, k))[0]
        U2 = np.linalg.qr(np.random.randn(d, k))[0]
        _, s, _ = np.linalg.svd(U1.T @ U2)
        a = np.arccos(np.clip(s, -1, 1)) * 180 / np.pi
        angles.append(a.mean())
    return float(np.mean(angles)), float(np.std(angles))


# ═══ Aggregation helpers ═══

def safe_nanmean(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) > 0 else np.nan

def safe_nanstd(vals):
    arr = np.array(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr)) if len(arr) > 1 else 0.0


# ═══ Main ═══

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # Load data
    print("Loading data...", flush=True)
    ts_windows = load_ts_windows(hf_token, N_SEQS)
    wiki_seqs = load_wikitext_sequences(max_sequences=N_SEQS, seq_len=T, hf_token=hf_token)
    print(f"  TS: {len(ts_windows)}, Wiki: {len(wiki_seqs)}")

    # Compute random baseline for principal angles
    print("Computing random baseline for principal angles...", flush=True)
    random_angle_mean, random_angle_std = compute_random_baseline_angles()
    print(f"  Random baseline: {random_angle_mean:.1f}° ± {random_angle_std:.1f}°")

    all_results = {}
    # FIX: store ALL sequence trajectories for alignment (not just 1 representative)
    all_trajectories = {}  # condition -> {layer: list of (T, D) tensors}

    for model_key, model_path in MODEL_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {model_key} ({model_path})")
        print(f"{'='*60}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16).model.to(DEVICE).eval()

        # All 3 models process both TS and text
        conditions = [(f"ts@{model_key}", ts_windows), (f"text@{model_key}", wiki_seqs)]

        for cond_name, sequences in conditions:
            print(f"\n  Condition: {cond_name}")
            layer_stats = {li: {"pca": [], "smooth": [], "spec": [], "ph": []}
                           for li in range(N_LAYERS)}
            layer_trajectories = {li: [] for li in range(N_LAYERS)}

            n_use = min(N_SEQS, len(sequences))
            for si, seq in enumerate(sequences[:n_use]):
                if si % 5 == 0:
                    print(f"    Seq {si+1}/{n_use}...", flush=True)
                layers = extract_per_layer(model, seq["input_ids"], DEVICE)
                for li in range(N_LAYERS):
                    H = layers[li]
                    layer_stats[li]["pca"].append(compute_pca_stats(H))
                    layer_stats[li]["smooth"].append(compute_smoothness(H))
                    layer_stats[li]["spec"].append(compute_spectrum(H))
                    layer_stats[li]["ph"].append(compute_ph(H))
                    layer_trajectories[li].append(H)
                del layers; gc.collect()

            # Aggregate with nanmean
            agg = {}
            for li in range(N_LAYERS):
                ls = layer_stats[li]
                agg[li] = {
                    "eff_rank": safe_nanmean([r["eff_rank"] for r in ls["pca"]]),
                    "eff_rank_std": safe_nanstd([r["eff_rank"] for r in ls["pca"]]),
                    "top1_frac": safe_nanmean([r["top1_frac"] for r in ls["pca"]]),
                    "pcs_90": safe_nanmean([r["pcs_90"] for r in ls["pca"]]),
                    "speed": safe_nanmean([r["speed_mean"] for r in ls["smooth"]]),
                    "speed_std": safe_nanstd([r["speed_mean"] for r in ls["smooth"]]),
                    "angle": safe_nanmean([r["angle_mean"] for r in ls["smooth"]]),
                    "tortuosity": safe_nanmean([r["tortuosity"] for r in ls["smooth"]]),
                    "low_energy": safe_nanmean([r["low_energy"] for r in ls["spec"]]),
                    "mid_energy": safe_nanmean([r["mid_energy"] for r in ls["spec"]]),
                    "high_energy": safe_nanmean([r["high_energy"] for r in ls["spec"]]),
                    "betti_1": safe_nanmean([r["betti"][1] for r in ls["ph"]]),
                }
            all_results[cond_name] = agg
            all_trajectories[cond_name] = layer_trajectories

        del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ Cross-model alignment (over ALL sequence pairs) ═══
    print(f"\n{'='*60}")
    print("INTER-MODEL SUBSPACE ALIGNMENT (averaged over all sequence pairs)")
    print(f"{'='*60}")

    pairs = [
        ("ts@PT", "ts@FT"), ("ts@PT", "ts@RI"), ("ts@FT", "ts@RI"),
        ("text@PT", "text@FT"), ("text@PT", "text@RI"),
        ("text@PT", "ts@FT"), ("text@PT", "ts@RI"),
        ("ts@PT", "text@PT"),  # same model, different modality
    ]

    cross = {}
    for c1, c2 in pairs:
        key = f"{c1}_vs_{c2}"
        cross[key] = {}
        for li in range(N_LAYERS):
            # FIX: compute principal angles for ALL pairs and average
            trajs1 = all_trajectories[c1][li]
            trajs2 = all_trajectories[c2][li]
            n_pairs = min(len(trajs1), len(trajs2))
            pair_angles = []
            for pi in range(n_pairs):
                pa = compute_principal_angles(trajs1[pi], trajs2[pi])
                pair_angles.append(pa["mean_angle"])
            cross[key][li] = {
                "mean_angle": float(np.mean(pair_angles)),
                "std_angle": float(np.std(pair_angles)),
            }
        print(f"  {key}: L8={cross[key][8]['mean_angle']:.1f}°±{cross[key][8]['std_angle']:.1f}°")

    # Save
    save_results = {}
    for cond in CONDITIONS:
        if cond in all_results:
            save_results[cond] = {str(li): v for li, v in all_results[cond].items()}
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    with open(f"{OUT_DIR}/cross_modality.json", "w") as f:
        json.dump({k: {str(li): v for li, v in d.items()} for k, d in cross.items()}, f, indent=2)
    with open(f"{OUT_DIR}/random_baseline.json", "w") as f:
        json.dump({"mean": random_angle_mean, "std": random_angle_std}, f)

    # ═══ Plots ═══
    print("\nGenerating plots...", flush=True)
    lx = list(range(N_LAYERS))
    active_conds = [c for c in CONDITIONS if c in all_results]

    # 1. Effective rank with error bands
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for cond in active_conds:
        vals = [all_results[cond][li]["eff_rank"] for li in lx]
        stds = [all_results[cond][li]["eff_rank_std"] for li in lx]
        axes[0].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=3, label=LABELS[cond])
        axes[0].fill_between(lx, np.array(vals)-np.array(stds), np.array(vals)+np.array(stds),
                             color=COLORS[cond], alpha=0.1)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Effective Rank")
    axes[0].set_title("Effective Rank (mean ± std)"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    for cond in active_conds:
        vals = [all_results[cond][li]["pcs_90"] for li in lx]
        axes[1].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=3, label=LABELS[cond])
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("PCs for 90%")
    axes[1].set_title("Dimensionality (PCs for 90%)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/dimensionality.png", dpi=150, bbox_inches="tight"); plt.close()

    # 2. Spectral
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Trajectory Spectral Energy", fontsize=13, fontweight='bold')
    for bi, (bname, bkey) in enumerate([("Low (0-10%)", "low_energy"),
                                        ("Mid (10-50%)", "mid_energy"),
                                        ("High (50-100%)", "high_energy")]):
        for cond in active_conds:
            vals = [all_results[cond][li][bkey] for li in lx]
            if not all(np.isnan(v) for v in vals):
                axes[bi].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=3, label=LABELS[cond])
        axes[bi].set_xlabel("Layer"); axes[bi].set_ylabel("Energy fraction")
        axes[bi].set_title(bname); axes[bi].legend(fontsize=6); axes[bi].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/spectral.png", dpi=150, bbox_inches="tight"); plt.close()

    # 3. Dynamics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Trajectory Dynamics", fontsize=13, fontweight='bold')
    for cond in active_conds:
        vals = [all_results[cond][li]["speed"] for li in lx]
        axes[0].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=3, label=LABELS[cond])
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean step size"); axes[0].set_title("Speed")
    axes[0].legend(fontsize=6); axes[0].grid(alpha=0.3)

    for cond in active_conds:
        vals = [all_results[cond][li]["tortuosity"] for li in lx]
        if not all(np.isnan(v) for v in vals):
            axes[1].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=3, label=LABELS[cond])
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Tortuosity"); axes[1].set_title("Tortuosity")
    axes[1].legend(fontsize=6); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/dynamics.png", dpi=150, bbox_inches="tight"); plt.close()

    # 4. Betti β₁
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in active_conds:
        vals = [all_results[cond][li]["betti_1"] for li in lx]
        ax.plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=3, label=LABELS[cond])
    ax.set_xlabel("Layer"); ax.set_ylabel("β₁"); ax.set_title("Persistent Homology: β₁ (loops)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/betti1.png", dpi=150, bbox_inches="tight"); plt.close()

    # 5. Subspace alignment WITH random baseline
    fig, ax = plt.subplots(figsize=(14, 6))
    pair_colors_list = ['#9C27B0', '#795548', '#009688', '#FF5722', '#607D8B', '#3F51B5', '#CDDC39', '#FF6F00']
    for i, key in enumerate(cross):
        vals = [cross[key][li]["mean_angle"] for li in lx]
        stds = [cross[key][li]["std_angle"] for li in lx]
        c = pair_colors_list[i % len(pair_colors_list)]
        label = key.replace("_vs_", " vs ")
        ax.plot(lx, vals, 'o-', color=c, linewidth=1.5, markersize=3, label=label)
        ax.fill_between(lx, np.array(vals)-np.array(stds), np.array(vals)+np.array(stds),
                        color=c, alpha=0.08)
    # Random baseline
    ax.axhline(random_angle_mean, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Random baseline ({random_angle_mean:.1f}°)')
    ax.axhspan(random_angle_mean - random_angle_std, random_angle_mean + random_angle_std,
               color='gray', alpha=0.1)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Principal Angle (°)")
    ax.set_title("Subspace Alignment (lower = more similar)\nwith random baseline (gray)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/alignment.png", dpi=150, bbox_inches="tight"); plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (Layer 8)")
    print(f"{'='*80}")
    print(f"Random baseline: {random_angle_mean:.1f}° ± {random_angle_std:.1f}°\n")
    print(f"{'Condition':<15} {'EffRank':>10} {'PCs90':>6} {'Speed':>8} {'LowE%':>7} {'β₁':>5}")
    print("-" * 55)
    for cond in active_conds:
        r = all_results[cond][8]
        low_e = f"{r['low_energy']*100:.1f}%" if not np.isnan(r['low_energy']) else "NaN"
        print(f"{LABELS[cond]:<15} {r['eff_rank']:>7.1f}±{r['eff_rank_std']:.1f} {r['pcs_90']:>6.0f} "
              f"{r['speed']:>8.1f} {low_e:>7} {r['betti_1']:>5.1f}")

    print(f"\nSubspace alignment (L8):")
    for key in cross:
        r = cross[key][8]
        sig = "***" if r['mean_angle'] < random_angle_mean - 2*random_angle_std else \
              "**" if r['mean_angle'] < random_angle_mean - random_angle_std else \
              "*" if r['mean_angle'] < random_angle_mean else ""
        print(f"  {key.replace('_vs_', ' vs ')}: {r['mean_angle']:.1f}°±{r['std_angle']:.1f}° {sig}")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
