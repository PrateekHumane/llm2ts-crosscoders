"""
Geometry comparison: same TS input through PT, FT, and RI.

All three models share the Qwen3-0.6B architecture (28 layers, d=1024).
- PT: pretrained on text only
- FT: PT finetuned on time series
- RI: random init trained on time series

We pass the same 20 binned TS windows through all 3 models and compare:
- Effective rank / PCA dimensionality
- Spectral profile (low/mid/high frequency energy)
- Trajectory smoothness (speed, curvature, tortuosity)
- Persistent homology (Betti numbers)
- Inter-model subspace alignment (principal angles)

Also: PT processing TEXT (its native modality) for comparison.

Usage:
    /usr/bin/python3 scripts/geometry_three_models.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, time, glob
from collections import defaultdict

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
OUT_DIR = "mapping_results/geometry_3models"
N_SEQS = 20
N_PH_POINTS = 300
SEED = 42

# 4 conditions: same TS through 3 models + text through PT
CONDITIONS = ["ts@PT", "ts@FT", "ts@RI", "text@PT"]

COLORS = {
    "ts@PT": "#2196F3",
    "ts@FT": "#4CAF50",
    "ts@RI": "#E91E63",
    "text@PT": "#FF9800",
}

LABELS = {
    "ts@PT": "TS → PT (text-only model)",
    "ts@FT": "TS → FT (PT finetuned on TS)",
    "ts@RI": "TS → RI (random init, trained on TS)",
    "text@PT": "Text → PT (native modality)",
}

MODEL_PATHS = {
    "PT": "Qwen/Qwen3-0.6B",
    "FT": "models/ft",
    "RI": "models/ri",
}


def load_ts_windows(hf_token, n_windows=20):
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    arrow_files = sorted(glob.glob(os.path.join(path, "**/*.arrow"), recursive=True))
    windows = []
    seen_domains = set()
    for f in arrow_files:
        domain = os.path.relpath(f, path).split("/")[0]
        if domain in seen_domains:
            continue
        with open(f, "rb") as fp:
            tbl = ipc.open_stream(fp).read_all()
        for t in tbl['target'].to_pylist():
            arr = np.array(t, dtype=np.float32)
            if len(arr) >= T:
                start = int(len(arr) * 0.7)
                if start + T > len(arr):
                    start = len(arr) - T
                w = arr[start:start + T]
                mu, sigma = w.mean(), w.std()
                if sigma < 1e-6:
                    continue
                w = (w - mu) / sigma
                clipped = np.clip(w, -5, 5)
                bins = ((clipped + 5) / 10 * 512).astype(np.int64).clip(0, 511)
                windows.append({"input_ids": bins, "values": w})
                seen_domains.add(domain)
                break
        if len(windows) >= n_windows:
            break
    return windows


def extract_per_layer(model, token_ids, device):
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
    # Skip position 0 (BOS outlier)
    layers = [captured[i].squeeze(0)[1:, :].float().cpu() for i in range(N_LAYERS)]
    for h in handles: h.remove()
    captured.clear()
    return layers


def compute_pca_stats(H):
    H_np = H.numpy()
    H_c = H_np - H_np.mean(axis=0)
    n_comp = min(100, H_np.shape[0], H_np.shape[1])
    pca = PCA(n_components=n_comp, svd_solver='full').fit(H_c)
    evals = pca.explained_variance_
    total_var = H_c.var(axis=0, ddof=1).sum()
    p = evals / total_var; p = p[p > 1e-15]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))
    pr = float(evals.sum() ** 2 / (evals ** 2).sum())
    cum = np.cumsum(pca.explained_variance_ratio_)
    return {
        "eff_rank": eff_rank,
        "pr": pr,
        "pcs_90": int(np.searchsorted(cum, 0.90) + 1),
        "pcs_95": int(np.searchsorted(cum, 0.95) + 1),
        "top1_frac": float(evals[0] / total_var) if total_var > 0 else 0,
        "top5_frac": float(evals[:5].sum() / total_var) if total_var > 0 else 0,
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
    return {
        "speed_mean": float(step_sizes.mean()),
        "speed_std": float(step_sizes.std()),
        "angle_mean": float(angles.mean()),
        "path_length": path_length,
        "tortuosity": path_length / e2e if e2e > 0 else float('inf'),
    }


def compute_spectrum(H):
    H_np = H.numpy()
    H_c = H_np - H_np.mean(axis=0)
    fft = np.fft.rfft(H_c, axis=0)
    psd = np.abs(fft) ** 2
    mean_psd = psd.mean(axis=1)
    mean_psd_norm = mean_psd / mean_psd.sum()
    F_len = len(mean_psd)
    low_end = int(F_len * 0.10)
    mid_end = int(F_len * 0.50)
    return {
        "low_energy": float(mean_psd_norm[:low_end].sum()),
        "mid_energy": float(mean_psd_norm[low_end:mid_end].sum()),
        "high_energy": float(mean_psd_norm[mid_end:].sum()),
    }


def compute_ph(H, n_points=N_PH_POINTS, n_pca_dims=10):
    T_len = H.shape[0]
    if T_len > n_points:
        idx = np.linspace(0, T_len - 1, n_points).astype(int)
        H_sub = H[idx].numpy()
    else:
        H_sub = H.numpy()
    H_sub = H_sub - H_sub.mean(axis=0)
    n_comp = min(n_pca_dims, H_sub.shape[0], H_sub.shape[1])
    H_sub = PCA(n_components=n_comp).fit_transform(H_sub)
    std = H_sub.std()
    if std > 1e-8:
        H_sub = H_sub / std
    try:
        result = ripser(H_sub, maxdim=2, thresh=2.0)
        betti = []
        for dim in range(3):
            if dim < len(result['dgms']):
                dgm = result['dgms'][dim]
                p = dgm[:, 1] - dgm[:, 0]
                betti.append(int((p[np.isfinite(p)] > 0.1).sum()))
            else:
                betti.append(0)
        return {"betti": betti}
    except:
        return {"betti": [0, 0, 0]}


def compute_principal_angles(H1, H2, k=10):
    H1_c = H1.numpy() - H1.numpy().mean(axis=0)
    H2_c = H2.numpy() - H2.numpy().mean(axis=0)
    U1 = PCA(n_components=k).fit(H1_c).components_.T
    U2 = PCA(n_components=k).fit(H2_c).components_.T
    _, s, _ = np.linalg.svd(U1.T @ U2)
    angles = np.arccos(np.clip(s, -1, 1)) * 180 / np.pi
    return {"mean_angle": float(angles.mean()), "angles": angles.tolist()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # Load data
    print("Loading TS windows...", flush=True)
    ts_windows = load_ts_windows(hf_token, N_SEQS)
    print(f"  TS: {len(ts_windows)}")

    print("Loading WikiText...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=N_SEQS, seq_len=T, hf_token=hf_token)
    print(f"  Wiki: {len(wiki_seqs)}")

    # Process each model
    all_results = {}
    all_representatives = {}  # condition -> {layer: tensor}

    for model_key, model_path in MODEL_PATHS.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {model_key} ({model_path})")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16).model.to(DEVICE).eval()

        # Determine conditions for this model
        if model_key == "PT":
            conditions = [("ts@PT", ts_windows), ("text@PT", wiki_seqs)]
        elif model_key == "FT":
            conditions = [("ts@FT", ts_windows)]
        else:
            conditions = [("ts@RI", ts_windows)]

        for cond_name, sequences in conditions:
            print(f"\n  Condition: {cond_name}")
            layer_stats = {li: {"pca": [], "smooth": [], "spec": [], "ph": []}
                           for li in range(N_LAYERS)}
            rep_layers = {}

            for si, seq in enumerate(sequences[:N_SEQS]):
                if si % 5 == 0:
                    print(f"    Seq {si+1}/{min(N_SEQS, len(sequences))}...", flush=True)
                layers = extract_per_layer(model, seq["input_ids"], DEVICE)

                for li in range(N_LAYERS):
                    H = layers[li]
                    layer_stats[li]["pca"].append(compute_pca_stats(H))
                    layer_stats[li]["smooth"].append(compute_smoothness(H))
                    layer_stats[li]["spec"].append(compute_spectrum(H))
                    layer_stats[li]["ph"].append(compute_ph(H))

                if si == 0:
                    rep_layers = {li: layers[li] for li in range(N_LAYERS)}
                del layers; gc.collect()

            # Aggregate
            agg = {}
            for li in range(N_LAYERS):
                ls = layer_stats[li]
                agg[li] = {
                    "eff_rank": float(np.mean([r["eff_rank"] for r in ls["pca"]])),
                    "eff_rank_std": float(np.std([r["eff_rank"] for r in ls["pca"]])),
                    "top1_frac": float(np.mean([r["top1_frac"] for r in ls["pca"]])),
                    "top5_frac": float(np.mean([r["top5_frac"] for r in ls["pca"]])),
                    "pcs_90": float(np.mean([r["pcs_90"] for r in ls["pca"]])),
                    "speed": float(np.mean([r["speed_mean"] for r in ls["smooth"]])),
                    "angle": float(np.mean([r["angle_mean"] for r in ls["smooth"]])),
                    "tortuosity": float(np.mean([r["tortuosity"] for r in ls["smooth"]])),
                    "low_energy": float(np.mean([r["low_energy"] for r in ls["spec"]])),
                    "mid_energy": float(np.mean([r["mid_energy"] for r in ls["spec"]])),
                    "high_energy": float(np.mean([r["high_energy"] for r in ls["spec"]])),
                    "betti_1": float(np.mean([r["betti"][1] for r in ls["ph"]])),
                }
            all_results[cond_name] = agg
            all_representatives[cond_name] = rep_layers

        del model; gc.collect(); torch.cuda.empty_cache()

    # Cross-model alignment
    print(f"\n{'='*60}")
    print("INTER-MODEL SUBSPACE ALIGNMENT")
    print(f"{'='*60}")

    pairs = [("ts@PT", "ts@FT"), ("ts@PT", "ts@RI"), ("ts@FT", "ts@RI"),
             ("text@PT", "ts@FT"), ("text@PT", "ts@RI")]
    cross = {}
    for c1, c2 in pairs:
        key = f"{c1}_vs_{c2}"
        cross[key] = {}
        for li in range(N_LAYERS):
            H1 = all_representatives[c1][li]
            H2 = all_representatives[c2][li]
            pa = compute_principal_angles(H1, H2)
            cross[key][li] = pa
        print(f"  {key}: L8 angle={cross[key][8]['mean_angle']:.1f}° L16={cross[key][16]['mean_angle']:.1f}°")

    # Save
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump({k: {str(li): v for li, v in agg.items()} for k, agg in all_results.items()}, f, indent=2)
    with open(f"{OUT_DIR}/cross_modality.json", "w") as f:
        json.dump({k: {str(li): v for li, v in d.items()} for k, d in cross.items()}, f, indent=2)

    # ═══ Plots ═══
    print("\nGenerating plots...", flush=True)
    lx = list(range(N_LAYERS))

    # 1. Effective rank
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for cond in CONDITIONS:
        vals = [all_results[cond][li]["eff_rank"] for li in lx]
        axes[0].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=LABELS[cond])
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Effective Rank")
    axes[0].set_title("Effective Rank of Trajectory"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for cond in CONDITIONS:
        vals = [all_results[cond][li]["pcs_90"] for li in lx]
        axes[1].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=LABELS[cond])
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("PCs for 90% variance")
    axes[1].set_title("Dimensionality (PCs for 90%)"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/dimensionality.png", dpi=150, bbox_inches="tight"); plt.close()

    # 2. Spectral profile
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Trajectory Spectral Energy Distribution", fontsize=14, fontweight='bold')
    for bi, (bname, bkey) in enumerate([("Low (0-10%)", "low_energy"),
                                        ("Mid (10-50%)", "mid_energy"),
                                        ("High (50-100%)", "high_energy")]):
        for cond in CONDITIONS:
            vals = [all_results[cond][li][bkey] for li in lx]
            axes[bi].plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=LABELS[cond])
        axes[bi].set_xlabel("Layer"); axes[bi].set_ylabel("Energy fraction")
        axes[bi].set_title(bname); axes[bi].legend(fontsize=7); axes[bi].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/spectral.png", dpi=150, bbox_inches="tight"); plt.close()

    # 3. Smoothness
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Trajectory Dynamics", fontsize=14, fontweight='bold')
    for cond in CONDITIONS:
        axes[0].plot(lx, [all_results[cond][li]["speed"] for li in lx], 'o-',
                     color=COLORS[cond], linewidth=1.5, markersize=4, label=LABELS[cond])
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean step size"); axes[0].set_title("Speed")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    for cond in CONDITIONS:
        axes[1].plot(lx, [all_results[cond][li]["angle"] for li in lx], 'o-',
                     color=COLORS[cond], linewidth=1.5, markersize=4, label=LABELS[cond])
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Mean turning angle (°)"); axes[1].set_title("Curvature")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    for cond in CONDITIONS:
        axes[2].plot(lx, [all_results[cond][li]["tortuosity"] for li in lx], 'o-',
                     color=COLORS[cond], linewidth=1.5, markersize=4, label=LABELS[cond])
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Tortuosity"); axes[2].set_title("Tortuosity")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/dynamics.png", dpi=150, bbox_inches="tight"); plt.close()

    # 4. Betti β₁
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in CONDITIONS:
        vals = [all_results[cond][li]["betti_1"] for li in lx]
        ax.plot(lx, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=LABELS[cond])
    ax.set_xlabel("Layer"); ax.set_ylabel("β₁ (loops)")
    ax.set_title("Persistent Homology: β₁ (topological loops in trajectory)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/betti1.png", dpi=150, bbox_inches="tight"); plt.close()

    # 5. Subspace alignment
    fig, ax = plt.subplots(figsize=(12, 5))
    pair_colors = {"ts@PT_vs_ts@FT": "#9C27B0", "ts@PT_vs_ts@RI": "#795548",
                   "ts@FT_vs_ts@RI": "#009688", "text@PT_vs_ts@FT": "#FF5722",
                   "text@PT_vs_ts@RI": "#607D8B"}
    pair_labels = {"ts@PT_vs_ts@FT": "TS@PT vs TS@FT",
                   "ts@PT_vs_ts@RI": "TS@PT vs TS@RI",
                   "ts@FT_vs_ts@RI": "TS@FT vs TS@RI",
                   "text@PT_vs_ts@FT": "Text@PT vs TS@FT",
                   "text@PT_vs_ts@RI": "Text@PT vs TS@RI"}
    for key in cross:
        vals = [cross[key][li]["mean_angle"] for li in lx]
        ax.plot(lx, vals, 'o-', color=pair_colors.get(key, 'gray'), linewidth=1.5, markersize=4,
                label=pair_labels.get(key, key))
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Principal Angle (°)")
    ax.set_title("Subspace Alignment Between Models (lower = more similar)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/plots/alignment.png", dpi=150, bbox_inches="tight"); plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (Layer 8)")
    print(f"{'='*80}")
    print(f"\n{'Condition':<30} {'EffRank':>8} {'PCs90':>6} {'Speed':>7} {'LowE%':>7} {'β₁':>5}")
    print("-" * 65)
    for cond in CONDITIONS:
        r = all_results[cond][8]
        print(f"{LABELS[cond]:<30} {r['eff_rank']:>8.1f} {r['pcs_90']:>6.0f} "
              f"{r['speed']:>7.1f} {r['low_energy']*100:>6.1f}% {r['betti_1']:>5.1f}")

    print(f"\nSubspace alignment at L8:")
    for key in cross:
        print(f"  {pair_labels.get(key, key)}: {cross[key][8]['mean_angle']:.1f}°")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
