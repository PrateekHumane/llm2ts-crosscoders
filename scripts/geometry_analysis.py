"""
Geometry and Topology of Sequential Representations Across Modalities.

Inspired by Moisescu-Pareja et al. (2025) "On the Geometry and Topology of
Representations", we treat hidden state trajectories as point clouds and
characterize their topology and geometry.

Conditions:
  1. text@PT     — WikiText through pretrained model
  2. ts@FT       — Time series (binned) through finetuned model
  3. rand@PT     — Random tokens through pretrained model
  4. rand@RI     — Random tokens through RandomInit model

Per condition, per layer, we compute:
  A. Persistent homology (Betti numbers β₀, β₁, β₂)
  B. PCA dimensionality (effective rank, PCs for 90/95/99%)
  C. Trajectory smoothness (step sizes, turning angles, autocorrelation)
  D. Trajectory power spectrum (in D-dimensional space)

Cross-modality:
  E. Principal angles between trajectory subspaces
  F. Grassmann distance between subspaces

We analyze MULTIPLE sequences per condition (not just one) to get distributions.

Usage:
    /usr/bin/python3 scripts/geometry_analysis.py
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
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.wikitext import load_wikitext_sequences

# ── Config ──
T = 512
D = 1024  # per-layer hidden dim
N_LAYERS = 28
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/geometry"

N_SEQS = 20       # sequences per condition (enough for distributions, fast enough)
N_PH_POINTS = 300 # subsample point cloud for persistent homology (ripser is O(n^3))
PH_MAX_DIM = 2    # compute up to H_2
SEED = 42

CONDITIONS = ["text@PT", "ts@FT", "rand@PT", "rand@RandomInit"]

COLORS = {
    "text@PT": "#2196F3",
    "ts@FT": "#4CAF50",
    "rand@PT": "#FF9800",
    "rand@RandomInit": "#E91E63",
}


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_ts_windows(hf_token, n_windows=20):
    """Load diverse TS windows, binned for FT model."""
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    arrow_files = sorted(glob.glob(os.path.join(path, "**/*.arrow"), recursive=True))
    windows = []
    for f in arrow_files:
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
                # Bin tokenize
                clipped = np.clip(w, -5, 5)
                bins = ((clipped + 5) / 10 * 512).astype(np.int64).clip(0, 511)
                windows.append({"input_ids": bins, "values": w})
                if len(windows) >= n_windows:
                    return windows
    return windows


def generate_random_sequences(n_seqs, seq_len, vocab_size=151936):
    rng = np.random.default_rng(SEED)
    return [{"input_ids": rng.integers(0, vocab_size, size=seq_len).astype(np.int64)}
            for _ in range(n_seqs)]


# ═══════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════

def extract_single_sequence_per_layer(model, token_ids, device):
    """Extract hidden states at each layer for a single sequence.
    Returns list of 28 (T, D) float32 tensors on CPU."""
    captured = {}
    handles = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook(m, i, o):
                captured[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))

    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=ids, use_cache=False)

    # Skip position 0 (BOS token has extreme norm ~5000, 200x larger than other tokens,
    # which creates a spurious rank-1 artifact dominating all PCA/geometry metrics)
    layers = [captured[i].squeeze(0)[1:, :].float().cpu() for i in range(N_LAYERS)]
    for h in handles:
        h.remove()
    captured.clear()
    return layers


# ═══════════════════════════════════════════════════════════════
# Analysis A: Persistent Homology
# ═══════════════════════════════════════════════════════════════

def compute_persistent_homology(H, n_points=N_PH_POINTS, max_dim=PH_MAX_DIM,
                                 n_pca_dims=10):
    """Compute persistent homology of point cloud H (T, D).
    Subsamples to n_points for computational feasibility.
    PCA-reduces to n_pca_dims before running ripser (high-D Euclidean
    distances are too large for a meaningful Rips filtration).
    Returns dict with betti numbers and persistence diagrams."""
    T_len, D_dim = H.shape

    # Subsample if needed
    if T_len > n_points:
        idx = np.linspace(0, T_len - 1, n_points).astype(int)
        H_sub = H[idx].numpy()
    else:
        H_sub = H.numpy()

    # Center
    H_sub = H_sub - H_sub.mean(axis=0)

    # PCA reduce: in D=1024, pairwise Euclidean distances scale as sqrt(D)
    # and overwhelm any reasonable filtration threshold.  Projecting onto the
    # top-k PCs preserves the dominant geometric structure while keeping
    # distances in a range where the Rips complex is informative.
    n_comp = min(n_pca_dims, H_sub.shape[0], H_sub.shape[1])
    pca = PCA(n_components=n_comp)
    H_sub = pca.fit_transform(H_sub)

    # Normalize for numerical stability
    std = H_sub.std()
    if std > 1e-8:
        H_sub = H_sub / std

    try:
        result = ripser(H_sub, maxdim=max_dim, thresh=2.0)
        diagrams = result['dgms']

        betti = []
        for dim in range(max_dim + 1):
            if dim < len(diagrams):
                dgm = diagrams[dim]
                # Count features with significant persistence (birth-death gap > threshold)
                persistence = dgm[:, 1] - dgm[:, 0]
                # Filter out infinite persistence (connected component)
                finite_mask = np.isfinite(persistence)
                significant = (persistence[finite_mask] > 0.1).sum()
                betti.append(int(significant))
            else:
                betti.append(0)

        # Also get the total persistence as a summary statistic
        total_persistence = []
        for dim in range(max_dim + 1):
            if dim < len(diagrams):
                dgm = diagrams[dim]
                p = dgm[:, 1] - dgm[:, 0]
                total_persistence.append(float(p[np.isfinite(p)].sum()))
            else:
                total_persistence.append(0.0)

        return {
            "betti": betti,  # [β₀, β₁, β₂]
            "total_persistence": total_persistence,
            "n_points": len(H_sub),
        }
    except Exception as e:
        print(f"    PH failed: {e}")
        return {"betti": [0, 0, 0], "total_persistence": [0, 0, 0], "n_points": len(H_sub)}


# ═══════════════════════════════════════════════════════════════
# Analysis B: PCA Dimensionality
# ═══════════════════════════════════════════════════════════════

def compute_pca_stats(H):
    """PCA analysis of trajectory H (T, D).
    Returns effective rank, participation ratio, PCs for various thresholds."""
    H_np = H.numpy()
    H_c = H_np - H_np.mean(axis=0)

    n_comp = min(100, H_np.shape[0], H_np.shape[1])
    pca = PCA(n_components=n_comp, svd_solver='full')
    pca.fit(H_c)

    evals = pca.explained_variance_
    # Use ddof=1 to match sklearn PCA's normalization (divides by N-1)
    total_var = H_c.var(axis=0, ddof=1).sum()

    # Effective rank using full trace
    p = evals / total_var
    p = p[p > 1e-15]
    eff_rank = float(np.exp(-np.sum(p * np.log(p))))

    # Participation ratio
    pr = float(evals.sum() ** 2 / (evals ** 2).sum())

    # PCs for thresholds
    cum = np.cumsum(pca.explained_variance_ratio_)
    pcs_90 = int(np.searchsorted(cum, 0.90) + 1)
    pcs_95 = int(np.searchsorted(cum, 0.95) + 1)
    pcs_99 = int(np.searchsorted(cum, 0.99) + 1)

    # Top eigenvalue fraction
    top1_frac = float(evals[0] / total_var) if total_var > 0 else 0

    return {
        "eff_rank": eff_rank,
        "participation_ratio": pr,
        "pcs_90": pcs_90,
        "pcs_95": pcs_95,
        "pcs_99": pcs_99,
        "top1_frac": top1_frac,
        "eigenvalues": evals[:20].tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# Analysis C: Trajectory Smoothness
# ═══════════════════════════════════════════════════════════════

def compute_smoothness(H):
    """Analyze trajectory smoothness.
    H: (T, D) tensor."""
    # Step sizes: ||h_{t+1} - h_t||
    diffs = H[1:] - H[:-1]  # (T-1, D)
    step_sizes = diffs.norm(dim=1).numpy()  # (T-1,)

    # Turning angles: angle between consecutive displacement vectors
    # cos(θ) = (d_t · d_{t+1}) / (|d_t| |d_{t+1}|)
    d_norm = F.normalize(diffs, dim=1)
    cos_angles = (d_norm[:-1] * d_norm[1:]).sum(dim=1).numpy()  # (T-2,)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(cos_angles) * 180 / np.pi  # degrees

    # Autocorrelation of step sizes
    ss_centered = step_sizes - step_sizes.mean()
    var = (ss_centered ** 2).sum()
    acf = []
    for lag in range(min(50, len(ss_centered) // 2)):
        if var > 0:
            acf.append(float((ss_centered[:len(ss_centered) - lag] *
                              ss_centered[lag:]).sum() / var))
        else:
            acf.append(0.0)

    # Speed statistics
    speed_mean = float(step_sizes.mean())
    speed_std = float(step_sizes.std())
    speed_cv = speed_std / speed_mean if speed_mean > 0 else 0  # coefficient of variation

    # Angle statistics
    angle_mean = float(angles.mean())
    angle_std = float(angles.std())

    # Trajectory length (total path length)
    path_length = float(step_sizes.sum())

    # End-to-end distance
    e2e = float((H[-1] - H[0]).norm())

    # Tortuosity = path_length / end-to-end
    tortuosity = path_length / e2e if e2e > 0 else float('inf')

    return {
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "speed_cv": speed_cv,
        "angle_mean": angle_mean,
        "angle_std": angle_std,
        "path_length": path_length,
        "e2e_distance": e2e,
        "tortuosity": tortuosity,
        "step_size_acf": acf[:20],
        "step_sizes_hist": np.histogram(step_sizes, bins=30)[0].tolist(),
        "angles_hist": np.histogram(angles, bins=30, range=(0, 180))[0].tolist(),
    }


# ═══════════════════════════════════════════════════════════════
# Analysis D: Trajectory Power Spectrum
# ═══════════════════════════════════════════════════════════════

def compute_trajectory_spectrum(H):
    """Power spectrum of the D-dimensional trajectory.
    Compute FFT along the time axis for each dimension, then aggregate."""
    H_np = H.numpy()
    H_c = H_np - H_np.mean(axis=0)

    # FFT along time axis: (T, D) -> (T//2+1, D) magnitudes
    fft = np.fft.rfft(H_c, axis=0)
    psd = np.abs(fft) ** 2  # (T//2+1, D)

    # Average PSD across dimensions
    mean_psd = psd.mean(axis=1)  # (T//2+1,)
    mean_psd_norm = mean_psd / mean_psd.sum()

    # Energy in frequency bands
    F = len(mean_psd)
    low_end = int(F * 0.10)
    mid_end = int(F * 0.50)

    low_energy = float(mean_psd_norm[:low_end].sum())
    mid_energy = float(mean_psd_norm[low_end:mid_end].sum())
    high_energy = float(mean_psd_norm[mid_end:].sum())

    # Spectral centroid
    freqs = np.arange(F) / F
    centroid = float((freqs * mean_psd_norm).sum())

    return {
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "spectral_centroid": centroid,
        "mean_psd": mean_psd_norm[:50].tolist(),  # first 50 freq bins
    }


# ═══════════════════════════════════════════════════════════════
# Analysis E: Cross-modality Subspace Alignment
# ═══════════════════════════════════════════════════════════════

def compute_principal_angles(H1, H2, k=10):
    """Compute principal angles between the k-dimensional subspaces
    spanned by the top-k PCs of H1 and H2.
    Returns angles in degrees."""
    H1_c = H1.numpy() - H1.numpy().mean(axis=0)
    H2_c = H2.numpy() - H2.numpy().mean(axis=0)

    pca1 = PCA(n_components=k).fit(H1_c)
    pca2 = PCA(n_components=k).fit(H2_c)

    U1 = pca1.components_.T  # (D, k)
    U2 = pca2.components_.T  # (D, k)

    # SVD of U1.T @ U2 gives cos(principal_angles)
    M = U1.T @ U2  # (k, k)
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, -1, 1)
    angles = np.arccos(s) * 180 / np.pi

    # Grassmann distance = sqrt(sum of squared angles in radians)
    angles_rad = np.arccos(s)
    grassmann = float(np.sqrt((angles_rad ** 2).sum()))

    return {
        "principal_angles": angles.tolist(),
        "grassmann_distance": grassmann,
        "mean_angle": float(angles.mean()),
        "max_angle": float(angles.max()),
        "min_angle": float(angles.min()),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for sub in ["plots", "data"]:
        os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # ── Load sequences ──
    print("Loading data...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=N_SEQS, seq_len=T, hf_token=hf_token)
    ts_windows = load_ts_windows(hf_token, N_SEQS)
    rand_seqs = generate_random_sequences(N_SEQS, T)

    print(f"  WikiText: {len(wiki_seqs)}, TS: {len(ts_windows)}, Random: {len(rand_seqs)}")

    # ── Models ──
    models_to_load = {
        "text@PT": ("Qwen/Qwen3-0.6B", wiki_seqs),
        "ts@FT": ("cerc-aai/Qwen3-0.6B-pretrain-normal_scale-uniform_bin-V512", ts_windows),
        "rand@PT": ("Qwen/Qwen3-0.6B", rand_seqs),
    }

    # RandomInit
    rand_init_seqs = rand_seqs  # same random tokens

    # ── Process each condition ──
    all_results = {}

    for cond in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond}")
        print(f"{'='*60}")

        # Load model
        if cond == "rand@RandomInit":
            config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
            model = AutoModelForCausalLM.from_config(config).to(
                dtype=torch.bfloat16).model.to(DEVICE).eval()
            sequences = rand_init_seqs
        else:
            model_name, sequences = models_to_load[cond]
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16).model.to(DEVICE).eval()

        # Per-layer results aggregated across sequences
        layer_results = {li: {
            "ph": [], "pca": [], "smooth": [], "spectrum": []
        } for li in range(N_LAYERS)}

        # Also collect one representative trajectory per layer for cross-modality analysis
        representative_trajectories = {}

        for si, seq in enumerate(sequences[:N_SEQS]):
            print(f"  Seq {si+1}/{N_SEQS}...", flush=True)
            layers = extract_single_sequence_per_layer(model, seq["input_ids"], DEVICE)

            for li in range(N_LAYERS):
                H = layers[li]  # (T, D)

                # A: Persistent Homology
                ph = compute_persistent_homology(H)
                layer_results[li]["ph"].append(ph)

                # B: PCA
                pca_stats = compute_pca_stats(H)
                layer_results[li]["pca"].append(pca_stats)

                # C: Smoothness
                smooth = compute_smoothness(H)
                layer_results[li]["smooth"].append(smooth)

                # D: Spectrum
                spec = compute_trajectory_spectrum(H)
                layer_results[li]["spectrum"].append(spec)

            # Save first sequence as representative
            if si == 0:
                representative_trajectories = {li: layers[li] for li in range(N_LAYERS)}

            del layers; gc.collect()

        del model; gc.collect(); torch.cuda.empty_cache()

        # Aggregate results
        agg = {}
        for li in range(N_LAYERS):
            lr = layer_results[li]
            agg[li] = {
                # PH: mean Betti numbers across sequences
                "betti_0_mean": float(np.mean([r["betti"][0] for r in lr["ph"]])),
                "betti_1_mean": float(np.mean([r["betti"][1] for r in lr["ph"]])),
                "betti_2_mean": float(np.mean([r["betti"][2] for r in lr["ph"]])),
                "total_persistence_1": float(np.mean([r["total_persistence"][1] for r in lr["ph"]])),

                # PCA: mean stats
                "eff_rank_mean": float(np.mean([r["eff_rank"] for r in lr["pca"]])),
                "eff_rank_std": float(np.std([r["eff_rank"] for r in lr["pca"]])),
                "pr_mean": float(np.mean([r["participation_ratio"] for r in lr["pca"]])),
                "pcs_90_mean": float(np.mean([r["pcs_90"] for r in lr["pca"]])),
                "top1_frac_mean": float(np.mean([r["top1_frac"] for r in lr["pca"]])),

                # Smoothness
                "speed_mean": float(np.mean([r["speed_mean"] for r in lr["smooth"]])),
                "speed_cv_mean": float(np.mean([r["speed_cv"] for r in lr["smooth"]])),
                "angle_mean": float(np.mean([r["angle_mean"] for r in lr["smooth"]])),
                "tortuosity_mean": float(np.mean([r["tortuosity"] for r in lr["smooth"]])),
                "path_length_mean": float(np.mean([r["path_length"] for r in lr["smooth"]])),

                # Spectrum
                "low_energy_mean": float(np.mean([r["low_energy"] for r in lr["spectrum"]])),
                "mid_energy_mean": float(np.mean([r["mid_energy"] for r in lr["spectrum"]])),
                "high_energy_mean": float(np.mean([r["high_energy"] for r in lr["spectrum"]])),
                "spectral_centroid_mean": float(np.mean([r["spectral_centroid"] for r in lr["spectrum"]])),
            }

        all_results[cond] = {
            "per_layer": agg,
            "representative": {li: representative_trajectories[li] for li in range(N_LAYERS)}
                              if representative_trajectories else {},
        }

    # ── Cross-modality analysis ──
    print(f"\n{'='*60}")
    print("CROSS-MODALITY SUBSPACE ALIGNMENT")
    print(f"{'='*60}")

    cross_results = {}
    pairs = [("text@PT", "ts@FT"), ("text@PT", "rand@PT"),
             ("text@PT", "rand@RandomInit"), ("ts@FT", "rand@PT")]

    for c1, c2 in pairs:
        pair_key = f"{c1}_vs_{c2}"
        print(f"\n  {pair_key}")
        cross_results[pair_key] = {}
        for li in range(N_LAYERS):
            H1 = all_results[c1]["representative"][li]
            H2 = all_results[c2]["representative"][li]
            pa = compute_principal_angles(H1, H2, k=10)
            cross_results[pair_key][li] = pa
            if li % 7 == 0 or li == N_LAYERS - 1:
                print(f"    L{li}: mean_angle={pa['mean_angle']:.1f}° grassmann={pa['grassmann_distance']:.3f}")

    # ── Save results (without tensors) ──
    save_results = {}
    for cond in CONDITIONS:
        save_results[cond] = all_results[cond]["per_layer"]

    with open(f"{OUT_DIR}/data/results.json", "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    with open(f"{OUT_DIR}/data/cross_modality.json", "w") as f:
        json.dump(cross_results, f, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")

    layers_x = list(range(N_LAYERS))

    # ── Plot 1: Effective Rank across layers ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in CONDITIONS:
        vals = [all_results[cond]["per_layer"][li]["eff_rank_mean"] for li in range(N_LAYERS)]
        ax.plot(layers_x, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=cond)
    ax.set_xlabel("Layer"); ax.set_ylabel("Effective Rank")
    ax.set_title("Effective Rank of Trajectory per Layer"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/effective_rank.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 2: Top-1 eigenvalue fraction ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in CONDITIONS:
        vals = [all_results[cond]["per_layer"][li]["top1_frac_mean"] for li in range(N_LAYERS)]
        ax.plot(layers_x, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=cond)
    ax.set_xlabel("Layer"); ax.set_ylabel("Fraction of variance in PC1")
    ax.set_title("Dominance of First Principal Component"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/top1_fraction.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 3: Trajectory smoothness (step size) ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for cond in CONDITIONS:
        vals = [all_results[cond]["per_layer"][li]["speed_mean"] for li in range(N_LAYERS)]
        axes[0].plot(layers_x, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=cond)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean step size ||h_{t+1} - h_t||")
    axes[0].set_title("Trajectory Speed"); axes[0].legend(); axes[0].grid(alpha=0.3)

    for cond in CONDITIONS:
        vals = [all_results[cond]["per_layer"][li]["angle_mean"] for li in range(N_LAYERS)]
        axes[1].plot(layers_x, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=cond)
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Mean turning angle (degrees)")
    axes[1].set_title("Trajectory Curvature"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/smoothness.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 4: Spectral energy distribution ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Energy Distribution of D-dimensional Trajectory Spectrum", fontsize=13, fontweight='bold')
    for bi, (bname, bkey) in enumerate([("Low (0-10%)", "low_energy_mean"),
                                        ("Mid (10-50%)", "mid_energy_mean"),
                                        ("High (50-100%)", "high_energy_mean")]):
        for cond in CONDITIONS:
            vals = [all_results[cond]["per_layer"][li][bkey] for li in range(N_LAYERS)]
            axes[bi].plot(layers_x, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=cond)
        axes[bi].set_xlabel("Layer"); axes[bi].set_ylabel("Energy fraction")
        axes[bi].set_title(bname); axes[bi].legend(fontsize=8); axes[bi].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/spectral_energy.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 5: Betti numbers ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Persistent Homology: Betti Numbers of Trajectory Point Cloud", fontsize=13, fontweight='bold')
    for bi in range(3):
        for cond in CONDITIONS:
            vals = [all_results[cond]["per_layer"][li][f"betti_{bi}_mean"] for li in range(N_LAYERS)]
            axes[bi].plot(layers_x, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=cond)
        axes[bi].set_xlabel("Layer"); axes[bi].set_ylabel(f"β_{bi}")
        axes[bi].set_title(f"β_{bi} (H_{bi})"); axes[bi].legend(fontsize=8); axes[bi].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/betti_numbers.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 6: Tortuosity ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for cond in CONDITIONS:
        vals = [all_results[cond]["per_layer"][li]["tortuosity_mean"] for li in range(N_LAYERS)]
        ax.plot(layers_x, vals, 'o-', color=COLORS[cond], linewidth=1.5, markersize=4, label=cond)
    ax.set_xlabel("Layer"); ax.set_ylabel("Tortuosity (path_length / end-to-end)")
    ax.set_title("Trajectory Tortuosity (higher = more winding)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/tortuosity.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 7: Cross-modality principal angles ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    pair_colors = {"text@PT_vs_ts@FT": "#9C27B0", "text@PT_vs_rand@PT": "#FF5722",
                   "text@PT_vs_rand@RandomInit": "#795548", "ts@FT_vs_rand@PT": "#009688"}
    pair_labels = {"text@PT_vs_ts@FT": "Text@PT vs TS@FT",
                   "text@PT_vs_rand@PT": "Text@PT vs Rand@PT",
                   "text@PT_vs_rand@RandomInit": "Text@PT vs Rand@RI",
                   "ts@FT_vs_rand@PT": "TS@FT vs Rand@PT"}

    for pair_key in cross_results:
        vals_angle = [cross_results[pair_key][li]["mean_angle"] for li in range(N_LAYERS)]
        vals_grass = [cross_results[pair_key][li]["grassmann_distance"] for li in range(N_LAYERS)]
        axes[0].plot(layers_x, vals_angle, 'o-', color=pair_colors.get(pair_key, 'gray'),
                     linewidth=1.5, markersize=4, label=pair_labels.get(pair_key, pair_key))
        axes[1].plot(layers_x, vals_grass, 'o-', color=pair_colors.get(pair_key, 'gray'),
                     linewidth=1.5, markersize=4, label=pair_labels.get(pair_key, pair_key))

    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Mean Principal Angle (degrees)")
    axes[0].set_title("Subspace Alignment (lower = more aligned)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Grassmann Distance")
    axes[1].set_title("Grassmann Distance Between Subspaces")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/subspace_alignment.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("SUMMARY TABLE (Layer 8 — representative mid layer)")
    print(f"{'='*80}")
    print(f"\n{'Condition':<20} {'EffRank':>8} {'Top1%':>7} {'Speed':>8} {'Angle':>7} "
          f"{'Tort':>7} {'Low%':>6} {'β₁':>4}")
    print("-" * 70)
    for cond in CONDITIONS:
        r = all_results[cond]["per_layer"][8]
        print(f"{cond:<20} {r['eff_rank_mean']:>8.1f} {r['top1_frac_mean']*100:>6.1f}% "
              f"{r['speed_mean']:>8.2f} {r['angle_mean']:>6.1f}° "
              f"{r['tortuosity_mean']:>7.1f} {r['low_energy_mean']*100:>5.1f}% "
              f"{r['betti_1_mean']:>4.1f}")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
