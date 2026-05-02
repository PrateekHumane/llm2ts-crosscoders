"""
Landscape spectrum analysis:
1. Eigenspectrum shape via stochastic trace estimation (many random v^T H v)
2. Per-example gradient alignment (pairwise cosine similarity)
3. Gradient-Hessian alignment (does gradient align with high or low curvature?)

Run on 4 GPUs:
  GPU 0: PT spectrum + gradient alignment (layers 6-10)
  GPU 1: RandomInit spectrum + gradient alignment (layers 6-10)
  GPU 2: PT per-example gradients
  GPU 3: RandomInit per-example gradients

Usage:
    CUDA_VISIBLE_DEVICES=X python scripts/landscape_spectrum.py --job spectrum_PT
    CUDA_VISIBLE_DEVICES=X python scripts/landscape_spectrum.py --job spectrum_RI
    CUDA_VISIBLE_DEVICES=X python scripts/landscape_spectrum.py --job gradients_PT
    CUDA_VISIBLE_DEVICES=X python scripts/landscape_spectrum.py --job gradients_RI
    python scripts/landscape_spectrum.py --job plot
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, glob, time, argparse

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = 512; SEED = 42
OUT_DIR = "mapping_results/loss_landscape"
LAYER_INDICES = [6, 7, 8, 9, 10]
N_SPECTRUM_DIRS = 200  # random directions for spectral density
N_DATA = 50


def load_ts_data(hf_token, n_windows=200):
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    windows = []
    for f in sorted(glob.glob(os.path.join(path, "**/*.arrow"), recursive=True)):
        with open(f, "rb") as fp:
            tbl = ipc.open_stream(fp).read_all()
        for t in tbl['target'].to_pylist():
            arr = np.array(t, dtype=np.float32)
            if len(arr) >= T:
                start = int(len(arr)*0.7)
                if start+T > len(arr): start = len(arr)-T
                w = arr[start:start+T]; mu, sigma = w.mean(), w.std()
                if sigma < 1e-6: continue
                w = (w-mu)/sigma
                bins = ((np.clip(w,-5,5)+5)/10*512).astype(np.int64).clip(0,511)
                windows.append(bins)
                if len(windows) >= n_windows: return windows
    return windows


def get_target_params(model, layer_indices):
    for p in model.parameters():
        p.requires_grad = False
    target = []
    for li in layer_indices:
        for p in model.model.layers[li].parameters():
            p.requires_grad = True
            target.append(p)
    return target


def compute_loss(model, batch_ids):
    logits = model(input_ids=batch_ids).logits.float()
    return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                           batch_ids[:, 1:].contiguous().view(-1))


def exact_hvp(model, batch_ids, vector_list, params):
    loss = compute_loss(model, batch_ids)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gv = sum((g * v).sum() for g, v in zip(grads, vector_list))
    hvp = torch.autograd.grad(gv, params, retain_graph=False)
    return [h.detach() for h in hvp]


def load_model(init_name, device):
    if init_name == "PT":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(device)
    else:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32).to(device)
    return model


def run_spectrum(init_name, device, ts_windows):
    """Compute spectral density via many random v^T H v."""
    print(f"\n{'='*60}")
    print(f"EIGENSPECTRUM: {init_name}")
    print(f"{'='*60}")

    model = load_model(init_name, device)
    params = get_target_params(model, LAYER_INDICES)
    n_params = sum(p.numel() for p in params)
    print(f"  Params: {n_params/1e6:.1f}M from layers {LAYER_INDICES}")

    # Data batches
    batches = [torch.tensor(ts_windows[i:i+2], dtype=torch.long, device=device)
               for i in range(0, min(N_DATA, len(ts_windows)), 2)]

    model.train()
    curvatures = []
    t0 = time.time()

    for di in range(N_SPECTRUM_DIRS):
        torch.manual_seed(SEED + di * 7)
        v = [torch.randn_like(p) for p in params]
        norm = torch.sqrt(sum((vi**2).sum() for vi in v))
        v = [vi / norm for vi in v]

        # Average HVP over a few batches
        n_batches_use = min(5, len(batches))
        hvp_sum = None
        for bi in range(n_batches_use):
            hvp = exact_hvp(model, batches[bi], v, params)
            if hvp_sum is None:
                hvp_sum = [h.clone() for h in hvp]
            else:
                for i in range(len(hvp_sum)):
                    hvp_sum[i] += hvp[i]
            torch.cuda.empty_cache()
        hvp_avg = [h / n_batches_use for h in hvp_sum]

        curv = sum((vi * hi).sum().item() for vi, hi in zip(v, hvp_avg))
        curvatures.append(curv)

        if (di + 1) % 50 == 0:
            print(f"    dir {di+1}/{N_SPECTRUM_DIRS}: mean={np.mean(curvatures):.6f} "
                  f"std={np.std(curvatures):.6f} ({time.time()-t0:.0f}s)", flush=True)

    # Also compute gradient and its alignment with Hessian
    print(f"  Computing mean gradient...", flush=True)
    model.zero_grad()
    total_loss = 0
    for bi in range(min(10, len(batches))):
        loss = compute_loss(model, batches[bi])
        loss.backward()
        total_loss += loss.item()

    # Extract mean gradient
    mean_grad = [p.grad.clone() / min(10, len(batches)) for p in params]
    grad_norm = torch.sqrt(sum((g**2).sum() for g in mean_grad)).item()

    # Compute curvature along gradient direction
    grad_dir = [g / grad_norm for g in mean_grad]
    model.zero_grad()
    hvp_grad = None
    for bi in range(min(5, len(batches))):
        hvp = exact_hvp(model, batches[bi], grad_dir, params)
        if hvp_grad is None:
            hvp_grad = [h.clone() for h in hvp]
        else:
            for i in range(len(hvp_grad)):
                hvp_grad[i] += hvp[i]
        torch.cuda.empty_cache()
    hvp_grad = [h / min(5, len(batches)) for h in hvp_grad]
    curv_along_grad = sum((gi * hi).sum().item() for gi, hi in zip(grad_dir, hvp_grad))

    result = {
        "curvatures": curvatures,
        "mean": float(np.mean(curvatures)),
        "std": float(np.std(curvatures)),
        "median": float(np.median(curvatures)),
        "min": float(np.min(curvatures)),
        "max": float(np.max(curvatures)),
        "grad_norm": grad_norm,
        "curvature_along_gradient": curv_along_grad,
        "n_params": n_params,
        "implied_trace": float(np.mean(curvatures) * n_params),
    }

    print(f"  Spectrum: mean={result['mean']:.6f} std={result['std']:.6f} "
          f"median={result['median']:.6f}")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"  Curvature along gradient: {curv_along_grad:.4f}")
    print(f"  Implied trace: {result['implied_trace']:.1f}")

    with open(f"{OUT_DIR}/spectrum_{init_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def run_gradients(init_name, device, ts_windows):
    """Compute per-example gradients and their pairwise alignment."""
    print(f"\n{'='*60}")
    print(f"PER-EXAMPLE GRADIENTS: {init_name}")
    print(f"{'='*60}")

    model = load_model(init_name, device)
    params = get_target_params(model, LAYER_INDICES)
    n_params = sum(p.numel() for p in params)

    # Compute per-example gradients
    n_examples = min(30, len(ts_windows))
    per_example_grads = []

    model.train()
    for ei in range(n_examples):
        model.zero_grad()
        batch = torch.tensor(ts_windows[ei:ei+1], dtype=torch.long, device=device)
        loss = compute_loss(model, batch)
        loss.backward()

        # Flatten gradient into single vector
        grad_flat = torch.cat([p.grad.flatten() for p in params]).cpu()
        per_example_grads.append(grad_flat)

        if (ei + 1) % 10 == 0:
            print(f"    example {ei+1}/{n_examples}", flush=True)

    # Stack into matrix (n_examples, n_params)
    G = torch.stack(per_example_grads)  # (N, D)

    # Pairwise cosine similarity
    G_norm = F.normalize(G, dim=1)
    cos_sim = (G_norm @ G_norm.T).numpy()

    # Extract upper triangle (exclude diagonal)
    mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
    pairwise_cos = cos_sim[mask]

    # Gradient norms
    grad_norms = G.norm(dim=1).numpy()

    # Mean gradient and its norm
    mean_grad = G.mean(dim=0)
    mean_grad_norm = mean_grad.norm().item()

    # Signal-to-noise ratio: ||mean_grad|| / mean(||grad_i - mean_grad||)
    noise_norms = (G - mean_grad.unsqueeze(0)).norm(dim=1).numpy()
    snr = mean_grad_norm / np.mean(noise_norms) if np.mean(noise_norms) > 0 else float('inf')

    # Gradient variance ratio: how much variance is in the mean direction vs orthogonal
    # Project each gradient onto mean direction
    mean_dir = mean_grad / mean_grad_norm if mean_grad_norm > 0 else mean_grad
    projections = (G @ mean_dir.unsqueeze(1)).squeeze(1).numpy()
    var_along_mean = np.var(projections)
    var_total = np.var(G.numpy(), axis=0).sum()
    var_ratio = var_along_mean / var_total if var_total > 0 else 0

    result = {
        "pairwise_cosine_mean": float(np.mean(pairwise_cos)),
        "pairwise_cosine_std": float(np.std(pairwise_cos)),
        "pairwise_cosine_median": float(np.median(pairwise_cos)),
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_std": float(np.std(grad_norms)),
        "mean_grad_norm": mean_grad_norm,
        "noise_norm_mean": float(np.mean(noise_norms)),
        "snr": snr,
        "var_ratio_along_mean": var_ratio,
        "pairwise_cosines": pairwise_cos.tolist(),
        "n_examples": n_examples,
    }

    print(f"  Pairwise cosine: mean={result['pairwise_cosine_mean']:.4f} "
          f"std={result['pairwise_cosine_std']:.4f}")
    print(f"  Gradient norm: mean={result['grad_norm_mean']:.6f}")
    print(f"  Mean gradient norm: {mean_grad_norm:.6f}")
    print(f"  SNR (||mean_grad|| / ||noise||): {snr:.4f}")
    print(f"  Var ratio along mean: {var_ratio:.4f}")

    # Convert numpy types for JSON
    def to_python(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, list):
            return [to_python(x) for x in obj]
        return obj
    result = {k: to_python(v) for k, v in result.items()}
    with open(f"{OUT_DIR}/gradients_{init_name}.json", "w") as f:
        json.dump(result, f, indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()
    return result


def make_plots():
    print("Generating plots...", flush=True)

    spec_pt = json.load(open(f"{OUT_DIR}/spectrum_PT.json"))
    spec_ri = json.load(open(f"{OUT_DIR}/spectrum_RandomInit.json"))
    grad_pt = json.load(open(f"{OUT_DIR}/gradients_PT.json"))
    grad_ri = json.load(open(f"{OUT_DIR}/gradients_RandomInit.json"))

    # ── 1. Spectral density histogram ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Curvature Distribution in Parameter Space (Layers 6-10)\n"
                 "200 random directions: v^T H v for each",
                 fontsize=13, fontweight='bold')

    # Side by side
    axes[0].hist(spec_pt["curvatures"], bins=40, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[0].axvline(spec_pt["mean"], color='black', linestyle='--', linewidth=1.5)
    axes[0].axvline(spec_pt["curvature_along_gradient"], color='red', linestyle='-', linewidth=2,
                    label=f'Along gradient ({spec_pt["curvature_along_gradient"]:.1f})')
    axes[0].set_xlabel("Curvature (v^T H v)"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"PT\nmean={spec_pt['mean']:.5f}, std={spec_pt['std']:.5f}")
    axes[0].legend(fontsize=8)

    axes[1].hist(spec_ri["curvatures"], bins=40, color='#FF9800', alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[1].axvline(spec_ri["mean"], color='black', linestyle='--', linewidth=1.5)
    axes[1].axvline(spec_ri["curvature_along_gradient"], color='red', linestyle='-', linewidth=2,
                    label=f'Along gradient ({spec_ri["curvature_along_gradient"]:.4f})')
    axes[1].set_xlabel("Curvature (v^T H v)")
    axes[1].set_title(f"RandomInit\nmean={spec_ri['mean']:.7f}, std={spec_ri['std']:.7f}")
    axes[1].legend(fontsize=8)

    # Overlay
    axes[2].hist(spec_pt["curvatures"], bins=40, color='#2196F3', alpha=0.5, label='PT', edgecolor='black', linewidth=0.2)
    axes[2].hist(spec_ri["curvatures"], bins=40, color='#FF9800', alpha=0.5, label='RandomInit', edgecolor='black', linewidth=0.2)
    axes[2].set_xlabel("Curvature"); axes[2].set_title("Overlay")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/spectral_density.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 2. Per-example gradient alignment ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Per-Example Gradient Alignment\n"
                 "Do individual example gradients point the same way?",
                 fontsize=13, fontweight='bold')

    if "pairwise_cosines" in grad_pt:
        axes[0].hist(grad_pt["pairwise_cosines"], bins=40, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[0].axvline(grad_pt["pairwise_cosine_mean"], color='black', linestyle='--', linewidth=1.5)
    axes[0].set_xlabel("Pairwise cosine similarity"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"PT\nmean={grad_pt['pairwise_cosine_mean']:.4f}")

    if "pairwise_cosines" in grad_ri:
        axes[1].hist(grad_ri["pairwise_cosines"], bins=40, color='#FF9800', alpha=0.7, edgecolor='black', linewidth=0.3)
    axes[1].axvline(grad_ri["pairwise_cosine_mean"], color='black', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel("Pairwise cosine similarity")
    axes[1].set_title(f"RandomInit\nmean={grad_ri['pairwise_cosine_mean']:.4f}")

    # Summary bar chart
    metrics = ['pairwise_cosine_mean', 'snr', 'var_ratio_along_mean']
    metric_labels = ['Gradient\nalignment', 'Signal-to-\nnoise ratio', 'Variance along\nmean direction']
    x = np.arange(len(metrics))
    w = 0.35
    pt_vals = [grad_pt[m] for m in metrics]
    ri_vals = [grad_ri[m] for m in metrics]
    axes[2].bar(x - w/2, pt_vals, w, color='#2196F3', label='PT')
    axes[2].bar(x + w/2, ri_vals, w, color='#FF9800', label='RandomInit')
    axes[2].set_xticks(x); axes[2].set_xticklabels(metric_labels, fontsize=9)
    axes[2].set_title("Gradient quality metrics\n(higher = more useful for SGD)")
    axes[2].legend()
    for i in range(len(metrics)):
        axes[2].text(i - w/2, pt_vals[i] + 0.005, f"{pt_vals[i]:.3f}", ha='center', fontsize=8)
        axes[2].text(i + w/2, ri_vals[i] + 0.005, f"{ri_vals[i]:.3f}", ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/gradient_alignment.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── 3. Combined: curvature along gradient vs random ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(spec_pt["curvatures"], bins=40, color='#2196F3', alpha=0.4, label='PT random dirs', density=True)
    ax.hist(spec_ri["curvatures"], bins=40, color='#FF9800', alpha=0.4, label='RI random dirs', density=True)
    ax.axvline(spec_pt["curvature_along_gradient"], color='#2196F3', linestyle='-', linewidth=3,
               label=f'PT gradient dir ({spec_pt["curvature_along_gradient"]:.1f})')
    ax.axvline(spec_ri["curvature_along_gradient"], color='#FF9800', linestyle='-', linewidth=3,
               label=f'RI gradient dir ({spec_ri["curvature_along_gradient"]:.4f})')
    ax.set_xlabel("Curvature (v^T H v)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Where Does the Gradient Point in the Curvature Spectrum?\n"
                 "Red line = curvature along gradient direction vs distribution of random directions",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/gradient_vs_spectrum.png", dpi=150, bbox_inches="tight"); plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("LANDSCAPE SPECTRUM SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Metric':<35} {'PT':>12} {'RandomInit':>12}")
    print("-" * 60)
    print(f"{'Random dir curvature (mean)':35} {spec_pt['mean']:>12.6f} {spec_ri['mean']:>12.8f}")
    print(f"{'Random dir curvature (std)':35} {spec_pt['std']:>12.6f} {spec_ri['std']:>12.8f}")
    print(f"{'Curvature along gradient':35} {spec_pt['curvature_along_gradient']:>12.4f} {spec_ri['curvature_along_gradient']:>12.6f}")
    print(f"{'Gradient norm':35} {spec_pt['grad_norm']:>12.6f} {spec_ri['grad_norm']:>12.6f}")
    print(f"{'Pairwise gradient cosine':35} {grad_pt['pairwise_cosine_mean']:>12.4f} {grad_ri['pairwise_cosine_mean']:>12.4f}")
    print(f"{'Gradient SNR':35} {grad_pt['snr']:>12.4f} {grad_ri['snr']:>12.4f}")
    print(f"{'Var ratio along mean grad':35} {grad_pt['var_ratio_along_mean']:>12.4f} {grad_ri['var_ratio_along_mean']:>12.4f}")

    print(f"\nAll plots saved to {OUT_DIR}/plots/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True,
                        choices=["spectrum_PT", "spectrum_RI", "gradients_PT", "gradients_RI", "plot"])
    args = parser.parse_args()

    if args.job == "plot":
        make_plots(); return

    device = torch.device("cuda:0")
    hf_token = os.environ.get("HF_TOKEN")
    ts_windows = load_ts_data(hf_token, N_DATA)
    print(f"Loaded {len(ts_windows)} TS sequences")

    if args.job == "spectrum_PT":
        run_spectrum("PT", device, ts_windows)
    elif args.job == "spectrum_RI":
        run_spectrum("RandomInit", device, ts_windows)
    elif args.job == "gradients_PT":
        run_gradients("PT", device, ts_windows)
    elif args.job == "gradients_RI":
        run_gradients("RandomInit", device, ts_windows)


if __name__ == "__main__":
    main()
