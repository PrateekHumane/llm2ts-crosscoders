"""
Loss Landscape Analysis: Is PT initialization in a region of low curvature
and good conditioning for the TS objective?

Experiment 1: Hessian eigenvalues of TS loss at PT init vs RandomInit
  - Compute top-k Hessian eigenvalues via power iteration (Lanczos)
  - Compare: PT should have smaller eigenvalues (flatter landscape)

Experiment 2: Sharpness of trained solutions (FT vs RI)
  - Add Gaussian noise to weights: W' = W + σ·ε
  - Measure TS loss at perturbed weights for varying σ
  - Sharper minimum = loss increases faster with σ

Usage:
    /usr/bin/python3 scripts/loss_landscape.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, glob, time, copy

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = 512
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/loss_landscape"
SEED = 42

MODEL_PATHS = {
    "PT": "Qwen/Qwen3-0.6B",
    "FT": "models/ft",
    "RI": "models/ri",
}
MODEL_COLORS = {"PT": "#2196F3", "FT": "#4CAF50", "RI": "#E91E63", "RandomInit": "#FF9800"}


def load_ts_data(hf_token, n_windows=100):
    """Load TS windows as bin tokens for evaluation."""
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
                if start + T > len(arr): start = len(arr) - T
                w = arr[start:start + T]
                mu, sigma = w.mean(), w.std()
                if sigma < 1e-6: continue
                w = (w - mu) / sigma
                bins = ((np.clip(w, -5, 5) + 5) / 10 * 512).astype(np.int64).clip(0, 511)
                windows.append(bins)
                if len(windows) >= n_windows:
                    return windows
    return windows


def compute_ts_loss(model, token_ids_list, device, batch_size=5):
    """Compute mean cross-entropy loss on TS data."""
    model.eval()
    total_loss = 0; n_batches = 0
    for i in range(0, len(token_ids_list), batch_size):
        batch = token_ids_list[i:i + batch_size]
        ids = torch.tensor(np.stack(batch), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               ids[:, 1:].contiguous().view(-1))
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Hessian top eigenvalues via power iteration
# ═══════════════════════════════════════════════════════════════

def hessian_vector_product(model, token_ids_batch, vector_dict, device):
    """Compute Hessian-vector product: H·v where H is the Hessian of
    the loss w.r.t. model parameters and v is a parameter-shaped vector.
    Uses finite differences: H·v ≈ (∇L(θ+εv) - ∇L(θ-εv)) / (2ε)"""
    eps = 1e-3

    # Save original params
    original_params = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    # Forward at θ + εv
    for name, p in model.named_parameters():
        if name in vector_dict:
            p.data.add_(eps * vector_dict[name].to(p.device))

    model.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=token_ids_batch).logits.float()
    loss_plus = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                 token_ids_batch[:, 1:].contiguous().view(-1))
    loss_plus.backward()
    grad_plus = {name: p.grad.clone() for name, p in model.named_parameters()
                 if p.requires_grad and p.grad is not None}

    # Restore and forward at θ - εv
    for name, p in model.named_parameters():
        if name in original_params:
            p.data.copy_(original_params[name])
            p.data.add_(-eps * vector_dict[name].to(p.device))

    model.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=token_ids_batch).logits.float()
    loss_minus = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                  token_ids_batch[:, 1:].contiguous().view(-1))
    loss_minus.backward()
    grad_minus = {name: p.grad.clone() for name, p in model.named_parameters()
                  if p.requires_grad and p.grad is not None}

    # Restore original
    for name, p in model.named_parameters():
        if name in original_params:
            p.data.copy_(original_params[name])

    # H·v = (∇L+ - ∇L-) / (2ε)
    hvp = {}
    for name in grad_plus:
        if name in grad_minus:
            hvp[name] = (grad_plus[name] - grad_minus[name]) / (2 * eps)

    return hvp


def power_iteration_hessian(model, token_ids_list, device, n_eigenvalues=5,
                             n_iterations=20, batch_size=10):
    """Estimate top-k Hessian eigenvalues via power iteration with deflation."""
    # Use a subset of data for Hessian computation
    batch_ids = torch.tensor(np.stack(token_ids_list[:batch_size]),
                             dtype=torch.long, device=device)

    param_names = [name for name, p in model.named_parameters() if p.requires_grad]

    eigenvalues = []
    eigenvectors = []

    for ei in range(n_eigenvalues):
        print(f"    Eigenvalue {ei+1}/{n_eigenvalues}...", flush=True)

        # Random initial vector
        torch.manual_seed(SEED + ei)
        v = {name: torch.randn_like(p) for name, p in model.named_parameters() if p.requires_grad}

        # Normalize
        norm = torch.sqrt(sum((v[n] ** 2).sum() for n in v))
        for n in v: v[n] /= norm

        eigenvalue = 0
        for it in range(n_iterations):
            # H·v
            hvp = hessian_vector_product(model, batch_ids, v, device)

            # Deflate: remove components along previous eigenvectors
            for prev_ev, prev_val in zip(eigenvectors, eigenvalues):
                dot = sum((hvp.get(n, torch.zeros(1)).to(prev_ev[n].device) * prev_ev[n]).sum() for n in prev_ev)
                for n in hvp:
                    if n in prev_ev:
                        hvp[n] -= dot * prev_ev[n].to(hvp[n].device)

            # Eigenvalue estimate: v · H·v
            eigenvalue = sum((v[n].to(hvp[n].device) * hvp[n]).sum().item() for n in v if n in hvp)

            # Update: v = H·v / ||H·v||
            norm = torch.sqrt(sum((hvp[n] ** 2).sum() for n in hvp))
            if norm > 1e-10:
                v = {n: hvp[n] / norm for n in hvp}

            if it % 5 == 0:
                print(f"      iter {it}: λ = {eigenvalue:.4f}", flush=True)

        eigenvalues.append(eigenvalue)
        eigenvectors.append({n: v[n].cpu() for n in v})
        print(f"    λ_{ei+1} = {eigenvalue:.4f}")

    return eigenvalues


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Weight perturbation sharpness
# ═══════════════════════════════════════════════════════════════

def measure_sharpness(model, token_ids_list, device, sigmas, n_samples=5):
    """Add Gaussian noise to weights and measure loss increase.
    For each sigma, sample n_samples perturbations and average."""
    base_loss = compute_ts_loss(model, token_ids_list, device)

    # Compute weight norm for relative scaling
    total_norm = 0
    n_params = 0
    for p in model.parameters():
        total_norm += (p.data ** 2).sum().item()
        n_params += p.numel()
    rms_weight = np.sqrt(total_norm / n_params)

    results = {"base_loss": base_loss, "rms_weight": rms_weight, "sigmas": {}}

    for sigma in sigmas:
        # Scale sigma relative to weight magnitude
        abs_sigma = sigma * rms_weight

        losses = []
        for sample in range(n_samples):
            # Save original weights
            original_state = {name: p.data.clone() for name, p in model.named_parameters()}

            # Add noise
            torch.manual_seed(SEED + sample * 1000 + int(sigma * 10000))
            for p in model.parameters():
                p.data.add_(abs_sigma * torch.randn_like(p))

            # Evaluate
            loss = compute_ts_loss(model, token_ids_list, device)
            losses.append(loss)

            # Restore
            for name, p in model.named_parameters():
                p.data.copy_(original_state[name])

        mean_loss = float(np.mean(losses))
        std_loss = float(np.std(losses))
        results["sigmas"][sigma] = {
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "abs_sigma": abs_sigma,
            "delta": mean_loss - base_loss,
            "relative_delta": (mean_loss - base_loss) / base_loss,
        }
        print(f"    σ={sigma:.4f} (abs={abs_sigma:.4f}): loss={mean_loss:.4f} "
              f"(Δ={mean_loss-base_loss:+.4f}, {(mean_loss-base_loss)/base_loss*100:+.1f}%)")

    return results


def main():
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # Load TS data
    print("Loading TS data...", flush=True)
    ts_windows = load_ts_data(hf_token, n_windows=50)
    eval_windows = ts_windows[:30]
    hessian_windows = ts_windows[:10]  # smaller set for Hessian (expensive)
    print(f"  Loaded {len(ts_windows)} windows")

    # ═══ Experiment 1: Hessian eigenvalues ═══
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: HESSIAN TOP EIGENVALUES")
    print(f"{'='*60}")

    hessian_results = {}

    for init_name in ["PT", "RandomInit"]:
        print(f"\n  {init_name}:")

        if init_name == "PT":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS["PT"], dtype=torch.float32).to(DEVICE)
        else:
            config = AutoConfig.from_pretrained(MODEL_PATHS["PT"])
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32).to(DEVICE)

        model.train()  # need gradients

        # First compute base loss
        base_loss = compute_ts_loss(model, hessian_windows, DEVICE)
        print(f"    Base TS loss: {base_loss:.4f}")

        # Top eigenvalues
        eigenvalues = power_iteration_hessian(
            model, hessian_windows, DEVICE,
            n_eigenvalues=5, n_iterations=15, batch_size=5)

        hessian_results[init_name] = {
            "base_loss": base_loss,
            "eigenvalues": eigenvalues,
        }
        print(f"    Top eigenvalues: {[f'{e:.4f}' for e in eigenvalues]}")

        del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ Experiment 2: Weight perturbation sharpness ═══
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: WEIGHT PERTURBATION SHARPNESS")
    print(f"{'='*60}")

    sigmas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    sharpness_results = {}

    for model_name in ["FT", "RI", "PT"]:
        print(f"\n  {model_name}:")

        if model_name == "PT":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS["PT"], dtype=torch.float32).to(DEVICE).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS[model_name], dtype=torch.float32).to(DEVICE).eval()

        results = measure_sharpness(model, eval_windows, DEVICE, sigmas, n_samples=3)
        sharpness_results[model_name] = results

        del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ Save results ═══
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump({
            "hessian": hessian_results,
            "sharpness": {k: {
                "base_loss": v["base_loss"],
                "rms_weight": v["rms_weight"],
                "sigmas": {str(sk): sv for sk, sv in v["sigmas"].items()}
            } for k, v in sharpness_results.items()},
        }, f, indent=2)

    # ═══ Plots ═══
    print(f"\nGenerating plots...", flush=True)

    # Plot 1: Hessian eigenvalues comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for init_name, color in [("PT", "#2196F3"), ("RandomInit", "#FF9800")]:
        eigs = hessian_results[init_name]["eigenvalues"]
        ax.bar(np.arange(len(eigs)) + (0 if init_name == "PT" else 0.35),
               eigs, 0.35, color=color, label=init_name)
    ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("Hessian eigenvalue")
    ax.set_title("Top-5 Hessian Eigenvalues of TS Loss\n"
                 "Lower = flatter landscape = easier to optimize",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis='y')
    ax.set_xticks(np.arange(5) + 0.175)
    ax.set_xticklabels([f"λ_{i+1}" for i in range(5)])
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/hessian_eigenvalues.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Sharpness comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Weight Perturbation Sharpness: FT vs RI vs PT\n"
                 "Add Gaussian noise σ to weights, measure loss increase",
                 fontsize=13, fontweight='bold')

    for model_name, color in [("FT", "#4CAF50"), ("RI", "#E91E63"), ("PT", "#2196F3")]:
        r = sharpness_results[model_name]
        sigs = sorted(r["sigmas"].keys())
        losses = [r["sigmas"][s]["mean_loss"] for s in sigs]
        stds = [r["sigmas"][s]["std_loss"] for s in sigs]

        axes[0].errorbar(sigs, losses, yerr=stds, fmt='o-', color=color,
                         linewidth=2, markersize=5, capsize=3, label=model_name)

        # Relative delta
        deltas = [r["sigmas"][s]["relative_delta"] * 100 for s in sigs]
        axes[1].plot(sigs, deltas, 'o-', color=color, linewidth=2, markersize=5, label=model_name)

    axes[0].set_xlabel("σ (relative to weight RMS)"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Absolute loss under perturbation")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xscale('log')

    axes[1].set_xlabel("σ (relative to weight RMS)"); axes[1].set_ylabel("Loss increase (%)")
    axes[1].set_title("Relative loss degradation")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/sharpness.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: Combined summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Loss Landscape: Is PT Initialization in a Favorable Region?",
                 fontsize=14, fontweight='bold')

    # Left: Hessian
    for i, (init_name, color) in enumerate([("PT", "#2196F3"), ("RandomInit", "#FF9800")]):
        eigs = hessian_results[init_name]["eigenvalues"]
        axes[0].semilogy(range(1, len(eigs)+1), [abs(e) for e in eigs], 'o-',
                         color=color, linewidth=2, markersize=8, label=init_name)
    axes[0].set_xlabel("Eigenvalue index"); axes[0].set_ylabel("|Hessian eigenvalue| (log)")
    axes[0].set_title("Curvature at initialization\n(lower = flatter)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Right: Sharpness at σ=0.01
    sigma_key = 0.01
    model_names = ["FT", "RI", "PT"]
    colors = [MODEL_COLORS[m] for m in model_names]
    deltas = [sharpness_results[m]["sigmas"][sigma_key]["relative_delta"] * 100 for m in model_names]
    bars = axes[1].bar(model_names, deltas, color=colors)
    for bar, d in zip(bars, deltas):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{d:.1f}%", ha='center', fontsize=11, fontweight='bold')
    axes[1].set_ylabel(f"Loss increase at σ={sigma_key} (%)")
    axes[1].set_title(f"Sharpness of trained solutions\n(higher = sharper minimum)")
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/landscape_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    print(f"\n{'='*70}")
    print("LOSS LANDSCAPE SUMMARY")
    print(f"{'='*70}")

    print(f"\nHessian top eigenvalues (TS loss):")
    for init_name in ["PT", "RandomInit"]:
        eigs = hessian_results[init_name]["eigenvalues"]
        print(f"  {init_name}: {[f'{e:.4f}' for e in eigs]}  (loss={hessian_results[init_name]['base_loss']:.4f})")

    print(f"\nSharpness (σ=0.01):")
    for m in ["FT", "RI", "PT"]:
        r = sharpness_results[m]
        s = r["sigmas"][0.01]
        print(f"  {m}: base={r['base_loss']:.4f} perturbed={s['mean_loss']:.4f} "
              f"(Δ={s['relative_delta']*100:+.1f}%) rms_weight={r['rms_weight']:.4f}")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
