"""
Loss Landscape Analysis v2: Proper implementation.

Fixes:
  1. Exact Hessian-vector products via autograd (Pearlmutter trick)
  2. Lanczos algorithm for stable top-k eigenvalue estimation
  3. Proper data size (100+ sequences, batched HVP averaging)

Experiment 1: Hessian eigenvalues at PT init vs RandomInit
Experiment 2: Weight perturbation sharpness for FT vs RI

Usage:
    /usr/bin/python3 scripts/loss_landscape_v2.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, glob, time

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

MODEL_PATHS = {"PT": "Qwen/Qwen3-0.6B", "FT": "models/ft", "RI": "models/ri"}
MODEL_COLORS = {"PT": "#2196F3", "FT": "#4CAF50", "RI": "#E91E63", "RandomInit": "#FF9800"}


def load_ts_data(hf_token, n_windows=200):
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
    model.eval()
    total_loss = 0; n_batches = 0
    for i in range(0, len(token_ids_list), batch_size):
        batch = token_ids_list[i:i + batch_size]
        ids = torch.tensor(np.stack(batch), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               ids[:, 1:].contiguous().view(-1))
        total_loss += loss.item(); n_batches += 1
    return total_loss / n_batches


def compute_loss_with_grad(model, batch_ids):
    """Compute loss with gradient graph intact for autograd HVP."""
    logits = model(input_ids=batch_ids).logits.float()
    loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                           batch_ids[:, 1:].contiguous().view(-1))
    return loss


# ═══════════════════════════════════════════════════════════════
# Exact Hessian-vector product via autograd
# ═══════════════════════════════════════════════════════════════

def exact_hvp(model, batch_ids, vector_list):
    """Compute exact H·v using the Pearlmutter trick.

    vector_list: list of tensors, same shapes as model.parameters()
    Returns: list of tensors (H·v components)
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # Forward pass with gradient graph
    loss = compute_loss_with_grad(model, batch_ids)

    # First derivative: ∇L
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # g · v (scalar)
    gv = sum((g * v).sum() for g, v in zip(grads, vector_list) if g is not None)

    # Second derivative: ∇(g·v) = H·v
    hvp = torch.autograd.grad(gv, params, retain_graph=False)

    return [h.detach() for h in hvp]


def batched_hvp(model, data_batches, vector_list):
    """Average HVP over multiple data batches for better estimates."""
    n = len(data_batches)
    hvp_sum = None

    for batch_ids in data_batches:
        hvp = exact_hvp(model, batch_ids, vector_list)
        if hvp_sum is None:
            hvp_sum = [h.clone() for h in hvp]
        else:
            for i in range(len(hvp_sum)):
                hvp_sum[i] += hvp[i]

    return [h / n for h in hvp_sum]


# ═══════════════════════════════════════════════════════════════
# Lanczos algorithm for top-k eigenvalues
# ═══════════════════════════════════════════════════════════════

def lanczos(model, data_batches, n_eigenvalues=10, n_iterations=50):
    """Lanczos algorithm to find top-k Hessian eigenvalues.

    More stable than power iteration with deflation.
    Builds a tridiagonal matrix T from Lanczos vectors,
    whose eigenvalues approximate the Hessian's extremal eigenvalues.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    model.train()

    # Random initial vector
    torch.manual_seed(SEED)
    q = [torch.randn_like(p) for p in params]
    # Normalize
    norm = torch.sqrt(sum((qi ** 2).sum() for qi in q))
    q = [qi / norm for qi in q]

    # Lanczos vectors and tridiagonal matrix entries
    Q_vectors = [q]  # store Lanczos vectors
    alphas = []  # diagonal of T
    betas = [0.0]   # off-diagonal of T

    n_iter = min(n_iterations, sum(p.numel() for p in params))  # can't exceed param count
    n_iter = min(n_iter, n_iterations)

    print(f"    Lanczos: {n_iter} iterations...", flush=True)
    t0 = time.time()

    for j in range(n_iter):
        # w = H·q_j
        w = batched_hvp(model, data_batches, q)

        # alpha_j = q_j · w
        alpha = sum((qi * wi).sum().item() for qi, wi in zip(q, w))
        alphas.append(alpha)

        # w = w - alpha_j * q_j - beta_j * q_{j-1}
        for i in range(len(w)):
            w[i] = w[i] - alpha * q[i]
            if j > 0:
                w[i] = w[i] - betas[j] * Q_vectors[j - 1][i]

        # Full reorthogonalization (important for stability)
        for k in range(j + 1):
            dot = sum((wi * Q_vectors[k][i]).sum().item() for i, wi in enumerate(w))
            for i in range(len(w)):
                w[i] = w[i] - dot * Q_vectors[k][i]

        # beta_{j+1} = ||w||
        beta = torch.sqrt(sum((wi ** 2).sum() for wi in w)).item()
        betas.append(beta)

        if beta < 1e-10:
            print(f"    Lanczos converged at iteration {j+1} (beta → 0)", flush=True)
            break

        # q_{j+1} = w / beta
        q = [wi / beta for wi in w]
        Q_vectors.append(q)

        if (j + 1) % 10 == 0:
            # Current eigenvalue estimates from tridiagonal matrix
            T_mat = np.diag(alphas) + np.diag(betas[1:len(alphas)], 1) + np.diag(betas[1:len(alphas)], -1)
            eigs = np.sort(np.linalg.eigvalsh(T_mat))[::-1]
            top = eigs[:min(3, len(eigs))]
            print(f"      iter {j+1}/{n_iter}: top eigs = {[f'{e:.2f}' for e in top]} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # Free old Lanczos vectors to save memory (keep last 2 for reorthogonalization)
        # Actually keep all for full reorthogonalization — but detach from graph
        for k in range(len(Q_vectors) - 1):
            Q_vectors[k] = [qi.detach() for qi in Q_vectors[k]]

    # Build tridiagonal matrix and get eigenvalues
    m = len(alphas)
    T_mat = np.zeros((m, m))
    for i in range(m):
        T_mat[i, i] = alphas[i]
    for i in range(m - 1):
        T_mat[i, i + 1] = betas[i + 1]
        T_mat[i + 1, i] = betas[i + 1]

    eigs = np.sort(np.linalg.eigvalsh(T_mat))[::-1]
    print(f"    Lanczos done ({time.time()-t0:.0f}s): top-{n_eigenvalues} = "
          f"{[f'{e:.2f}' for e in eigs[:n_eigenvalues]]}", flush=True)

    return eigs[:n_eigenvalues].tolist(), eigs.tolist()


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Sharpness (same as v1, it was fine)
# ═══════════════════════════════════════════════════════════════

def measure_sharpness(model, token_ids_list, device, sigmas, n_samples=5):
    base_loss = compute_ts_loss(model, token_ids_list, device)
    total_norm = sum((p.data ** 2).sum().item() for p in model.parameters())
    n_params = sum(p.numel() for p in model.parameters())
    rms_weight = np.sqrt(total_norm / n_params)

    results = {"base_loss": base_loss, "rms_weight": rms_weight, "sigmas": {}}

    for sigma in sigmas:
        abs_sigma = sigma * rms_weight
        losses = []
        for sample in range(n_samples):
            original_state = {name: p.data.clone() for name, p in model.named_parameters()}
            torch.manual_seed(SEED + sample * 1000 + int(sigma * 10000))
            for p in model.parameters():
                p.data.add_(abs_sigma * torch.randn_like(p))
            loss = compute_ts_loss(model, token_ids_list, device)
            losses.append(loss)
            for name, p in model.named_parameters():
                p.data.copy_(original_state[name])

        results["sigmas"][sigma] = {
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "abs_sigma": abs_sigma,
            "delta": float(np.mean(losses)) - base_loss,
            "relative_delta": (float(np.mean(losses)) - base_loss) / base_loss,
        }
        print(f"    σ={sigma:.4f}: loss={np.mean(losses):.4f} "
              f"({(np.mean(losses)-base_loss)/base_loss*100:+.1f}%)")

    return results


def main():
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # Load TS data — use more sequences for proper estimates
    print("Loading TS data...", flush=True)
    ts_windows = load_ts_data(hf_token, n_windows=200)
    print(f"  Loaded {len(ts_windows)} windows")

    hessian_windows = ts_windows[:100]
    eval_windows = ts_windows[:50]

    # Prepare batches for Hessian (multiple batches, averaged)
    HESSIAN_BS = 10
    hessian_batches = [
        torch.tensor(np.stack(hessian_windows[i:i + HESSIAN_BS]),
                     dtype=torch.long, device=DEVICE)
        for i in range(0, min(50, len(hessian_windows)), HESSIAN_BS)
    ]
    print(f"  Hessian: {len(hessian_batches)} batches of {HESSIAN_BS}")

    # ═══ Experiment 1: Hessian eigenvalues ═══
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: HESSIAN TOP EIGENVALUES (Lanczos + exact HVP)")
    print(f"{'='*60}")

    hessian_results = {}
    N_LANCZOS = 30  # Lanczos iterations
    N_EIGS = 10     # eigenvalues to report

    for init_name in ["PT", "RandomInit"]:
        print(f"\n  {init_name}:")
        if init_name == "PT":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS["PT"], dtype=torch.float32).to(DEVICE)
        else:
            config = AutoConfig.from_pretrained(MODEL_PATHS["PT"])
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32).to(DEVICE)

        base_loss = compute_ts_loss(model, hessian_windows[:50], DEVICE)
        print(f"    Base TS loss: {base_loss:.4f}")

        model.train()
        top_eigs, all_eigs = lanczos(model, hessian_batches, n_eigenvalues=N_EIGS,
                                      n_iterations=N_LANCZOS)

        hessian_results[init_name] = {
            "base_loss": base_loss,
            "top_eigenvalues": top_eigs,
            "all_lanczos_eigenvalues": all_eigs,
        }

        del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ Experiment 2: Sharpness ═══
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: WEIGHT PERTURBATION SHARPNESS")
    print(f"{'='*60}")

    sigmas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    sharpness_results = {}

    for model_name in ["FT", "RI", "PT"]:
        print(f"\n  {model_name}:")
        if model_name in MODEL_PATHS:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS[model_name], dtype=torch.float32).to(DEVICE).eval()
        results = measure_sharpness(model, eval_windows, DEVICE, sigmas, n_samples=5)
        sharpness_results[model_name] = results
        del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ Save ═══
    with open(f"{OUT_DIR}/results_v2.json", "w") as f:
        json.dump({
            "hessian": hessian_results,
            "sharpness": {k: {
                "base_loss": v["base_loss"], "rms_weight": v["rms_weight"],
                "sigmas": {str(sk): sv for sk, sv in v["sigmas"].items()}
            } for k, v in sharpness_results.items()},
        }, f, indent=2)

    # ═══ Plots ═══
    print(f"\nGenerating plots...", flush=True)

    # Plot 1: Hessian eigenvalue spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hessian Eigenvalue Spectrum of TS Loss at Initialization\n"
                 "(Lanczos algorithm with exact autograd HVP, averaged over 5 batches)",
                 fontsize=13, fontweight='bold')

    for init_name, color in [("PT", "#2196F3"), ("RandomInit", "#FF9800")]:
        eigs = hessian_results[init_name]["top_eigenvalues"]
        axes[0].plot(range(1, len(eigs)+1), eigs, 'o-', color=color, linewidth=2,
                     markersize=8, label=f'{init_name} (loss={hessian_results[init_name]["base_loss"]:.1f})')
        axes[1].semilogy(range(1, len(eigs)+1), [abs(e) for e in eigs], 'o-', color=color,
                         linewidth=2, markersize=8, label=init_name)

    axes[0].set_xlabel("Eigenvalue index"); axes[0].set_ylabel("Hessian eigenvalue")
    axes[0].set_title("Linear scale"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Eigenvalue index"); axes[1].set_ylabel("|eigenvalue| (log)")
    axes[1].set_title("Log scale"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/hessian_eigenvalues.png", dpi=150, bbox_inches="tight"); plt.close()

    # Plot 2: Sharpness
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Weight Perturbation Sharpness: FT vs RI vs PT\n"
                 "Gaussian noise σ·ε added to all weights (σ relative to weight RMS)",
                 fontsize=13, fontweight='bold')

    for model_name, color in [("FT", "#4CAF50"), ("RI", "#E91E63"), ("PT", "#2196F3")]:
        r = sharpness_results[model_name]
        sigs = sorted(r["sigmas"].keys())
        losses = [r["sigmas"][s]["mean_loss"] for s in sigs]
        stds = [r["sigmas"][s]["std_loss"] for s in sigs]
        deltas = [r["sigmas"][s]["relative_delta"]*100 for s in sigs]

        axes[0].errorbar(sigs, losses, yerr=stds, fmt='o-', color=color, linewidth=2,
                         markersize=5, capsize=3, label=f'{model_name} (base={r["base_loss"]:.2f})')
        axes[1].plot(sigs, deltas, 'o-', color=color, linewidth=2, markersize=5, label=model_name)

    axes[0].set_xlabel("σ (relative)"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Absolute loss"); axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xscale('log')
    axes[1].set_xlabel("σ (relative)"); axes[1].set_ylabel("Loss increase (%)")
    axes[1].set_title("Relative degradation"); axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xscale('log')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/sharpness.png", dpi=150, bbox_inches="tight"); plt.close()

    # Summary
    print(f"\n{'='*70}")
    print("LOSS LANDSCAPE SUMMARY")
    print(f"{'='*70}")

    print(f"\nHessian top eigenvalues:")
    for init_name in ["PT", "RandomInit"]:
        eigs = hessian_results[init_name]["top_eigenvalues"]
        print(f"  {init_name}: {[f'{e:.2f}' for e in eigs[:5]]}  "
              f"(loss={hessian_results[init_name]['base_loss']:.4f})")

    print(f"\nSharpness (σ=0.01):")
    for m in ["FT", "RI", "PT"]:
        r = sharpness_results[m]
        s = r["sigmas"][0.01]
        print(f"  {m}: base={r['base_loss']:.4f} → {s['mean_loss']:.4f} "
              f"({s['relative_delta']*100:+.1f}%)")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
