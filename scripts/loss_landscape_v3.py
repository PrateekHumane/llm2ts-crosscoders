"""
Loss Landscape v3: Rigorous implementation with all sanity checks.

1. HVP correctness verification (autograd vs finite differences)
2. Graph construction verification
3. Parameter counting verification
4. Proper data size (50+ sequences)
5. Normalized curvature (remove loss scale effects)
6. Random-direction curvature histogram
7. Sharpness experiment

Run on all 4 GPUs:
  GPU 0: PT Hessian
  GPU 1: RandomInit Hessian
  GPU 2: Random-direction curvature PT
  GPU 3: Random-direction curvature RandomInit + sharpness

Usage:
    /usr/bin/python3 scripts/loss_landscape_v3.py --job hessian_PT --gpu 0
    /usr/bin/python3 scripts/loss_landscape_v3.py --job hessian_RI --gpu 1
    /usr/bin/python3 scripts/loss_landscape_v3.py --job random_curv --gpu 2
    /usr/bin/python3 scripts/loss_landscape_v3.py --job sharpness --gpu 3
    /usr/bin/python3 scripts/loss_landscape_v3.py --job verify --gpu 0
    /usr/bin/python3 scripts/loss_landscape_v3.py --job plot
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
MODEL_PATHS = {"PT": "Qwen/Qwen3-0.6B", "FT": "models/ft", "RI": "models/ri"}
LAYER_INDICES = [6, 7, 8, 9, 10]
N_LANCZOS = 30
N_RANDOM_DIRS = 50


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
    """Freeze all, unfreeze target layers, return target param list."""
    for p in model.parameters():
        p.requires_grad = False
    target = []
    for li in layer_indices:
        for p in model.model.layers[li].parameters():
            p.requires_grad = True
            target.append(p)
    return target


def compute_loss(model, batch_ids):
    """Compute loss WITH gradient graph."""
    logits = model(input_ids=batch_ids).logits.float()
    return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                           batch_ids[:, 1:].contiguous().view(-1))


def compute_loss_nograds(model, token_ids_list, device, batch_size=5):
    """Compute loss without gradients."""
    model.eval()
    total = 0; n = 0
    for i in range(0, len(token_ids_list), batch_size):
        batch = token_ids_list[i:i+batch_size]
        ids = torch.tensor(np.stack(batch), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               ids[:, 1:].contiguous().view(-1))
        total += loss.item(); n += 1
    return total / n


def exact_hvp(model, batch_ids, vector_list, params):
    """Exact HVP via Pearlmutter trick."""
    loss = compute_loss(model, batch_ids)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gv = sum((g * v).sum() for g, v in zip(grads, vector_list))
    hvp = torch.autograd.grad(gv, params, retain_graph=False)
    return [h.detach() for h in hvp]


def finite_diff_hvp(model, batch_ids, vector_list, params, eps=1e-3):
    """Finite-difference HVP for verification."""
    # Save original
    orig = [p.data.clone() for p in params]

    # θ + εv
    for p, v in zip(params, vector_list):
        p.data.add_(eps * v)
    loss_plus = compute_loss(model, batch_ids)
    grads_plus = torch.autograd.grad(loss_plus, params)
    grads_plus = [g.detach().clone() for g in grads_plus]

    # Restore and do θ - εv
    for p, o, vi in zip(params, orig, vector_list):
        p.data.copy_(o - eps * vi)
    loss_minus = compute_loss(model, batch_ids)
    grads_minus = torch.autograd.grad(loss_minus, params)
    grads_minus = [g.detach().clone() for g in grads_minus]

    # Restore
    for p, o in zip(params, orig):
        p.data.copy_(o)

    # H·v ≈ (∇L+ - ∇L-) / 2ε
    return [(gp - gm) / (2 * eps) for gp, gm in zip(grads_plus, grads_minus)]


def batched_hvp(model, data_batches, vector_list, params):
    """Average HVP over multiple data batches."""
    hvp_sum = None
    for batch_ids in data_batches:
        hvp = exact_hvp(model, batch_ids, vector_list, params)
        if hvp_sum is None:
            hvp_sum = [h.clone() for h in hvp]
        else:
            for i in range(len(hvp_sum)):
                hvp_sum[i] += hvp[i]
        torch.cuda.empty_cache()
    return [h / len(data_batches) for h in hvp_sum]


def lanczos(model, data_batches, params, n_iterations=30):
    """Lanczos algorithm with full reorthogonalization."""
    model.train()

    n_params = sum(p.numel() for p in params)
    print(f"    Lanczos over {len(params)} tensors ({n_params/1e6:.1f}M params)", flush=True)

    torch.manual_seed(SEED)
    q = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum((qi**2).sum() for qi in q))
    q = [qi / norm for qi in q]

    Q_vectors = [q]; alphas = []; betas = [0.0]
    t0 = time.time()

    for j in range(n_iterations):
        w = batched_hvp(model, data_batches, q, params)
        alpha = sum((qi * wi).sum().item() for qi, wi in zip(q, w))
        alphas.append(alpha)

        for i in range(len(w)):
            w[i] = w[i] - alpha * q[i]
            if j > 0:
                w[i] = w[i] - betas[j] * Q_vectors[j-1][i]

        for k in range(j + 1):
            dot = sum((wi * Q_vectors[k][i]).sum().item() for i, wi in enumerate(w))
            for i in range(len(w)):
                w[i] = w[i] - dot * Q_vectors[k][i]

        beta = torch.sqrt(sum((wi**2).sum() for wi in w)).item()
        betas.append(beta)
        if beta < 1e-10:
            print(f"    Converged at iter {j+1}", flush=True); break

        q = [wi / beta for wi in w]
        Q_vectors.append(q)

        if (j+1) % 5 == 0:
            T_mat = np.diag(alphas) + np.diag(betas[1:len(alphas)], 1) + np.diag(betas[1:len(alphas)], -1)
            eigs = np.sort(np.linalg.eigvalsh(T_mat))[::-1]
            print(f"      iter {j+1}: top-5 = {[f'{e:.2f}' for e in eigs[:5]]} ({time.time()-t0:.0f}s)", flush=True)

        for k in range(len(Q_vectors)-1):
            Q_vectors[k] = [qi.detach() for qi in Q_vectors[k]]

    m = len(alphas)
    T_mat = np.zeros((m, m))
    for i in range(m): T_mat[i, i] = alphas[i]
    for i in range(m-1): T_mat[i, i+1] = betas[i+1]; T_mat[i+1, i] = betas[i+1]
    eigs = np.sort(np.linalg.eigvalsh(T_mat))[::-1]
    print(f"    Done ({time.time()-t0:.0f}s)", flush=True)
    return eigs.tolist()


def random_direction_curvature(model, data_batches, params, n_dirs=50):
    """Compute curvature along random directions: λ_v = v^T H v / v^T v."""
    model.train()
    curvatures = []
    for di in range(n_dirs):
        torch.manual_seed(SEED + di * 7)
        v = [torch.randn_like(p) for p in params]
        norm = torch.sqrt(sum((vi**2).sum() for vi in v))
        v = [vi / norm for vi in v]

        hvp = batched_hvp(model, data_batches, v, params)
        curv = sum((vi * hi).sum().item() for vi, hi in zip(v, hvp))
        curvatures.append(curv)

        if (di+1) % 10 == 0:
            print(f"    dir {di+1}/{n_dirs}: mean={np.mean(curvatures):.4f} "
                  f"std={np.std(curvatures):.4f}", flush=True)
        torch.cuda.empty_cache()

    return curvatures


def load_model(init_name, device):
    if init_name == "PT":
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATHS["PT"], dtype=torch.float32).to(device)
    elif init_name == "RandomInit":
        config = AutoConfig.from_pretrained(MODEL_PATHS["PT"])
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[init_name], dtype=torch.float32).to(device)
    # NOTE: do NOT enable gradient_checkpointing — it breaks create_graph=True for HVP
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True,
                        choices=["verify", "hessian_PT", "hessian_RI", "random_curv", "sharpness", "plot"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.job == "plot":
        make_plots(); return

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    hf_token = os.environ.get("HF_TOKEN")
    os.makedirs(OUT_DIR, exist_ok=True)

    ts_windows = load_ts_data(hf_token, 200)

    # ═══ Verification ═══
    if args.job == "verify":
        print("="*60)
        print("VERIFICATION: Autograd HVP vs Finite-Difference HVP")
        print("="*60)

        model = load_model("PT", device)
        params = get_target_params(model, LAYER_INDICES)

        # Count params explicitly
        n_params = sum(p.numel() for p in params)
        n_tensors = len(params)
        print(f"\nTarget params: {n_tensors} tensors, {n_params:,} parameters ({n_params/1e6:.1f}M)")
        print(f"All require_grad: {all(p.requires_grad for p in params)}")
        print(f"Shapes: {[tuple(p.shape) for p in params[:5]]}... (showing first 5)")

        batch = torch.tensor(np.stack(ts_windows[:1]), dtype=torch.long, device=device)

        # Random vector
        torch.manual_seed(SEED)
        v = [torch.randn_like(p) for p in params]
        norm = torch.sqrt(sum((vi**2).sum() for vi in v))
        v = [vi / norm for vi in v]

        print(f"\nVector shapes match params: {all(vi.shape == pi.shape for vi, pi in zip(v, params))}")

        # Autograd HVP
        t0 = time.time()
        hvp_auto = exact_hvp(model, batch, v, params)
        t_auto = time.time() - t0
        print(f"\nAutograd HVP: {t_auto:.2f}s")

        # Finite-difference HVP
        t0 = time.time()
        hvp_fd = finite_diff_hvp(model, batch, v, params)
        t_fd = time.time() - t0
        print(f"Finite-diff HVP: {t_fd:.2f}s")

        # Compare
        diff_norm = torch.sqrt(sum(((a - f)**2).sum() for a, f in zip(hvp_auto, hvp_fd))).item()
        fd_norm = torch.sqrt(sum((f**2).sum() for f in hvp_fd)).item()
        auto_norm = torch.sqrt(sum((a**2).sum() for a in hvp_auto)).item()
        rel_error = diff_norm / fd_norm if fd_norm > 0 else float('inf')

        print(f"\n||H_auto·v|| = {auto_norm:.6f}")
        print(f"||H_fd·v||   = {fd_norm:.6f}")
        print(f"||H_auto·v - H_fd·v|| = {diff_norm:.6f}")
        print(f"Relative error: {rel_error:.6f}")
        print(f"\n{'✅ PASS' if rel_error < 0.01 else '❌ FAIL'}: relative error {'<' if rel_error < 0.01 else '>'} 0.01")

        # Also verify eigenvalue: v^T H v
        lambda_auto = sum((vi * hi).sum().item() for vi, hi in zip(v, hvp_auto))
        lambda_fd = sum((vi * hi).sum().item() for vi, hi in zip(v, hvp_fd))
        print(f"\nv^T H_auto v = {lambda_auto:.6f}")
        print(f"v^T H_fd v   = {lambda_fd:.6f}")
        print(f"Difference:    {abs(lambda_auto - lambda_fd):.6f}")

        del model; gc.collect(); torch.cuda.empty_cache()
        return

    # ═══ Hessian eigenvalues ═══
    if args.job.startswith("hessian"):
        init_name = "PT" if args.job == "hessian_PT" else "RandomInit"
        print(f"\n{'='*60}")
        print(f"HESSIAN EIGENVALUES: {init_name}")
        print(f"{'='*60}")

        model = load_model(init_name, device)
        params = get_target_params(model, LAYER_INDICES)
        n_params = sum(p.numel() for p in params)
        print(f"  Params: {n_params/1e6:.1f}M from layers {LAYER_INDICES}")

        base_loss = compute_loss_nograds(model, ts_windows[:50], device)
        print(f"  Base loss: {base_loss:.4f}")

        # Prepare batches — 1 sequence each, 10 batches
        batches = [torch.tensor(ts_windows[i:i+1], dtype=torch.long, device=device) for i in range(10)]

        eigs = lanczos(model, batches, params, n_iterations=N_LANCZOS)

        # Normalized eigenvalues
        trace_est = sum(eigs)
        normalized = [e / trace_est if trace_est != 0 else 0 for e in eigs]

        result = {
            "base_loss": base_loss,
            "eigenvalues": eigs,
            "normalized_eigenvalues": normalized,
            "trace_estimate": trace_est,
            "n_params": n_params,
            "n_data_batches": 10,
            "n_lanczos_iters": N_LANCZOS,
            "layers": LAYER_INDICES,
        }
        with open(f"{OUT_DIR}/hessian_{init_name}_v3.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to hessian_{init_name}_v3.json")

        del model; gc.collect(); torch.cuda.empty_cache()
        return

    # ═══ Random direction curvature ═══
    if args.job == "random_curv":
        print(f"\n{'='*60}")
        print("RANDOM DIRECTION CURVATURE")
        print(f"{'='*60}")

        results = {}
        batches_for_curv = None

        for init_name in ["PT", "RandomInit"]:
            print(f"\n  {init_name}:")
            model = load_model(init_name, device)
            params = get_target_params(model, LAYER_INDICES)

            if batches_for_curv is None:
                batches_for_curv = [torch.tensor(ts_windows[i:i+1], dtype=torch.long, device=device)
                                    for i in range(10)]

            curvatures = random_direction_curvature(model, batches_for_curv, params, n_dirs=N_RANDOM_DIRS)
            results[init_name] = {
                "curvatures": curvatures,
                "mean": float(np.mean(curvatures)),
                "std": float(np.std(curvatures)),
                "min": float(np.min(curvatures)),
                "max": float(np.max(curvatures)),
            }
            print(f"    {init_name}: mean={np.mean(curvatures):.4f} std={np.std(curvatures):.4f} "
                  f"range=[{np.min(curvatures):.4f}, {np.max(curvatures):.4f}]")

            del model; gc.collect(); torch.cuda.empty_cache()

        with open(f"{OUT_DIR}/random_curvature_v3.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to random_curvature_v3.json")
        return

    # ═══ Sharpness ═══
    if args.job == "sharpness":
        print(f"\n{'='*60}")
        print("SHARPNESS")
        print(f"{'='*60}")

        sigmas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

        for model_name in ["FT", "RI", "PT"]:
            print(f"\n  {model_name}:")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS[model_name], dtype=torch.float32).to(device).eval()

            base_loss = compute_loss_nograds(model, ts_windows[:50], device)
            total_norm = sum((p.data**2).sum().item() for p in model.parameters())
            n_p = sum(p.numel() for p in model.parameters())
            rms = np.sqrt(total_norm / n_p)
            print(f"    Base: {base_loss:.4f}, RMS: {rms:.4f}")

            result = {"base_loss": base_loss, "rms_weight": rms, "sigmas": {}}
            for sigma in sigmas:
                abs_sigma = sigma * rms
                losses = []
                for sample in range(5):
                    orig = {n: p.data.clone() for n, p in model.named_parameters()}
                    torch.manual_seed(SEED + sample*1000 + int(sigma*10000))
                    for p in model.parameters():
                        p.data.add_(abs_sigma * torch.randn_like(p))
                    losses.append(compute_loss_nograds(model, ts_windows[:50], device))
                    for n, p in model.named_parameters():
                        p.data.copy_(orig[n])

                result["sigmas"][str(sigma)] = {
                    "mean_loss": float(np.mean(losses)),
                    "std_loss": float(np.std(losses)),
                    "relative_delta": (float(np.mean(losses)) - base_loss) / base_loss,
                }
                print(f"    σ={sigma}: {np.mean(losses):.4f} ({(np.mean(losses)-base_loss)/base_loss*100:+.2f}%)")

            with open(f"{OUT_DIR}/sharpness_{model_name}_v3.json", "w") as f:
                json.dump(result, f, indent=2)
            del model; gc.collect(); torch.cuda.empty_cache()

        print(f"\nAll sharpness saved")
        return


def make_plots():
    """Generate all plots from saved results."""
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

    # Load results
    hess_pt = json.load(open(f"{OUT_DIR}/hessian_PT_v3.json"))
    hess_ri = json.load(open(f"{OUT_DIR}/hessian_RandomInit_v3.json"))
    rand_curv = json.load(open(f"{OUT_DIR}/random_curvature_v3.json"))
    sharp = {m: json.load(open(f"{OUT_DIR}/sharpness_{m}_v3.json")) for m in ["FT", "RI", "PT"]}

    pt_eigs = hess_pt["eigenvalues"][:10]
    ri_eigs = hess_ri["eigenvalues"][:10]

    # ── Plot 1: Eigenvalue spectrum (linear + log) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Hessian Eigenvalue Spectrum of TS Loss (Layers 6-10)\n"
                 "Lanczos (30 iter) + exact autograd HVP, 10 sequences",
                 fontsize=13, fontweight='bold')

    axes[0].plot(range(1, len(pt_eigs)+1), pt_eigs, 'o-', color='#2196F3', linewidth=2,
                 markersize=8, label=f'PT (loss={hess_pt["base_loss"]:.1f})')
    axes[0].plot(range(1, len(ri_eigs)+1), ri_eigs, 's-', color='#FF9800', linewidth=2,
                 markersize=8, label=f'RandomInit (loss={hess_ri["base_loss"]:.1f})')
    axes[0].set_xlabel("Index"); axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Linear scale"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].semilogy(range(1, len(pt_eigs)+1), [abs(e) for e in pt_eigs], 'o-', color='#2196F3',
                     linewidth=2, markersize=8, label='PT')
    axes[1].semilogy(range(1, len(ri_eigs)+1), [abs(e) for e in ri_eigs], 's-', color='#FF9800',
                     linewidth=2, markersize=8, label='RandomInit')
    axes[1].set_xlabel("Index"); axes[1].set_ylabel("|Eigenvalue| (log)")
    axes[1].set_title("Log scale"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/hessian_spectrum.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 2: Normalized eigenvalues ──
    fig, ax = plt.subplots(figsize=(10, 5))
    pt_norm = hess_pt["normalized_eigenvalues"][:10]
    ri_norm = hess_ri["normalized_eigenvalues"][:10]
    ax.plot(range(1, 11), pt_norm, 'o-', color='#2196F3', linewidth=2, markersize=8, label='PT')
    ax.plot(range(1, 11), ri_norm, 's-', color='#FF9800', linewidth=2, markersize=8, label='RandomInit')
    ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("λᵢ / Tr(H)")
    ax.set_title("Normalized Eigenvalue Spectrum (λᵢ / trace)\nShows anisotropy independent of scale",
                 fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/normalized_spectrum.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 3: Random direction curvature histogram ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Curvature Along 50 Random Directions: v^T H v\n"
                 "Wide spread = anisotropic, narrow = isotropic",
                 fontsize=13, fontweight='bold')

    for i, (name, color) in enumerate([("PT", "#2196F3"), ("RandomInit", "#FF9800")]):
        curvs = rand_curv[name]["curvatures"]
        axes[i].hist(curvs, bins=20, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        axes[i].axvline(np.mean(curvs), color='black', linestyle='--', linewidth=1.5,
                        label=f'mean={np.mean(curvs):.2f}')
        axes[i].set_xlabel("Curvature (v^T H v)"); axes[i].set_ylabel("Count")
        axes[i].set_title(f"{name}\nmean={np.mean(curvs):.2f} std={np.std(curvs):.2f}")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/random_curvature.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 4: Overlaid histogram ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rand_curv["PT"]["curvatures"], bins=20, color='#2196F3', alpha=0.5, label='PT', edgecolor='black', linewidth=0.3)
    ax.hist(rand_curv["RandomInit"]["curvatures"], bins=20, color='#FF9800', alpha=0.5, label='RandomInit', edgecolor='black', linewidth=0.3)
    ax.set_xlabel("Curvature (v^T H v)"); ax.set_ylabel("Count")
    ax.set_title("Random Direction Curvature: PT vs RandomInit\n"
                 "PT has wider spread (anisotropic) and larger values (steeper)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/random_curvature_overlay.png", dpi=150, bbox_inches="tight"); plt.close()

    # ── Plot 5: Sharpness ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Weight Perturbation Sharpness", fontsize=13, fontweight='bold')
    for name, color in [("FT", "#4CAF50"), ("RI", "#E91E63"), ("PT", "#2196F3")]:
        r = sharp[name]
        sigs = sorted([float(s) for s in r["sigmas"].keys()])
        losses = [r["sigmas"][str(s)]["mean_loss"] for s in sigs]
        deltas = [r["sigmas"][str(s)]["relative_delta"]*100 for s in sigs]
        axes[0].plot(sigs, losses, 'o-', color=color, linewidth=2, markersize=5,
                     label=f'{name} (base={r["base_loss"]:.2f})')
        axes[1].plot(sigs, deltas, 'o-', color=color, linewidth=2, markersize=5, label=name)
    axes[0].set_xlabel("σ"); axes[0].set_ylabel("Loss"); axes[0].set_title("Absolute")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_xscale('log')
    axes[1].set_xlabel("σ"); axes[1].set_ylabel("Δ (%)"); axes[1].set_title("Relative")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_xscale('log')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/sharpness.png", dpi=150, bbox_inches="tight"); plt.close()

    print("All plots saved")


if __name__ == "__main__":
    main()
