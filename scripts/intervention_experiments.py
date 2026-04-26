"""
Intervention Experiments: Testing the geometric transfer hypothesis.

A. Rank-constrained FT: project hidden states onto top-K PCs at layer L,
   continue forward pass. Does low-rank preserve prediction accuracy?

B. Cross-model direction transfer: project PT's hidden states onto FT's
   learned PC directions, inject into FT's remaining layers. Does PT
   already have the right information in FT's subspace?

C. Controls: random directions, RI's directions.

Metric: cross-entropy loss on next-token prediction (model's native objective).

Usage:
    /usr/bin/python3 scripts/intervention_experiments.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, glob, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = 512
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/interventions"
SEED = 42
N_TS = 50  # number of TS windows for evaluation

# Layers to intervene at
INTERVENTION_LAYERS = [4, 8, 12, 16]

# Rank values to test
RANK_VALUES = [2, 5, 10, 15, 20, 30, 50, 100]

MODEL_PATHS = {"PT": "Qwen/Qwen3-0.6B", "FT": "models/ft", "RI": "models/ri"}


def load_ts_windows(hf_token, n_windows):
    """Load TS windows, bin-tokenized."""
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
                bins = ((np.clip(w, -5, 5) + 5) / 10 * 512).astype(np.int64).clip(0, 511)
                windows.append(bins)
                if len(windows) >= n_windows:
                    return windows
    return windows


def compute_baseline_loss(model, lm_head, token_ids_batch, device):
    """Compute cross-entropy loss without intervention (baseline)."""
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(input_ids=token_ids_batch, use_cache=False)
        h_final = output.last_hidden_state  # (B, T, D) — this is after final norm
        # Use the LM head to get logits
        logits = lm_head(h_final).float()  # (B, T, vocab)
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids_batch[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), reduction='mean')
    return loss.item()


def intervened_forward(model, lm_head, token_ids_batch, intervention_layer,
                       projection_matrix, device):
    """Forward pass with rank-K intervention at a specific layer.

    projection_matrix: (D, D) — projects hidden states onto a subspace.
                       For rank-K: P = V_k @ V_k.T where V_k are top-K PCs.
    """
    captured_pre = {}
    captured_post = {}

    # Hook to capture and replace hidden states at intervention layer
    def make_intervention_hook(proj_mat):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Center, project, uncenter
            h_mean = h.mean(dim=1, keepdim=True)
            h_centered = h - h_mean
            h_projected = h_centered.float() @ proj_mat.to(h.device)
            h_projected = h_projected.to(h.dtype) + h_mean
            if isinstance(output, tuple):
                return (h_projected,) + output[1:]
            return h_projected
        return hook

    handle = model.layers[intervention_layer].register_forward_hook(
        make_intervention_hook(projection_matrix))

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(input_ids=token_ids_batch, use_cache=False)
        h_final = output.last_hidden_state

    handle.remove()

    logits = lm_head(h_final).float()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids_batch[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), reduction='mean')
    return loss.item()


def compute_pca_directions(model, token_ids_list, layer_idx, device, n_components=100):
    """Extract hidden states at a layer and compute PCA directions.
    Returns: PCA object fitted on centered hidden states."""
    all_h = []
    captured = {}

    def hook(m, i, o):
        captured['h'] = (o[0] if isinstance(o, tuple) else o).detach()

    handle = model.layers[layer_idx].register_forward_hook(hook)

    for ids in token_ids_list:
        input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model(input_ids=input_ids, use_cache=False)
        h = captured['h'].squeeze(0).float().cpu().numpy()
        all_h.append(h)

    handle.remove()

    # Stack all tokens from all sequences
    stacked = np.concatenate(all_h, axis=0)  # (N*T, D)
    stacked_c = stacked - stacked.mean(axis=0)

    n_comp = min(n_components, stacked_c.shape[0] - 1, stacked_c.shape[1])
    pca = PCA(n_components=n_comp).fit(stacked_c)
    return pca


def make_projection_matrix(pca, k):
    """Create (D, D) projection matrix onto top-k PCA directions."""
    V_k = pca.components_[:k].T  # (D, k)
    P = V_k @ V_k.T  # (D, D)
    return torch.tensor(P, dtype=torch.float32)


def make_random_projection_matrix(d, k, seed=None):
    """Create (D, D) projection matrix onto random k-dim subspace."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((d, k))
    V, _ = np.linalg.qr(V)  # orthogonalize
    P = V @ V.T
    return torch.tensor(P, dtype=torch.float32)


def main():
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    # Load TS data
    print("Loading TS windows...", flush=True)
    ts_windows = load_ts_windows(hf_token, N_TS)
    print(f"  Got {len(ts_windows)} windows")

    # Split: first 20 for computing PCA directions, rest for evaluation
    pca_windows = ts_windows[:20]
    eval_windows = ts_windows[20:]
    print(f"  PCA set: {len(pca_windows)}, Eval set: {len(eval_windows)}")

    # Prepare eval — process in small batches to avoid OOM
    EVAL_BS = 5
    eval_ids_list = [torch.tensor(np.stack(eval_windows[i:i+EVAL_BS]), dtype=torch.long, device=DEVICE)
                     for i in range(0, len(eval_windows), EVAL_BS)]

    def eval_loss(fn, *args):
        """Average loss over eval batches."""
        losses = []
        for batch in eval_ids_list:
            losses.append(fn(*args, batch))
        return float(np.mean(losses))

    # ═══ Step 1: Compute PCA directions for each model ═══
    print("\n" + "=" * 60)
    print("COMPUTING PCA DIRECTIONS")
    print("=" * 60)

    pca_dirs = {}  # (model, layer) -> PCA object
    for mname, mpath in MODEL_PATHS.items():
        print(f"\n  {mname}:")
        model = AutoModelForCausalLM.from_pretrained(
            mpath, dtype=torch.bfloat16).to(DEVICE).eval()
        transformer = model.model

        for li in INTERVENTION_LAYERS:
            pca = compute_pca_directions(transformer, pca_windows, li, DEVICE)
            pca_dirs[(mname, li)] = pca
            print(f"    Layer {li}: top-5 var ratio = {pca.explained_variance_ratio_[:5].sum()*100:.1f}%")

        del model; gc.collect(); torch.cuda.empty_cache()

    # ═══ Step 2: Experiment A — Rank-constrained FT ═══
    print("\n" + "=" * 60)
    print("EXPERIMENT A: RANK-CONSTRAINED FT PREDICTION")
    print("=" * 60)

    model_ft = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["FT"], dtype=torch.bfloat16).to(DEVICE).eval()
    transformer_ft = model_ft.model
    lm_head_ft = model_ft.lm_head

    # Helper: evaluate with batching
    def batched_baseline(transformer, lm_head):
        losses = []
        for batch in eval_ids_list:
            losses.append(compute_baseline_loss(transformer, lm_head, batch, DEVICE))
        return float(np.mean(losses))

    def batched_intervene(transformer, lm_head, layer, proj):
        losses = []
        for batch in eval_ids_list:
            losses.append(intervened_forward(transformer, lm_head, batch, layer, proj, DEVICE))
        return float(np.mean(losses))

    # Baseline (no intervention)
    baseline_loss = batched_baseline(transformer_ft, lm_head_ft)
    print(f"\n  FT baseline loss: {baseline_loss:.4f}")

    exp_a_results = {}
    for li in INTERVENTION_LAYERS:
        print(f"\n  Layer {li}:")
        pca = pca_dirs[("FT", li)]
        layer_results = {"baseline": baseline_loss, "ranks": {}}

        for k in RANK_VALUES:
            if k > pca.n_components_:
                continue
            P = make_projection_matrix(pca, k)
            loss = batched_intervene(transformer_ft, lm_head_ft, li, P)
            layer_results["ranks"][k] = loss
            degradation = (loss - baseline_loss) / baseline_loss * 100
            print(f"    K={k:>3}: loss={loss:.4f} (Δ={degradation:+.1f}%)")

        exp_a_results[li] = layer_results

    del model_ft; gc.collect(); torch.cuda.empty_cache()

    # ═══ Step 3: Experiment B — Cross-model direction transfer ═══
    print("\n" + "=" * 60)
    print("EXPERIMENT B: CROSS-MODEL DIRECTION TRANSFER")
    print("=" * 60)

    model_ft = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["FT"], dtype=torch.bfloat16).to(DEVICE).eval()
    transformer_ft = model_ft.model
    lm_head_ft = model_ft.lm_head

    baseline_loss_ft = batched_baseline(transformer_ft, lm_head_ft)

    exp_b_results = {}
    K_TRANSFER = 15

    for li in INTERVENTION_LAYERS:
        print(f"\n  Layer {li} (K={K_TRANSFER}):")
        results = {"baseline": baseline_loss_ft}

        P_ft = make_projection_matrix(pca_dirs[("FT", li)], K_TRANSFER)
        loss_ft_dirs = batched_intervene(transformer_ft, lm_head_ft, li, P_ft)
        results["ft_dirs"] = loss_ft_dirs
        print(f"    FT dirs:     loss={loss_ft_dirs:.4f}")

        P_pt = make_projection_matrix(pca_dirs[("PT", li)], K_TRANSFER)
        loss_pt_dirs = batched_intervene(transformer_ft, lm_head_ft, li, P_pt)
        results["pt_dirs"] = loss_pt_dirs
        print(f"    PT dirs:     loss={loss_pt_dirs:.4f}")

        P_ri = make_projection_matrix(pca_dirs[("RI", li)], K_TRANSFER)
        loss_ri_dirs = batched_intervene(transformer_ft, lm_head_ft, li, P_ri)
        results["ri_dirs"] = loss_ri_dirs
        print(f"    RI dirs:     loss={loss_ri_dirs:.4f}")

        random_losses = []
        for seed in range(5):
            P_rand = make_random_projection_matrix(1024, K_TRANSFER, seed=seed + 100)
            loss_rand = batched_intervene(transformer_ft, lm_head_ft, li, P_rand)
            random_losses.append(loss_rand)
        results["random_dirs"] = float(np.mean(random_losses))
        results["random_dirs_std"] = float(np.std(random_losses))
        print(f"    Random dirs: loss={results['random_dirs']:.4f} ± {results['random_dirs_std']:.4f}")

        P_full = torch.eye(1024)
        loss_full = batched_intervene(transformer_ft, lm_head_ft, li, P_full)
        results["full_rank"] = loss_full
        print(f"    Full rank:   loss={loss_full:.4f} (should ≈ baseline)")

        exp_b_results[li] = results

    del model_ft; gc.collect(); torch.cuda.empty_cache()

    # ═══ Step 4: Experiment B variant — sweep K for PT dirs ═══
    print("\n" + "=" * 60)
    print("EXPERIMENT B2: PT DIRECTIONS AT VARYING K")
    print("=" * 60)

    model_ft = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["FT"], dtype=torch.bfloat16).to(DEVICE).eval()
    transformer_ft = model_ft.model
    lm_head_ft = model_ft.lm_head

    exp_b2_results = {}
    li = 8

    print(f"\n  Layer {li}:")
    for source in ["FT", "PT", "RI"]:
        print(f"\n    {source} directions:")
        pca = pca_dirs[(source, li)]
        losses = {}
        for k in RANK_VALUES:
            if k > pca.n_components_:
                continue
            P = make_projection_matrix(pca, k)
            loss = batched_intervene(transformer_ft, lm_head_ft, li, P)
            losses[k] = loss
            print(f"      K={k:>3}: loss={loss:.4f}")
        exp_b2_results[source] = losses

    print(f"\n    Random directions:")
    rand_losses = {}
    for k in RANK_VALUES:
        seeds_losses = []
        for seed in range(3):
            P = make_random_projection_matrix(1024, k, seed=seed + 200)
            loss = batched_intervene(transformer_ft, lm_head_ft, li, P)
            seeds_losses.append(loss)
        rand_losses[k] = float(np.mean(seeds_losses))
        print(f"      K={k:>3}: loss={rand_losses[k]:.4f}")
    exp_b2_results["Random"] = rand_losses

    del model_ft; gc.collect(); torch.cuda.empty_cache()

    # ═══ Save results ═══
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump({
            "exp_a": {str(k): v for k, v in exp_a_results.items()},
            "exp_b": {str(k): v for k, v in exp_b_results.items()},
            "exp_b2": exp_b2_results,
            "baseline_ft": baseline_loss,
        }, f, indent=2)

    # ═══ Plots ═══
    print("\nGenerating plots...", flush=True)

    # Plot A: Rank-constrained FT
    fig, axes = plt.subplots(1, len(INTERVENTION_LAYERS), figsize=(5 * len(INTERVENTION_LAYERS), 5))
    fig.suptitle("Experiment A: Rank-Constrained FT Prediction\n"
                 "Project hidden states onto top-K PCs at each layer — does accuracy survive?",
                 fontsize=13, fontweight='bold')

    for i, li in enumerate(INTERVENTION_LAYERS):
        ax = axes[i]
        r = exp_a_results[li]
        ks = sorted(r["ranks"].keys())
        losses = [r["ranks"][k] for k in ks]
        ax.plot(ks, losses, 'o-', color='#4CAF50', linewidth=2, markersize=6)
        ax.axhline(r["baseline"], color='gray', linestyle='--', linewidth=1.5,
                   label=f'Baseline ({r["baseline"]:.3f})')
        ax.set_xlabel("Rank K"); ax.set_ylabel("Cross-entropy loss")
        ax.set_title(f"Layer {li}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/exp_a_rank_constrained.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot B: Cross-model at fixed K
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(INTERVENTION_LAYERS))
    width = 0.18
    sources = ["ft_dirs", "pt_dirs", "ri_dirs", "random_dirs"]
    source_labels = ["FT dirs", "PT dirs", "RI dirs", "Random dirs"]
    source_colors = ["#4CAF50", "#2196F3", "#E91E63", "#9E9E9E"]

    for si, (src, label, color) in enumerate(zip(sources, source_labels, source_colors)):
        vals = [exp_b_results[li][src] for li in INTERVENTION_LAYERS]
        if src == "random_dirs":
            errs = [exp_b_results[li]["random_dirs_std"] for li in INTERVENTION_LAYERS]
            ax.bar(x + si * width, vals, width, color=color, label=label, yerr=errs, capsize=3)
        else:
            ax.bar(x + si * width, vals, width, color=color, label=label)

    baseline = exp_b_results[INTERVENTION_LAYERS[0]]["baseline"]
    ax.axhline(baseline, color='black', linestyle='--', linewidth=1.5, label=f'FT baseline ({baseline:.3f})')
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"Layer {li}" for li in INTERVENTION_LAYERS])
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Experiment B: Whose Directions Work Best in FT? (K={K_TRANSFER})\n"
                 "Project hidden states onto K directions from each model, continue FT forward pass",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/exp_b_cross_model.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot B2: K sweep for different direction sources at layer 8
    fig, ax = plt.subplots(figsize=(10, 6))
    source_colors_b2 = {"FT": "#4CAF50", "PT": "#2196F3", "RI": "#E91E63", "Random": "#9E9E9E"}
    for source in ["FT", "PT", "RI", "Random"]:
        ks = sorted(exp_b2_results[source].keys())
        losses = [exp_b2_results[source][k] for k in ks]
        ax.plot(ks, losses, 'o-', color=source_colors_b2[source], linewidth=2,
                markersize=6, label=f"{source} directions")

    ax.axhline(baseline_loss, color='black', linestyle='--', linewidth=1.5,
               label=f'FT baseline ({baseline_loss:.3f})')
    ax.set_xlabel("Rank K"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Layer 8: FT Prediction with K Directions from Each Model\n"
                 "FT's own directions are best; PT's are second best (shared subspace)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xscale('log', base=2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/exp_b2_k_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ═══ Summary ═══
    print(f"\n{'=' * 70}")
    print("INTERVENTION EXPERIMENT RESULTS")
    print(f"{'=' * 70}")

    print(f"\nFT baseline loss: {baseline_loss:.4f}")

    print(f"\n--- Experiment A: Rank-constrained FT at Layer 8 ---")
    r = exp_a_results[8]
    for k in sorted(r["ranks"].keys()):
        deg = (r["ranks"][k] - r["baseline"]) / r["baseline"] * 100
        print(f"  K={k:>3}: loss={r['ranks'][k]:.4f} ({deg:+.1f}%)")

    print(f"\n--- Experiment B: Cross-model directions at Layer 8 (K={K_TRANSFER}) ---")
    r = exp_b_results[8]
    for src, label in [("ft_dirs", "FT"), ("pt_dirs", "PT"), ("ri_dirs", "RI"), ("random_dirs", "Random")]:
        deg = (r[src] - r["baseline"]) / r["baseline"] * 100
        print(f"  {label:>8} dirs: loss={r[src]:.4f} ({deg:+.1f}%)")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
