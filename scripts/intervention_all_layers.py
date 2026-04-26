"""
All-layers-simultaneous intervention: project hidden states onto K PCs
at EVERY layer, not just one. This is the strongest test — if PT's
directions still work when we constrain all 28 layers, the shared
subspace is real and functional throughout the model.

Usage:
    /usr/bin/python3 scripts/intervention_all_layers.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, glob

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = 512; N_LAYERS = 28; DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/interventions"
SEED = 42; N_TS = 50; EVAL_BS = 5
RANK_VALUES = [2, 5, 10, 15, 20, 30, 50, 100]
MODEL_PATHS = {"PT": "Qwen/Qwen3-0.6B", "FT": "models/ft", "RI": "models/ri"}


def load_ts_windows(hf_token, n_windows):
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
                w = arr[start:start+T]
                mu, sigma = w.mean(), w.std()
                if sigma < 1e-6: continue
                w = (w - mu) / sigma
                bins = ((np.clip(w, -5, 5) + 5) / 10 * 512).astype(np.int64).clip(0, 511)
                windows.append(bins)
                if len(windows) >= n_windows: return windows
    return windows


def compute_pca_all_layers(model, token_ids_list, device, n_components=100):
    """Compute PCA directions at every layer. Returns dict: layer -> PCA."""
    pcas = {}
    for li in range(N_LAYERS):
        all_h = []
        captured = {}
        def hook(m, i, o): captured['h'] = (o[0] if isinstance(o, tuple) else o).detach()
        handle = model.layers[li].register_forward_hook(hook)
        for ids in token_ids_list:
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model(input_ids=input_ids, use_cache=False)
            all_h.append(captured['h'].squeeze(0).float().cpu().numpy())
        handle.remove()
        stacked = np.concatenate(all_h, axis=0)
        stacked_c = stacked - stacked.mean(axis=0)
        n_comp = min(n_components, stacked_c.shape[0] - 1, stacked_c.shape[1])
        pcas[li] = PCA(n_components=n_comp).fit(stacked_c)
        if li % 7 == 0:
            print(f"    L{li}: top-5 = {pcas[li].explained_variance_ratio_[:5].sum()*100:.1f}%", flush=True)
    return pcas


def make_proj(pca, k):
    V = pca.components_[:k].T
    return torch.tensor(V @ V.T, dtype=torch.float32)


def make_random_proj(d, k, seed=None):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((d, k))
    V, _ = np.linalg.qr(V)
    return torch.tensor(V @ V.T, dtype=torch.float32)


def forward_with_all_layer_intervention(model, lm_head, token_ids_batch, proj_matrices, device):
    """Forward pass projecting at EVERY layer simultaneously.
    proj_matrices: dict layer_idx -> (D, D) projection matrix."""
    handles = []
    for li, P in proj_matrices.items():
        def make_hook(proj):
            def hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                h_mean = h.mean(dim=1, keepdim=True)
                h_c = h - h_mean
                h_p = h_c.float() @ proj.to(h.device)
                h_p = h_p.to(h.dtype) + h_mean
                if isinstance(output, tuple):
                    return (h_p,) + output[1:]
                return h_p
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(P)))

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        output = model(input_ids=token_ids_batch, use_cache=False)
        logits = lm_head(output.last_hidden_state).float()

    for h in handles:
        h.remove()

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = token_ids_batch[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1), reduction='mean')
    return loss.item()


def batched_eval(model, lm_head, eval_batches, proj_matrices, device):
    losses = []
    for batch in eval_batches:
        losses.append(forward_with_all_layer_intervention(
            model, lm_head, batch, proj_matrices, device))
    return float(np.mean(losses))


def batched_baseline(model, lm_head, eval_batches, device):
    return batched_eval(model, lm_head, eval_batches, {}, device)


def main():
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")
    np.random.seed(SEED)

    print("Loading TS...", flush=True)
    ts_windows = load_ts_windows(hf_token, N_TS)
    pca_windows = ts_windows[:20]
    eval_windows = ts_windows[20:]
    eval_batches = [torch.tensor(np.stack(eval_windows[i:i+EVAL_BS]), dtype=torch.long, device=DEVICE)
                    for i in range(0, len(eval_windows), EVAL_BS)]
    print(f"  PCA: {len(pca_windows)}, Eval: {len(eval_windows)}")

    # Compute PCA at every layer for each model
    print("\nComputing PCA directions at all 28 layers...", flush=True)
    all_pcas = {}
    for mname, mpath in MODEL_PATHS.items():
        print(f"\n  {mname}:")
        model = AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.bfloat16).to(DEVICE).eval()
        all_pcas[mname] = compute_pca_all_layers(model.model, pca_windows, DEVICE)
        del model; gc.collect(); torch.cuda.empty_cache()

    # Load FT for interventions
    print("\nLoading FT for interventions...", flush=True)
    model_ft = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["FT"], dtype=torch.bfloat16).to(DEVICE).eval()

    baseline = batched_baseline(model_ft.model, model_ft.lm_head, eval_batches, DEVICE)
    print(f"  Baseline: {baseline:.4f}")

    # ═══ Experiment: all-layers simultaneous projection ═══
    print(f"\n{'='*60}")
    print("ALL-LAYERS SIMULTANEOUS INTERVENTION")
    print(f"{'='*60}")

    results = {}

    # For each direction source and K, project at ALL 28 layers
    for source in ["FT", "PT", "RI"]:
        print(f"\n  {source} directions:")
        results[source] = {}
        for k in RANK_VALUES:
            proj_matrices = {}
            skip = False
            for li in range(N_LAYERS):
                if k > all_pcas[source][li].n_components_:
                    skip = True; break
                proj_matrices[li] = make_proj(all_pcas[source][li], k)
            if skip:
                continue
            loss = batched_eval(model_ft.model, model_ft.lm_head, eval_batches, proj_matrices, DEVICE)
            results[source][k] = loss
            deg = (loss - baseline) / baseline * 100
            print(f"    K={k:>3}: loss={loss:.4f} ({deg:+.1f}%)")

    # Random baseline
    print(f"\n  Random directions:")
    results["Random"] = {}
    for k in RANK_VALUES:
        seed_losses = []
        for seed in range(3):
            proj_matrices = {li: make_random_proj(1024, k, seed=seed*100+li) for li in range(N_LAYERS)}
            loss = batched_eval(model_ft.model, model_ft.lm_head, eval_batches, proj_matrices, DEVICE)
            seed_losses.append(loss)
        results["Random"][k] = float(np.mean(seed_losses))
        deg = (results["Random"][k] - baseline) / baseline * 100
        print(f"    K={k:>3}: loss={results['Random'][k]:.4f} ({deg:+.1f}%)")

    # Also: single-layer intervention at layer 8 for comparison
    print(f"\n  Single-layer (L8 only) for comparison:")
    results["FT_single_L8"] = {}
    for k in RANK_VALUES:
        if k > all_pcas["FT"][8].n_components_: continue
        proj_matrices = {8: make_proj(all_pcas["FT"][8], k)}
        loss = batched_eval(model_ft.model, model_ft.lm_head, eval_batches, proj_matrices, DEVICE)
        results["FT_single_L8"][k] = loss
        deg = (loss - baseline) / baseline * 100
        print(f"    K={k:>3}: loss={loss:.4f} ({deg:+.1f}%)")

    del model_ft; gc.collect(); torch.cuda.empty_cache()

    # Save
    with open(f"{OUT_DIR}/all_layers_results.json", "w") as f:
        json.dump({"baseline": baseline, "results": results}, f, indent=2)

    # ═══ Plot ═══
    print("\nGenerating plots...", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("All-Layers Simultaneous Intervention vs Single-Layer\n"
                 "Project onto K PCs at every layer simultaneously — the strongest test",
                 fontsize=13, fontweight='bold')

    colors = {"FT": "#4CAF50", "PT": "#2196F3", "RI": "#E91E63", "Random": "#9E9E9E",
              "FT_single_L8": "#8BC34A"}
    labels = {"FT": "FT dirs (all layers)", "PT": "PT dirs (all layers)",
              "RI": "RI dirs (all layers)", "Random": "Random (all layers)",
              "FT_single_L8": "FT dirs (L8 only)"}
    linestyles = {"FT": "-", "PT": "-", "RI": "-", "Random": "--", "FT_single_L8": ":"}

    # Left: all sources
    for source in ["FT", "PT", "RI", "Random", "FT_single_L8"]:
        ks = sorted(results[source].keys())
        vals = [results[source][k] for k in ks]
        axes[0].plot(ks, vals, 'o-', color=colors[source], linewidth=2, markersize=5,
                     linestyle=linestyles[source], label=labels[source])
    axes[0].axhline(baseline, color='black', linestyle='--', linewidth=1.5,
                    label=f'Baseline ({baseline:.3f})')
    axes[0].set_xlabel("Rank K"); axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("All sources comparison"); axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3); axes[0].set_xscale('log', base=2)

    # Right: degradation percentage
    for source in ["FT", "PT", "RI", "Random"]:
        ks = sorted(results[source].keys())
        degs = [(results[source][k] - baseline) / baseline * 100 for k in ks]
        axes[1].plot(ks, degs, 'o-', color=colors[source], linewidth=2, markersize=5,
                     label=f"{source} dirs")
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel("Rank K"); axes[1].set_ylabel("Loss degradation (%)")
    axes[1].set_title("Degradation from baseline"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3); axes[1].set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/all_layers_intervention.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    print(f"\n{'='*70}")
    print("ALL-LAYERS INTERVENTION SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline: {baseline:.4f}\n")
    print(f"{'Source':<20} ", end="")
    for k in RANK_VALUES:
        print(f"{'K='+str(k):>8}", end="")
    print()
    print("-" * (20 + 8 * len(RANK_VALUES)))
    for source in ["FT", "PT", "RI", "Random", "FT_single_L8"]:
        print(f"{labels[source]:<20} ", end="")
        for k in RANK_VALUES:
            if k in results[source]:
                deg = (results[source][k] - baseline) / baseline * 100
                print(f"{deg:>+7.1f}%", end="")
            else:
                print(f"{'n/a':>8}", end="")
        print()

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
