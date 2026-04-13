"""
Generate comparison plots for all 4 ablation conditions.
Uses saved mappers from mapping_results/ablation/.

Plots:
  1. Best-4 per model grid (4×4)
  2. Quality levels (#1, #25, #50, #100) per model
  3. Diversity overlay (50 predictions per model)
  4. Fair K=4 side-by-side
"""
import torch, torch.nn as nn, numpy as np, os, sys, gc, time, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences
from scripts.mapping_experiment import load_ts_windows, compute_acf

D_CONCAT = 28 * 1024
T = 512
N_PLOT = 500  # sequences to use for plotting (subset of train)
DEVICE = torch.device("cuda:0")

ABLATION_DIR = "mapping_results/ablation"
PLOT_DIR = f"{ABLATION_DIR}/plots"

MODELS = {
    "text_PT":        {"model_type": "pt",          "tokens": "text"},
    "text_RandInit":  {"model_type": "random_init",  "tokens": "text"},
    "rand_PT":        {"model_type": "pt",          "tokens": "random"},
    "rand_RandInit":  {"model_type": "random_init",  "tokens": "random"},
}

COLORS = {
    "text_PT": "#2196F3",        # blue
    "text_RandInit": "#FF9800",  # orange
    "rand_PT": "#4CAF50",        # green
    "rand_RandInit": "#E91E63",  # pink
}

LABELS = {
    "text_PT": "Text + PT",
    "text_RandInit": "Text + RandInit",
    "rand_PT": "Random + PT",
    "rand_RandInit": "Random + RandInit",
}


def extract_all_layers(model, sequences, device, desc=""):
    captured = {}
    handles = []
    for li in range(28):
        def make_hook(idx):
            def hook(m, i, o):
                captured[idx] = (o[0] if isinstance(o, tuple) else o).detach()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))
    all_hs = []
    BS = 8
    N = len(sequences)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, BS):
            batch = sequences[start:start + BS]
            ids = torch.tensor(np.stack([s["input_ids"] for s in batch]),
                               dtype=torch.long, device=device)
            model(input_ids=ids, use_cache=False)
            layers = [captured[i].float().cpu()[:, :T, :] for i in range(28)]
            all_hs.append(torch.cat(layers, dim=-1).half())
            captured.clear()
            if start % 80 == 0 and start > 0:
                print(f"    {desc} {start}/{N} ({time.time()-t0:.0f}s)", flush=True)
    for h in handles:
        h.remove()
    result = torch.cat(all_hs, dim=0)
    print(f"    {desc} Done: {result.shape} ({time.time()-t0:.0f}s)", flush=True)
    return result


def predict(mapper, h_data, device):
    mapper.eval()
    all_pred = []
    with torch.no_grad():
        for start in range(0, h_data.shape[0], 32):
            h_b = h_data[start:start + 32].float().to(device)
            yr = mapper(h_b).squeeze(-1)
            ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
            all_pred.append(((yr - yr.mean(-1, keepdim=True)) / ys).cpu())
    return torch.cat(all_pred, dim=0)


def compute_nn(pred, ts_bank):
    nn_indices = []; nn_dists = []
    for i in range(pred.shape[0]):
        d2 = ((pred[i:i + 1] - ts_bank) ** 2).mean(-1).squeeze(0)
        nn_indices.append(d2.argmin().item())
        nn_dists.append(d2.min().item())
    return np.array(nn_indices), np.array(nn_dists)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")

    # Load shared data
    print("Loading WikiText...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=2000, seq_len=T, hf_token=hf_token)
    plot_wiki = wiki_seqs[:N_PLOT]

    print("Loading TS bank...", flush=True)
    ts = load_ts_windows(cfg, 10000, hf_token)

    # Random token sequences
    vocab_size = 151936
    rng = np.random.default_rng(42)
    rand_seqs = [{"input_ids": rng.integers(0, vocab_size, size=T).astype(np.int64)} for _ in range(N_PLOT)]

    # Extract + predict for each ablation
    all_preds = {}
    all_nn_idx = {}
    all_nn_dist = {}

    for name, info in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Processing {name}")
        print(f"{'='*50}")

        # Load model
        if info["model_type"] == "pt":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
            ).model.to(DEVICE).eval()
        else:
            config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).model.to(DEVICE).eval()

        seqs = plot_wiki if info["tokens"] == "text" else rand_seqs

        # Extract
        h = extract_all_layers(model, seqs, DEVICE, desc=name)
        del model; gc.collect(); torch.cuda.empty_cache()

        # Load mapper and predict
        mapper = nn.Linear(D_CONCAT, 1)
        mapper.load_state_dict(torch.load(
            f"{ABLATION_DIR}/mapper_{name}.pt", map_location="cpu", weights_only=True))
        mapper = mapper.to(DEVICE).eval()

        pred = predict(mapper, h, DEVICE)
        nn_idx, nn_dist = compute_nn(pred, ts)

        all_preds[name] = pred
        all_nn_idx[name] = nn_idx
        all_nn_dist[name] = nn_dist

        n_unique = len(set(nn_idx.tolist()))
        print(f"  {name}: unique={n_unique}, mean_dist={nn_dist.mean():.4f}")

        del h, mapper; gc.collect(); torch.cuda.empty_cache()

    # ── Plot 1: Best 4 per model ──
    print("\nGenerating Plot 1: Best 4 per model grid...", flush=True)
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle("Best 4 Predictions per Ablation Condition (overlaid with nearest real TS)",
                 fontsize=16, fontweight='bold', y=0.98)

    for col, name in enumerate(MODELS.keys()):
        pred = all_preds[name]
        nn_idx = all_nn_idx[name]
        nn_dist = all_nn_dist[name]
        best_order = np.argsort(nn_dist)

        axes[0, col].set_title(LABELS[name], fontsize=13, fontweight='bold',
                               color=COLORS[name])
        for row in range(4):
            ax = axes[row, col]
            idx = best_order[row]
            ti = nn_idx[idx]
            ax.plot(ts[ti].numpy(), color="green", linewidth=1.2, alpha=0.6, label="Real TS")
            ax.plot(pred[idx].numpy(), color=COLORS[name], linewidth=1.2, alpha=0.8,
                    label="Predicted")
            ax.set_ylim(-4, 4)
            ax.set_ylabel(f"#{row+1}\nd={nn_dist[idx]:.3f}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{PLOT_DIR}/best4_per_model.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Quality levels (#1, #25, #50, #100) per model ──
    print("Generating Plot 2: Quality levels per model...", flush=True)
    ranks = [0, 24, 49, 99]  # 0-indexed: #1, #25, #50, #100
    rank_labels = ["#1 (best)", "#25", "#50", "#100"]

    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle("Prediction Quality at Different Ranks (overlaid with nearest real TS)",
                 fontsize=16, fontweight='bold', y=0.98)

    for col, name in enumerate(MODELS.keys()):
        pred = all_preds[name]
        nn_idx = all_nn_idx[name]
        nn_dist = all_nn_dist[name]
        best_order = np.argsort(nn_dist)
        n_unique = len(set(nn_idx.tolist()))

        axes[0, col].set_title(f"{LABELS[name]}\n({n_unique} unique)", fontsize=12,
                               fontweight='bold', color=COLORS[name])
        for row, (rank, rlabel) in enumerate(zip(ranks, rank_labels)):
            ax = axes[row, col]
            if rank >= len(best_order):
                ax.text(0.5, 0.5, f"Only {len(best_order)} preds",
                        ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.set_ylabel(rlabel, fontsize=10)
                continue
            idx = best_order[rank]
            ti = nn_idx[idx]
            ax.plot(ts[ti].numpy(), color="green", linewidth=1.2, alpha=0.6, label="Real TS")
            ax.plot(pred[idx].numpy(), color=COLORS[name], linewidth=1.2, alpha=0.8,
                    label="Predicted")
            ax.set_ylim(-4, 4)
            ax.set_ylabel(f"{rlabel}\nd={nn_dist[idx]:.3f}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{PLOT_DIR}/quality_levels.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Diversity overlay (50 predictions per model) ──
    print("Generating Plot 3: Diversity overlay...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle("Prediction Diversity: 50 Random Predictions per Ablation",
                 fontsize=16, fontweight='bold')

    for i, name in enumerate(MODELS.keys()):
        ax = axes[i // 2, i % 2]
        pred = all_preds[name]
        n_unique = len(set(all_nn_idx[name].tolist()))
        sample = np.random.choice(pred.shape[0], min(50, pred.shape[0]), replace=False)
        for j in sample:
            ax.plot(pred[j].numpy(), alpha=0.15, linewidth=0.6, color=COLORS[name])
        ax.set_title(f"{LABELS[name]} ({n_unique} unique matches)", fontsize=13,
                     fontweight='bold')
        ax.set_ylim(-4, 4)
        ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/diversity_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Fair K=4 side-by-side (best unique match per model) ──
    print("Generating Plot 4: Fair K=4 comparison...", flush=True)
    fig, axes = plt.subplots(4, 4, figsize=(24, 16))
    fig.suptitle("Fair Comparison: Best 4 Unique Matches per Model\n(deduped — one prediction per unique TS)",
                 fontsize=15, fontweight='bold', y=0.98)

    for col, name in enumerate(MODELS.keys()):
        pred = all_preds[name]
        nn_idx = all_nn_idx[name]
        nn_dist = all_nn_dist[name]

        # Deduplicate: keep best match per unique TS index
        unique_best = {}
        for i in range(pred.shape[0]):
            ti = nn_idx[i]
            if ti not in unique_best or nn_dist[i] < unique_best[ti][1]:
                unique_best[ti] = (i, nn_dist[i])
        sorted_unique = sorted(unique_best.items(), key=lambda x: x[1][1])

        n_unique = len(sorted_unique)
        axes[0, col].set_title(f"{LABELS[name]}\n({n_unique} unique)", fontsize=12,
                               fontweight='bold', color=COLORS[name])

        for row in range(4):
            ax = axes[row, col]
            if row >= len(sorted_unique):
                ax.text(0.5, 0.5, f"Only {n_unique} unique",
                        ha='center', va='center', transform=ax.transAxes, fontsize=11)
                continue
            ti, (pred_idx, dist) = sorted_unique[row]
            ax.plot(ts[ti].numpy(), color="green", linewidth=1.2, alpha=0.6, label="Real TS")
            ax.plot(pred[pred_idx].numpy(), color=COLORS[name], linewidth=1.2, alpha=0.8,
                    label="Predicted")
            ax.set_ylim(-4, 4)
            ax.set_ylabel(f"#{row+1}\nd={dist:.3f}", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{PLOT_DIR}/fair_k4_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 5: Summary bar chart ──
    print("Generating Plot 5: Summary metrics...", flush=True)
    with open(f"{ABLATION_DIR}/results.json") as f:
        results = json.load(f)

    names = list(MODELS.keys())
    labels = [LABELS[n] for n in names]
    colors = [COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Ablation Study Summary (Training Set)", fontsize=15, fontweight='bold')

    # Unique matches
    vals = [results[n]["train"]["n_unique"] for n in names]
    axes[0].bar(labels, vals, color=colors)
    axes[0].set_ylabel("Unique TS Matches")
    axes[0].set_title("Diversity (unique matches / 10K bank)")
    for j, v in enumerate(vals):
        axes[0].text(j, v + 10, str(v), ha='center', fontsize=11, fontweight='bold')

    # Dedup NN distance
    vals = [results[n]["train"]["dedup_nn_mean"] for n in names]
    axes[1].bar(labels, vals, color=colors)
    axes[1].set_ylabel("Mean Dedup NN Distance")
    axes[1].set_title("Quality (lower = better match)")
    for j, v in enumerate(vals):
        axes[1].text(j, v + 0.01, f"{v:.3f}", ha='center', fontsize=10, fontweight='bold')

    # Clusters
    vals = [results[n]["train"]["clusters"] for n in names]
    axes[2].bar(labels, vals, color=colors)
    axes[2].set_ylabel("Clusters Covered (/20)")
    axes[2].set_title("Coverage (ACF clusters)")
    axes[2].set_ylim(0, 22)
    for j, v in enumerate(vals):
        axes[2].text(j, v + 0.3, str(v), ha='center', fontsize=11, fontweight='bold')

    for ax in axes:
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/summary_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll plots saved to {PLOT_DIR}/")
    for f in sorted(os.listdir(PLOT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
