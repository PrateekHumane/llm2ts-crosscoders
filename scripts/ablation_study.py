"""
Proper ablation study: Architecture × Input tokens.

Ablations:
  1. Text tokens + PT           (language model processing real text)
  2. Text tokens + RandomInit   (untrained architecture processing real text)
  3. Random tokens + PT         (language model processing random tokens)
  4. Random tokens + RandomInit (untrained architecture processing random tokens)

Each ablation:
  - Extract all 28 layers concatenated (D = 28672)
  - Train linear mapper with PSD diversity penalty (λ = 0.5)
  - Evaluate on training data AND held-out data against full 10K TS bank
  - Save mapper weights

Usage:
    /usr/bin/python3 scripts/ablation_study.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, time, gc

from transformers import AutoModelForCausalLM, AutoConfig
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences
from scripts.mapping_experiment import load_ts_windows, compute_acf, compute_psd


D_CONCAT = 28 * 1024
T = 512
N_TRAIN = 1920
N_HELD = 1000  # held-out sequences
N_EPOCHS = 100
LAMBDA_DIV = 0.5
BATCH_SIZE = 32
LR = 1e-3
SEED = 1  # seed 1 was the best in stability test


def psd_diversity_penalty(y_pred):
    psd = compute_psd(y_pred)
    psd_norm = F.normalize(psd, dim=-1)
    sim = psd_norm @ psd_norm.T
    mask = 1 - torch.eye(sim.shape[0], device=sim.device)
    return (sim * mask).sum() / mask.sum()


def extract_all_layers(model, sequences, device, desc=""):
    """Extract all 28 layers from a model for given token sequences.
    sequences: list of dicts with 'input_ids' (numpy int64 array of shape (512,))
    Returns: (N, T, D_CONCAT) float16 tensor
    """
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


def generate_random_token_sequences(n_seqs, seq_len, vocab_size):
    """Generate sequences of random token IDs."""
    rng = np.random.default_rng(42)
    sequences = []
    for i in range(n_seqs):
        ids = rng.integers(0, vocab_size, size=seq_len).astype(np.int64)
        sequences.append({"input_ids": ids})
    return sequences


def train_mapper(h_train, ts_data, device, seed=SEED):
    """Train linear mapper with PSD diversity penalty. Returns (predictions, mapper, final_loss)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    mapper = nn.Linear(D_CONCAT, 1).to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=LR)
    N = h_train.shape[0]
    N_ts = ts_data.shape[0]

    final_loss = 0
    for epoch in range(N_EPOCHS):
        perm = torch.randperm(N)
        epoch_loss = 0; n_batches = 0
        for start in range(0, N, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            h_b = h_train[idx].float().to(device)
            ts_b = ts_data[torch.randint(0, N_ts, (128,))].to(device)
            y_raw = mapper(h_b).squeeze(-1)
            y_std = y_raw.std(dim=-1, keepdim=True).clamp(min=1e-4)
            y = (y_raw - y_raw.mean(dim=-1, keepdim=True)) / y_std
            with torch.no_grad():
                d2 = ((y.unsqueeze(1) - ts_b.unsqueeze(0)) ** 2).mean(-1)
                tgt = ts_b[d2.argmin(1)]
            loss = F.mse_loss(y, tgt) + LAMBDA_DIV * psd_diversity_penalty(y)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1
        final_loss = epoch_loss / n_batches
        if epoch % 25 == 0 or epoch == N_EPOCHS - 1:
            print(f"      Epoch {epoch}/{N_EPOCHS} loss={final_loss:.4f}", flush=True)

    return mapper, final_loss


def predict(mapper, h_data, device):
    """Generate predictions from hidden states using trained mapper."""
    mapper.eval()
    all_pred = []
    with torch.no_grad():
        for start in range(0, h_data.shape[0], BATCH_SIZE):
            h_b = h_data[start:start + BATCH_SIZE].float().to(device)
            yr = mapper(h_b).squeeze(-1)
            ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
            all_pred.append(((yr - yr.mean(-1, keepdim=True)) / ys).cpu())
    return torch.cat(all_pred, dim=0)


def evaluate(pred, ts_bank, ts_labels, name):
    """Full evaluation against 10K TS bank."""
    N = pred.shape[0]
    nn_indices = []; nn_dists = []
    for i in range(N):
        d2 = ((pred[i:i + 1] - ts_bank) ** 2).mean(-1).squeeze(0)
        nn_indices.append(d2.argmin().item())
        nn_dists.append(d2.min().item())
    nn_dists = np.array(nn_dists)
    nn_indices = np.array(nn_indices)

    unique_set = set(nn_indices.tolist())
    unique_best = {}
    for i in range(N):
        ti = nn_indices[i]
        if ti not in unique_best or nn_dists[i] < unique_best[ti]:
            unique_best[ti] = nn_dists[i]
    dedup_dists = np.array(list(unique_best.values()))
    sorted_unique = sorted(unique_best.items(), key=lambda x: x[1])
    matched_clusters = set(ts_labels[list(unique_set)])

    result = {
        "n_preds": N,
        "raw_nn_mean": float(nn_dists.mean()),
        "raw_nn_median": float(np.median(nn_dists)),
        "n_unique": len(unique_set),
        "dedup_nn_mean": float(dedup_dists.mean()),
        "dedup_nn_median": float(np.median(dedup_dists)),
        "clusters": int(len(matched_clusters)),
        "top50": float(np.mean([d for _, d in sorted_unique[:50]])) if len(sorted_unique) >= 50 else None,
        "top100": float(np.mean([d for _, d in sorted_unique[:100]])) if len(sorted_unique) >= 100 else None,
        "top200": float(np.mean([d for _, d in sorted_unique[:200]])) if len(sorted_unique) >= 200 else None,
    }

    print(f"    [{name}] ({N} preds)")
    print(f"      Raw NN:  mean={result['raw_nn_mean']:.4f}")
    print(f"      Unique:  {result['n_unique']} / 10000")
    print(f"      Dedup:   mean={result['dedup_nn_mean']:.4f}")
    print(f"      Clusters: {result['clusters']}/20")
    if result['top50']:
        print(f"      Top-50:  {result['top50']:.4f}")
    if result['top100']:
        print(f"      Top-100: {result['top100']:.4f}")

    return result


def main():
    device = torch.device("cuda:0")
    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")

    out_dir = "mapping_results/ablation"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ──
    print("Loading WikiText...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=N_TRAIN + N_HELD, seq_len=T, hf_token=hf_token)
    wiki_train = wiki_seqs[:N_TRAIN]
    wiki_held = wiki_seqs[N_TRAIN:N_TRAIN + N_HELD]
    print(f"  WikiText: {len(wiki_train)} train, {len(wiki_held)} held-out")

    print("Loading TS...", flush=True)
    ts = load_ts_windows(cfg, 10000, hf_token)
    ts_labels = KMeans(n_clusters=20, n_init=5, random_state=42).fit_predict(
        compute_acf(ts, max_lag=32).numpy())

    print("Generating random token sequences...", flush=True)
    vocab_size = 151936  # Qwen3 vocab size
    rand_train = generate_random_token_sequences(N_TRAIN, T, vocab_size)
    rand_held = generate_random_token_sequences(N_HELD, T, vocab_size)
    # Use different seed for held-out
    rng2 = np.random.default_rng(99)
    rand_held = [{"input_ids": rng2.integers(0, vocab_size, size=T).astype(np.int64)} for _ in range(N_HELD)]

    # ── Define ablations ──
    ablations = [
        ("text_PT",       "pt",          wiki_train, wiki_held),
        ("text_RandInit",  "random_init", wiki_train, wiki_held),
        ("rand_PT",       "pt",          rand_train, rand_held),
        ("rand_RandInit",  "random_init", rand_train, rand_held),
    ]

    all_results = {}

    for name, model_type, train_seqs, held_seqs in ablations:
        print(f"\n{'='*60}")
        print(f"ABLATION: {name}")
        print(f"{'='*60}")

        # Load model
        print(f"  Loading {model_type} model...", flush=True)
        if model_type == "pt":
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
            ).model.to(device).eval()
        else:
            config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16).model.to(device).eval()

        # Extract training hidden states
        print(f"  Extracting train ({len(train_seqs)} seqs)...", flush=True)
        h_train = extract_all_layers(model, train_seqs, device, desc=f"{name} train")

        # Extract held-out hidden states
        print(f"  Extracting held-out ({len(held_seqs)} seqs)...", flush=True)
        h_held = extract_all_layers(model, held_seqs, device, desc=f"{name} held")

        del model; gc.collect(); torch.cuda.empty_cache()

        # Train mapper
        print(f"  Training mapper...", flush=True)
        mapper, final_loss = train_mapper(h_train, ts, device)

        # Save mapper
        mapper_path = os.path.join(out_dir, f"mapper_{name}.pt")
        torch.save(mapper.state_dict(), mapper_path)
        print(f"  Saved mapper to {mapper_path}")

        # Predict + evaluate training
        print(f"  Evaluating training...", flush=True)
        pred_train = predict(mapper, h_train, device)
        r_train = evaluate(pred_train, ts, ts_labels, f"{name} TRAIN")

        # Predict + evaluate held-out
        print(f"  Evaluating held-out...", flush=True)
        pred_held = predict(mapper, h_held, device)
        r_held = evaluate(pred_held, ts, ts_labels, f"{name} HELD-OUT")

        all_results[name] = {
            "train": r_train,
            "held_out": r_held,
            "final_loss": final_loss,
        }

        del h_train, h_held, pred_train, pred_held, mapper
        gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*70}")

    print(f"\n{'Ablation':<20} {'Loss':>6} | {'Train':>6} {'Uniq':>5} {'Clust':>6} {'Top50':>6} | "
          f"{'Held':>6} {'Uniq':>5} {'Clust':>6} {'Top50':>6}")
    print("-" * 90)
    for name, r in all_results.items():
        tr = r["train"]; he = r["held_out"]
        t50 = f"{tr['top50']:.3f}" if tr['top50'] else "  n/a"
        h50 = f"{he['top50']:.3f}" if he['top50'] else "  n/a"
        print(f"{name:<20} {r['final_loss']:>6.3f} | "
              f"{tr['raw_nn_mean']:>6.3f} {tr['n_unique']:>5} {tr['clusters']:>3}/20 {t50:>6} | "
              f"{he['raw_nn_mean']:>6.3f} {he['n_unique']:>5} {he['clusters']:>3}/20 {h50:>6}")

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
