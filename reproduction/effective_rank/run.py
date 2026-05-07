"""
Effective rank and subspace alignment across PT, FT, FT_IO, RI models.

For each (model, input_type, layer), extracts hidden states, accumulates
the covariance matrix in streaming fashion, then eigendecomposes to get:
  - Effective rank: erank = exp(H(p)) where p_i = λ_i / Σλ_j
  - Subspace alignment: ||V1[:,:k]^T V2[:,:k]||_F^2 / k

Computations:
  erank:     TS through PT, FT, FT_IO, RI  |  Text through PT, FT, FT_IO
  alignment: all pairwise among {PT, FT, FT_IO, RI} on TS, plus random baseline

Runs all 4 models in parallel across 4 GPUs.

Usage:
    python reproduction/effective_rank/run.py
    python reproduction/effective_rank/run.py --n_windows 1000 --batch_size 256
"""
import argparse
import json
import os
import sys
import threading
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from src.data.dataset import load_gifteval_series, temporal_split, WindowDataset
from src.data.wikitext import load_wikitext_sequences

# ── Paths ────────────────────────────────────────────────────────────────────

PT_PATH = "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
MODELS = {
    "PT": PT_PATH,
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420/checkpoint-8192",
    "FT_IO": "/workspace/NanoTS_v2/checkpoints/io_only/checkpoint-8192",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-8192",
}

# Model → GPU assignment
MODEL_DEVICES = {
    "PT": "cuda:0",
    "FT": "cuda:1",
    "FT_IO": "cuda:2",
    "RI": "cuda:3",
}

N_LAYERS = 28
HIDDEN_SIZE = 1024
CONTEXT_LENGTH = 512

# Tokenizer (uniform binning shared by FT, FT_IO, RI — and used for TS through PT too)
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0

# ── TS tokenization ─────────────────────────────────────────────────────────

def tokenize_ts_batch(values_batch: np.ndarray) -> torch.Tensor:
    """
    Z-score normalize each window independently, then uniform-bin to [0, N_BINS-1].

    Parameters
    ----------
    values_batch : (B, context_length) float32

    Returns
    -------
    (B, context_length) int64 tensor of token IDs
    """
    mean = values_batch.mean(axis=1, keepdims=True)
    std = values_batch.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normed = (values_batch - mean) / std
    clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
    bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
    bins = np.clip(bins, 0, N_BINS - 1)
    return torch.from_numpy(bins)


# ── Covariance accumulator ───────────────────────────────────────────────────

class CovarianceAccumulator:
    """Streaming covariance accumulation in float64 for numerical stability."""

    def __init__(self, n_layers: int, hidden_size: int):
        self.n_layers = n_layers
        self.h = hidden_size
        self.count = 0
        self.sum_x = [torch.zeros(hidden_size, dtype=torch.float64) for _ in range(n_layers)]
        self.sum_xx = [torch.zeros(hidden_size, hidden_size, dtype=torch.float64) for _ in range(n_layers)]

    def update(self, layer_idx: int, hidden_states: torch.Tensor):
        """
        Accumulate a batch of hidden states.

        Parameters
        ----------
        hidden_states : (N, hidden_size) float32 on GPU
        """
        # Compute matmuls on GPU in float32, then move small results to CPU for float64 accumulation
        self.sum_x[layer_idx] += hidden_states.sum(dim=0).to(dtype=torch.float64, device="cpu")
        self.sum_xx[layer_idx] += (hidden_states.T @ hidden_states).to(dtype=torch.float64, device="cpu")
        if layer_idx == 0:
            self.count += hidden_states.shape[0]

    def eigendecompose(self, layer_idx: int):
        """
        Returns (eigenvalues, eigenvectors) of centered covariance, sorted descending.
        eigenvalues: (hidden_size,)  eigenvectors: (hidden_size, hidden_size) columns
        """
        mean = self.sum_x[layer_idx] / self.count
        cov = self.sum_xx[layer_idx] / self.count - mean.outer(mean)
        cov = (cov + cov.T) / 2  # ensure symmetry
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        idx = eigenvalues.argsort(descending=True)
        return eigenvalues[idx], eigenvectors[:, idx]


# ── Activation extraction ────────────────────────────────────────────────────

@torch.no_grad()
def extract_and_accumulate(
    model: AutoModelForCausalLM,
    input_ids_list: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
    accumulator: CovarianceAccumulator,
    desc: str = "",
):
    """
    Run input_ids through model in batches, accumulate covariance per layer.

    Parameters
    ----------
    input_ids_list : list of (seq_len,) int64 tensors
    """
    n = len(input_ids_list)
    n_batches = (n + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch = torch.stack(input_ids_list[start:end]).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch, output_hidden_states=True)
        # outputs.hidden_states: tuple of (B, seq_len, H) for embedding + 28 layers
        # We want layers 0..27 (indices 1..28 in hidden_states)
        for layer_idx in range(N_LAYERS):
            hs = outputs.hidden_states[layer_idx + 1]  # (B, seq_len, H)
            # Skip position 0: attention sink creates ~200x norm outlier that
            # dominates covariance as a rank-1 artifact
            hs = hs[:, 1:, :]
            flat = hs.reshape(-1, HIDDEN_SIZE).float()
            accumulator.update(layer_idx, flat)

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            print(f"  {desc} batch {batch_idx + 1}/{n_batches}  ({accumulator.count:,} vectors)", flush=True)


# ── Per-model worker ─────────────────────────────────────────────────────────

def process_model(
    model_name: str,
    device_str: str,
    ts_inputs: list[torch.Tensor],
    text_inputs: list[torch.Tensor] | None,
    batch_size: int,
    results_dict: dict,
    lock: threading.Lock,
):
    """Load one model on one GPU, run TS (and optionally text), store accumulators."""
    device = torch.device(device_str)
    print(f"[{model_name}] Loading on {device_str} from {MODELS[model_name]}", flush=True)

    full_model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_name], dtype=torch.bfloat16, device_map=device,
    )
    # Use inner transformer only — skip lm_head (saves ~37 GiB per forward pass)
    model = full_model.model
    model.eval()

    # TS
    acc_ts = CovarianceAccumulator(N_LAYERS, HIDDEN_SIZE)
    extract_and_accumulate(model, ts_inputs, batch_size, device, acc_ts,
                           desc=f"{model_name}/TS")

    with lock:
        results_dict[(model_name, "TS")] = acc_ts

    # Text (skip RI)
    if text_inputs is not None:
        acc_text = CovarianceAccumulator(N_LAYERS, HIDDEN_SIZE)
        extract_and_accumulate(model, text_inputs, batch_size, device, acc_text,
                               desc=f"{model_name}/Text")
        with lock:
            results_dict[(model_name, "Text")] = acc_text

    del full_model, model
    torch.cuda.empty_cache()
    print(f"[{model_name}] Done.", flush=True)


# ── Metrics ──────────────────────────────────────────────────────────────────

def effective_rank(eigenvalues: torch.Tensor) -> float:
    ev = eigenvalues[eigenvalues > 0]
    p = ev / ev.sum()
    H = -(p * torch.log(p)).sum()
    return torch.exp(H).item()


def subspace_alignment(V1: torch.Tensor, V2: torch.Tensor, k: int) -> float:
    U1 = V1[:, :k]
    U2 = V2[:, :k]
    return (U1.T @ U2).pow(2).sum().item() / k


def random_alignment_baseline(hidden_size: int, k: int, n_trials: int = 200) -> tuple[float, float]:
    alignments = []
    for _ in range(n_trials):
        Q1, _ = torch.linalg.qr(torch.randn(hidden_size, k, dtype=torch.float64))
        Q2, _ = torch.linalg.qr(torch.randn(hidden_size, k, dtype=torch.float64))
        a = (Q1.T @ Q2).pow(2).sum().item() / k
        alignments.append(a)
    return float(np.mean(alignments)), float(np.std(alignments))


# ── Data loading ─────────────────────────────────────────────────────────────

def load_ts_inputs(n_windows: int) -> list[torch.Tensor]:
    """Load GiftEval val windows, tokenize to bin IDs."""
    print("Loading GiftEval series...", flush=True)
    hf_token = os.environ.get("HF_TOKEN")
    series_list = load_gifteval_series(hf_token=hf_token)
    print(f"  {len(series_list)} series loaded", flush=True)

    val_splits = []
    for s in series_list:
        _, va, _ = temporal_split(s, 0.70, 0.15)
        if len(va) >= CONTEXT_LENGTH:
            val_splits.append(va)

    ds = WindowDataset(val_splits, CONTEXT_LENGTH, stride=CONTEXT_LENGTH)
    n_use = min(n_windows, len(ds))
    print(f"  Using {n_use} / {len(ds)} val windows", flush=True)

    values_batch = np.stack([ds[i]["values"] for i in range(n_use)])
    token_ids = tokenize_ts_batch(values_batch)
    return [token_ids[i] for i in range(n_use)]


def load_text_inputs(n_sequences: int) -> list[torch.Tensor]:
    """Load WikiText-103 sequences, tokenized to 512 tokens."""
    print("Loading WikiText-103...", flush=True)
    hf_token = os.environ.get("HF_TOKEN")
    sequences = load_wikitext_sequences(
        max_sequences=n_sequences, seq_len=CONTEXT_LENGTH, hf_token=hf_token
    )
    n_use = min(n_sequences, len(sequences))
    print(f"  Using {n_use} text sequences", flush=True)
    return [torch.from_numpy(seq["input_ids"]) for seq in sequences[:n_use]]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_windows", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results/effective_rank")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────
    ts_inputs = load_ts_inputs(args.n_windows)
    text_inputs = load_text_inputs(args.n_windows)

    # ── Run all 4 models in parallel across 4 GPUs ───────────────────────
    print(f"\nLaunching 4 models on 4 GPUs (batch_size={args.batch_size})...", flush=True)

    text_models = {"PT", "FT", "FT_IO"}
    accumulators: dict[tuple[str, str], CovarianceAccumulator] = {}
    lock = threading.Lock()
    threads = []

    for model_name in MODELS:
        t = threading.Thread(
            target=process_model,
            args=(
                model_name,
                MODEL_DEVICES[model_name],
                ts_inputs,
                text_inputs if model_name in text_models else None,
                args.batch_size,
                accumulators,
                lock,
            ),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"\nAll models done. Accumulators: {list(accumulators.keys())}", flush=True)

    # ── Compute eigendecompositions ──────────────────────────────────────
    print("Computing eigendecompositions...", flush=True)

    # eigen[key][layer] = (eigenvalues, eigenvectors)
    eigen: dict[tuple[str, str], list] = {}
    for key, acc in accumulators.items():
        eigen[key] = []
        for layer in range(N_LAYERS):
            evals, evecs = acc.eigendecompose(layer)
            eigen[key].append((evals, evecs))

    # ── Effective rank ───────────────────────────────────────────────────
    print("Computing effective ranks...", flush=True)
    erank_results = {}
    for key in sorted(eigen.keys()):
        model_name, input_type = key
        eranks = [effective_rank(eigen[key][l][0]) for l in range(N_LAYERS)]
        erank_results[f"{model_name}_{input_type}"] = eranks
        print(f"  {model_name}/{input_type}: min={min(eranks):.1f}  max={max(eranks):.1f}  "
              f"mean={np.mean(eranks):.1f}", flush=True)

    # ── Subspace alignment (TS only, all pairwise) ───────────────────────
    print("\nComputing subspace alignments...", flush=True)
    alignment_results = {}
    ts_models = ["PT", "FT", "FT_IO", "RI"]
    ts_keys = [(m, "TS") for m in ts_models]

    for (m1, _), (m2, _) in combinations(ts_keys, 2):
        pair_name = f"{m1}_vs_{m2}"
        alignments = []
        for layer in range(N_LAYERS):
            evals1, evecs1 = eigen[(m1, "TS")][layer]
            evals2, evecs2 = eigen[(m2, "TS")][layer]
            k = min(
                round(effective_rank(evals1)),
                round(effective_rank(evals2)),
            )
            k = max(k, 1)
            a = subspace_alignment(evecs1, evecs2, k)
            alignments.append({"layer": layer, "alignment": a, "k": k})
        alignment_results[pair_name] = alignments
        vals = [x["alignment"] for x in alignments]
        print(f"  {pair_name}: min={min(vals):.3f}  max={max(vals):.3f}  mean={np.mean(vals):.3f}",
              flush=True)

    # ── Random baseline ──────────────────────────────────────────────────
    print("\nComputing random alignment baselines...", flush=True)
    ks_used = set()
    for pair in alignment_results.values():
        for entry in pair:
            ks_used.add(entry["k"])

    random_baselines = {}
    for k in sorted(ks_used):
        mean_a, std_a = random_alignment_baseline(HIDDEN_SIZE, k)
        random_baselines[k] = {"mean": mean_a, "std": std_a}
        print(f"  k={k}: {mean_a:.4f} ± {std_a:.4f}", flush=True)

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "config": {
            "n_windows": args.n_windows,
            "context_length": CONTEXT_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "n_layers": N_LAYERS,
            "n_bins": N_BINS,
            "models": {k: v for k, v in MODELS.items()},
            "skip_position_0": True,
        },
        "effective_rank": erank_results,
        "subspace_alignment": {
            pair: [{"layer": e["layer"], "alignment": e["alignment"], "k": e["k"]}
                   for e in entries]
            for pair, entries in alignment_results.items()
        },
        "random_baseline": {str(k): v for k, v in random_baselines.items()},
    }

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Also save eigenvalues for potential later use
    eigenvalue_data = {}
    for key in sorted(eigen.keys()):
        label = f"{key[0]}_{key[1]}"
        eigenvalue_data[label] = [eigen[key][l][0].tolist() for l in range(N_LAYERS)]

    ev_path = os.path.join(args.output_dir, "eigenvalues.json")
    with open(ev_path, "w") as f:
        json.dump(eigenvalue_data, f)
    print(f"Eigenvalues saved to {ev_path}", flush=True)


if __name__ == "__main__":
    main()
