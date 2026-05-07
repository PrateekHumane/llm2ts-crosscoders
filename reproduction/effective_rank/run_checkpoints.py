"""
Effective rank across all training checkpoints for FT, FT_IO, RI.

Runs 4 checkpoints in parallel across 4 GPUs, iterating through all
checkpoints for each model family. Also computes text erank for FT/FT_IO
to track catastrophic forgetting across training.

Usage:
    python reproduction/effective_rank/run_checkpoints.py
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))
from src.data.dataset import load_gifteval_series, temporal_split, WindowDataset
from src.data.wikitext import load_wikitext_sequences

# ── Config ───────────────────────────────────────────────────────────────────

N_LAYERS = 28
HIDDEN_SIZE = 1024
CONTEXT_LENGTH = 512
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0
N_WINDOWS = 10000
BATCH_SIZE = 256

PT_PATH = "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

CHECKPOINT_DIRS = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "FT_IO": "/workspace/NanoTS_v2/checkpoints/io_only",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
}

# Text erank for FT and FT_IO only (not RI)
TEXT_MODELS = {"FT", "FT_IO"}

OUTPUT_DIR = "results/effective_rank_checkpoints"

# ── Reused from run.py ───────────────────────────────────────────────────────

def tokenize_ts_batch(values_batch):
    mean = values_batch.mean(axis=1, keepdims=True)
    std = values_batch.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normed = (values_batch - mean) / std
    clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
    bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
    bins = np.clip(bins, 0, N_BINS - 1)
    return torch.from_numpy(bins)


class CovarianceAccumulator:
    def __init__(self):
        self.count = 0
        self.sum_x = [torch.zeros(HIDDEN_SIZE, dtype=torch.float64) for _ in range(N_LAYERS)]
        self.sum_xx = [torch.zeros(HIDDEN_SIZE, HIDDEN_SIZE, dtype=torch.float64) for _ in range(N_LAYERS)]

    def update(self, layer_idx, hidden_states):
        self.sum_x[layer_idx] += hidden_states.sum(dim=0).to(dtype=torch.float64, device="cpu")
        self.sum_xx[layer_idx] += (hidden_states.T @ hidden_states).to(dtype=torch.float64, device="cpu")
        if layer_idx == 0:
            self.count += hidden_states.shape[0]

    def eigenvalues(self, layer_idx):
        mean = self.sum_x[layer_idx] / self.count
        cov = self.sum_xx[layer_idx] / self.count - mean.outer(mean)
        cov = (cov + cov.T) / 2
        evals = torch.linalg.eigvalsh(cov)
        return evals.flip(0)


def effective_rank(eigenvalues):
    ev = eigenvalues[eigenvalues > 0]
    p = ev / ev.sum()
    H = -(p * torch.log(p)).sum()
    return torch.exp(H).item()


@torch.no_grad()
def run_accumulator(model, input_ids_list, device, desc=""):
    acc = CovarianceAccumulator()
    n = len(input_ids_list)
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        batch = torch.stack(input_ids_list[start:end]).to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch, output_hidden_states=True)

        for layer_idx in range(N_LAYERS):
            hs = outputs.hidden_states[layer_idx + 1][:, 1:, :]
            acc.update(layer_idx, hs.reshape(-1, HIDDEN_SIZE).float())

    eranks = [effective_rank(acc.eigenvalues(l)) for l in range(N_LAYERS)]
    print(f"    {desc}: mean_erank={np.mean(eranks):.1f}", flush=True)
    return eranks


# ── Worker ───────────────────────────────────────────────────────────────────

def process_checkpoint(ckpt_path, model_family, device_str, ts_inputs, text_inputs, results_dict, lock):
    device = torch.device(device_str)
    step = int(ckpt_path.split("-")[-1])
    key = f"{model_family}_step{step}"

    full_model = AutoModelForCausalLM.from_pretrained(ckpt_path, dtype=torch.bfloat16, device_map=device)
    model = full_model.model
    model.eval()

    # TS
    ts_eranks = run_accumulator(model, ts_inputs, device, desc=f"{key}/TS")

    # Text
    text_eranks = None
    if model_family in TEXT_MODELS:
        text_eranks = run_accumulator(model, text_inputs, device, desc=f"{key}/Text")

    with lock:
        results_dict[key] = {
            "model": model_family,
            "step": step,
            "checkpoint": ckpt_path,
            "ts_erank": ts_eranks,
        }
        if text_eranks is not None:
            results_dict[key]["text_erank"] = text_eranks

    del full_model, model
    torch.cuda.empty_cache()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data...", flush=True)
    hf_token = os.environ.get("HF_TOKEN")
    series_list = load_gifteval_series(hf_token=hf_token)
    val_splits = []
    for s in series_list:
        _, va, _ = temporal_split(s, 0.70, 0.15)
        if len(va) >= CONTEXT_LENGTH:
            val_splits.append(va)
    ds = WindowDataset(val_splits, CONTEXT_LENGTH, stride=CONTEXT_LENGTH)
    n_use = min(N_WINDOWS, len(ds))
    values_batch = np.stack([ds[i]["values"] for i in range(n_use)])
    ts_inputs = [tokenize_ts_batch(values_batch)[i] for i in range(n_use)]
    print(f"  {n_use} TS windows", flush=True)

    sequences = load_wikitext_sequences(max_sequences=N_WINDOWS, seq_len=CONTEXT_LENGTH, hf_token=hf_token)
    text_inputs = [torch.from_numpy(seq["input_ids"]) for seq in sequences[:n_use]]
    print(f"  {len(text_inputs)} text sequences", flush=True)

    # PT baseline (compute once)
    print("\nComputing PT baseline...", flush=True)
    full_model = AutoModelForCausalLM.from_pretrained(PT_PATH, dtype=torch.bfloat16, device_map="cuda:0")
    model = full_model.model
    model.eval()
    pt_ts_eranks = run_accumulator(model, ts_inputs, torch.device("cuda:0"), desc="PT/TS")
    pt_text_eranks = run_accumulator(model, text_inputs, torch.device("cuda:0"), desc="PT/Text")
    del full_model, model
    torch.cuda.empty_cache()

    # Collect all checkpoints
    all_jobs = []
    for family, base_dir in CHECKPOINT_DIRS.items():
        ckpts = sorted(
            [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        for ckpt in ckpts:
            all_jobs.append((os.path.join(base_dir, ckpt), family))

    print(f"\n{len(all_jobs)} checkpoints to process across 4 GPUs", flush=True)

    # Process in rounds of 4
    results_dict = {}
    lock = threading.Lock()
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    t_start = time.time()

    for round_idx in range(0, len(all_jobs), 4):
        batch_jobs = all_jobs[round_idx:round_idx + 4]
        print(f"\nRound {round_idx // 4 + 1}/{(len(all_jobs) + 3) // 4}: "
              f"{[j[1] + '-' + j[0].split('-')[-1] for j in batch_jobs]}", flush=True)

        threads = []
        for i, (ckpt_path, family) in enumerate(batch_jobs):
            t = threading.Thread(
                target=process_checkpoint,
                args=(ckpt_path, family, gpus[i], ts_inputs, text_inputs, results_dict, lock),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    elapsed = time.time() - t_start
    print(f"\nAll checkpoints done in {elapsed:.0f}s", flush=True)

    # Save results
    output = {
        "config": {
            "n_windows": N_WINDOWS,
            "context_length": CONTEXT_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "n_layers": N_LAYERS,
            "n_bins": N_BINS,
            "skip_position_0": True,
        },
        "PT": {
            "ts_erank": pt_ts_eranks,
            "text_erank": pt_text_eranks,
        },
        "checkpoints": results_dict,
    }

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
