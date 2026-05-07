"""
Sine wave hidden-state PCA across ALL training checkpoints for FT, FT_IO, RI.

Processes 4 checkpoints in parallel across 4 GPUs. For each checkpoint and
each of the 28 layers, extracts hidden states from a sine wave input and
computes 2D PCA projections with explained variance.

PT baseline computed once. Results saved to results/sine_evolution_checkpoints/.

Usage:
    python3 reproduction/sine_evolution/run_checkpoints.py
"""
import json
import os
import sys
import threading
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────────────

T = 512
PERIOD = 64
SKIP = 5
N_LAYERS = 28
HIDDEN_SIZE = 1024
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0

PT_PATH = "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

CHECKPOINT_DIRS = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "FT_IO": "/workspace/NanoTS_v2/checkpoints/io_only",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
}

OUTPUT_DIR = "results/sine_evolution_checkpoints"

# ── Sine wave ────────────────────────────────────────────────────────────────

def make_sine_tokens():
    t_arr = np.arange(T, dtype=np.float32)
    sine = np.sin(2 * np.pi * t_arr / PERIOD)
    mean, std = float(np.mean(sine)), float(np.std(sine))
    if std < 1e-8:
        std = 1.0
    normed = (sine - mean) / std
    clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
    bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
    bins = np.clip(bins, 0, N_BINS - 1)
    phase = (t_arr % PERIOD) / PERIOD
    return torch.from_numpy(bins).unsqueeze(0), phase


# ── Extract + project ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_and_project(model_path, input_ids, phase, device):
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    model = full_model.model
    model.eval()

    captured = {}
    handles = []
    for li in range(N_LAYERS):
        def make_hook(l):
            def hook(module, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[l] = o.detach().float().cpu()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))

    model(input_ids=input_ids.to(device), use_cache=False)

    for h in handles:
        h.remove()
    del full_model, model
    torch.cuda.empty_cache()

    phase_clean = phase[SKIP:]
    layer_results = {}
    for li in range(N_LAYERS):
        hs = captured[li].squeeze(0).numpy()[SKIP:]
        pca = PCA(n_components=2)
        proj = pca.fit_transform(hs)
        var_exp = pca.explained_variance_ratio_.sum() * 100
        layer_results[li] = {
            "pca": proj.tolist(),
            "pca_var": float(var_exp),
        }

    return layer_results


# ── Worker ───────────────────────────────────────────────────────────────────

def process_checkpoint(ckpt_path, model_family, device_str, input_ids, phase, results_dict, lock):
    step = int(ckpt_path.split("-")[-1])
    key = f"{model_family}_step{step}"

    layer_results = extract_and_project(ckpt_path, input_ids, phase, device_str)

    with lock:
        results_dict[key] = {
            "model": model_family,
            "step": step,
            "layers": {str(k): v for k, v in layer_results.items()},
        }

    print(f"    {key} done", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_ids, phase = make_sine_tokens()
    print(f"Sine wave: T={T}, period={PERIOD}, skip={SKIP}", flush=True)

    # PT baseline
    print("\nComputing PT baseline...", flush=True)
    pt_results = extract_and_project(PT_PATH, input_ids, phase, "cuda:0")
    print("  PT done", flush=True)

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

    results_dict = {}
    lock = threading.Lock()
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    t_start = time.time()

    for round_idx in range(0, len(all_jobs), 4):
        batch_jobs = all_jobs[round_idx:round_idx + 4]
        tags = [f"{j[1]}-{j[0].split('-')[-1]}" for j in batch_jobs]
        print(f"\nRound {round_idx // 4 + 1}/{(len(all_jobs) + 3) // 4}: {tags}", flush=True)

        threads = []
        for i, (ckpt_path, family) in enumerate(batch_jobs):
            t = threading.Thread(
                target=process_checkpoint,
                args=(ckpt_path, family, gpus[i], input_ids, phase, results_dict, lock),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    elapsed = time.time() - t_start
    print(f"\nAll checkpoints done in {elapsed:.0f}s", flush=True)

    # Save
    output = {
        "config": {
            "T": T, "period": PERIOD, "skip": SKIP,
            "n_layers": N_LAYERS, "n_bins": N_BINS,
        },
        "phase": phase[SKIP:].tolist(),
        "PT": {str(k): v for k, v in pt_results.items()},
        "checkpoints": results_dict,
    }

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    print(f"Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
