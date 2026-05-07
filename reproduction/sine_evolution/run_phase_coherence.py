"""
Phase coherence for periodic inputs across PT, FT, RI models.

Phase coherence = mean(same-phase pairwise distances) / mean(all pairwise distances)
computed in the full 1024-dim hidden space. Lower = more periodic representation.

Inputs: sine p=64, sine p=128, two-frequency (sin(2πt/64) + 0.5*sin(2πt/17))
Models: Random (RI step-1), Base (PT), LangInit (FT), RandInit (RI)

Usage:
    python3 reproduction/sine_evolution/run_phase_coherence.py
"""
import json
import os
import threading

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from transformers import AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────────────

T = 512
SKIP = 5
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0
N_LAYERS = 28

MODEL_PATHS = {
    "Random": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-1",
    "Base": "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "LangInit": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420/checkpoint-8192",
    "RandInit": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss/checkpoint-8192",
}

OUTPUT_DIR = "results/sine_evolution"

t_arr = np.arange(T, dtype=np.float32)

INPUTS = {
    "Sine p=64": {"signal": np.sin(2 * np.pi * t_arr / 64), "period": 64},
    "Sine p=128": {"signal": np.sin(2 * np.pi * t_arr / 128), "period": 128},
    "Two freq": {"signal": np.sin(2 * np.pi * t_arr / 64) + 0.5 * np.sin(2 * np.pi * t_arr / 17), "period": 64},
}

# ── Tokenization ─────────────────────────────────────────────────────────────

def tokenize(signal):
    mean, std = float(np.mean(signal)), float(np.std(signal))
    if std < 1e-8:
        std = 1.0
    normed = (signal - mean) / std
    clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
    bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
    bins = np.clip(bins, 0, N_BINS - 1)
    return torch.from_numpy(bins).unsqueeze(0)


# ── Phase coherence ──────────────────────────────────────────────────────────

def compute_phase_coherence(hidden_states, period):
    """
    Phase coherence = mean(same-phase distances) / mean(all distances).
    Computed over positions SKIP..T in the full hidden dimension.
    """
    hs = hidden_states[SKIP:]  # (T-SKIP, 1024)
    n = hs.shape[0]
    positions = np.arange(SKIP, T)
    phases = positions % period

    # All pairwise distances
    dists = squareform(pdist(hs, metric="euclidean"))

    # Same-phase mask
    phase_matrix = phases[:, None] == phases[None, :]
    np.fill_diagonal(phase_matrix, False)

    # All-pairs mask (exclude diagonal)
    all_mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(all_mask, False)

    same_phase_mean = dists[phase_matrix].mean()
    all_pairs_mean = dists[all_mask].mean()

    return float(same_phase_mean / all_pairs_mean)


# ── Extract hidden states ────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_layers(model_name, model_path, all_input_ids, device):
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

    results = {}
    for name, input_ids in all_input_ids.items():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=input_ids.to(device), use_cache=False)
        results[name] = {li: captured[li].squeeze(0).numpy() for li in range(N_LAYERS)}

    for h in handles:
        h.remove()
    del full_model, model
    torch.cuda.empty_cache()
    print(f"  {model_name}: extracted {N_LAYERS} layers for {len(results)} inputs", flush=True)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tokenize
    all_input_ids = {}
    for name, cfg in INPUTS.items():
        all_input_ids[name] = tokenize(cfg["signal"])

    # Extract (4 models in parallel on 4 GPUs)
    print("Extracting hidden states...", flush=True)
    all_hidden = {}
    lock = threading.Lock()
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    threads = []

    def worker(model_name, model_path, device):
        hs = extract_all_layers(model_name, model_path, all_input_ids, device)
        with lock:
            all_hidden[model_name] = hs

    for (model_name, model_path), gpu in zip(MODEL_PATHS.items(), gpus):
        t = threading.Thread(target=worker, args=(model_name, model_path, gpu))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    # Compute phase coherence at every layer
    print("\nComputing phase coherence...", flush=True)
    results = {}
    for model_name in MODEL_PATHS:
        results[model_name] = {}
        for input_name, cfg in INPUTS.items():
            period = cfg["period"]
            layer_scores = []
            for li in range(N_LAYERS):
                hs = all_hidden[model_name][input_name][li]
                pc = compute_phase_coherence(hs, period)
                layer_scores.append(pc)
            results[model_name][input_name] = layer_scores

    # Print table for key layers
    for layer in [8, 13]:
        print(f"\n{'='*60}")
        print(f"Phase coherence at Layer {layer} (lower = more periodic)")
        print(f"{'='*60}")
        header = f"{'Input':<20}" + "".join(f"{m:>12}" for m in MODEL_PATHS)
        print(header)
        print("-" * len(header))
        for input_name in INPUTS:
            row = f"{input_name:<20}"
            for model_name in MODEL_PATHS:
                val = results[model_name][input_name][layer]
                row += f"{val:>12.3f}"
            print(row)

    # Print mean across all layers
    print(f"\n{'='*60}")
    print(f"Phase coherence averaged over all 28 layers")
    print(f"{'='*60}")
    header = f"{'Input':<20}" + "".join(f"{m:>12}" for m in MODEL_PATHS)
    print(header)
    print("-" * len(header))
    for input_name in INPUTS:
        row = f"{input_name:<20}"
        for model_name in MODEL_PATHS:
            val = np.mean(results[model_name][input_name])
            row += f"{val:>12.3f}"
        print(row)

    # Save full results
    out_path = os.path.join(OUTPUT_DIR, "phase_coherence.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
