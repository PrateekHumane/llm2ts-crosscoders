"""
Phase coherence at Layer 13 across all training checkpoints for FT, FT_IO, RI.

Processes 4 checkpoints in parallel across 4 GPUs. Also computes PT baseline.

Usage:
    python3 reproduction/sine_evolution/run_phase_coherence_checkpoints.py
"""
import json
import os
import threading
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from transformers import AutoModelForCausalLM

# ── Config ───────────────────────────────────────────────────────────────────

T = 512
SKIP = 5
LAYER = 13
N_BINS = 1024
BIN_LOW = -5.0
BIN_HIGH = 5.0

PT_PATH = "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

CHECKPOINT_DIRS = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "FT_IO": "/workspace/NanoTS_v2/checkpoints/io_only",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
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
    hs = hidden_states[SKIP:]
    n = hs.shape[0]
    positions = np.arange(SKIP, T)
    phases = positions % period

    dists = squareform(pdist(hs, metric="euclidean"))

    phase_matrix = phases[:, None] == phases[None, :]
    np.fill_diagonal(phase_matrix, False)

    all_mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(all_mask, False)

    same_phase_mean = dists[phase_matrix].mean()
    all_pairs_mean = dists[all_mask].mean()

    return float(same_phase_mean / all_pairs_mean)


# ── Extract + compute ────────────────────────────────────────────────────────

@torch.no_grad()
def process_checkpoint(ckpt_path, model_family, device_str, all_input_ids, results_dict, lock):
    step = int(ckpt_path.split("-")[-1])
    key = f"{model_family}_step{step}"

    full_model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, dtype=torch.bfloat16, device_map=device_str
    )
    model = full_model.model
    model.eval()

    captured = {}
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["hs"] = o.detach().float().cpu()
    handle = model.layers[LAYER].register_forward_hook(hook)

    coherences = {}
    for input_name, input_ids in all_input_ids.items():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=input_ids.to(device_str), use_cache=False)
        hs = captured["hs"].squeeze(0).numpy()
        period = INPUTS[input_name]["period"]
        coherences[input_name] = compute_phase_coherence(hs, period)

    handle.remove()
    del full_model, model
    torch.cuda.empty_cache()

    with lock:
        results_dict[key] = {
            "model": model_family,
            "step": step,
            "coherence": coherences,
        }
    print(f"    {key} done", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_input_ids = {}
    for name, cfg in INPUTS.items():
        all_input_ids[name] = tokenize(cfg["signal"])

    # PT baseline
    print("Computing PT baseline...", flush=True)
    full_model = AutoModelForCausalLM.from_pretrained(
        PT_PATH, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model = full_model.model
    model.eval()

    captured = {}
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["hs"] = o.detach().float().cpu()
    handle = model.layers[LAYER].register_forward_hook(hook)

    pt_coherences = {}
    for input_name, input_ids in all_input_ids.items():
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=input_ids.to("cuda:0"), use_cache=False)
        hs = captured["hs"].squeeze(0).numpy()
        period = INPUTS[input_name]["period"]
        pt_coherences[input_name] = compute_phase_coherence(hs, period)

    handle.remove()
    del full_model, model
    torch.cuda.empty_cache()
    print(f"  PT done: {pt_coherences}", flush=True)

    # Collect checkpoints
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
                args=(ckpt_path, family, gpus[i], all_input_ids, results_dict, lock),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    elapsed = time.time() - t_start
    print(f"\nAll checkpoints done in {elapsed:.0f}s", flush=True)

    # Save
    output = {
        "config": {"T": T, "skip": SKIP, "layer": LAYER, "n_bins": N_BINS},
        "inputs": {name: {"period": cfg["period"]} for name, cfg in INPUTS.items()},
        "PT": pt_coherences,
        "checkpoints": results_dict,
    }
    out_path = os.path.join(OUTPUT_DIR, "phase_coherence_checkpoints.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}", flush=True)

    # Print summary tables
    families = {}
    for key, val in results_dict.items():
        m = val["model"]
        if m not in families:
            families[m] = []
        families[m].append(val)
    for m in families:
        families[m].sort(key=lambda x: x["step"])

    for input_name in INPUTS:
        print(f"\n{'='*70}")
        print(f"Phase coherence at Layer {LAYER}: {input_name}")
        print(f"{'='*70}")
        print(f"{'Step':>8}{'FT':>10}{'FT_IO':>10}{'RI':>10}")
        print("-" * 38)
        print(f"{'PT':>8}{pt_coherences[input_name]:>10.3f}")
        max_steps = max(len(families[m]) for m in families)
        for i in range(max_steps):
            row = ""
            step_label = ""
            for m in ["FT", "FT_IO", "RI"]:
                if i < len(families[m]):
                    if not step_label:
                        step_label = str(families[m][i]["step"])
                    row += f"{families[m][i]['coherence'][input_name]:>10.3f}"
                else:
                    row += f"{'':>10}"
            print(f"{step_label:>8}{row}")


if __name__ == "__main__":
    main()
