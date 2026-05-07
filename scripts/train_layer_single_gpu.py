"""
Train linear crosscoder on a single GPU: precompute activations → load into VRAM → train.
Supports running multiple instances on different GPUs for different layers.

Usage:
    python3 scripts/train_layer_single_gpu.py --layer 0  --gpu 1
    python3 scripts/train_layer_single_gpu.py --layer 6  --gpu 2
    python3 scripts/train_layer_single_gpu.py --layer 20 --gpu 3
"""
import argparse
import os
import sys
import json
import time
import math

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.crosscoder.model import Crosscoder

N_WINDOWS = 3000
EARLY_STOP_PATIENCE = 1500
EARLY_STOP_WINDOW = 200


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def precompute_to_gpu(layer_idx, cfg, device, n_windows):
    """Load models, extract activations for n_windows, return GPU tensors."""
    from src.data.dataset import build_datasets
    from src.models.extractor import ModelExtractor

    hf_token = os.environ.get("HF_TOKEN")
    print(f"[GPU {device.index}] Loading dataset...", flush=True)
    train_ds, _, _ = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=hf_token
    )

    n_use = min(n_windows, len(train_ds))
    T, H = cfg.context_length, cfg.hidden_size

    print(f"[GPU {device.index}] Loading 3 models for layer {layer_idx} extraction...", flush=True)
    extractor = ModelExtractor(cfg, device, hf_token=hf_token, pt_sub_batch=16)

    pt_chunks, ft_chunks, ri_chunks = [], [], []
    batch_size = 32
    t0 = time.time()

    print(f"[GPU {device.index}] Extracting {n_use} windows...", flush=True)
    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        windows = [train_ds[i] for i in range(start, end)]

        with torch.no_grad():
            pt_acts, ft_acts, ri_acts = extractor.extract_all(windows, [layer_idx])

        pt_chunks.append(pt_acts[layer_idx].float())
        ft_chunks.append(ft_acts[layer_idx].float())
        ri_chunks.append(ri_acts[layer_idx].float())

        if (start // batch_size) % 10 == 0:
            done = end
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_use - done) / rate / 60 if rate > 0 else 0
            print(f"  [{device.index}] {done}/{n_use} ({done/n_use*100:.0f}%) "
                  f"{rate:.1f} win/s  ETA {eta:.1f}m", flush=True)

    pt_gpu = torch.cat(pt_chunks, dim=0).to(device)
    ft_gpu = torch.cat(ft_chunks, dim=0).to(device)
    ri_gpu = torch.cat(ri_chunks, dim=0).to(device)

    del extractor, pt_chunks, ft_chunks, ri_chunks
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    gb = (pt_gpu.nbytes + ft_gpu.nbytes + ri_gpu.nbytes) / 1e9
    print(f"[GPU {device.index}] Extraction done: {pt_gpu.shape[0]:,} samples, "
          f"{gb:.1f}GB on GPU ({elapsed/60:.1f}m)", flush=True)

    return pt_gpu, ft_gpu, ri_gpu


def load_from_disk(layer_idx, cfg, device, n_windows):
    """Load precomputed activations from disk memmaps into GPU."""
    T, H = cfg.context_length, cfg.hidden_size
    N_full = cfg.n_precompute_windows
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")
    shape = (N_full, T, H)

    n_use = min(n_windows, N_full)
    print(f"[GPU {device.index}] Loading {n_use} windows from disk...", flush=True)
    t0 = time.time()

    pt_mm = np.memmap(os.path.join(layer_dir, "pt.bin"), dtype="float16", mode="r", shape=shape)
    ft_mm = np.memmap(os.path.join(layer_dir, "ft.bin"), dtype="float16", mode="r", shape=shape)
    ri_mm = np.memmap(os.path.join(layer_dir, "ri.bin"), dtype="float16", mode="r", shape=shape)

    pt_gpu = torch.from_numpy(pt_mm[:n_use].copy()).float().reshape(n_use * T, H).to(device)
    ft_gpu = torch.from_numpy(ft_mm[:n_use].copy()).float().reshape(n_use * T, H).to(device)
    ri_gpu = torch.from_numpy(ri_mm[:n_use].copy()).float().reshape(n_use * T, H).to(device)
    del pt_mm, ft_mm, ri_mm

    gb = (pt_gpu.nbytes + ft_gpu.nbytes + ri_gpu.nbytes) / 1e9
    print(f"[GPU {device.index}] Loaded {pt_gpu.shape[0]:,} samples, "
          f"{gb:.1f}GB ({time.time()-t0:.1f}s)", flush=True)
    return pt_gpu, ft_gpu, ri_gpu


def train(layer_idx, cfg, device, pt_gpu, ft_gpu, ri_gpu):
    """Train crosscoder with early stopping."""
    T = cfg.context_length
    N_samples = pt_gpu.shape[0]

    cc = Crosscoder(cfg).to(device)
    n_params = sum(p.numel() for p in cc.parameters())
    print(f"[GPU {device.index}] Layer {layer_idx}: {n_params/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(
        cc.parameters(), lr=cfg.lr,
        betas=(cfg.adam_b1, cfg.adam_b2),
        weight_decay=cfg.weight_decay,
    )
    dead_counter = torch.zeros(cfg.latent_dim, device=device)
    dead_threshold = cfg.dead_neuron_threshold * cfg.dead_neuron_window

    BS = cfg.batch_size * T
    log_freq = 200
    t_start = time.time()

    best_avg_loss = float("inf")
    best_step = 0
    loss_accum = 0.0
    loss_count = 0

    print(f"[GPU {device.index}] Training: batch={cfg.batch_size} windows ({BS} samples), "
          f"steps={cfg.total_steps}, early_stop_patience={EARLY_STOP_PATIENCE}", flush=True)

    final_step = 0
    for step in range(cfg.total_steps):
        lr = cosine_lr(step, cfg.total_steps, cfg.warmup_steps, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        idx = torch.randint(0, N_samples, (BS,), device=device)
        x_pt = pt_gpu[idx]
        x_ft = ft_gpu[idx]
        x_ri = ri_gpu[idx]

        dead_mask = None
        if step >= cfg.auxk_start_step:
            dead_mask = (dead_counter < dead_threshold)

        opt.zero_grad()
        loss, (z_pt, z_ft, z_ri) = cc(x_pt, x_ft, x_ri, update_stats=True, dead_mask=dead_mask)
        loss.backward()
        nn.utils.clip_grad_norm_(cc.parameters(), max_norm=1.0)
        opt.step()
        cc.normalize_decoder_columns()

        with torch.no_grad():
            active = (
                (z_pt.abs() > 0) | (z_ft.abs() > 0) | (z_ri.abs() > 0)
            ).any(dim=0).float()
            dead_counter = dead_counter * 0.999 + active

        loss_val = loss.item()
        loss_accum += loss_val
        loss_count += 1
        final_step = step

        if step % log_freq == 0:
            avg_loss = loss_accum / max(loss_count, 1)
            dead_n = cc.count_dead_neurons(dead_counter, cfg.dead_neuron_window)
            elapsed = time.time() - t_start
            steps_per_sec = max(step, 1) / elapsed
            eta_min = (cfg.total_steps - step) / steps_per_sec / 60 if steps_per_sec > 0 else 0
            auxk_str = "off" if dead_mask is None else f"on({int(dead_mask.sum())}dead)"
            print(f"[{device.index}] step={step:6d}/{cfg.total_steps}  lr={lr:.2e}  "
                  f"loss={loss_val:.6f}  avg={avg_loss:.6f}  dead={dead_n:.0f}  "
                  f"auxk={auxk_str}  {steps_per_sec:.1f} step/s  ETA={eta_min:.1f}m",
                  flush=True)

            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                best_step = step
            loss_accum = 0.0
            loss_count = 0

        if step - best_step >= EARLY_STOP_PATIENCE and step >= cfg.warmup_steps + EARLY_STOP_PATIENCE:
            print(f"[{device.index}] Early stopping at step {step} "
                  f"(best avg={best_avg_loss:.6f} at step {best_step})", flush=True)
            break

    elapsed_total = time.time() - t_start
    final_loss = loss_val
    print(f"[GPU {device.index}] Training done: {final_step+1} steps in {elapsed_total/60:.1f}m, "
          f"final loss={final_loss:.6f}", flush=True)

    # Save
    out_dir = os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}")
    os.makedirs(out_dir, exist_ok=True)

    cc_cpu = cc.cpu()
    torch.save(cc_cpu.state_dict(), os.path.join(out_dir, "crosscoder.pt"))

    norm_stats = {}
    for domain in Crosscoder.DOMAINS:
        norm_stats[domain] = {
            "mean": getattr(cc_cpu, f"{domain}_mean").tolist(),
            "std": getattr(cc_cpu, f"{domain}_std").tolist(),
        }
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f)

    with open(os.path.join(out_dir, "train_info.json"), "w") as f:
        json.dump({
            "layer": layer_idx,
            "latent_dim": cfg.latent_dim,
            "linear": True,
            "batch_size": cfg.batch_size,
            "n_windows": N_WINDOWS,
            "n_samples": N_samples,
            "steps": final_step + 1,
            "final_loss": final_loss,
            "best_avg_loss": best_avg_loss,
            "elapsed_min": elapsed_total / 60,
        }, f, indent=2)

    print(f"[GPU {device.index}] Saved to {out_dir}/", flush=True)
    return final_loss, final_step + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    cfg = Config()
    cfg.linear_crosscoder = True
    cfg.latent_dim = 4096
    cfg.top_k = 64
    cfg.batch_size = 256
    cfg.total_steps = 10_000
    cfg.warmup_steps = 500
    cfg.checkpoint_dir = "checkpoints/linear_d4096"

    device = torch.device(f"cuda:{args.gpu}")

    # Check if precomputed activations exist on disk for this layer
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{args.layer}")
    T, H = cfg.context_length, cfg.hidden_size
    expected = cfg.n_precompute_windows * T * H * 2
    disk_exists = all(
        os.path.exists(os.path.join(layer_dir, f"{d}.bin"))
        and os.path.getsize(os.path.join(layer_dir, f"{d}.bin")) == expected
        for d in ("pt", "ft", "ri")
    )

    if disk_exists:
        pt_gpu, ft_gpu, ri_gpu = load_from_disk(args.layer, cfg, device, N_WINDOWS)
    else:
        pt_gpu, ft_gpu, ri_gpu = precompute_to_gpu(args.layer, cfg, device, N_WINDOWS)

    train(args.layer, cfg, device, pt_gpu, ft_gpu, ri_gpu)


if __name__ == "__main__":
    main()
