"""
Fast training of linear crosscoder on layer 13.
Loads a subset of precomputed activations directly into GPU VRAM.

Usage:
    python3 scripts/train_layer13_fast.py
"""
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

LAYER = 13
N_WINDOWS = 3000  # ~19GB on GPU in float32 (3 domains)


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    cfg = Config()
    cfg.linear_crosscoder = True
    cfg.latent_dim = 4096
    cfg.top_k = 64
    cfg.batch_size = 256
    cfg.total_steps = 10_000
    cfg.warmup_steps = 500
    cfg.checkpoint_dir = "checkpoints/linear_d4096"

    device = torch.device("cuda:0")
    H = cfg.hidden_size
    T = cfg.context_length
    N_full = cfg.n_precompute_windows

    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{LAYER}")
    shape = (N_full, T, H)

    # Load subset into GPU VRAM
    print(f"Loading {N_WINDOWS} windows into GPU...", flush=True)
    t0 = time.time()
    pt_mm = np.memmap(os.path.join(layer_dir, "pt.bin"), dtype="float16", mode="r", shape=shape)
    ft_mm = np.memmap(os.path.join(layer_dir, "ft.bin"), dtype="float16", mode="r", shape=shape)
    ri_mm = np.memmap(os.path.join(layer_dir, "ri.bin"), dtype="float16", mode="r", shape=shape)

    # Read sequentially (fast) and move to GPU
    pt_gpu = torch.from_numpy(pt_mm[:N_WINDOWS].copy()).float().reshape(N_WINDOWS * T, H).to(device)
    ft_gpu = torch.from_numpy(ft_mm[:N_WINDOWS].copy()).float().reshape(N_WINDOWS * T, H).to(device)
    ri_gpu = torch.from_numpy(ri_mm[:N_WINDOWS].copy()).float().reshape(N_WINDOWS * T, H).to(device)
    del pt_mm, ft_mm, ri_mm

    N_samples = N_WINDOWS * T
    gb_used = (pt_gpu.nbytes + ft_gpu.nbytes + ri_gpu.nbytes) / 1e9
    print(f"  Loaded: {N_samples:,} samples, {gb_used:.1f}GB on GPU ({time.time()-t0:.1f}s)", flush=True)

    cc = Crosscoder(cfg).to(device)
    print(f"Linear crosscoder: {sum(p.numel() for p in cc.parameters())/1e6:.2f}M params", flush=True)

    opt = torch.optim.AdamW(
        cc.parameters(), lr=cfg.lr,
        betas=(cfg.adam_b1, cfg.adam_b2),
        weight_decay=cfg.weight_decay,
    )
    dead_counter = torch.zeros(cfg.latent_dim, device=device)
    dead_threshold = cfg.dead_neuron_threshold * cfg.dead_neuron_window

    BS = cfg.batch_size * T  # samples per step (256 windows × 512 timesteps)
    log_freq = 200
    t_start = time.time()

    print(f"Training: batch={cfg.batch_size} windows ({BS} samples), "
          f"steps={cfg.total_steps}, lr={cfg.lr}", flush=True)

    for step in range(cfg.total_steps):
        lr = cosine_lr(step, cfg.total_steps, cfg.warmup_steps, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Random sample from GPU tensor
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

        if step % log_freq == 0:
            dead_n = cc.count_dead_neurons(dead_counter, cfg.dead_neuron_window)
            elapsed = time.time() - t_start
            steps_per_sec = max(step, 1) / elapsed
            eta_min = (cfg.total_steps - step) / steps_per_sec / 60 if steps_per_sec > 0 else 0
            auxk_str = "off" if dead_mask is None else f"on({int(dead_mask.sum())}dead)"
            print(f"step={step:6d}/{cfg.total_steps}  lr={lr:.2e}  loss={loss.item():.6f}  "
                  f"dead={dead_n:.0f}  auxk={auxk_str}  "
                  f"{steps_per_sec:.1f} step/s  ETA={eta_min:.1f}m", flush=True)

    elapsed_total = time.time() - t_start
    final_loss = loss.item()
    print(f"\nTraining done: {cfg.total_steps} steps in {elapsed_total/60:.1f}m, "
          f"final loss={final_loss:.6f}", flush=True)

    # Save
    out_dir = os.path.join(cfg.checkpoint_dir, f"layer_{LAYER}")
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
            "layer": LAYER,
            "latent_dim": cfg.latent_dim,
            "linear": True,
            "batch_size": cfg.batch_size,
            "n_windows": N_WINDOWS,
            "n_samples": N_samples,
            "steps": cfg.total_steps,
            "final_loss": final_loss,
            "elapsed_min": elapsed_total / 60,
        }, f, indent=2)

    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
