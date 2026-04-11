"""
DDP training script for a single layer. Launched via torchrun:

    torchrun --nproc_per_node=4 scripts/train_layer_ddp.py \
        --layer 13 --total_steps 100000

Each process opens its own memmaps (no fork issues with large files).
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.crosscoder.model import Crosscoder


class PrecomputedDataset(Dataset):
    def __init__(self, layer_dir, N, T, H):
        shape = (N, T, H)
        self.N = N
        self.pt = np.memmap(os.path.join(layer_dir, "pt.bin"),
                            dtype="float16", mode="r", shape=shape)
        self.ft = np.memmap(os.path.join(layer_dir, "ft.bin"),
                            dtype="float16", mode="r", shape=shape)
        self.ri = np.memmap(os.path.join(layer_dir, "ri.bin"),
                            dtype="float16", mode="r", shape=shape)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (torch.from_numpy(self.pt[idx].copy()).float(),
                torch.from_numpy(self.ft[idx].copy()).float(),
                torch.from_numpy(self.ri[idx].copy()).float())


def collate(batch):
    pt, ft, ri = zip(*batch)
    return torch.stack(pt), torch.stack(ft), torch.stack(ri)


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def check_convergence(loss_history, window=2000, threshold=0.005):
    if len(loss_history) < 2 * window:
        return False
    recent = sum(loss_history[-window:]) / window
    prev = sum(loss_history[-2 * window:-window]) / window
    rel_improvement = (prev - recent) / (abs(prev) + 1e-8)
    return rel_improvement < threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--early_stop", action="store_true", default=True,
                        help="Enable convergence-based early stopping")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    cfg = Config()
    cfg.auxk_coeff = 1 / 4
    cfg.auxk_start_step = 1000
    cfg.total_steps = args.total_steps
    layer_idx = args.layer

    N, T, H = cfg.n_precompute_windows, cfg.context_length, cfg.hidden_size
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")

    ds = PrecomputedDataset(layer_dir, N, T, H)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True)
    per_gpu_batch = cfg.batch_size // world_size
    loader = DataLoader(ds, batch_size=per_gpu_batch, sampler=sampler,
                        collate_fn=collate, num_workers=2, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)

    cc = Crosscoder(cfg).to(device)
    cc_ddp = DDP(cc, device_ids=[rank])
    opt = torch.optim.AdamW(cc_ddp.parameters(), lr=cfg.lr,
                            betas=(cfg.adam_b1, cfg.adam_b2),
                            weight_decay=cfg.weight_decay)
    dead_counter = torch.zeros(cfg.latent_dim, device=device)
    dead_threshold = cfg.dead_neuron_threshold * cfg.dead_neuron_window

    step = 0
    epoch = 0
    t_start = time.time()
    log_freq = 500
    loss_history = []
    early_stop_step = cfg.total_steps

    if rank == 0:
        print(f"[L{layer_idx}] DDP Training: "
              f"{N:,} windows, batch={cfg.batch_size} "
              f"({per_gpu_batch}/GPU × {world_size} GPUs), "
              f"steps≤{cfg.total_steps:,}, auxk={cfg.auxk_coeff:.4f}",
              flush=True)

    while step < early_stop_step:
        sampler.set_epoch(epoch)
        epoch += 1
        for pt_b, ft_b, ri_b in loader:
            if step >= early_stop_step:
                break

            lr = cosine_lr(step, cfg.total_steps, cfg.warmup_steps, cfg.lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            x_pt = pt_b.to(device, non_blocking=True).reshape(-1, H)
            x_ft = ft_b.to(device, non_blocking=True).reshape(-1, H)
            x_ri = ri_b.to(device, non_blocking=True).reshape(-1, H)

            dead_mask = None
            if step >= cfg.auxk_start_step:
                dead_mask = (dead_counter < dead_threshold)

            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, (z_pt, z_ft, z_ri) = cc_ddp(
                    x_pt, x_ft, x_ri,
                    update_stats=True,
                    dead_mask=dead_mask,
                )
            loss.backward()
            nn.utils.clip_grad_norm_(cc_ddp.parameters(), max_norm=1.0)
            opt.step()
            cc.normalize_decoder_columns()

            with torch.no_grad():
                active = (
                    (z_pt.abs() > 0) | (z_ft.abs() > 0) | (z_ri.abs() > 0)
                ).any(dim=0).float()
                dead_counter = dead_counter * 0.999 + active

            loss_val = loss.item()
            loss_history.append(loss_val)

            if rank == 0 and step % log_freq == 0:
                dead_n = cc.count_dead_neurons(dead_counter, cfg.dead_neuron_window)
                elapsed = time.time() - t_start
                steps_remain = early_stop_step - step
                eta_h = (elapsed / max(step, 1)) * steps_remain / 3600
                auxk_str = "off" if dead_mask is None else f"on({int(dead_mask.sum())}dead)"
                avg_loss = sum(loss_history[-log_freq:]) / min(len(loss_history), log_freq)
                print(
                    f"[L{layer_idx}] step={step:6d}/{early_stop_step}  "
                    f"lr={lr:.2e}  loss={avg_loss:.4f}  "
                    f"dead={dead_n:.0f}  auxk={auxk_str}  "
                    f"t={elapsed/60:.1f}m  ETA={eta_h:.1f}h",
                    flush=True,
                )

            # Convergence check every 2000 steps
            if args.early_stop and step > 0 and step % 2000 == 0:
                converged = False
                if rank == 0 and step >= cfg.warmup_steps + 4000:
                    converged = check_convergence(loss_history)
                    if converged:
                        print(f"[L{layer_idx}] Loss converged at step {step}.",
                              flush=True)
                conv_tensor = torch.tensor([1 if converged else 0],
                                           device=device)
                dist.broadcast(conv_tensor, src=0)
                if conv_tensor.item() == 1:
                    early_stop_step = step
                    break

            step += 1

    # Sync norm stats across ranks
    for d in Crosscoder.DOMAINS:
        for stat in ["mean", "std"]:
            buf = getattr(cc, f"{d}_{stat}")
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf.div_(world_size)

    elapsed_total = time.time() - t_start

    if rank == 0:
        dead_n = cc.count_dead_neurons(dead_counter, cfg.dead_neuron_window)
        avg_loss = sum(loss_history[-500:]) / min(len(loss_history), 500)
        print(f"[L{layer_idx}] Training done: {step} steps in "
              f"{elapsed_total/60:.1f}m. Dead={dead_n:.0f} "
              f"Loss={avg_loss:.4f}", flush=True)

        # Save checkpoint
        ckpt_dir = os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}")
        os.makedirs(ckpt_dir, exist_ok=True)
        cc_cpu = cc.cpu()
        torch.save(cc_cpu.state_dict(), os.path.join(ckpt_dir, "crosscoder.pt"))

        norm_stats = {}
        for domain in Crosscoder.DOMAINS:
            norm_stats[domain] = {
                "mean": getattr(cc_cpu, f"{domain}_mean").tolist(),
                "std": getattr(cc_cpu, f"{domain}_std").tolist(),
            }
        with open(os.path.join(ckpt_dir, "norm_stats.json"), "w") as f:
            json.dump(norm_stats, f)

        # Save training info for the pipeline to read
        info = {
            "layer": layer_idx,
            "steps": step,
            "elapsed_min": elapsed_total / 60,
            "final_loss": avg_loss,
            "dead_features": int(dead_n),
        }
        with open(os.path.join(ckpt_dir, "train_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"[L{layer_idx}] Checkpoint → {ckpt_dir}/", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
