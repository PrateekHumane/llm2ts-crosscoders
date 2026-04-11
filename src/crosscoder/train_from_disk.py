"""
Train a single crosscoder from precomputed activation memmaps using DDP.

Expected files (written by precompute.py):
  precompute_dir/layer_{i}/pt.bin   (N, T, H) float16
  precompute_dir/layer_{i}/ft.bin
  precompute_dir/layer_{i}/ri.bin
"""
import os
import json
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from src.config import Config
from src.crosscoder.model import Crosscoder


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PrecomputedDataset(Dataset):
    """
    Memory-mapped dataset of precomputed activations for one layer.
    Returns float32 tensors of shape (T, H) per sample (window).
    """

    def __init__(self, layer_dir: str, cfg: Config):
        N, T, H = cfg.n_precompute_windows, cfg.context_length, cfg.hidden_size
        self.shape = (N, T, H)
        self.N = N
        self.pt = np.memmap(os.path.join(layer_dir, "pt.bin"),
                            dtype="float16", mode="r", shape=self.shape)
        self.ft = np.memmap(os.path.join(layer_dir, "ft.bin"),
                            dtype="float16", mode="r", shape=self.shape)
        self.ri = np.memmap(os.path.join(layer_dir, "ri.bin"),
                            dtype="float16", mode="r", shape=self.shape)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.pt[idx].copy()).float(),
            torch.from_numpy(self.ft[idx].copy()).float(),
            torch.from_numpy(self.ri[idx].copy()).float(),
        )


def _collate(batch):
    pt, ft, ri = zip(*batch)
    return torch.stack(pt), torch.stack(ft), torch.stack(ri)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------

def check_convergence(loss_history: list[float], window: int = 2000,
                      threshold: float = 0.005) -> bool:
    """
    Check if loss has converged: compare mean of last `window` losses
    to mean of the `window` before that. If relative improvement < threshold,
    we've converged.
    """
    if len(loss_history) < 2 * window:
        return False
    recent = sum(loss_history[-window:]) / window
    prev = sum(loss_history[-2*window:-window]) / window
    rel_improvement = (prev - recent) / (abs(prev) + 1e-8)
    return rel_improvement < threshold


# ---------------------------------------------------------------------------
# DDP Training
# ---------------------------------------------------------------------------

def _ddp_train_worker(
    rank: int,
    world_size: int,
    layer_idx: int,
    cfg: Config,
    result_dict: dict,
):
    """DDP worker: trains crosscoder on one GPU shard of the data."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")
    H = cfg.hidden_size

    ds = PrecomputedDataset(layer_dir, cfg)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True)
    # batch_size is per-GPU; effective batch = batch_size * world_size
    per_gpu_batch = cfg.batch_size // world_size
    # num_workers=0: avoid forking with large memmaps (409GB).
    # Memmap reads are fast enough without prefetching.
    loader = DataLoader(
        ds, batch_size=per_gpu_batch, sampler=sampler,
        collate_fn=_collate, num_workers=0, pin_memory=True,
    )

    cc = Crosscoder(cfg).to(device)
    cc_ddp = DDP(cc, device_ids=[rank])
    opt = torch.optim.AdamW(
        cc_ddp.parameters(), lr=cfg.lr,
        betas=(cfg.adam_b1, cfg.adam_b2),
        weight_decay=cfg.weight_decay,
    )
    dead_counter = torch.zeros(cfg.latent_dim, device=device)
    dead_threshold = cfg.dead_neuron_threshold * cfg.dead_neuron_window

    step = 0
    epoch = 0
    t_start = time.time()
    log_freq = 500
    loss_history = []
    early_stop = cfg.total_steps  # may be lowered by convergence check

    if rank == 0:
        print(f"[L{layer_idx}] DDP Training: "
              f"{len(ds):,} windows, batch={cfg.batch_size} "
              f"({per_gpu_batch}/GPU × {world_size} GPUs), "
              f"steps≤{cfg.total_steps:,}, auxk={cfg.auxk_coeff:.4f}",
              flush=True)

    while step < early_stop:
        sampler.set_epoch(epoch)
        epoch += 1
        for pt_b, ft_b, ri_b in loader:
            if step >= early_stop:
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
                steps_remain = early_stop - step
                eta_h = (elapsed / max(step, 1)) * steps_remain / 3600
                auxk_str = "off" if dead_mask is None else f"on({int(dead_mask.sum())}dead)"
                avg_loss = sum(loss_history[-log_freq:]) / min(len(loss_history), log_freq)
                print(
                    f"[L{layer_idx}] step={step:6d}/{early_stop}  "
                    f"lr={lr:.2e}  loss={avg_loss:.4f}  "
                    f"dead={dead_n:.0f}  auxk={auxk_str}  "
                    f"t={elapsed/60:.1f}m  ETA={eta_h:.1f}h",
                    flush=True,
                )

            # Convergence check on rank 0, broadcast decision
            if step > 0 and step % 2000 == 0:
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
                    early_stop = step
                    break

            step += 1

    elapsed_total = time.time() - t_start

    # Sync normalization stats across ranks (average them)
    for d in Crosscoder.DOMAINS:
        for stat in ["mean", "std"]:
            buf = getattr(cc, f"{d}_{stat}")
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            buf.div_(world_size)

    if rank == 0:
        dead_n = cc.count_dead_neurons(dead_counter, cfg.dead_neuron_window)
        print(f"[L{layer_idx}] Training done: {step} steps in "
              f"{elapsed_total/60:.1f}m ({elapsed_total/3600:.2f}h). "
              f"Dead={dead_n:.0f}", flush=True)
        result_dict["model_state"] = cc.cpu().state_dict()
        result_dict["steps"] = step
        result_dict["elapsed_min"] = elapsed_total / 60
        result_dict["final_loss"] = sum(loss_history[-500:]) / min(len(loss_history), 500)

    dist.destroy_process_group()


def train_crosscoder_ddp(
    layer_idx: int,
    cfg: Config,
) -> tuple[Crosscoder, dict]:
    """
    Train crosscoder using DDP across all GPUs.
    Returns (trained Crosscoder on CPU, info dict).
    """
    world_size = cfg.num_gpus
    manager = mp.Manager()
    result_dict = manager.dict()

    mp.spawn(
        _ddp_train_worker,
        args=(world_size, layer_idx, cfg, result_dict),
        nprocs=world_size,
        join=True,
    )

    cc = Crosscoder(cfg)
    cc.load_state_dict(result_dict["model_state"])
    info = {
        "steps": result_dict["steps"],
        "elapsed_min": result_dict["elapsed_min"],
        "final_loss": result_dict["final_loss"],
    }
    return cc, info


# ---------------------------------------------------------------------------
# Single-GPU training (kept for backward compat / small experiments)
# ---------------------------------------------------------------------------

def train_crosscoder_from_disk(
    layer_idx: int,
    cfg: Config,
    device: torch.device | None = None,
) -> Crosscoder:
    """
    Train crosscoder for layer_idx from precomputed activations (single GPU).
    Returns the trained Crosscoder on CPU.
    """
    if device is None:
        device = torch.device("cuda:0")

    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")

    ds = PrecomputedDataset(layer_dir, cfg)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    cc = Crosscoder(cfg).to(device)
    opt = torch.optim.AdamW(
        cc.parameters(), lr=cfg.lr,
        betas=(cfg.adam_b1, cfg.adam_b2),
        weight_decay=cfg.weight_decay,
    )
    dead_counter = torch.zeros(cfg.latent_dim, device=device)
    dead_threshold = cfg.dead_neuron_threshold * cfg.dead_neuron_window

    step = 0
    epoch = 0
    t_start = time.time()
    log_freq = 200
    H = cfg.hidden_size

    print(f"[L{layer_idx}] Training: "
          f"{len(ds):,} windows, batch={cfg.batch_size}, "
          f"steps={cfg.total_steps:,}", flush=True)

    while step < cfg.total_steps:
        epoch += 1
        for pt_b, ft_b, ri_b in loader:
            if step >= cfg.total_steps:
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
                loss, (z_pt, z_ft, z_ri) = cc(x_pt, x_ft, x_ri,
                                               update_stats=True,
                                               dead_mask=dead_mask)
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
                steps_remain = cfg.total_steps - step
                eta_h = (elapsed / max(step, 1)) * steps_remain / 3600
                print(
                    f"[L{layer_idx}] step={step:6d}/{cfg.total_steps}  "
                    f"lr={lr:.2e}  loss={loss.item():.4f}  "
                    f"dead={dead_n:.0f}  "
                    f"t={elapsed/60:.1f}m  ETA={eta_h:.1f}h",
                    flush=True,
                )
            step += 1

    elapsed_total = time.time() - t_start
    print(f"[L{layer_idx}] Training done in {elapsed_total/60:.1f}m.", flush=True)
    return cc.cpu()


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_crosscoder(cc: Crosscoder, layer_idx: int, cfg: Config):
    """Save crosscoder state dict and normalization stats to checkpoint_dir."""
    layer_dir = os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)

    torch.save(cc.state_dict(), os.path.join(layer_dir, "crosscoder.pt"))

    norm_stats = {}
    for domain in Crosscoder.DOMAINS:
        norm_stats[domain] = {
            "mean": getattr(cc, f"{domain}_mean").cpu().tolist(),
            "std":  getattr(cc, f"{domain}_std").cpu().tolist(),
        }
    with open(os.path.join(layer_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f)

    print(f"[L{layer_idx}] Checkpoint → {layer_dir}/", flush=True)
