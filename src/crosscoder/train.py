"""
Training loop with data-parallel activation extraction.

Design:
- Each GPU processes batch_size // num_gpus windows (a shard of the full batch).
- Each GPU extracts activations for ALL 28 layers from its shard.
- torch.distributed.all_gather combines shards so every GPU has the full batch's
  activations for its own assigned layers.
- Each GPU trains its 7 crosscoders on the full combined batch.

This eliminates redundant PT computation: instead of 4 GPUs each running PT on 64
windows independently, all 4 GPUs collectively run PT on 64 total unique windows
(16 each), then share results. Wall-time speedup: ~4× for extraction.
"""
import os
import json
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from src.config import Config
from src.crosscoder.model import Crosscoder
from src.models.extractor import ModelExtractor
from src.data.dataset import WindowDataset


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def collate_fn(batch):
    return batch


def _all_gather_activations(
    local_acts: dict[int, torch.Tensor],
    world_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """
    All-gather per-layer activations across GPUs.
    local_acts: {layer_idx: (local_B*T, H)}
    Returns: {layer_idx: (global_B*T, H)} — concatenated across all GPUs.
    """
    gathered = {}
    for layer_idx, x in local_acts.items():
        # x: (local_B*T, H)
        shards = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(shards, x.contiguous())
        gathered[layer_idx] = torch.cat(shards, dim=0)   # (global_B*T, H)
    return gathered


def train_worker(
    rank: int,
    cfg: Config,
    train_ds: WindowDataset,
    hf_token: str | None = None,
):
    """
    Main training function for one GPU worker.

    Each worker:
    - Is assigned layers [rank*layers_per_gpu, (rank+1)*layers_per_gpu).
    - Processes a shard of the batch (batch_size // num_gpus windows).
    - Participates in all_gather to collect full-batch activations.
    - Trains its assigned crosscoders on the full-batch activations.
    """
    device = torch.device(f"cuda:{rank}")
    layer_start   = rank * cfg.layers_per_gpu
    layer_end     = min(layer_start + cfg.layers_per_gpu, cfg.num_layers)
    n_layers      = layer_end - layer_start
    layer_indices = list(range(layer_start, layer_end))

    # ── Distributed init ─────────────────────────────────────────────────────
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=cfg.num_gpus)
    torch.cuda.set_device(device)

    print(f"[GPU {rank}] Starting — layers {layer_start}–{layer_end-1}")

    # ── Load models ───────────────────────────────────────────────────────────
    # pt_sub_batch sized for the shard: batch_size / num_gpus windows
    shard_size    = cfg.batch_size // cfg.num_gpus
    pt_sub_batch  = min(16, shard_size)
    extractor     = ModelExtractor(cfg, device, hf_token=hf_token,
                                   pt_sub_batch=pt_sub_batch)

    # ── Create crosscoders ────────────────────────────────────────────────────
    crosscoders = [Crosscoder(cfg).to(device) for _ in range(n_layers)]
    optimizers  = [
        torch.optim.AdamW(
            cc.parameters(), lr=cfg.lr,
            betas=(cfg.adam_b1, cfg.adam_b2),
            weight_decay=cfg.weight_decay,
        )
        for cc in crosscoders
    ]
    dead_counters = [torch.zeros(cfg.latent_dim, device=device)
                     for _ in range(n_layers)]

    # ── Data loader (distributed sampler: each GPU sees a disjoint shard) ────
    sampler = DistributedSampler(
        train_ds,
        num_replicas=cfg.num_gpus,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    dataloader = DataLoader(
        train_ds,
        batch_size=shard_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    step    = 0
    t_start = time.time()
    log_freq = 100
    # Each pass through the dataloader is one "epoch" of the shard.
    # We loop epochs until we reach total_steps.
    epoch = 0

    while step < cfg.total_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for shard_batch in dataloader:
            if step >= cfg.total_steps:
                break

            lr = cosine_lr(step, cfg.total_steps, cfg.warmup_steps, cfg.lr)
            for opt in optimizers:
                for pg in opt.param_groups:
                    pg["lr"] = lr

            # ── Extract activations from this GPU's shard ──────────────────
            # Each GPU extracts ALL 28 layers from its shard_size windows.
            # We only need our own layers, but we share with others.
            # Using extract_all with all layer indices so each GPU can
            # all_gather to give every GPU its needed full-batch activations.
            with torch.no_grad():
                pt_local, ft_local, ri_local = extractor.extract_all(
                    shard_batch, layer_indices
                )

            # ── All-gather: combine shards into full-batch activations ─────
            pt_full = _all_gather_activations(pt_local, cfg.num_gpus, device)
            ft_full = _all_gather_activations(ft_local, cfg.num_gpus, device)
            ri_full = _all_gather_activations(ri_local, cfg.num_gpus, device)

            # ── Train each assigned crosscoder on the full batch ───────────
            losses = []
            for i, layer_idx in enumerate(layer_indices):
                cc  = crosscoders[i]
                opt = optimizers[i]

                x_pt = pt_full[layer_idx]
                x_ft = ft_full[layer_idx]
                x_ri = ri_full[layer_idx]

                opt.zero_grad()
                loss, (z_pt, z_ft, z_ri) = cc(x_pt, x_ft, x_ri, update_stats=True)
                loss.backward()
                nn.utils.clip_grad_norm_(cc.parameters(), max_norm=1.0)
                opt.step()
                cc.normalize_decoder_columns()

                with torch.no_grad():
                    active = (
                        (z_pt.abs() > 0) | (z_ft.abs() > 0) | (z_ri.abs() > 0)
                    ).any(dim=0).float()
                    dead_counters[i] = dead_counters[i] * 0.999 + active
                losses.append(loss.item())

            if step % log_freq == 0:
                avg_loss  = sum(losses) / len(losses)
                elapsed   = time.time() - t_start
                dead_counts = [cc.count_dead_neurons(dc, cfg.dead_neuron_window)
                               for cc, dc in zip(crosscoders, dead_counters)]
                avg_dead  = sum(dead_counts) / len(dead_counts)
                print(
                    f"[GPU {rank}] step={step:6d}/{cfg.total_steps}  "
                    f"lr={lr:.2e}  loss={avg_loss:.4f}  "
                    f"dead_avg={avg_dead:.0f}  "
                    f"t={elapsed/60:.1f}m"
                )
            step += 1

    # ── Save checkpoints ─────────────────────────────────────────────────────
    if rank == 0:
        print("Saving checkpoints...")
    for i, layer_idx in enumerate(layer_indices):
        layer_dir = os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        cc = crosscoders[i]
        torch.save(cc.state_dict(), os.path.join(layer_dir, "crosscoder.pt"))
        norm_stats = {}
        for domain in Crosscoder.DOMAINS:
            norm_stats[domain] = {
                "mean": getattr(cc, f"{domain}_mean").cpu().tolist(),
                "std":  getattr(cc, f"{domain}_std").cpu().tolist(),
            }
        with open(os.path.join(layer_dir, "norm_stats.json"), "w") as f:
            json.dump(norm_stats, f)

    dist.destroy_process_group()
    print(f"[GPU {rank}] Done.")
