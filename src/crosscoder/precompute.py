"""
Precompute and save hidden-state activations.

Runs each model through N windows with hooks, saving float16 numpy memmaps:
  precompute_dir/layer_{i}/pt.bin   shape (N, T, H) float16
  precompute_dir/layer_{i}/ft.bin
  precompute_dir/layer_{i}/ri.bin

Supports extracting all layers or a subset (for single-layer experiments).
"""
import os
import json
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.models.extractor import ModelExtractor


def _precompute_worker(
    rank: int,
    cfg: Config,
    train_ds,
    hf_token: str | None,
    layer_indices: list[int],
):
    device = torch.device(f"cuda:{rank}")
    N = cfg.n_precompute_windows
    T = cfg.context_length
    H = cfg.hidden_size
    shard = N // cfg.num_gpus
    row_start = rank * shard
    row_end = row_start + shard

    print(f"[GPU {rank}] Loading models...", flush=True)
    extractor = ModelExtractor(cfg, device, hf_token=hf_token, pt_sub_batch=16)

    sub_ds = Subset(train_ds, list(range(row_start, row_end)))
    loader = DataLoader(
        sub_ds, batch_size=32, shuffle=False,
        collate_fn=lambda b: b, num_workers=0,
    )

    shape = (N, T, H)
    memmaps = {}
    for li in layer_indices:
        layer_dir = os.path.join(cfg.precompute_dir, f"layer_{li}")
        memmaps[li] = {
            "pt": np.memmap(os.path.join(layer_dir, "pt.bin"),
                            dtype="float16", mode="r+", shape=shape),
            "ft": np.memmap(os.path.join(layer_dir, "ft.bin"),
                            dtype="float16", mode="r+", shape=shape),
            "ri": np.memmap(os.path.join(layer_dir, "ri.bin"),
                            dtype="float16", mode="r+", shape=shape),
        }

    cursor = row_start
    t0 = time.time()

    for batch_idx, windows in enumerate(loader):
        with torch.no_grad():
            pt_acts, ft_acts, ri_acts = extractor.extract_all(
                windows, layer_indices
            )

        B = len(windows)
        for li in layer_indices:
            memmaps[li]["pt"][cursor:cursor + B] = (
                pt_acts[li].cpu().half().numpy().reshape(B, T, H))
            memmaps[li]["ft"][cursor:cursor + B] = (
                ft_acts[li].cpu().half().numpy().reshape(B, T, H))
            memmaps[li]["ri"][cursor:cursor + B] = (
                ri_acts[li].cpu().half().numpy().reshape(B, T, H))
        cursor += B

        if batch_idx % 10 == 0 or cursor >= row_end:
            done = cursor - row_start
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (shard - done) / rate if rate > 0 else 0
            print(
                f"[GPU {rank}] {done}/{shard} "
                f"({done/shard*100:.0f}%)  "
                f"{rate:.1f} win/s  ETA {eta/60:.1f}m",
                flush=True,
            )

    for li in layer_indices:
        for mm in memmaps[li].values():
            mm.flush()
    print(f"[GPU {rank}] Done (rows {row_start}–{row_end-1}).", flush=True)


def precompute(
    cfg: Config,
    train_ds,
    hf_token: str | None,
    layer_indices: list[int] | None = None,
):
    """
    Pre-allocate memmaps and spawn GPU workers.
    layer_indices: which layers to extract (default: all).
    """
    if layer_indices is None:
        layer_indices = list(range(cfg.num_layers))

    N, T, H = cfg.n_precompute_windows, cfg.context_length, cfg.hidden_size
    shape = (N, T, H)
    expected_bytes = N * T * H * 2

    for li in layer_indices:
        layer_dir = os.path.join(cfg.precompute_dir, f"layer_{li}")
        os.makedirs(layer_dir, exist_ok=True)
        for domain in ("pt", "ft", "ri"):
            path = os.path.join(layer_dir, f"{domain}.bin")
            meta = os.path.join(layer_dir, f"{domain}.json")
            if not (os.path.exists(path) and os.path.getsize(path) == expected_bytes):
                mm = np.memmap(path, dtype="float16", mode="w+", shape=shape)
                del mm
            with open(meta, "w") as f:
                json.dump({"shape": list(shape), "dtype": "float16"}, f)

    total_gb = len(layer_indices) * 3 * expected_bytes / 1e9
    print(f"Extracting layers {layer_indices}: "
          f"{N:,} windows, {total_gb:.1f} GB on disk", flush=True)

    t0 = time.time()
    mp.spawn(
        _precompute_worker,
        args=(cfg, train_ds, hf_token, layer_indices),
        nprocs=cfg.num_gpus,
        join=True,
    )
    elapsed = time.time() - t0
    print(f"Precompute done in {elapsed/60:.1f}m.", flush=True)


def precompute_layer(
    layer_idx: int,
    cfg: Config,
    train_ds,
    hf_token: str | None,
):
    """Convenience wrapper: precompute activations for a single layer."""
    precompute(cfg, train_ds, hf_token, layer_indices=[layer_idx])
