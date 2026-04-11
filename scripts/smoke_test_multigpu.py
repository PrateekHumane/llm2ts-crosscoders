"""
Multi-GPU smoke test with data-parallel extraction (10 steps).
"""
import os
import sys
import copy
import glob
import time
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import build_datasets
from src.crosscoder.train import train_worker

if __name__ == "__main__":
    HF_TOKEN = os.environ.get("HF_TOKEN")
    cfg = Config()

    test_cfg = copy.copy(cfg)
    test_cfg.total_steps    = 10
    test_cfg.batch_size     = 16   # 4 windows per GPU shard
    test_cfg.num_gpus       = 4
    test_cfg.layers_per_gpu = 7
    test_cfg.checkpoint_dir = "/tmp/crosscoder_smoke_test2"

    print("Loading GiftEval...")
    train_ds, _, _ = build_datasets(
        test_cfg.context_length, test_cfg.train_frac, test_cfg.val_frac, HF_TOKEN
    )
    print(f"Train windows: {len(train_ds):,}")
    print(f"\nLaunching {test_cfg.num_gpus} GPU workers, 10 steps each...\n")

    t0 = time.time()
    mp.spawn(
        train_worker,
        args=(test_cfg, train_ds, HF_TOKEN),
        nprocs=test_cfg.num_gpus,
        join=True,
    )
    elapsed = time.time() - t0

    ckpts = sorted(glob.glob(f"{test_cfg.checkpoint_dir}/layer_*/crosscoder.pt"))
    print(f"\nCheckpoints saved: {len(ckpts)}")
    print(f"Total time for 10 steps: {elapsed:.1f}s  ({elapsed/10:.2f}s/step)")

    # Estimate full training time
    step_time = elapsed / 10
    for n_steps in [20_000, 100_000]:
        hours = step_time * n_steps / 3600
        print(f"  Estimated {n_steps:,} steps: {hours:.1f}h ({hours/24:.1f} days)")

    print("\n=== MULTI-GPU SMOKE TEST PASSED ✓ ===")
