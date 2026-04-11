"""
Smoke test for the precompute-then-train pipeline.

Runs precompute for layers 0 and 1 with a tiny window count (64 windows),
then trains each crosscoder for 10 steps, then deletes the files.
Expected runtime: a few minutes.

Usage:
    source ~/.bashrc && python scripts/smoke_test_precompute.py
"""
import os
import sys
import copy
import glob
import shutil
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import build_datasets
from src.crosscoder.precompute import precompute_layer
from src.crosscoder.train_from_disk import train_crosscoder_from_disk, save_crosscoder


def main():
    hf_token = os.environ.get("HF_TOKEN")
    cfg = Config()

    # Override for smoke test
    test_cfg = copy.copy(cfg)
    test_cfg.n_precompute_windows = 64    # 64 total (16 per GPU)
    test_cfg.total_steps          = 10
    test_cfg.batch_size           = 16
    test_cfg.warmup_steps         = 2
    test_cfg.precompute_dir       = "/tmp/smoke_precompute"
    test_cfg.checkpoint_dir       = "/tmp/smoke_checkpoints"

    print("Loading GiftEval (tiny subset)...")
    train_ds, _, _ = build_datasets(
        test_cfg.context_length, test_cfg.train_frac, test_cfg.val_frac,
        hf_token=hf_token,
    )
    print(f"  Available windows: {len(train_ds):,}")

    os.makedirs(test_cfg.precompute_dir, exist_ok=True)
    os.makedirs(test_cfg.checkpoint_dir, exist_ok=True)

    test_layers = [0, 1]
    t0 = time.time()

    for layer_idx in test_layers:
        print(f"\n--- Layer {layer_idx} ---")

        print("  Precomputing activations...")
        t_pre = time.time()
        precompute_layer(layer_idx, test_cfg, train_ds, hf_token)
        print(f"  Precompute: {time.time()-t_pre:.1f}s")

        print("  Training crosscoder (10 steps)...")
        t_train = time.time()
        cc = train_crosscoder_from_disk(layer_idx, test_cfg)
        print(f"  Train: {time.time()-t_train:.1f}s")

        save_crosscoder(cc, layer_idx, test_cfg)

        # Clean up layer acts
        layer_dir = os.path.join(test_cfg.precompute_dir, f"layer_{layer_idx}")
        shutil.rmtree(layer_dir)
        print(f"  Deleted activation files for layer {layer_idx}.")

    elapsed = time.time() - t0
    ckpts = sorted(glob.glob(f"{test_cfg.checkpoint_dir}/layer_*/crosscoder.pt"))
    print(f"\nCheckpoints saved: {len(ckpts)}/{len(test_layers)}")
    for c in ckpts:
        print(f"  {c}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("\n=== SMOKE TEST PASSED ✓ ===")


if __name__ == "__main__":
    main()
