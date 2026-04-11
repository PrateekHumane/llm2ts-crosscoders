"""
Iterate on a single layer: precompute activations, train crosscoder, monitor.

Usage:
    source ~/.bashrc && python3 scripts/iterate_layer.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import time
from src.config import Config
from src.data.dataset import build_datasets
from src.crosscoder.precompute import precompute
from src.crosscoder.train_from_disk import train_crosscoder_from_disk, save_crosscoder

LAYER = 16
N_WINDOWS = 5000
TOTAL_STEPS = 20_000
AUXK_COEFF = 1 / 4   # much stronger than default 1/32

def main():
    hf_token = os.environ.get("HF_TOKEN")
    cfg = Config()

    # Override for experiment
    cfg.n_precompute_windows = (N_WINDOWS // cfg.num_gpus) * cfg.num_gpus
    cfg.total_steps = TOTAL_STEPS
    cfg.auxk_coeff = AUXK_COEFF
    cfg.precompute_dir = "precomputed_acts"
    cfg.checkpoint_dir = "checkpoints"

    print(f"=== Layer {LAYER} experiment ===")
    print(f"  windows={cfg.n_precompute_windows}, steps={cfg.total_steps}, "
          f"auxk_coeff={cfg.auxk_coeff:.4f}")

    # Load dataset
    train_ds, _, _ = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=hf_token,
    )
    n_avail = len(train_ds)
    if n_avail < cfg.n_precompute_windows:
        cfg.n_precompute_windows = (n_avail // cfg.num_gpus) * cfg.num_gpus
    print(f"  Available windows: {n_avail:,}, using: {cfg.n_precompute_windows:,}")

    # Check if activations already exist
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{LAYER}")
    expected = cfg.n_precompute_windows * cfg.context_length * cfg.hidden_size * 2
    acts_ready = all(
        os.path.exists(os.path.join(layer_dir, f"{d}.bin"))
        and os.path.getsize(os.path.join(layer_dir, f"{d}.bin")) == expected
        for d in ("pt", "ft", "ri")
    )

    if acts_ready:
        print("Activations already on disk, skipping extraction.")
    else:
        os.makedirs(cfg.precompute_dir, exist_ok=True)
        precompute(cfg, train_ds, hf_token, layer_indices=[LAYER])

    # Train
    import torch
    cc = train_crosscoder_from_disk(LAYER, cfg, device=torch.device("cuda:0"))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    save_crosscoder(cc, LAYER, cfg)

if __name__ == "__main__":
    main()
