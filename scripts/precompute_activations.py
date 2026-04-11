"""
Standalone script: precompute activations for all 28 layers.

Useful if you want to precompute everything first (e.g. to verify disk/speed)
before running training. train_all_layers.py calls precompute_layer() directly
and deletes files after each layer, so this script is optional.

Usage:
    source ~/.bashrc && python scripts/precompute_activations.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import build_datasets
from src.crosscoder.precompute import precompute_layer


def main():
    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading GiftEval training set...")
    train_ds, _, _ = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=hf_token
    )
    n_avail = len(train_ds)
    n_use = min(cfg.n_precompute_windows, n_avail)
    n_use = (n_use // cfg.num_gpus) * cfg.num_gpus
    cfg.n_precompute_windows = n_use
    print(f"Training windows: {n_avail:,}  (using {n_use:,})")

    os.makedirs(cfg.precompute_dir, exist_ok=True)

    for layer_idx in range(cfg.num_layers):
        layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")
        N, T, H = cfg.n_precompute_windows, cfg.context_length, cfg.hidden_size
        expected = N * T * H * 2
        already = all(
            os.path.exists(os.path.join(layer_dir, f"{d}.bin"))
            and os.path.getsize(os.path.join(layer_dir, f"{d}.bin")) == expected
            for d in ("pt", "ft", "ri")
        )
        if already:
            print(f"Layer {layer_idx}: already done, skipping.")
            continue

        print(f"\n=== Precomputing layer {layer_idx}/{cfg.num_layers-1} ===")
        precompute_layer(layer_idx, cfg, train_ds, hf_token)


if __name__ == "__main__":
    main()
