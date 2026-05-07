"""
Test pipeline: precompute layer 13, train linear crosscoders with
latent_dim=1024 and 4096, then run analysis on both.

Usage:
    python3 scripts/run_layer13_test.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import build_datasets
from src.crosscoder.precompute import precompute_layer
from src.crosscoder.train_from_disk import train_crosscoder_ddp, save_crosscoder

LAYER = 13
HF_TOKEN = os.environ.get("HF_TOKEN")


def main():
    cfg = Config()
    cfg.linear_crosscoder = True

    # Load dataset
    print("Loading GiftEval...", flush=True)
    train_ds, val_ds, test_ds = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=HF_TOKEN
    )

    # Precompute layer 13 activations
    n_use = min(cfg.n_precompute_windows, len(train_ds))
    n_use = (n_use // cfg.num_gpus) * cfg.num_gpus
    cfg.n_precompute_windows = n_use

    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{LAYER}")
    expected = n_use * cfg.context_length * cfg.hidden_size * 2
    acts_exist = all(
        os.path.exists(os.path.join(layer_dir, f"{d}.bin"))
        and os.path.getsize(os.path.join(layer_dir, f"{d}.bin")) == expected
        for d in ("pt", "ft", "ri")
    )

    if not acts_exist:
        print(f"\nPrecomputing layer {LAYER} activations ({n_use:,} windows)...", flush=True)
        precompute_layer(LAYER, cfg, train_ds, HF_TOKEN)
    else:
        print(f"\nLayer {LAYER} activations already on disk.", flush=True)

    # Train with both latent dims
    for latent_dim in [1024, 4096]:
        print(f"\n{'='*60}")
        print(f"Training linear crosscoder: layer={LAYER}, latent_dim={latent_dim}")
        print(f"{'='*60}", flush=True)

        cfg.latent_dim = latent_dim
        cfg.checkpoint_dir = f"checkpoints/linear_d{latent_dim}"

        t0 = time.time()
        cc, info = train_crosscoder_ddp(LAYER, cfg)
        elapsed = time.time() - t0

        save_crosscoder(cc, LAYER, cfg)

        print(f"\nlatent_dim={latent_dim}: "
              f"{info['steps']} steps, "
              f"loss={info['final_loss']:.4f}, "
              f"{elapsed/60:.1f}m", flush=True)

        # Save training info
        info_path = os.path.join(cfg.checkpoint_dir, f"layer_{LAYER}", "train_info.json")
        with open(info_path, "w") as f:
            json.dump({
                "layer": LAYER,
                "latent_dim": latent_dim,
                "linear": True,
                "steps": info["steps"],
                "final_loss": info["final_loss"],
                "elapsed_min": info["elapsed_min"],
            }, f, indent=2)

    print("\nDone! Check checkpoints/linear_d1024/ and checkpoints/linear_d4096/")


if __name__ == "__main__":
    main()
