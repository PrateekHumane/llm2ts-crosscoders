"""
Train linear crosscoders on precomputed layer 13 activations.
Activations must already exist in precomputed_acts/layer_13/.

Usage:
    python3 scripts/train_layer13.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.crosscoder.train_from_disk import train_crosscoder_ddp, save_crosscoder

LAYER = 13


def main():
    cfg = Config()
    cfg.linear_crosscoder = True

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

    print("\nDone!")


if __name__ == "__main__":
    main()
