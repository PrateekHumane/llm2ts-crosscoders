"""
Full pipeline: precompute activations then train crosscoders, one layer at a time.

For each layer (in priority order):
  1. Precompute PT/FT/RI activations (4-GPU parallel extraction → float16 memmaps).
  2. Train crosscoder via torchrun DDP on 4 GPUs (with early stopping on convergence).
  3. Delete activation files to reclaim disk space.
  Then move to the next layer.

Layer order: binary subdivision for maximum early coverage across network depth.
First layer (layer 13) trains with 100k step budget + convergence detection.
Subsequent layers use the step count discovered from layer 13 (+20% buffer).

Usage:
    source ~/.bashrc && /usr/bin/python3 scripts/train_all_layers.py
"""
import os
import sys
import json
import glob
import shutil
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.dataset import build_datasets
from src.crosscoder.precompute import precompute_layer

# Priority order: center → extremes → fill in gaps
LAYER_ORDER = [13, 0, 27, 6, 20, 3, 10, 16, 24, 1, 4, 8, 11, 14, 18, 22, 26,
               2, 5, 7, 9, 12, 15, 17, 19, 21, 23, 25]

FIRST_LAYER_STEPS = 100_000


def layer_is_trained(layer_idx, cfg):
    return os.path.exists(
        os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}", "crosscoder.pt")
    )


def layer_acts_exist(layer_idx, cfg):
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")
    N, T, H = cfg.n_precompute_windows, cfg.context_length, cfg.hidden_size
    expected = N * T * H * 2
    return all(
        os.path.exists(os.path.join(layer_dir, f"{d}.bin"))
        and os.path.getsize(os.path.join(layer_dir, f"{d}.bin")) == expected
        for d in ("pt", "ft", "ri")
    )


def delete_layer_acts(layer_idx, cfg):
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{layer_idx}")
    if os.path.isdir(layer_dir):
        shutil.rmtree(layer_dir)
        print(f"  Deleted {layer_dir}", flush=True)


def train_layer_ddp(layer_idx, total_steps, cfg):
    """Launch DDP training via torchrun subprocess. Returns train_info dict."""
    cmd = [
        "torchrun",
        f"--nproc_per_node={cfg.num_gpus}",
        "--master_port=29501",
        "scripts/train_layer_ddp.py",
        f"--layer={layer_idx}",
        f"--total_steps={total_steps}",
        "--early_stop",
    ]
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"torchrun failed with exit code {result.returncode}")

    info_path = os.path.join(cfg.checkpoint_dir, f"layer_{layer_idx}", "train_info.json")
    with open(info_path) as f:
        return json.load(f)


def main():
    cfg = Config()
    hf_token = os.environ.get("HF_TOKEN")
    discovered_steps = None

    print("=" * 60)
    print("Crosscoder Training Pipeline — All 28 Layers (DDP)")
    print("=" * 60)
    print(f"  n_precompute_windows : {cfg.n_precompute_windows:,}")
    print(f"  batch_size           : {cfg.batch_size}")
    print(f"  latent_dim / top_k   : {cfg.latent_dim} / {cfg.top_k}")
    print(f"  num_gpus (DDP)       : {cfg.num_gpus}")
    print(f"  Layer order          : {LAYER_ORDER[:10]}...")
    print()

    print("Loading GiftEval training set...")
    train_ds, _, _ = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=hf_token
    )
    n_avail = len(train_ds)
    print(f"  Available windows: {n_avail:,}")

    n_use = min(cfg.n_precompute_windows, n_avail)
    n_use = (n_use // cfg.num_gpus) * cfg.num_gpus
    if n_use != cfg.n_precompute_windows:
        print(f"  Adjusting: {cfg.n_precompute_windows:,} → {n_use:,}")
        cfg.n_precompute_windows = n_use
    print()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.precompute_dir, exist_ok=True)

    n_trained = sum(layer_is_trained(i, cfg) for i in range(cfg.num_layers))
    print(f"Already trained: {n_trained}/{cfg.num_layers} layers")

    log_path = os.path.join(cfg.checkpoint_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            training_log = json.load(f)
        # Try to recover discovered_steps from first trained layer
        if "13" in training_log:
            discovered_steps = training_log["13"]["steps"]
            print(f"Recovered convergence point from layer 13: {discovered_steps} steps")
    else:
        training_log = {}

    t_pipeline_start = time.time()

    for priority, layer_idx in enumerate(LAYER_ORDER):
        if layer_is_trained(layer_idx, cfg):
            print(f"\nLayer {layer_idx:2d}: checkpoint exists, skipping.")
            if layer_acts_exist(layer_idx, cfg):
                delete_layer_acts(layer_idx, cfg)
            continue

        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx} (priority {priority+1}/{len(LAYER_ORDER)})")
        print(f"{'='*60}")
        t_layer_start = time.time()

        if discovered_steps is None:
            total_steps = FIRST_LAYER_STEPS
            print(f"  First layer: up to {total_steps:,} steps (convergence detection on)")
        else:
            total_steps = int(discovered_steps * 1.2)
            print(f"  Using {total_steps:,} steps (120% of convergence at {discovered_steps})")

        # ── Step 1: Precompute ─────────────────────────────────────────
        if layer_acts_exist(layer_idx, cfg):
            print(f"  Step 1: Activations already on disk.")
        else:
            print(f"  Step 1: Precomputing activations...")
            t_pre = time.time()
            precompute_layer(layer_idx, cfg, train_ds, hf_token)
            print(f"  Precompute: {(time.time()-t_pre)/60:.1f}m")

        # ── Step 2: Train (DDP via torchrun) ───────────────────────────
        print(f"  Step 2: Training crosscoder (DDP, {cfg.num_gpus} GPUs)...")
        info = train_layer_ddp(layer_idx, total_steps, cfg)

        if discovered_steps is None:
            discovered_steps = info["steps"]
            print(f"  >>> First layer converged at step {discovered_steps}")
            print(f"  >>> Subsequent layers: {int(discovered_steps * 1.2):,} steps")

        # ── Step 3: Free disk ──────────────────────────────────────────
        print(f"  Step 3: Deleting activation files...")
        delete_layer_acts(layer_idx, cfg)

        elapsed = time.time() - t_layer_start
        training_log[str(layer_idx)] = {
            "priority": priority + 1,
            "steps": info["steps"],
            "elapsed_min": elapsed / 60,
            "train_min": info["elapsed_min"],
            "final_loss": info["final_loss"],
            "dead_features": info["dead_features"],
        }
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)

        print(f"  Layer {layer_idx} complete in {elapsed/60:.1f}m "
              f"(train: {info['elapsed_min']:.1f}m, "
              f"loss: {info['final_loss']:.4f}, "
              f"dead: {info['dead_features']}).", flush=True)

    total_elapsed = time.time() - t_pipeline_start
    ckpts = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "layer_*/crosscoder.pt")))
    print()
    print("=" * 60)
    print(f"Pipeline complete.  {len(ckpts)}/28 checkpoints saved.")
    print(f"Total wall time: {total_elapsed/3600:.1f}h")
    print("=" * 60)


if __name__ == "__main__":
    main()
