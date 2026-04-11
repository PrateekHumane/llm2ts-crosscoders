"""
Compare 30k vs 130k window training for layer 16.

Runs both experiments sequentially:
  1. Precompute 30k windows → train 50k steps → save results → delete acts
  2. Precompute 130k windows → train 50k steps → save results → delete acts

Usage:
    /usr/bin/python3 scripts/compare_window_counts.py
"""
import os
import sys
import json
import time
import math
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.crosscoder.model import Crosscoder
from src.data.dataset import build_datasets
from src.crosscoder.precompute import precompute_layer

LAYER = 16
EXPERIMENTS = [30_000, 130_000]
TOTAL_STEPS = 50_000
LOG_FREQ = 500
EVAL_FREQ = 5000  # full eval every N steps
RESULTS_DIR = "experiments"


class PrecomputedDataset(Dataset):
    def __init__(self, layer_dir, N, T, H):
        shape = (N, T, H)
        self.N = N
        self.pt = np.memmap(os.path.join(layer_dir, "pt.bin"),
                            dtype="float16", mode="r", shape=shape)
        self.ft = np.memmap(os.path.join(layer_dir, "ft.bin"),
                            dtype="float16", mode="r", shape=shape)
        self.ri = np.memmap(os.path.join(layer_dir, "ri.bin"),
                            dtype="float16", mode="r", shape=shape)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return (torch.from_numpy(self.pt[idx].copy()).float(),
                torch.from_numpy(self.ft[idx].copy()).float(),
                torch.from_numpy(self.ri[idx].copy()).float())


def collate(batch):
    pt, ft, ri = zip(*batch)
    return torch.stack(pt), torch.stack(ft), torch.stack(ri)


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(cc, layer_dir, cfg, n_windows, n_eval=200):
    """Evaluate R² on a sample of windows."""
    N, T, H = n_windows, cfg.context_length, cfg.hidden_size
    pt_mm = np.memmap(os.path.join(layer_dir, "pt.bin"),
                      dtype="float16", mode="r", shape=(N, T, H))
    ft_mm = np.memmap(os.path.join(layer_dir, "ft.bin"),
                      dtype="float16", mode="r", shape=(N, T, H))
    ri_mm = np.memmap(os.path.join(layer_dir, "ri.bin"),
                      dtype="float16", mode="r", shape=(N, T, H))

    # Use last n_eval windows
    start = max(0, N - n_eval)
    idx = list(range(start, N))

    mse_pt = mse_ft = mse_ri = 0.0
    var_pt = var_ft = var_ri = 0.0
    n_dead_total = 0
    n_batches = 0

    with torch.no_grad():
        for b_start in range(0, len(idx), 20):
            b_idx = idx[b_start:b_start+20]
            x_pt = torch.from_numpy(pt_mm[b_idx].copy()).float().reshape(-1, H)
            x_ft = torch.from_numpy(ft_mm[b_idx].copy()).float().reshape(-1, H)
            x_ri = torch.from_numpy(ri_mm[b_idx].copy()).float().reshape(-1, H)

            z_pt, _, n_pt = cc.encode_single(x_pt, "PT")
            z_ft, _, n_ft = cc.encode_single(x_ft, "FT")
            z_ri, _, n_ri = cc.encode_single(x_ri, "RI")

            r_pt = cc.decoders["PT"](z_pt)
            r_ft = cc.decoders["FT"](z_ft)
            r_ri = cc.decoders["RI"](z_ri)

            mse_pt += ((r_pt - n_pt)**2).mean().item()
            mse_ft += ((r_ft - n_ft)**2).mean().item()
            mse_ri += ((r_ri - n_ri)**2).mean().item()
            var_pt += n_pt.var().item()
            var_ft += n_ft.var().item()
            var_ri += n_ri.var().item()
            n_batches += 1

    mse_pt /= n_batches; mse_ft /= n_batches; mse_ri /= n_batches
    var_pt /= n_batches; var_ft /= n_batches; var_ri /= n_batches

    return {
        "r2_pt": 1 - mse_pt / var_pt,
        "r2_ft": 1 - mse_ft / var_ft,
        "r2_ri": 1 - mse_ri / var_ri,
        "mse_total": mse_pt + mse_ft + mse_ri,
    }


def run_experiment(n_windows, cfg, train_ds, hf_token):
    """Run one experiment: precompute, train, evaluate, cleanup."""
    device = torch.device("cuda:0")
    T, H = cfg.context_length, cfg.hidden_size
    layer_dir = os.path.join(cfg.precompute_dir, f"layer_{LAYER}")

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {n_windows:,} windows, {TOTAL_STEPS:,} steps")
    print(f"{'='*70}")

    # -- Precompute --
    cfg.n_precompute_windows = (n_windows // cfg.num_gpus) * cfg.num_gpus
    actual_n = cfg.n_precompute_windows

    expected = actual_n * T * H * 2
    acts_ready = os.path.isdir(layer_dir) and all(
        os.path.exists(os.path.join(layer_dir, f"{d}.bin"))
        and os.path.getsize(os.path.join(layer_dir, f"{d}.bin")) == expected
        for d in ("pt", "ft", "ri")
    )

    if acts_ready:
        print(f"Activations already on disk ({actual_n:,} windows).")
    else:
        print(f"Precomputing {actual_n:,} windows...")
        if os.path.isdir(layer_dir):
            shutil.rmtree(layer_dir)
        t0 = time.time()
        precompute_layer(LAYER, cfg, train_ds, hf_token)
        print(f"Precompute done in {(time.time()-t0)/60:.1f}m")

    # -- Train --
    ds = PrecomputedDataset(layer_dir, actual_n, T, H)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=4, pin_memory=True,
                        persistent_workers=True, prefetch_factor=2)

    cc = Crosscoder(cfg).to(device)
    opt = torch.optim.AdamW(cc.parameters(), lr=cfg.lr,
                            betas=(cfg.adam_b1, cfg.adam_b2),
                            weight_decay=cfg.weight_decay)
    dead_counter = torch.zeros(cfg.latent_dim, device=device)
    dead_threshold = cfg.dead_neuron_threshold * cfg.dead_neuron_window

    log = []  # (step, loss, dead, lr)
    evals = []  # (step, r2_pt, r2_ft, r2_ri, mse_total)

    step = 0
    epoch = 0
    t_start = time.time()

    print(f"Training: {actual_n:,} windows, batch={cfg.batch_size}, "
          f"steps={TOTAL_STEPS:,}, auxk={cfg.auxk_coeff:.4f}")

    while step < TOTAL_STEPS:
        epoch += 1
        for pt_b, ft_b, ri_b in loader:
            if step >= TOTAL_STEPS:
                break

            lr = cosine_lr(step, TOTAL_STEPS, cfg.warmup_steps, cfg.lr)
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

            if step % LOG_FREQ == 0:
                dead_n = cc.count_dead_neurons(dead_counter, cfg.dead_neuron_window)
                elapsed = time.time() - t_start
                eta_h = (elapsed / max(step, 1)) * (TOTAL_STEPS - step) / 3600
                log.append((step, loss.item(), dead_n, lr))
                print(f"  step={step:6d}/{TOTAL_STEPS}  lr={lr:.2e}  "
                      f"loss={loss.item():.4f}  dead={dead_n}  "
                      f"t={elapsed/60:.1f}m  ETA={eta_h:.1f}h", flush=True)

            if step > 0 and step % EVAL_FREQ == 0:
                cc_eval = cc.cpu().eval()
                ev = evaluate(cc_eval, layer_dir, cfg, actual_n)
                evals.append((step, ev["r2_pt"], ev["r2_ft"], ev["r2_ri"],
                              ev["mse_total"]))
                print(f"  [EVAL] step={step}  R²: PT={ev['r2_pt']:.4f} "
                      f"FT={ev['r2_ft']:.4f} RI={ev['r2_ri']:.4f}  "
                      f"MSE_total={ev['mse_total']:.4f}", flush=True)
                cc.to(device).train()

            step += 1

    # Final eval
    elapsed = time.time() - t_start
    cc_eval = cc.cpu().eval()
    ev = evaluate(cc_eval, layer_dir, cfg, actual_n)
    evals.append((step, ev["r2_pt"], ev["r2_ft"], ev["r2_ri"], ev["mse_total"]))
    dead_n = cc.count_dead_neurons(dead_counter.cpu(), cfg.dead_neuron_window)

    print(f"\nTraining done in {elapsed/60:.1f}m ({elapsed/3600:.2f}h)")
    print(f"Final: loss={log[-1][1]:.4f}  dead={dead_n}  "
          f"R²: PT={ev['r2_pt']:.4f} FT={ev['r2_ft']:.4f} RI={ev['r2_ri']:.4f}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "n_windows": actual_n,
        "total_steps": TOTAL_STEPS,
        "elapsed_minutes": elapsed / 60,
        "final_loss": log[-1][1],
        "final_dead": dead_n,
        "final_r2": {"PT": ev["r2_pt"], "FT": ev["r2_ft"], "RI": ev["r2_ri"]},
        "log": [{"step": s, "loss": l, "dead": d, "lr": lr_}
                for s, l, d, lr_ in log],
        "evals": [{"step": s, "r2_pt": p, "r2_ft": f, "r2_ri": r, "mse": m}
                  for s, p, f, r, m in evals],
    }
    fname = os.path.join(RESULTS_DIR, f"layer16_{actual_n}win.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {fname}")

    # Save checkpoint
    ckpt_dir = os.path.join(RESULTS_DIR, f"layer16_{actual_n}win")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(cc_eval.state_dict(), os.path.join(ckpt_dir, "crosscoder.pt"))

    # Cleanup acts
    if os.path.isdir(layer_dir):
        shutil.rmtree(layer_dir)
        print(f"Deleted {layer_dir}")

    return result


def main():
    hf_token = os.environ.get("HF_TOKEN")
    cfg = Config()
    cfg.auxk_coeff = 1 / 4
    cfg.auxk_start_step = 1000

    print("Loading GiftEval training set...")
    train_ds, _, _ = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=hf_token
    )
    n_avail = len(train_ds)
    print(f"Available windows: {n_avail:,}")

    results = {}
    for n_win in EXPERIMENTS:
        n_use = min(n_win, n_avail)
        results[n_use] = run_experiment(n_use, cfg, train_ds, hf_token)

    # -- Comparison --
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<25}", end="")
    for n in results:
        print(f"  {n:>10,} win", end="")
    print()
    print("-" * 55)
    for key in ["elapsed_minutes", "final_loss", "final_dead"]:
        print(f"{key:<25}", end="")
        for n in results:
            v = results[n][key]
            if key == "elapsed_minutes":
                print(f"  {v:>10.1f} min", end="")
            else:
                print(f"  {v:>14.4f}", end="")
        print()
    for domain in ["PT", "FT", "RI"]:
        print(f"R² {domain:<22}", end="")
        for n in results:
            print(f"  {results[n]['final_r2'][domain]:>14.4f}", end="")
        print()


if __name__ == "__main__":
    main()
