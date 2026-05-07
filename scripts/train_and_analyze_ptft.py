"""
Streamlined PT+FT-only crosscoder: extract → train → analyze on a single GPU.
Skips RI model entirely for ~33% faster throughput.
Loads big models once — extracts train, analysis, and wiki data, then unloads.

Usage (run up to 4 in parallel):
    python3 scripts/train_and_analyze_ptft.py --layer 7  --gpu 0
    python3 scripts/train_and_analyze_ptft.py --layer 8  --gpu 1
    python3 scripts/train_and_analyze_ptft.py --layer 9  --gpu 2
    python3 scripts/train_and_analyze_ptft.py --layer 10 --gpu 3
"""
import argparse
import os
import sys
import json
import time
import math
import heapq
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.crosscoder.model import Crosscoder
from src.data.dataset import build_datasets, load_gifteval_series, temporal_split, WindowDataset
from src.data.wikitext import load_wikitext_sequences
from src.data.tokenize import normalize_window, uniform_bin_tokenize, window_to_text, get_timestep_token_spans
from src.models.extractor import _extract_with_hooks, _vectorized_pool, _build_idx_map

# ─── Constants ───────────────────────────────────────────────────────────────
N_TRAIN_WINDOWS = 3000
N_ANALYSIS_WINDOWS = 50_000
N_WIKI_SEQS = 30_000
WIKI_SEQ_LEN = 512
TOP_FEATURES = 30
TOP_WINDOWS = 10
TOP_WIKI = 10
WIKI_CONTEXT_TOKENS = 50
FIRING_THRESHOLD = 0.01
EARLY_STOP_PATIENCE = 1500
ANALYSIS_DIR = "analysis"
CHECKPOINT_DIR = "checkpoints/linear_d4096"


# ─── PT+FT Extractor (loads models once, extracts all data) ──────────────────

class PTFTExtractor:
    def __init__(self, cfg, device, hf_token=None, pt_sub_batch=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.device = device
        self.pt_sub_batch = pt_sub_batch
        self._span_cache = {}

        print(f"[GPU {device.index}] Loading PT model...", flush=True)
        self.pt_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_pt, dtype=torch.bfloat16, token=hf_token
        ).model.to(device).eval()

        print(f"[GPU {device.index}] Loading FT model...", flush=True)
        self.ft_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_ft, dtype=torch.bfloat16, token=hf_token
        ).model.to(device).eval()

        self.pt_tokenizer = AutoTokenizer.from_pretrained(cfg.model_pt, token=hf_token)
        if self.pt_tokenizer.pad_token is None:
            self.pt_tokenizer.pad_token = self.pt_tokenizer.eos_token

    def _get_spans(self, window):
        key = (window["series_idx"], window["offset"])
        if key not in self._span_cache:
            norm, _, _ = normalize_window(window["values"])
            text = window_to_text(norm)
            self._span_cache[key] = get_timestep_token_spans(text, norm, self.pt_tokenizer)
        return self._span_cache[key]

    @torch.no_grad()
    def extract_batch(self, windows, layer_idx):
        """Returns (pt_acts, ft_acts) each shape (B*T, H) float32."""
        cfg = self.cfg
        B, T = len(windows), cfg.context_length

        # FT: simple bin tokenization → forward → capture layer output
        all_tokens = []
        for w in windows:
            norm, _, _ = normalize_window(w["values"])
            all_tokens.append(uniform_bin_tokenize(norm, cfg.n_bins, cfg.bin_low, cfg.bin_high))
        input_ids = torch.from_numpy(np.stack(all_tokens)).long().to(self.device)

        ft_captured = _extract_with_hooks(self.ft_model, input_ids, [layer_idx])
        ft_acts = ft_captured[layer_idx].float().reshape(B * T, -1)

        # PT: sub-batched text tokenization → forward → mean-pool
        pt_chunks = []
        for sub_start in range(0, B, self.pt_sub_batch):
            sub = windows[sub_start:sub_start + self.pt_sub_batch]
            sub_B = len(sub)

            texts, all_spans = [], []
            for w in sub:
                norm, _, _ = normalize_window(w["values"])
                texts.append(window_to_text(norm))
                all_spans.append(self._get_spans(w))

            enc = self.pt_tokenizer(
                texts, return_tensors="pt", padding=True, truncation=False,
                add_special_tokens=False, return_attention_mask=True,
            )
            pt_ids = enc["input_ids"].to(self.device)
            attn = enc["attention_mask"].to(self.device)
            L = pt_ids.size(1)

            idx_map = _build_idx_map(all_spans, L, T, self.device)
            pt_captured = _extract_with_hooks(self.pt_model, pt_ids, [layer_idx], attn)
            hs = pt_captured[layer_idx].float()
            pooled = _vectorized_pool(hs, idx_map, T)
            pt_chunks.append(pooled)

        pt_acts = torch.cat(pt_chunks, dim=0).reshape(B * T, -1)
        return pt_acts, ft_acts

    @torch.no_grad()
    def extract_wiki_batch(self, input_ids_batch, layer_idx):
        """Run PT model on WikiText batch. Returns (B, T, H) float16 numpy."""
        captured = {}
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["hs"] = hs.detach()

        handle = self.pt_model.layers[layer_idx].register_forward_hook(hook)
        original_layers = self.pt_model.layers
        self.pt_model.layers = original_layers[:layer_idx + 1]
        try:
            self.pt_model(input_ids=input_ids_batch, use_cache=False)
        finally:
            self.pt_model.layers = original_layers
            handle.remove()

        return captured["hs"].float().cpu().half().numpy()

    def unload(self):
        del self.pt_model, self.ft_model
        torch.cuda.empty_cache()


# ─── Training ────────────────────────────────────────────────────────────────

def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_crosscoder(layer_idx, cfg, device, pt_gpu, ft_gpu):
    """Train 3-domain crosscoder with PT+FT loss only (RI decoder unused)."""
    T = cfg.context_length
    N_samples = pt_gpu.shape[0]

    cc = Crosscoder(cfg).to(device)
    n_params = sum(p.numel() for p in cc.parameters())
    print(f"[GPU {device.index}] Layer {layer_idx}: {n_params/1e6:.2f}M params, PT+FT loss only",
          flush=True)

    opt = torch.optim.AdamW(
        cc.parameters(), lr=cfg.lr,
        betas=(cfg.adam_b1, cfg.adam_b2),
        weight_decay=cfg.weight_decay,
    )
    dead_counter = torch.zeros(cfg.latent_dim, device=device)
    dead_threshold = cfg.dead_neuron_threshold * cfg.dead_neuron_window

    BS = cfg.batch_size * T
    log_freq = 200
    t_start = time.time()
    best_avg_loss = float("inf")
    best_step = 0
    loss_accum = 0.0
    loss_count = 0

    print(f"[GPU {device.index}] Training: batch={cfg.batch_size} windows ({BS} samples), "
          f"steps={cfg.total_steps}", flush=True)

    final_step = 0
    final_loss = 0.0
    for step in range(cfg.total_steps):
        lr = cosine_lr(step, cfg.total_steps, cfg.warmup_steps, cfg.lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        idx = torch.randint(0, N_samples, (BS,), device=device)
        x_pt, x_ft = pt_gpu[idx], ft_gpu[idx]

        cc.update_stats(x_pt, "PT")
        cc.update_stats(x_ft, "FT")

        z_pt, pre_pt, n_pt = cc.encode_single(x_pt, "PT")
        z_ft, pre_ft, n_ft = cc.encode_single(x_ft, "FT")

        recon_pt = cc.decoders["PT"](z_pt)
        recon_ft = cc.decoders["FT"](z_ft)
        loss = F.mse_loss(recon_pt, n_pt) + F.mse_loss(recon_ft, n_ft)

        if step >= cfg.auxk_start_step:
            dead_mask = dead_counter < dead_threshold
            if dead_mask.any():
                k_aux = min(cfg.auxk_k, int(dead_mask.sum().item()))
                if k_aux > 0:
                    aux_loss = torch.tensor(0.0, device=device)
                    for pre, n, recon, dec in [
                        (pre_pt, n_pt, recon_pt, cc.decoders["PT"]),
                        (pre_ft, n_ft, recon_ft, cc.decoders["FT"]),
                    ]:
                        res = (n - recon).detach()
                        pre_dead = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
                        vals, aux_idx = torch.topk(pre_dead, k_aux, dim=-1)
                        z_aux = torch.zeros_like(pre).scatter(-1, aux_idx, F.relu(vals))
                        aux_loss = aux_loss + F.mse_loss(dec(z_aux), res)
                    loss = loss + cfg.auxk_coeff * aux_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(cc.parameters(), max_norm=1.0)
        opt.step()
        cc.normalize_decoder_columns()

        with torch.no_grad():
            active = ((z_pt.abs() > 0) | (z_ft.abs() > 0)).any(dim=0).float()
            dead_counter = dead_counter * 0.999 + active

        loss_val = loss.item()
        loss_accum += loss_val
        loss_count += 1
        final_step = step
        final_loss = loss_val

        if step % log_freq == 0:
            avg_loss = loss_accum / max(loss_count, 1)
            dead_n = int((dead_counter < dead_threshold).sum().item())
            elapsed = time.time() - t_start
            sps = max(step, 1) / elapsed
            eta = (cfg.total_steps - step) / sps / 60 if sps > 0 else 0
            print(f"[{device.index}] step={step:6d}/{cfg.total_steps}  lr={lr:.2e}  "
                  f"loss={loss_val:.6f}  avg={avg_loss:.6f}  dead={dead_n}  "
                  f"{sps:.1f} step/s  ETA={eta:.1f}m", flush=True)

            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                best_step = step
            loss_accum = 0.0
            loss_count = 0

        if step - best_step >= EARLY_STOP_PATIENCE and step >= cfg.warmup_steps + EARLY_STOP_PATIENCE:
            print(f"[{device.index}] Early stopping at step {step} "
                  f"(best avg={best_avg_loss:.6f} at step {best_step})", flush=True)
            break

    elapsed_total = time.time() - t_start
    print(f"[GPU {device.index}] Training done: {final_step+1} steps in {elapsed_total/60:.1f}m, "
          f"final loss={final_loss:.6f}", flush=True)

    out_dir = os.path.join(CHECKPOINT_DIR, f"layer_{layer_idx}")
    os.makedirs(out_dir, exist_ok=True)

    cc_cpu = cc.cpu()
    torch.save(cc_cpu.state_dict(), os.path.join(out_dir, "crosscoder.pt"))

    norm_stats = {}
    for domain in Crosscoder.DOMAINS:
        norm_stats[domain] = {
            "mean": getattr(cc_cpu, f"{domain}_mean").tolist(),
            "std": getattr(cc_cpu, f"{domain}_std").tolist(),
        }
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump(norm_stats, f)

    with open(os.path.join(out_dir, "train_info.json"), "w") as f:
        json.dump({
            "layer": layer_idx,
            "latent_dim": cfg.latent_dim,
            "linear": True,
            "batch_size": cfg.batch_size,
            "n_windows": N_TRAIN_WINDOWS,
            "n_samples": N_samples,
            "steps": final_step + 1,
            "final_loss": final_loss,
            "best_avg_loss": best_avg_loss,
            "elapsed_min": elapsed_total / 60,
            "domains": ["PT", "FT"],
        }, f, indent=2)

    print(f"[GPU {device.index}] Saved to {out_dir}/", flush=True)
    return final_loss


# ─── Extraction helpers ──────────────────────────────────────────────────────

def extract_train_data(layer_idx, cfg, device, extractor, train_ds):
    """Extract training activations into GPU tensors."""
    n_use = min(N_TRAIN_WINDOWS, len(train_ds))
    T, H = cfg.context_length, cfg.hidden_size

    pt_chunks, ft_chunks = [], []
    batch_size = 32
    t0 = time.time()

    print(f"[GPU {device.index}] Extracting {n_use} training windows...", flush=True)
    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        windows = [train_ds[i] for i in range(start, end)]
        pt_acts, ft_acts = extractor.extract_batch(windows, layer_idx)
        pt_chunks.append(pt_acts)
        ft_chunks.append(ft_acts)

        if (start // batch_size) % 10 == 0:
            done = end
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_use - done) / rate / 60 if rate > 0 else 0
            print(f"  [{device.index}] Train: {done}/{n_use} ({done/n_use*100:.0f}%) "
                  f"{rate:.1f} win/s  ETA {eta:.1f}m", flush=True)

    pt_gpu = torch.cat(pt_chunks, dim=0).to(device)
    ft_gpu = torch.cat(ft_chunks, dim=0).to(device)
    gb = (pt_gpu.nbytes + ft_gpu.nbytes) / 1e9
    print(f"[GPU {device.index}] Train data: {pt_gpu.shape[0]:,} samples, "
          f"{gb:.1f}GB ({(time.time()-t0)/60:.1f}m)", flush=True)
    return pt_gpu, ft_gpu


def extract_analysis_data(layer_idx, cfg, device, extractor, val_ds, act_dir):
    """Extract analysis activations to disk memmaps. Returns n_windows used."""
    n_use = min(N_ANALYSIS_WINDOWS, len(val_ds))
    T, H = cfg.context_length, cfg.hidden_size
    shape = (n_use, T, H)

    pt_path = os.path.join(act_dir, "pt.bin")
    ft_path = os.path.join(act_dir, "ft.bin")
    expected = n_use * T * H * 2

    if (os.path.exists(pt_path) and os.path.getsize(pt_path) == expected and
        os.path.exists(ft_path) and os.path.getsize(ft_path) == expected):
        print(f"  Analysis activations already on disk.", flush=True)
        return n_use

    pt_mm = np.memmap(pt_path, dtype="float16", mode="w+", shape=shape)
    ft_mm = np.memmap(ft_path, dtype="float16", mode="w+", shape=shape)

    batch_size = 32
    cursor = 0
    t0 = time.time()

    print(f"[GPU {device.index}] Extracting {n_use} analysis windows...", flush=True)
    for start in range(0, n_use, batch_size):
        end = min(start + batch_size, n_use)
        windows = [val_ds[i] for i in range(start, end)]
        B = len(windows)

        pt_acts, ft_acts = extractor.extract_batch(windows, layer_idx)
        pt_mm[cursor:cursor + B] = pt_acts.cpu().half().numpy().reshape(B, T, H)
        ft_mm[cursor:cursor + B] = ft_acts.cpu().half().numpy().reshape(B, T, H)
        cursor += B

        if (start // batch_size) % 50 == 0 and start > 0:
            done = cursor
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_use - done) / rate / 60 if rate > 0 else 0
            print(f"  [{device.index}] Analysis: {done}/{n_use} ({done/n_use*100:.0f}%) "
                  f"{rate:.1f} win/s  ETA {eta:.1f}m", flush=True)

    pt_mm.flush()
    ft_mm.flush()
    print(f"[GPU {device.index}] Analysis extraction done in {(time.time()-t0)/60:.1f}m", flush=True)
    return n_use


def extract_wiki_data(layer_idx, cfg, device, extractor, wiki_sequences, act_dir):
    """Extract WikiText hidden states to disk memmap using PT model."""
    N = len(wiki_sequences)
    T, H = WIKI_SEQ_LEN, cfg.hidden_size

    out_path = os.path.join(act_dir, "wiki_pt.bin")
    expected = N * T * H * 2

    if os.path.exists(out_path) and os.path.getsize(out_path) == expected:
        print(f"  WikiText activations already on disk.", flush=True)
        return

    mm = np.memmap(out_path, dtype="float16", mode="w+", shape=(N, T, H))
    batch_size = 32
    cursor = 0
    t0 = time.time()

    print(f"[GPU {device.index}] Extracting {N} WikiText sequences...", flush=True)
    for start in range(0, N, batch_size):
        batch = wiki_sequences[start:start + batch_size]
        input_ids = torch.tensor(
            np.stack([s["input_ids"] for s in batch]),
            dtype=torch.long, device=device,
        )
        hs = extractor.extract_wiki_batch(input_ids, layer_idx)
        B = hs.shape[0]
        mm[cursor:cursor + B] = hs[:, :T, :]
        cursor += B

        if (start // batch_size) % 20 == 0 and start > 0:
            done = cursor
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (N - done) / rate if rate > 0 else 0
            print(f"    Wiki: {done}/{N} ({done/N*100:.0f}%) "
                  f"{rate:.0f} seq/s  ETA {eta:.0f}s", flush=True)

    mm.flush()
    print(f"[GPU {device.index}] WikiText extraction done in {(time.time()-t0)/60:.1f}m", flush=True)


# ─── Analysis: categorize + extract top features ─────────────────────────────

def compute_wiki_norm(act_dir, n_wiki, H):
    wiki_mm = np.memmap(os.path.join(act_dir, "wiki_pt.bin"), dtype="float16",
                        mode="r", shape=(n_wiki, WIKI_SEQ_LEN, H))
    mean_acc = np.zeros(H, dtype=np.float64)
    sq_acc = np.zeros(H, dtype=np.float64)
    n = 0
    for start in range(0, n_wiki, 50):
        end = min(start + 50, n_wiki)
        chunk = wiki_mm[start:end].astype(np.float32).reshape(-1, H)
        mean_acc += chunk.sum(axis=0)
        sq_acc += (chunk ** 2).sum(axis=0)
        n += chunk.shape[0]
    wiki_mean = torch.tensor(mean_acc / n, dtype=torch.float32)
    wiki_std = torch.tensor(np.sqrt(sq_acc / n - (mean_acc / n) ** 2),
                            dtype=torch.float32).clamp(min=1e-6)
    return wiki_mean, wiki_std


def encode_wiki(x, encoder, wiki_mean, wiki_std):
    device = x.device
    n = (x - wiki_mean.to(device)) / (wiki_std.to(device) + 1e-8)
    z, pre = encoder(n)
    return z


def categorize_and_extract(layer_idx, cfg, device, act_dir, n_ts, n_wiki,
                           wiki_sequences, val_ds):
    """Full categorization and top-feature extraction, PT_FT focused."""
    T, H = cfg.context_length, cfg.hidden_size

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"layer_{layer_idx}", "crosscoder.pt")
    cc = Crosscoder(cfg)
    cc.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    cc = cc.to(device).eval()

    pt_mm = np.memmap(os.path.join(act_dir, "pt.bin"), dtype="float16", mode="r",
                      shape=(n_ts, T, H))
    ft_mm = np.memmap(os.path.join(act_dir, "ft.bin"), dtype="float16", mode="r",
                      shape=(n_ts, T, H))
    wiki_mm = np.memmap(os.path.join(act_dir, "wiki_pt.bin"), dtype="float16",
                        mode="r", shape=(n_wiki, WIKI_SEQ_LEN, H))

    # ── Pass 1: firing rates ──
    fire_pt = torch.zeros(cfg.latent_dim, device=device)
    fire_ft = torch.zeros(cfg.latent_dim, device=device)
    fire_wiki = torch.zeros(cfg.latent_dim, device=device)
    act_sum_pt = torch.zeros(cfg.latent_dim, device=device)
    act_sum_ft = torch.zeros(cfg.latent_dim, device=device)
    n_ts_samples = 0
    n_wiki_samples = 0

    BS = 20
    print(f"  Pass 1: firing rates ({n_ts:,} TS windows)...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_ts, BS):
            end = min(start + BS, n_ts)
            x_pt = torch.from_numpy(pt_mm[start:end].copy()).float().reshape(-1, H).to(device)
            x_ft = torch.from_numpy(ft_mm[start:end].copy()).float().reshape(-1, H).to(device)
            z_pt, _, _ = cc.encode_single(x_pt, "PT")
            z_ft, _, _ = cc.encode_single(x_ft, "FT")
            fire_pt += (z_pt > 0).float().sum(dim=0)
            fire_ft += (z_ft > 0).float().sum(dim=0)
            act_sum_pt += z_pt.sum(dim=0)
            act_sum_ft += z_ft.sum(dim=0)
            n_ts_samples += x_pt.shape[0]

            if (start // BS) % 300 == 0 and start > 0:
                print(f"    TS: {start}/{n_ts} ({start/n_ts*100:.0f}%) "
                      f"t={(time.time()-t0)/60:.1f}m", flush=True)

    wiki_mean, wiki_std = compute_wiki_norm(act_dir, n_wiki, H)

    print(f"  Wiki firing rates ({n_wiki:,} seqs)...", flush=True)
    with torch.no_grad():
        for start in range(0, n_wiki, BS):
            end = min(start + BS, n_wiki)
            x_wiki = torch.from_numpy(wiki_mm[start:end].copy()).float().reshape(-1, H).to(device)
            z_wiki = encode_wiki(x_wiki, cc.encoder, wiki_mean, wiki_std)
            fire_wiki += (z_wiki > 0).float().sum(dim=0)
            n_wiki_samples += x_wiki.shape[0]

    print(f"  Rates computed in {(time.time()-t0)/60:.1f}m", flush=True)

    rate_pt = (fire_pt / n_ts_samples).cpu()
    rate_ft = (fire_ft / n_ts_samples).cpu()
    rate_wiki = (fire_wiki / n_wiki_samples).cpu()
    mean_act_pt = (act_sum_pt / n_ts_samples).cpu()
    mean_act_ft = (act_sum_ft / n_ts_samples).cpu()

    active_pt = rate_pt > FIRING_THRESHOLD
    active_ft = rate_ft > FIRING_THRESHOLD

    categories = {"PT_FT": [], "PT_only": [], "FT_only": [], "None": []}
    feature_stats = {}
    for j in range(cfg.latent_dim):
        pt_on, ft_on = active_pt[j].item(), active_ft[j].item()
        if pt_on and ft_on:
            cat = "PT_FT"
        elif pt_on:
            cat = "PT_only"
        elif ft_on:
            cat = "FT_only"
        else:
            cat = "None"
        categories[cat].append(j)
        feature_stats[j] = {
            "category": cat,
            "rate_pt": rate_pt[j].item(),
            "rate_ft": rate_ft[j].item(),
            "rate_ri": 0.0,
            "rate_wiki": rate_wiki[j].item(),
            "mean_act_pt": mean_act_pt[j].item(),
            "mean_act_ft": mean_act_ft[j].item(),
            "mean_act_ri": 0.0,
        }

    for cat_name, feats in categories.items():
        print(f"    {cat_name}: {len(feats)}", flush=True)

    # ── Rank and extract top PT_FT features ──
    pt_ft_feats = categories["PT_FT"]
    if not pt_ft_feats:
        print("  WARNING: No PT_FT features found!", flush=True)
        del cc
        torch.cuda.empty_cache()
        return categories, feature_stats, [], n_ts_samples, n_wiki_samples

    ranked = sorted(pt_ft_feats,
                    key=lambda j: min(feature_stats[j]["rate_pt"],
                                      feature_stats[j]["rate_ft"]),
                    reverse=True)
    top_feats = ranked[:TOP_FEATURES]
    print(f"  Pass 2: extracting top {len(top_feats)} PT_FT features...", flush=True)

    ts_topk = {j: [] for j in top_feats}
    wiki_topk = {j: [] for j in top_feats}

    print(f"    Scanning TS windows...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n_ts, BS):
            end = min(start + BS, n_ts)
            B = end - start
            x_pt = torch.from_numpy(pt_mm[start:end].copy()).float().to(device)
            x_ft = torch.from_numpy(ft_mm[start:end].copy()).float().to(device)

            z_pt, _, _ = cc.encode_single(x_pt.reshape(-1, H), "PT")
            z_ft, _, _ = cc.encode_single(x_ft.reshape(-1, H), "FT")
            z_pt = z_pt.reshape(B, T, -1)
            z_ft = z_ft.reshape(B, T, -1)

            for j in top_feats:
                z_combined = z_pt[:, :, j] + z_ft[:, :, j]
                max_vals, max_ts = z_combined.max(dim=1)
                for b_idx in range(B):
                    val = max_vals[b_idx].item()
                    if val <= 0:
                        continue
                    entry = (val, start + b_idx, max_ts[b_idx].item())
                    if len(ts_topk[j]) < TOP_WINDOWS:
                        heapq.heappush(ts_topk[j], entry)
                    elif val > ts_topk[j][0][0]:
                        heapq.heapreplace(ts_topk[j], entry)

            if (start // BS) % 500 == 0 and start > 0:
                print(f"      TS: {start}/{n_ts} ({start/n_ts*100:.0f}%) "
                      f"t={(time.time()-t0)/60:.1f}m", flush=True)

    print(f"    Scanning WikiText...", flush=True)
    with torch.no_grad():
        for start in range(0, n_wiki, BS):
            end = min(start + BS, n_wiki)
            B = end - start
            x_wiki = torch.from_numpy(wiki_mm[start:end].copy()).float().to(device)
            z_wiki = encode_wiki(x_wiki.reshape(-1, H), cc.encoder, wiki_mean, wiki_std)
            z_wiki = z_wiki.reshape(B, WIKI_SEQ_LEN, -1)

            for j in top_feats:
                z_feat = z_wiki[:, :, j]
                max_vals, max_ts = z_feat.max(dim=1)
                for b_idx in range(B):
                    val = max_vals[b_idx].item()
                    if val <= 0:
                        continue
                    entry = (val, start + b_idx, max_ts[b_idx].item())
                    if len(wiki_topk[j]) < TOP_WIKI:
                        heapq.heappush(wiki_topk[j], entry)
                    elif val > wiki_topk[j][0][0]:
                        heapq.heapreplace(wiki_topk[j], entry)

    # ── Collect detailed data ──
    top_features_data = []
    for j in top_feats:
        feat_data = {
            "feature_id": j,
            "stats": feature_stats[j],
            "top_windows": [],
            "top_wiki_spans": [],
        }

        for val, win_idx, ts_idx in sorted(ts_topk[j], key=lambda x: -x[0]):
            window = val_ds[win_idx]
            x_pt_w = torch.from_numpy(pt_mm[win_idx].copy()).float().to(device)
            x_ft_w = torch.from_numpy(ft_mm[win_idx].copy()).float().to(device)

            with torch.no_grad():
                z_pt_w, _, _ = cc.encode_single(x_pt_w, "PT")
                z_ft_w, _, _ = cc.encode_single(x_ft_w, "FT")

            feat_data["top_windows"].append({
                "activation_value": val,
                "window_idx": win_idx,
                "peak_timestep": ts_idx,
                "series_idx": window["series_idx"],
                "offset": window["offset"],
                "raw_values": window["values"].tolist(),
                "activations_pt": z_pt_w[:, j].cpu().tolist(),
                "activations_ft": z_ft_w[:, j].cpu().tolist(),
                "activations_ri": [0.0] * T,
            })

        for val, seq_idx, tok_idx in sorted(wiki_topk[j], key=lambda x: -x[0]):
            seq = wiki_sequences[seq_idx]
            x_w = torch.from_numpy(wiki_mm[seq_idx].copy()).float().to(device)
            with torch.no_grad():
                z_w = encode_wiki(x_w, cc.encoder, wiki_mean, wiki_std)
            activations = z_w[:, j].cpu().tolist()

            ctx_start = max(0, tok_idx - WIKI_CONTEXT_TOKENS)
            ctx_end = min(WIKI_SEQ_LEN, tok_idx + WIKI_CONTEXT_TOKENS + 1)

            feat_data["top_wiki_spans"].append({
                "activation_value": val,
                "seq_idx": seq_idx,
                "peak_token_idx": tok_idx,
                "full_text": seq["text"],
                "context_start": ctx_start,
                "context_end": ctx_end,
                "activations": activations[ctx_start:ctx_end],
            })

        top_features_data.append(feat_data)

    del cc
    torch.cuda.empty_cache()
    return categories, feature_stats, top_features_data, n_ts_samples, n_wiki_samples


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_top_windows(feat_data, output_path):
    windows = feat_data["top_windows"][:TOP_WINDOWS]
    if not windows:
        return

    n_plots = len(windows)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.5 * n_plots), squeeze=False)

    for i, win in enumerate(windows):
        ax = axes[i, 0]
        values = np.array(win["raw_values"], dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        acts = np.array(win["activations_pt"]) + np.array(win["activations_ft"])

        x = np.arange(len(values))
        points = np.array([x, values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        act_colors = (acts[:-1] + acts[1:]) / 2
        act_max = np.nanmax(act_colors) if len(act_colors) > 0 else 0
        norm = mcolors.Normalize(vmin=0, vmax=max(act_max, 1e-6))

        lc = LineCollection(segments, cmap="YlOrRd", norm=norm)
        lc.set_array(act_colors)
        lc.set_linewidth(1.5)
        ax.add_collection(lc)

        v_min, v_max = float(values.min()), float(values.max())
        v_range = max(abs(v_max - v_min), 1e-6)
        ax.set_xlim(0, len(values))
        ax.set_ylim(v_min - 0.1 * v_range, v_max + 0.1 * v_range)

        peak = win["peak_timestep"]
        ax.axvline(x=peak, color="red", alpha=0.5, linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"#{i+1}\nact={win['activation_value']:.2f}", fontsize=8)
        if i == 0:
            ax.set_title(f"Feature {feat_data['feature_id']} — Top {n_plots} Windows",
                        fontsize=10)
        if i < n_plots - 1:
            ax.set_xticklabels([])

    axes[-1, 0].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Save ────────────────────────────────────────────────────────────────────

def save_results(layer_idx, categories, feature_stats, top_features_data,
                 n_ts_samples, n_wiki_samples):
    layer_dir = os.path.join(ANALYSIS_DIR, f"layer_{layer_idx}")
    os.makedirs(layer_dir, exist_ok=True)

    cat_summary = {
        "layer": layer_idx,
        "n_ts_samples": n_ts_samples,
        "n_wiki_samples": n_wiki_samples,
        "category_counts": {k: len(v) for k, v in categories.items()},
        "feature_stats": {str(k): v for k, v in feature_stats.items()},
        "note": "PT+FT only (RI skipped)",
    }
    with open(os.path.join(layer_dir, "categorization.json"), "w") as f:
        json.dump(cat_summary, f, indent=2)

    cat_dir = os.path.join(layer_dir, "PT_FT")
    os.makedirs(cat_dir, exist_ok=True)

    ranking = [
        {
            "rank": i + 1,
            "feature_id": f["feature_id"],
            "rate_pt": f["stats"]["rate_pt"],
            "rate_ft": f["stats"]["rate_ft"],
            "rate_wiki": f["stats"]["rate_wiki"],
            "mean_act_pt": f["stats"]["mean_act_pt"],
            "mean_act_ft": f["stats"]["mean_act_ft"],
        }
        for i, f in enumerate(top_features_data)
    ]
    with open(os.path.join(cat_dir, "ranking.json"), "w") as f:
        json.dump(ranking, f, indent=2)

    for feat_data in top_features_data:
        fid = feat_data["feature_id"]
        feat_dir = os.path.join(cat_dir, f"feature_{fid}")
        os.makedirs(feat_dir, exist_ok=True)

        with open(os.path.join(feat_dir, "windows.json"), "w") as f:
            json.dump(feat_data["top_windows"], f)

        if feat_data["top_wiki_spans"]:
            with open(os.path.join(feat_dir, "wiki_spans.json"), "w") as f:
                json.dump(feat_data["top_wiki_spans"], f)

        info = {
            "feature_id": fid,
            "layer": layer_idx,
            "category": feat_data["stats"]["category"],
            "stats": feat_data["stats"],
            "qualitative_score": None,
            "interpretation": None,
        }
        with open(os.path.join(feat_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=2)

        plot_top_windows(feat_data, os.path.join(feat_dir, "plot.png"))

    print(f"  Saved to {layer_dir}/", flush=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()

    t_total = time.time()

    cfg = Config()
    cfg.linear_crosscoder = True
    cfg.latent_dim = 4096
    cfg.top_k = 64
    cfg.batch_size = 256
    cfg.total_steps = 10_000
    cfg.warmup_steps = 500
    cfg.checkpoint_dir = CHECKPOINT_DIR

    device = torch.device(f"cuda:{args.gpu}")
    hf_token = os.environ.get("HF_TOKEN")

    layer_idx = args.layer
    act_dir = os.path.join("precomputed_acts_ptft", f"layer_{layer_idx}")

    # Skip if already complete
    analysis_done = os.path.exists(
        os.path.join(ANALYSIS_DIR, f"layer_{layer_idx}", "PT_FT", "ranking.json"))
    if analysis_done:
        print(f"Layer {layer_idx} already analyzed. Skipping.", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"LAYER {layer_idx} on GPU {args.gpu} — PT+FT PIPELINE", flush=True)
    print(f"{'='*60}", flush=True)

    # ── Load datasets ──
    print("Loading GiftEval...", flush=True)
    train_ds, _, _ = build_datasets(
        cfg.context_length, cfg.train_frac, cfg.val_frac, hf_token=hf_token)

    series_list = load_gifteval_series(hf_token)
    val_splits = []
    for s in series_list:
        _, va, _ = temporal_split(s, cfg.train_frac, cfg.val_frac)
        if len(va) >= cfg.context_length:
            val_splits.append(va)
    val_ds = WindowDataset(val_splits, cfg.context_length, stride=cfg.context_length)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}", flush=True)

    print(f"Loading WikiText ({N_WIKI_SEQS:,} seqs)...", flush=True)
    wiki_sequences = load_wikitext_sequences(
        max_sequences=N_WIKI_SEQS, seq_len=WIKI_SEQ_LEN, hf_token=hf_token)
    print(f"  WikiText: {len(wiki_sequences):,}", flush=True)

    # ── Load models once, do all extraction ──
    print(f"\n--- Loading models & extracting all data ---", flush=True)
    os.makedirs(act_dir, exist_ok=True)

    extractor = PTFTExtractor(cfg, device, hf_token=hf_token, pt_sub_batch=16)

    ckpt_exists = os.path.exists(
        os.path.join(CHECKPOINT_DIR, f"layer_{layer_idx}", "crosscoder.pt"))

    if not ckpt_exists:
        pt_gpu, ft_gpu = extract_train_data(layer_idx, cfg, device, extractor, train_ds)
    else:
        pt_gpu, ft_gpu = None, None
        print("  Checkpoint exists, skipping train extraction.", flush=True)

    n_analysis = extract_analysis_data(layer_idx, cfg, device, extractor, val_ds, act_dir)
    extract_wiki_data(layer_idx, cfg, device, extractor, wiki_sequences, act_dir)

    extractor.unload()
    print(f"  Models unloaded.", flush=True)

    # ── Train crosscoder ──
    if not ckpt_exists:
        print(f"\n--- Training crosscoder ---", flush=True)
        final_loss = train_crosscoder(layer_idx, cfg, device, pt_gpu, ft_gpu)
        del pt_gpu, ft_gpu
        torch.cuda.empty_cache()
    else:
        print(f"  Using existing checkpoint.", flush=True)

    # ── Categorize + extract top features ──
    print(f"\n--- Analyzing features ---", flush=True)
    cats, fstats, top_data, n_ts, n_wiki = categorize_and_extract(
        layer_idx, cfg, device, act_dir, n_analysis,
        len(wiki_sequences), wiki_sequences, val_ds)

    # ── Save + cleanup ──
    print(f"\n--- Saving results ---", flush=True)
    save_results(layer_idx, cats, fstats, top_data, n_ts, n_wiki)

    if os.path.isdir(act_dir):
        shutil.rmtree(act_dir)
        print(f"  Cleaned up {act_dir}", flush=True)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"Layer {layer_idx} COMPLETE in {elapsed/60:.1f}m", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
