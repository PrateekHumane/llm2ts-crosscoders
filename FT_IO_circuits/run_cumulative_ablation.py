"""Cumulative ablation — test whether selective components form a circuit.

Ablates combinations of the top selective components and checks for
superadditivity (combined effect >> sum of individual effects).
Then runs the strongest combinations on WikiText to find affected text.

Usage:  python3 FT_IO_circuits/run_cumulative_ablation.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json
import time
import numpy as np
import torch
from itertools import combinations
from dataclasses import dataclass
from contextlib import ExitStack
from transformers import AutoTokenizer

from FT_IO_circuits.synthetic import generate_all, tokenize_windows
from FT_IO_circuits.extract import load_model
from FT_IO_circuits.ablation import ablate_head, ablate_mlp, compute_loss
from FT_IO_circuits.run import load_wikitext_ids


@dataclass
class Cfg:
    model_ft_io: str = "/workspace/NanoTS_v2/checkpoints/io_only/checkpoint-8192"
    model_pt: str = (
        "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/"
        "c1899de289a04d12100db370d81485cdf75e47ca"
    )
    n_per_type: int = 20
    context_length: int = 512
    n_bins: int = 1024
    batch_size: int = 16
    device: str = "cuda:0"
    num_heads: int = 16
    head_dim: int = 128
    n_wikitext: int = 2000
    wiki_seq_len: int = 512
    output_dir: str = "FT_IO_circuits/ablation_results"


PERIODIC = ["sine", "square_wave", "sawtooth", "seasonal", "damped_sine"]
CONTROL = ["white_noise", "constant", "linear_trend", "random_walk"]
ALL_TYPES = PERIODIC + CONTROL

# Top selective components from Phase 1
COMPONENTS = [
    {"comp": "head_L1_H4",  "kind": "head", "layer": 1,  "head": 4,  "d_individual": 4.50},
    {"comp": "mlp_L1",      "kind": "mlp",  "layer": 1,  "head": None, "d_individual": 5.75},
    {"comp": "head_L20_H0", "kind": "head", "layer": 20, "head": 0,  "d_individual": 3.40},
    {"comp": "head_L21_H8", "kind": "head", "layer": 21, "head": 8,  "d_individual": 1.85},
    {"comp": "head_L23_H6", "kind": "head", "layer": 23, "head": 6,  "d_individual": 2.00},
    {"comp": "head_L13_H6", "kind": "head", "layer": 13, "head": 6,  "d_individual": 3.35},
    {"comp": "head_L15_H3", "kind": "head", "layer": 15, "head": 3,  "d_individual": 1.55},
    {"comp": "head_L11_H8", "kind": "head", "layer": 11, "head": 8,  "d_individual": 1.25},
]


def _ablation_ctx(model, comp, cfg):
    if comp["kind"] == "head":
        return ablate_head(model, comp["layer"], comp["head"],
                           cfg.num_heads, cfg.head_dim)
    return ablate_mlp(model, comp["layer"])


def _batched_loss(model, ids, bs, device):
    parts = []
    for s in range(0, len(ids), bs):
        parts.append(compute_loss(model, ids[s:s+bs].to(device)))
    return torch.cat(parts)


def _per_type_mean(losses, type_idx, names):
    return {t: losses[type_idx == i].mean().item() for i, t in enumerate(names)}


def run_ts_cumulative(cfg):
    print("=" * 60)
    print("Cumulative ablation on periodic TS")
    print("=" * 60)

    all_ts = generate_all(cfg.n_per_type, cfg.context_length)
    tokens = {t: tokenize_windows(all_ts[t], n_bins=cfg.n_bins) for t in ALL_TYPES}
    all_ids = torch.from_numpy(np.concatenate([tokens[t] for t in ALL_TYPES]))
    type_idx = torch.cat([torch.full((cfg.n_per_type,), i)
                          for i, _ in enumerate(ALL_TYPES)]).long()

    model = load_model(cfg.model_ft_io, cfg.device)

    bl = _batched_loss(model, all_ids, cfg.batch_size, cfg.device)
    baseline = _per_type_mean(bl, type_idx, ALL_TYPES)
    bl_periodic = np.mean([baseline[t] for t in PERIODIC])
    bl_control = np.mean([baseline[t] for t in CONTROL])
    print(f"  baseline: periodic={bl_periodic:.4f}  control={bl_control:.4f}")

    results = []

    # Individual (re-confirm)
    print("\n  --- Individual ablations ---")
    for comp in COMPONENTS:
        with _ablation_ctx(model, comp, cfg):
            losses = _batched_loss(model, all_ids, cfg.batch_size, cfg.device)
        abl = _per_type_mean(losses, type_idx, ALL_TYPES)
        dp = np.mean([abl[t] - baseline[t] for t in PERIODIC])
        dc = np.mean([abl[t] - baseline[t] for t in CONTROL])
        results.append({"combo": [comp["comp"]], "d_periodic": float(dp), "d_control": float(dc),
                         "sum_individual": comp["d_individual"],
                         "superadditive": False, "ratio": 1.0})
        print(f"    {comp['comp']:>16s}  ΔL_p={dp:.4f}  ΔL_c={dc:.4f}")

    # Pairs
    print("\n  --- Pairs ---")
    for i, j in combinations(range(len(COMPONENTS)), 2):
        c1, c2 = COMPONENTS[i], COMPONENTS[j]
        sum_ind = c1["d_individual"] + c2["d_individual"]
        with ExitStack() as stack:
            stack.enter_context(_ablation_ctx(model, c1, cfg))
            stack.enter_context(_ablation_ctx(model, c2, cfg))
            losses = _batched_loss(model, all_ids, cfg.batch_size, cfg.device)
        abl = _per_type_mean(losses, type_idx, ALL_TYPES)
        dp = np.mean([abl[t] - baseline[t] for t in PERIODIC])
        dc = np.mean([abl[t] - baseline[t] for t in CONTROL])
        ratio = dp / sum_ind if sum_ind > 0 else 0
        results.append({"combo": [c1["comp"], c2["comp"]],
                         "d_periodic": float(dp), "d_control": float(dc),
                         "sum_individual": float(sum_ind),
                         "superadditive": bool(dp > sum_ind * 1.2),
                         "ratio": float(ratio)})
        flag = " ***SUPER***" if dp > sum_ind * 1.2 else ""
        print(f"    {c1['comp']:>16s} + {c2['comp']:<16s}  "
              f"ΔL_p={dp:.4f}  sum_ind={sum_ind:.4f}  ratio={ratio:.2f}{flag}")

    # Growing cumulative (add components one at a time in selectivity order)
    print("\n  --- Cumulative (growing) ---")
    for k in range(2, len(COMPONENTS) + 1):
        subset = COMPONENTS[:k]
        sum_ind = sum(c["d_individual"] for c in subset)
        with ExitStack() as stack:
            for c in subset:
                stack.enter_context(_ablation_ctx(model, c, cfg))
            losses = _batched_loss(model, all_ids, cfg.batch_size, cfg.device)
        abl = _per_type_mean(losses, type_idx, ALL_TYPES)
        dp = np.mean([abl[t] - baseline[t] for t in PERIODIC])
        dc = np.mean([abl[t] - baseline[t] for t in CONTROL])
        ratio = dp / sum_ind if sum_ind > 0 else 0
        names = [c["comp"] for c in subset]
        results.append({"combo": names, "d_periodic": float(dp), "d_control": float(dc),
                         "sum_individual": float(sum_ind),
                         "superadditive": bool(dp > sum_ind * 1.2),
                         "ratio": float(ratio)})
        flag = " ***SUPER***" if dp > sum_ind * 1.2 else ""
        print(f"    top-{k}: ΔL_p={dp:.4f}  sum_ind={sum_ind:.4f}  "
              f"ratio={ratio:.2f}  ΔL_c={dc:.4f}{flag}")

    del model
    torch.cuda.empty_cache()
    return results, baseline


def run_wiki_circuit(cfg, best_combo):
    """Ablate the best circuit combo on WikiText."""
    print("\n" + "=" * 60)
    print(f"WikiText ablation — circuit: {[c['comp'] for c in best_combo]}")
    print("=" * 60)

    model = load_model(cfg.model_pt, cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_pt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wiki_np = load_wikitext_ids(cfg.n_wikitext, cfg.wiki_seq_len, tokenizer)
    wiki_ids = torch.from_numpy(wiki_np)

    print("  baseline …")
    bl = _batched_loss(model, wiki_ids, cfg.batch_size, cfg.device)
    print(f"    mean wiki loss: {bl.mean():.4f}")

    print("  ablating circuit …")
    with ExitStack() as stack:
        for c in best_combo:
            stack.enter_context(_ablation_ctx(model, c, cfg))
        abl = _batched_loss(model, wiki_ids, cfg.batch_size, cfg.device)

    delta = abl - bl
    print(f"    mean ΔL: {delta.mean():.4f}   max ΔL: {delta.max():.4f}")

    # Top-50 most affected passages
    top_idx = delta.argsort(descending=True)[:50]
    top_passages = []
    for idx in top_idx:
        i = idx.item()
        top_passages.append(dict(
            idx=i, delta=delta[i].item(),
            baseline=bl[i].item(), ablated=abl[i].item(),
            text=tokenizer.decode(wiki_np[i])[:600],
        ))

    # Bottom-50 (least affected / improved)
    bot_idx = delta.argsort()[:50]
    bot_passages = []
    for idx in bot_idx:
        i = idx.item()
        bot_passages.append(dict(
            idx=i, delta=delta[i].item(),
            text=tokenizer.decode(wiki_np[i])[:600],
        ))

    del model
    torch.cuda.empty_cache()
    return dict(
        combo=[c["comp"] for c in best_combo],
        mean_delta=delta.mean().item(),
        max_delta=delta.max().item(),
        top_passages=top_passages,
        bot_passages=bot_passages,
    )


def report(cfg, ts_results, wiki_results):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "cumulative_ts.json", "w") as f:
        json.dump(ts_results, f, indent=2)
    if wiki_results:
        with open(out / "cumulative_wiki.json", "w") as f:
            json.dump(wiki_results, f, indent=2, ensure_ascii=False)

    if wiki_results:
        print("\n" + "=" * 60)
        print("Top 20 WikiText passages MOST degraded by circuit ablation")
        print("=" * 60)
        for p in wiki_results["top_passages"][:20]:
            txt = p["text"][:160].replace("\n", " ")
            print(f"  ΔL={p['delta']:.4f}  \"{txt}\"")

        print("\n" + "=" * 60)
        print("Top 10 WikiText passages LEAST affected (control)")
        print("=" * 60)
        for p in wiki_results["bot_passages"][:10]:
            txt = p["text"][:160].replace("\n", " ")
            print(f"  ΔL={p['delta']:.4f}  \"{txt}\"")

    print(f"\nSaved to {out}")


def main():
    cfg = Cfg()

    ts_results, baseline = run_ts_cumulative(cfg)

    # Pick the best circuit for WikiText testing:
    # Use the growing cumulative that shows strongest superadditivity
    # or just the top-5 selective components
    best_combo = COMPONENTS[:5]
    wiki_results = run_wiki_circuit(cfg, best_combo)

    report(cfg, ts_results, wiki_results)


if __name__ == "__main__":
    main()
