"""Ablation-based circuit identification for FT_IO.

Phase 1: Zero-ablate each attention head and MLP on periodic TS,
         rank components by loss increase.
Phase 2: For the top-K critical components, ablate on WikiText
         and find which passages degrade most.

Usage:  python3 FT_IO_circuits/run_ablation.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json
import time
import numpy as np
import torch
from dataclasses import dataclass
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
    num_layers: int = 28
    num_heads: int = 16
    head_dim: int = 128
    n_wikitext: int = 2000
    wiki_seq_len: int = 512
    top_k_components: int = 20
    output_dir: str = "FT_IO_circuits/ablation_results"


PERIODIC = ["sine", "square_wave", "sawtooth", "seasonal", "damped_sine"]
CONTROL = ["white_noise", "constant", "linear_trend", "random_walk"]
ALL_TYPES = PERIODIC + CONTROL


# ── helpers ──────────────────────────────────────────────────────────────────

def _batched_loss(model, all_ids: torch.Tensor, bs: int, device: str) -> torch.Tensor:
    """Compute per-sequence loss over a full dataset. Returns (N,) CPU tensor."""
    parts = []
    for s in range(0, len(all_ids), bs):
        batch = all_ids[s : s + bs].to(device)
        parts.append(compute_loss(model, batch))
    return torch.cat(parts)


def _per_type_mean(losses: torch.Tensor, type_idx: torch.Tensor,
                   type_names: list[str]) -> dict[str, float]:
    return {t: losses[type_idx == i].mean().item()
            for i, t in enumerate(type_names)}


# ── Phase 1 ─────────────────────────────────────────────────────────────────

def run_phase1(cfg: Cfg):
    print("=" * 60)
    print("Phase 1 — ablation on synthetic time series")
    print("=" * 60)

    all_ts = generate_all(cfg.n_per_type, cfg.context_length)
    tokens_per_type = {t: tokenize_windows(all_ts[t], n_bins=cfg.n_bins) for t in ALL_TYPES}

    all_ids = torch.from_numpy(np.concatenate([tokens_per_type[t] for t in ALL_TYPES]))
    type_idx = torch.cat([torch.full((cfg.n_per_type,), i) for i, _ in enumerate(ALL_TYPES)]).long()

    print(f"  {len(ALL_TYPES)} types × {cfg.n_per_type} = {len(all_ids)} windows")

    model = load_model(cfg.model_ft_io, cfg.device)

    # baseline
    print("  baseline …")
    bl = _batched_loss(model, all_ids, cfg.batch_size, cfg.device)
    baseline = _per_type_mean(bl, type_idx, ALL_TYPES)
    for t in ALL_TYPES:
        print(f"    {t:>14s}: {baseline[t]:.4f}")

    # sweep
    total = cfg.num_layers * (cfg.num_heads + 1)
    results = []
    t0 = time.time()
    count = 0

    for li in range(cfg.num_layers):
        for hi in range(cfg.num_heads):
            with ablate_head(model, li, hi, cfg.num_heads, cfg.head_dim):
                losses = _batched_loss(model, all_ids, cfg.batch_size, cfg.device)
            abl = _per_type_mean(losses, type_idx, ALL_TYPES)
            deltas = {t: abl[t] - baseline[t] for t in ALL_TYPES}
            dp = np.mean([deltas[t] for t in PERIODIC])
            dc = np.mean([deltas[t] for t in CONTROL])
            results.append(dict(
                comp=f"head_L{li}_H{hi}", kind="head", layer=li, head=hi,
                d_periodic=dp, d_control=dc, d_selective=dp - dc, per_type=deltas,
            ))
            count += 1
            if count % 50 == 0:
                print(f"  {count}/{total}  ({time.time()-t0:.0f}s)")

        with ablate_mlp(model, li):
            losses = _batched_loss(model, all_ids, cfg.batch_size, cfg.device)
        abl = _per_type_mean(losses, type_idx, ALL_TYPES)
        deltas = {t: abl[t] - baseline[t] for t in ALL_TYPES}
        dp = np.mean([deltas[t] for t in PERIODIC])
        dc = np.mean([deltas[t] for t in CONTROL])
        results.append(dict(
            comp=f"mlp_L{li}", kind="mlp", layer=li, head=None,
            d_periodic=dp, d_control=dc, d_selective=dp - dc, per_type=deltas,
        ))
        count += 1
        if count % 50 == 0 or li == cfg.num_layers - 1:
            print(f"  {count}/{total}  ({time.time()-t0:.0f}s)")

    results.sort(key=lambda r: -r["d_periodic"])
    del model
    torch.cuda.empty_cache()
    return results, baseline


# ── Phase 2 ─────────────────────────────────────────────────────────────────

def run_phase2(cfg: Cfg, critical: list[dict]):
    print()
    print("=" * 60)
    print(f"Phase 2 — ablation on WikiText for top {len(critical)} components")
    print("=" * 60)

    model = load_model(cfg.model_pt, cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_pt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wiki_ids_np = load_wikitext_ids(cfg.n_wikitext, cfg.wiki_seq_len, tokenizer)
    wiki_ids = torch.from_numpy(wiki_ids_np)
    n = len(wiki_ids)

    print("  baseline …")
    bl = _batched_loss(model, wiki_ids, cfg.batch_size, cfg.device)
    print(f"    mean wiki loss: {bl.mean():.4f}")

    wiki_results = []
    for ci, comp in enumerate(critical):
        name = comp["comp"]
        print(f"  [{ci+1}/{len(critical)}] {name} …")

        if comp["kind"] == "head":
            ctx = ablate_head(model, comp["layer"], comp["head"],
                              cfg.num_heads, cfg.head_dim)
        else:
            ctx = ablate_mlp(model, comp["layer"])

        with ctx:
            abl = _batched_loss(model, wiki_ids, cfg.batch_size, cfg.device)

        delta = abl - bl                               # (n,)
        top_idx = delta.argsort(descending=True)[:30]

        top_passages = []
        for idx in top_idx:
            i = idx.item()
            top_passages.append(dict(
                idx=i,
                delta=delta[i].item(),
                baseline=bl[i].item(),
                ablated=abl[i].item(),
                text=tokenizer.decode(wiki_ids_np[i])[:600],
            ))

        wiki_results.append(dict(
            comp=name,
            mean_delta=delta.mean().item(),
            max_delta=delta.max().item(),
            top_passages=top_passages,
        ))

    del model
    torch.cuda.empty_cache()
    return wiki_results


# ── output ───────────────────────────────────────────────────────────────────

def report(cfg, p1, baseline, p2):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "phase1.json", "w") as f:
        json.dump(p1, f, indent=2)
    with open(out / "baseline_ts.json", "w") as f:
        json.dump(baseline, f, indent=2)
    if p2:
        with open(out / "phase2.json", "w") as f:
            json.dump(p2, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Top 30 components — ranked by ΔLoss on periodic TS")
    print("=" * 60)
    print(f"{'component':>18}  {'ΔL periodic':>12}  {'ΔL control':>12}  {'selective':>10}")
    for r in p1[:30]:
        print(f"{r['comp']:>18}  {r['d_periodic']:>12.4f}  "
              f"{r['d_control']:>12.4f}  {r['d_selective']:>10.4f}")

    if p2:
        print("\n" + "=" * 60)
        print("WikiText passages most degraded by each critical component")
        print("=" * 60)
        for wr in p2[:10]:
            print(f"\n--- {wr['comp']} (mean ΔL wiki = {wr['mean_delta']:.4f}) ---")
            for p in wr["top_passages"][:3]:
                txt = p["text"][:160].replace("\n", " ")
                print(f"  ΔL={p['delta']:.4f}  \"{txt}\"")

    print(f"\nSaved to {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = Cfg()
    p1, baseline = run_phase1(cfg)
    critical = p1[: cfg.top_k_components]
    p2 = run_phase2(cfg, critical)
    report(cfg, p1, baseline, p2)


if __name__ == "__main__":
    main()
