"""FT_IO circuit analysis pipeline.

Finds WikiText passages whose internal activations match those produced by
synthetic time series in the FT_IO model.  Since FT_IO only trained embeddings
+ LM-head, the 28 transformer layers are identical to PT — so we extract
synthetic-TS activations from FT_IO and WikiText activations from PT, then
compare in the shared representation space.

Usage (from repo root):
    python FT_IO_circuits/run.py
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
from FT_IO_circuits.extract import load_model, extract_batch
from FT_IO_circuits.match import (
    ACT_TYPES,
    TopKTracker,
    compute_type_profiles,
    match_batch_tokens,
    match_batch_windows,
    _normalize,
    _gather_profiles,
)


@dataclass
class Config:
    model_ft_io: str = "/workspace/NanoTS_v2/checkpoints/io_only/checkpoint-8192"
    model_pt: str = (
        "/workspace/.hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/"
        "c1899de289a04d12100db370d81485cdf75e47ca"
    )

    n_per_type: int = 50
    context_length: int = 512
    n_bins: int = 1024

    n_wikitext: int = 10_000
    wiki_seq_len: int = 512

    batch_size: int = 16
    device: str = "cuda:0"

    top_k_tokens: int = 50
    top_k_windows: int = 20
    window_size: int = 50
    window_stride: int = 25

    output_dir: str = "FT_IO_circuits/results"


# ── WikiText loading ─────────────────────────────────────────────────────────

def load_wikitext_ids(n_seqs: int, seq_len: int, tokenizer) -> np.ndarray:
    """Load WikiText-103, tokenize, chunk into (n_seqs, seq_len) int64 array."""
    from datasets import load_dataset

    ds = load_dataset(
        "wikitext", "wikitext-103-v1", split="train",
        cache_dir="/workspace/.hf_home/hub",
    )
    all_ids: list[int] = []
    target = n_seqs * seq_len * 2
    for row in ds:
        text = row["text"]
        if not text.strip():
            continue
        enc = tokenizer(text, add_special_tokens=False)
        all_ids.extend(enc["input_ids"])
        if len(all_ids) >= target:
            break

    all_ids = np.array(all_ids[:target], dtype=np.int64)
    n_full = len(all_ids) // seq_len
    n_full = min(n_full, n_seqs)
    arr = all_ids[: n_full * seq_len].reshape(n_full, seq_len)
    print(f"WikiText: {n_full} sequences of length {seq_len}")
    return arr


# ── Phase 1: synthetic TS profiles ──────────────────────────────────────────

def extract_synthetic_profiles(cfg: Config):
    print("=== Phase 1: Synthetic TS → FT_IO activations ===")
    all_ts = generate_all(cfg.n_per_type, cfg.context_length)
    type_names = list(all_ts.keys())
    n_types = len(type_names)
    print(f"  {n_types} types × {cfg.n_per_type} windows = "
          f"{n_types * cfg.n_per_type} total")

    # save example windows for later visualisation
    examples = {}
    for name, wins in all_ts.items():
        examples[name] = wins[:3].tolist()

    # tokenize
    all_tokens = np.concatenate(
        [tokenize_windows(all_ts[n], n_bins=cfg.n_bins) for n in type_names]
    )
    N = all_tokens.shape[0]
    T = cfg.context_length
    type_labels = torch.from_numpy(
        np.repeat(np.arange(n_types), cfg.n_per_type * T)
    ).long()

    print("  Loading FT_IO model …")
    model = load_model(cfg.model_ft_io, cfg.device)
    layers = list(range(28))

    chunks: dict[tuple, list[torch.Tensor]] = {}
    for s in range(0, N, cfg.batch_size):
        e = min(s + cfg.batch_size, N)
        ids = torch.from_numpy(all_tokens[s:e]).to(cfg.device)
        acts = extract_batch(model, ids, layers)
        for k, v in acts.items():
            chunks.setdefault(k, []).append(v.reshape(-1, v.shape[-1]).cpu())

    merged = {k: torch.cat(vs) for k, vs in chunks.items()}
    profiles = compute_type_profiles(merged, type_labels, type_names, layers)
    for k in profiles:
        profiles[k] = profiles[k].to(cfg.device)
    print(f"  {len(profiles)} profiles computed")

    del model, merged, chunks
    torch.cuda.empty_cache()
    return profiles, type_names, examples


# ── Phase 2: WikiText matching ───────────────────────────────────────────────

def run_wikitext_matching(cfg: Config, profiles, type_names):
    print("\n=== Phase 2: WikiText → PT activations, matching ===")
    print("  Loading PT model …")
    model = load_model(cfg.model_pt, cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_pt)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wiki_ids = load_wikitext_ids(cfg.n_wikitext, cfg.wiki_seq_len, tokenizer)
    n_wiki = wiki_ids.shape[0]

    layers = list(range(28))
    trackers: dict[tuple, TopKTracker] = {}
    for at in ACT_TYPES:
        for li in layers:
            for ts in type_names:
                trackers[(at, li, ts, "token")] = TopKTracker(cfg.top_k_tokens)
                trackers[(at, li, ts, "window")] = TopKTracker(cfg.top_k_windows)

    overview = {at: np.zeros((28, len(type_names)), dtype=np.float64)
                for at in ACT_TYPES}
    n_passages = 0

    n_batches = (n_wiki + cfg.batch_size - 1) // cfg.batch_size
    t0 = time.time()
    for bi in range(n_batches):
        s = bi * cfg.batch_size
        e = min(s + cfg.batch_size, n_wiki)
        ids_np = wiki_ids[s:e]
        ids = torch.from_numpy(ids_np).to(cfg.device)
        B = ids.shape[0]

        acts = extract_batch(model, ids, layers)

        # overview: per-passage mean similarity (batched across types)
        for at in ACT_TYPES:
            for li in layers:
                a = acts.get((at, li))
                if a is None:
                    continue
                pm = _normalize(a.mean(dim=1))         # (B, D)
                prof_mat, valid = _gather_profiles(profiles, at, li, type_names)
                if prof_mat is None:
                    continue
                sims = (pm @ prof_mat.T).sum(dim=0)    # (n_valid,)
                for ti, ts in enumerate(valid):
                    idx = type_names.index(ts)
                    overview[at][li, idx] += sims[ti].item()
        n_passages += B

        pidxs = list(range(s, e))
        ids_cpu = ids.cpu()
        match_batch_tokens(acts, profiles, tokenizer, ids_cpu,
                           pidxs, trackers, layers, type_names)
        match_batch_windows(acts, profiles, tokenizer, ids_cpu,
                            pidxs, trackers, layers, type_names,
                            window_size=cfg.window_size,
                            stride=cfg.window_stride)

        if (bi + 1) % 25 == 0 or bi == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  batch {bi+1}/{n_batches}  ({elapsed:.0f}s)")

    for at in overview:
        overview[at] /= n_passages

    del model
    torch.cuda.empty_cache()
    return trackers, overview, tokenizer


# ── save results ─────────────────────────────────────────────────────────────

def save_results(cfg, type_names, examples, trackers, overview):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # overview arrays
    for at in ACT_TYPES:
        np.save(out / f"overview_{at}.npy", overview[at])
    with open(out / "type_names.json", "w") as f:
        json.dump(type_names, f)
    with open(out / "examples.json", "w") as f:
        json.dump(examples, f)

    # detailed matches — one JSON per (act_type, layer, ts_type, match_type)
    match_dir = out / "matches"
    match_dir.mkdir(exist_ok=True)
    n_saved = 0
    for key, tracker in trackers.items():
        matches = tracker.get_sorted()
        if not matches:
            continue
        at, li, ts, mt = key
        payload = {
            "act_type": at, "layer": li, "ts_type": ts, "match_type": mt,
            "matches": [m.to_dict() for m in matches],
        }
        with open(match_dir / f"{at}_L{li:02d}_{ts}_{mt}.json", "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        n_saved += 1
    print(f"\nSaved {n_saved} match files to {match_dir}")

    # overview summary table (MLP)
    mlp = overview["mlp"]
    print("\n=== Mean cosine similarity — MLP ===")
    header = f"{'Layer':>6}" + "".join(f"  {t[:7]:>7}" for t in type_names)
    print(header)
    for li in range(28):
        row = f"  L{li:02d} " + "".join(f"  {mlp[li,ti]:7.4f}"
                                         for ti in range(len(type_names)))
        print(row)

    avg = mlp.mean(axis=1)
    top5 = np.argsort(avg)[::-1][:5]
    print("\nTop-5 layers by mean MLP similarity:")
    for li in top5:
        print(f"  Layer {li}: {avg[li]:.4f}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()
    profiles, type_names, examples = extract_synthetic_profiles(cfg)
    trackers, overview, tokenizer = run_wikitext_matching(cfg, profiles, type_names)
    save_results(cfg, type_names, examples, trackers, overview)


if __name__ == "__main__":
    main()
