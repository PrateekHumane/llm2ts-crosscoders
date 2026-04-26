"""
Experiment G2: Pass TS through FT model, find WikiText with most similar
hidden states in the same FT model.

For each domain TS:
  1. Tokenize TS via uniform binning → pass through FT → hidden states per layer
  2. Pass WikiText through FT → hidden states per layer
  3. Compare using CKA (per-layer) and cosine similarity
  4. Find best-matching WikiText per layer and overall

Usage:
    /usr/bin/python3 scripts/reverse_mapping_ft.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, time, glob

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.wikitext import load_wikitext_sequences

# ── Config ──
T = 512
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/reverse_mapping_ft"
N_WIKI = 10000  # WikiText sequences to search (10K for speed, can increase)
N_LAYERS = 28
TOP_K = 5

DOMAIN_LABELS = {
    "electricity": "Electricity Load",
    "solar": "Solar Energy",
    "LOOP_SEATTLE": "Traffic (Seattle)",
    "SZ_TAXI": "Taxi (Shenzhen)",
    "temperature_rain_with_missing": "Temperature/Rain",
    "us_births": "US Births",
    "saugeenday": "River Flow (Saugeen)",
    "kdd_cup_2018_with_missing": "Air Quality (KDD)",
}


def load_domain_windows(hf_token, target_domains):
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    arrow_files = sorted(glob.glob(os.path.join(path, "**/*.arrow"), recursive=True))
    domain_series = {}
    for f in arrow_files:
        rel = os.path.relpath(f, path)
        domain = rel.split("/")[0]
        if domain not in target_domains:
            continue
        with open(f, "rb") as fp:
            reader = ipc.open_stream(fp)
            tbl = reader.read_all()
        for t in tbl['target'].to_pylist():
            arr = np.array(t, dtype=np.float32)
            if len(arr) >= T:
                domain_series.setdefault(domain, []).append(arr)
    results = []
    for domain in target_domains:
        if domain not in domain_series:
            continue
        longest = max(domain_series[domain], key=len)
        start = int(len(longest) * 0.7)
        if start + T > len(longest):
            start = len(longest) - T
        window = longest[start:start + T]
        mu, sigma = window.mean(), window.std()
        if sigma < 1e-6:
            start = int(len(longest) * 0.5)
            window = longest[start:start + T]
            mu, sigma = window.mean(), window.std()
        if sigma < 1e-6:
            continue
        window = (window - mu) / sigma
        results.append((domain, torch.tensor(window, dtype=torch.float32)))
    return results


def ts_to_bin_tokens(ts_window, n_bins=512):
    """Convert z-scored TS window to uniform bin token IDs.
    Matches cerc-aai uniform_bin-V512 tokenization."""
    # Clip to [-5, 5], map to [0, n_bins-1]
    clipped = np.clip(ts_window.numpy(), -5, 5)
    bins = ((clipped + 5) / 10 * n_bins).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)
    return torch.tensor(bins, dtype=torch.long)


def linear_cka(X, Y):
    """Compute linear CKA between two matrices X (n, p) and Y (n, q).
    Both should be centered. Returns scalar similarity in [0, 1]."""
    # Center
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    # HSIC
    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-12)
    return (hsic_xy / denom).item()


def cosine_sim_trajectory(X, Y):
    """Average cosine similarity between two (T, D) trajectories."""
    # Per-timestep cosine sim, averaged
    cos = F.cosine_similarity(X, Y, dim=-1)  # (T,)
    return cos.mean().item()


def extract_per_layer(model, token_ids, device):
    """Extract hidden states at each layer. Returns list of 28 (T, D) tensors on CPU."""
    captured = {}
    handles = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook(m, inp, out):
                captured[idx] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=token_ids.unsqueeze(0).to(device), use_cache=False)

    layers = [captured[i].squeeze(0).float().cpu() for i in range(N_LAYERS)]
    for h in handles:
        h.remove()
    return layers


def extract_per_layer_batch(model, sequences, device, desc="", batch_size=8):
    """Extract per-layer hidden states for many sequences.
    Returns: list of N_LAYERS tensors, each (N, T, D) float16 on CPU."""
    captured = {}
    handles = []
    for li in range(N_LAYERS):
        def make_hook(idx):
            def hook(m, inp, out):
                captured[idx] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        handles.append(model.layers[li].register_forward_hook(make_hook(li)))

    # Accumulate per layer
    per_layer = [[] for _ in range(N_LAYERS)]
    N = len(sequences)
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = sequences[start:start + batch_size]
            ids = torch.tensor(np.stack([s["input_ids"] for s in batch]),
                               dtype=torch.long, device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model(input_ids=ids, use_cache=False)
            for li in range(N_LAYERS):
                per_layer[li].append(captured[li].float().cpu().half())
            captured.clear()
            if start % 200 == 0 and start > 0:
                elapsed = time.time() - t0
                rate = start / elapsed
                print(f"    {desc} {start}/{N} ({elapsed:.0f}s, {rate:.0f} seq/s)", flush=True)

    for h in handles:
        h.remove()

    # Concatenate
    result = [torch.cat(per_layer[li], dim=0) for li in range(N_LAYERS)]
    print(f"    {desc} Done: {N} seqs, {result[0].shape} per layer ({time.time()-t0:.0f}s)", flush=True)
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    # ── 1. Load domain TS ──
    print("Loading domain TS windows...", flush=True)
    domain_windows = load_domain_windows(hf_token, list(DOMAIN_LABELS.keys()))
    print(f"  Got {len(domain_windows)} domains\n")

    # ── 2. Load FT model ──
    print("Loading FT model...", flush=True)
    ft_model = AutoModelForCausalLM.from_pretrained(
        "cerc-aai/Qwen3-0.6B-pretrain-normal_scale-uniform_bin-V512",
        dtype=torch.bfloat16
    ).model.to(DEVICE).eval()

    # ── 3. Extract TS hidden states through FT (all layers, small — only 8 sequences) ──
    print("\nExtracting TS hidden states through FT...", flush=True)
    ts_hidden = {}  # domain -> list of 28 (T, D) tensors
    for domain, ts_window in domain_windows:
        label = DOMAIN_LABELS.get(domain, domain)
        bin_tokens = ts_to_bin_tokens(ts_window)
        layers = extract_per_layer(ft_model, bin_tokens, DEVICE)
        ts_hidden[domain] = layers
        print(f"  {label}: {layers[0].shape}")

    # ── 4. Load WikiText ──
    print(f"\nLoading {N_WIKI} WikiText sequences...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=N_WIKI, seq_len=T, hf_token=hf_token)
    actual_n = len(wiki_seqs)
    print(f"  Got {actual_n} sequences")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", token=hf_token)

    # ── 5. Process ONE LAYER AT A TIME to avoid OOM ──
    # For each layer: extract WikiText hidden states → compare to all TS → discard
    print(f"\nProcessing layer-by-layer (one at a time to save RAM)...", flush=True)

    # Initialize results structure
    all_per_layer = {domain: [] for domain, _ in domain_windows}

    for li in range(N_LAYERS):
        print(f"\n  === Layer {li} ===", flush=True)

        # Extract WikiText hidden states for this layer only
        captured = {}
        def make_hook():
            def hook(m, inp, out):
                captured['h'] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        handle = ft_model.layers[li].register_forward_hook(make_hook())

        wiki_h_layer = []  # (N, T, D) for this layer only
        t0 = time.time()
        with torch.no_grad():
            for start in range(0, actual_n, 8):
                batch = wiki_seqs[start:start + 8]
                ids = torch.tensor(np.stack([s["input_ids"] for s in batch]),
                                   dtype=torch.long, device=DEVICE)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    ft_model(input_ids=ids, use_cache=False)
                wiki_h_layer.append(captured['h'].float().cpu().half())
                captured.clear()
                if start % 500 == 0 and start > 0:
                    print(f"    extract {start}/{actual_n} ({time.time()-t0:.0f}s)", flush=True)

        handle.remove()
        wiki_h_layer = torch.cat(wiki_h_layer, dim=0)  # (N, T, D) ~10GB in fp16
        print(f"    Extracted L{li}: {wiki_h_layer.shape} ({time.time()-t0:.0f}s)", flush=True)

        # Compare to each domain's TS at this layer
        for domain, ts_window in domain_windows:
            ts_h = ts_hidden[domain][li].float()  # (T, D)
            ts_norm = F.normalize(ts_h, dim=-1)

            # Batched cosine similarity
            cos_sims = []
            for start in range(0, actual_n, 500):
                batch = wiki_h_layer[start:start + 500].float()
                batch_norm = F.normalize(batch, dim=-1)
                cos = (batch_norm * ts_norm.unsqueeze(0)).sum(-1).mean(-1)
                cos_sims.append(cos)
            cos_sims = torch.cat(cos_sims)
            best_cos_idx = cos_sims.argmax().item()
            best_cos = cos_sims[best_cos_idx].item()

            # CKA on top-50 by cosine
            top_indices = cos_sims.argsort(descending=True)[:50]
            best_cka = -1; best_cka_idx = top_indices[0].item()
            for idx in top_indices:
                wiki_h_i = wiki_h_layer[idx.item()].float()
                cka = linear_cka(ts_h, wiki_h_i)
                if cka > best_cka:
                    best_cka = cka
                    best_cka_idx = idx.item()

            all_per_layer[domain].append({
                "layer": li,
                "best_cos_idx": best_cos_idx,
                "best_cos": best_cos,
                "best_cka_idx": best_cka_idx,
                "best_cka": best_cka,
            })

            label = DOMAIN_LABELS.get(domain, domain)
            if li % 7 == 0 or li == N_LAYERS - 1:
                print(f"    {label}: cos={best_cos:.4f}, CKA={best_cka:.4f}", flush=True)

        # Free this layer's data
        del wiki_h_layer; gc.collect()

    del ft_model; gc.collect(); torch.cuda.empty_cache()

    # ── 6. Compile results ──
    all_results = []
    for domain, ts_window in domain_windows:
        label = DOMAIN_LABELS.get(domain, domain)
        per_layer_results = all_per_layer[domain]

        best_layer = max(per_layer_results, key=lambda x: x["best_cka"])
        best_overall_idx = best_layer["best_cka_idx"]
        best_text = tokenizer.decode(wiki_seqs[best_overall_idx]["input_ids"], skip_special_tokens=True)

        best_cos_layer = max(per_layer_results, key=lambda x: x["best_cos"])
        best_cos_overall_idx = best_cos_layer["best_cos_idx"]
        best_cos_text = tokenizer.decode(wiki_seqs[best_cos_overall_idx]["input_ids"], skip_special_tokens=True)

        print(f"\n  [{label}]")
        print(f"    Best CKA: L{best_layer['layer']} CKA={best_layer['best_cka']:.4f} seq={best_overall_idx}")
        print(f"    Text: \"{best_text[:150]}...\"")

        all_results.append({
            "domain": domain,
            "label": label,
            "per_layer": per_layer_results,
            "best_cka_layer": best_layer["layer"],
            "best_cka_score": best_layer["best_cka"],
            "best_cka_seq": best_overall_idx,
            "best_cka_text": best_text[:500],
            "best_cos_layer": best_cos_layer["layer"],
            "best_cos_score": best_cos_layer["best_cos"],
            "best_cos_seq": best_cos_overall_idx,
            "best_cos_text": best_cos_text[:500],
        })

        # Per-domain plot: CKA and cosine across layers
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"FT Activation Similarity: {label}", fontsize=13, fontweight='bold')

        layers_x = list(range(N_LAYERS))
        cka_vals = [r["best_cka"] for r in per_layer_results]
        cos_vals = [r["best_cos"] for r in per_layer_results]

        axes[0].plot(layers_x, cka_vals, 'o-', color='red', linewidth=1.5)
        axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Best CKA")
        axes[0].set_title("CKA (structural similarity)")

        axes[1].plot(layers_x, cos_vals, 'o-', color='blue', linewidth=1.5)
        axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Best Cosine Sim")
        axes[1].set_title("Cosine Similarity (mean over timesteps)")

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/{domain}_layers.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ── 6. Save results ──
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary plot: CKA across layers for all domains
    fig, ax = plt.subplots(figsize=(12, 6))
    for result in all_results:
        cka_vals = [r["best_cka"] for r in result["per_layer"]]
        ax.plot(range(N_LAYERS), cka_vals, 'o-', linewidth=1.2, markersize=3,
                label=result["label"])
    ax.set_xlabel("Layer"); ax.set_ylabel("Best CKA (TS vs WikiText)")
    ax.set_title("Per-Layer CKA: TS through FT vs WikiText through FT")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/cka_all_domains.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary table
    print(f"\n{'='*70}")
    print("EXPERIMENT G2 RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Domain':<25} {'Best Layer':>10} {'CKA':>8} {'Cos':>8} {'Seq':>6}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['label']:<25} L{r['best_cka_layer']:>8} {r['best_cka_score']:>8.4f} "
              f"{r['best_cos_score']:>8.4f} {r['best_cka_seq']:>6}")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
