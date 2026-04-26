"""
Reverse Mapping (Experiment G1):
For representative TS from distinct GiftEval domains, find WikiText passages
whose PT hidden states (projected through the trained mapper W) produce
the closest match.

Usage:
    /usr/bin/python3 scripts/reverse_mapping.py
"""
import torch
import torch.nn as nn
import numpy as np
import json, os, sys, gc, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences

# ── Config ──
T = 512
D_CONCAT = 28 * 1024
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/reverse_mapping"
TOP_K = 5  # top-K text matches per target TS

# Domains to sample from, with human-readable labels
DOMAIN_LABELS = {
    "electricity": "Electricity Load",
    "solar": "Solar Energy",
    "LOOP_SEATTLE": "Traffic (Seattle)",
    "SZ_TAXI": "Taxi (Shenzhen)",
    "jena_weather": "Weather (Jena)",
    "temperature_rain_with_missing": "Temperature/Rain",
    "covid_deaths": "COVID Deaths",
    "us_births": "US Births",
    "saugeenday": "River Flow (Saugeen)",
    "restaurant": "Restaurant Sales",
    "hospital": "Hospital Admissions",
    "kdd_cup_2018_with_missing": "Air Quality (KDD)",
}


def load_domain_windows(hf_token, target_domains, n_per_domain=1):
    """Load representative TS windows from specific GiftEval domains.
    Returns list of (domain, window_tensor) tuples."""
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    import glob
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
        targets = tbl['target'].to_pylist()
        for t in targets:
            arr = np.array(t, dtype=np.float32)
            if len(arr) >= T:
                if domain not in domain_series:
                    domain_series[domain] = []
                domain_series[domain].append(arr)

    results = []
    for domain in target_domains:
        if domain not in domain_series:
            print(f"  Skipping {domain}: no series with len >= {T}")
            continue
        series_list = domain_series[domain]
        # Pick a window from the middle of the longest series (most representative)
        longest = max(series_list, key=len)
        # Take a window from 70% into the series (validation region)
        start = int(len(longest) * 0.7)
        if start + T > len(longest):
            start = len(longest) - T
        window = longest[start:start + T]
        # Z-score normalize
        mu, sigma = window.mean(), window.std()
        if sigma < 1e-6:
            # Try another window
            start = int(len(longest) * 0.5)
            window = longest[start:start + T]
            mu, sigma = window.mean(), window.std()
        if sigma < 1e-6:
            print(f"  Skipping {domain}: constant window")
            continue
        window = (window - mu) / sigma
        results.append((domain, torch.tensor(window, dtype=torch.float32)))
        print(f"  {domain}: window from series len={len(longest)}, start={start}")

    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    # ── 1. Load target TS from distinct domains ──
    print("Loading domain-specific TS windows...", flush=True)
    target_domains = list(DOMAIN_LABELS.keys())
    domain_windows = load_domain_windows(hf_token, target_domains)
    print(f"  Got {len(domain_windows)} domain windows")

    # ── 2. Load WikiText sequences and their text ──
    print("\nLoading WikiText sequences...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=2000, seq_len=T, hf_token=hf_token)
    train_seqs = wiki_seqs[:1920]

    # Decode token IDs back to text
    print("Loading tokenizer for text decoding...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", token=hf_token)
    wiki_texts = []
    for seq in train_seqs:
        text = tokenizer.decode(seq["input_ids"], skip_special_tokens=True)
        wiki_texts.append(text)

    # ── 3. Load saved predictions ──
    print("\nLoading saved predictions...", flush=True)
    pred_path = "mapping_results/additional_experiments/pred_text_PT.pt"
    if os.path.exists(pred_path):
        pred = torch.load(pred_path, map_location="cpu", weights_only=True)
        print(f"  Loaded predictions: {pred.shape}")
        # These are from 500 sequences. We need predictions for all 1920.
        # Check if we have the full set
        if pred.shape[0] < 1920:
            print(f"  Only {pred.shape[0]} predictions saved, need 1920. Regenerating...")
            pred = None
    else:
        pred = None

    if pred is None:
        # Generate predictions for all 1920 training sequences
        print("  Extracting hidden states and generating predictions for all 1920 seqs...", flush=True)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", dtype=torch.bfloat16).model.to(DEVICE).eval()

        captured = {}
        handles = []
        for li in range(28):
            def make_hook(idx):
                def hook(m, i, o):
                    captured[idx] = (o[0] if isinstance(o, tuple) else o).detach()
                return hook
            handles.append(model.layers[li].register_forward_hook(make_hook(li)))

        mapper = nn.Linear(D_CONCAT, 1)
        mapper.load_state_dict(torch.load(
            "mapping_results/ablation/mapper_text_PT.pt", map_location="cpu", weights_only=True))
        mapper = mapper.to(DEVICE).eval()

        all_pred = []
        BS = 8
        t0 = time.time()
        with torch.no_grad():
            for start in range(0, len(train_seqs), BS):
                batch = train_seqs[start:start + BS]
                ids = torch.tensor(np.stack([s["input_ids"] for s in batch]),
                                   dtype=torch.long, device=DEVICE)
                model(input_ids=ids, use_cache=False)
                layers = [captured[i].float().cpu()[:, :T, :] for i in range(28)]
                h = torch.cat(layers, dim=-1).to(DEVICE)
                yr = mapper(h).squeeze(-1)
                ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
                pred_batch = ((yr - yr.mean(-1, keepdim=True)) / ys).cpu()
                all_pred.append(pred_batch)
                captured.clear()
                if start % 80 == 0 and start > 0:
                    print(f"    {start}/{len(train_seqs)} ({time.time()-t0:.0f}s)", flush=True)

        for h in handles:
            h.remove()
        del model, mapper; gc.collect(); torch.cuda.empty_cache()
        pred = torch.cat(all_pred, dim=0)
        # Save for reuse
        torch.save(pred, f"{OUT_DIR}/pred_all_1920.pt")
        print(f"  Generated {pred.shape[0]} predictions ({time.time()-t0:.0f}s)")

    # ── 4. For each domain target, find top-K matching WikiText sequences ──
    print(f"\nFinding top-{TOP_K} WikiText matches per domain target...", flush=True)
    all_results = []

    for domain, target_ts in domain_windows:
        # Compute MSE between each prediction and this target
        # pred: (N, T), target_ts: (T,)
        d2 = ((pred - target_ts.unsqueeze(0)) ** 2).mean(-1)  # (N,)
        top_indices = d2.argsort()[:TOP_K].numpy()
        top_dists = d2[top_indices].numpy()

        label = DOMAIN_LABELS.get(domain, domain)
        print(f"\n  [{label}] (domain={domain})")
        matches = []
        for rank, (idx, dist) in enumerate(zip(top_indices, top_dists)):
            text_preview = wiki_texts[idx][:200].replace('\n', ' ')
            print(f"    #{rank+1} (seq {idx}, d={dist:.4f}): {text_preview}...")
            matches.append({
                "rank": rank + 1,
                "seq_idx": int(idx),
                "distance": float(dist),
                "text_preview": wiki_texts[idx][:500],
                "full_text": wiki_texts[idx],
            })

        all_results.append({
            "domain": domain,
            "label": label,
            "matches": matches,
        })

    # ── 5. Save results ──
    # Save JSON (without full text for readability)
    results_json = []
    for r in all_results:
        entry = {"domain": r["domain"], "label": r["label"], "matches": []}
        for m in r["matches"]:
            entry["matches"].append({
                "rank": m["rank"],
                "seq_idx": m["seq_idx"],
                "distance": m["distance"],
                "text_preview": m["text_preview"],
            })
        results_json.append(entry)
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Save full texts
    with open(f"{OUT_DIR}/full_texts.json", "w") as f:
        json.dump([{
            "domain": r["domain"],
            "label": r["label"],
            "matches": [{"rank": m["rank"], "seq_idx": m["seq_idx"],
                         "distance": m["distance"], "full_text": m["full_text"]}
                        for m in r["matches"]]
        } for r in all_results], f, indent=2)

    # ── 6. Generate plots ──
    print(f"\nGenerating plots...", flush=True)
    n_domains = len(all_results)

    # Plot: For each domain, show target TS + top-3 decoded matches
    for ri, result in enumerate(all_results):
        domain = result["domain"]
        label = result["label"]
        target_ts = dict(domain_windows)[domain]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle(f"Reverse Mapping: {label}\nTarget TS (green) vs decoded WikiText (blue)",
                     fontsize=13, fontweight='bold')

        # Top row: target TS alone
        axes[0].plot(target_ts.numpy(), color="green", linewidth=1.5)
        axes[0].set_ylabel("Target TS", fontsize=10)
        axes[0].set_ylim(-4, 4)
        axes[0].set_title(f"Target: {label}", fontsize=11)

        # Rows 1-3: top 3 matches overlaid
        for row in range(min(3, len(result["matches"]))):
            ax = axes[row + 1]
            m = result["matches"][row]
            idx = m["seq_idx"]
            ax.plot(target_ts.numpy(), color="green", linewidth=1.2, alpha=0.5, label="Target TS")
            ax.plot(pred[idx].numpy(), color="blue", linewidth=1.2, alpha=0.8, label="Decoded")
            ax.set_ylim(-4, 4)
            text_short = m["text_preview"][:120].replace('\n', ' ')
            ax.set_ylabel(f"#{row+1} (d={m['distance']:.3f})", fontsize=9)
            ax.set_title(f'"{text_short}..."', fontsize=8, style='italic', color='gray')
            if row == 0:
                ax.legend(fontsize=9, loc='upper right')

        axes[-1].set_xlabel("Timestep")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/{domain}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Summary grid: all domains, top-1 match each
    n_cols = 3
    n_rows = (n_domains + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Reverse Mapping: Best WikiText Match per Domain",
                 fontsize=15, fontweight='bold', y=1.01)

    for i, result in enumerate(all_results):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        domain = result["domain"]
        target_ts = dict(domain_windows)[domain]
        m = result["matches"][0]
        idx = m["seq_idx"]
        ax.plot(target_ts.numpy(), color="green", linewidth=1.2, alpha=0.6, label="Real TS")
        ax.plot(pred[idx].numpy(), color="blue", linewidth=1.2, alpha=0.8, label="Decoded")
        ax.set_ylim(-4, 4)
        ax.set_title(f"{result['label']} (d={m['distance']:.3f})", fontsize=10, fontweight='bold')
        text_short = m["text_preview"][:80].replace('\n', ' ')
        ax.text(0.02, 0.02, f'"{text_short}..."', transform=ax.transAxes,
                fontsize=6, style='italic', color='gray', va='bottom')
        if i == 0:
            ax.legend(fontsize=8)

    # Hide empty subplots
    for i in range(n_domains, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/summary_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAll results saved to {OUT_DIR}/")
    print(f"Plots saved to {OUT_DIR}/plots/")


if __name__ == "__main__":
    main()
