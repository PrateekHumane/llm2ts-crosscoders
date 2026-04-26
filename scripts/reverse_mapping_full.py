"""
Reverse Mapping with full WikiText-103 corpus (100K sequences).
Extract hidden states in batches, project through mapper immediately,
discard hidden states — only keep 1D predictions (~200MB).

Usage:
    /usr/bin/python3 scripts/reverse_mapping_full.py
"""
import torch
import torch.nn as nn
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
D_CONCAT = 28 * 1024
N_SEQS = 100000
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/reverse_mapping_100k"
TOP_K = 5
EXTRACT_BS = 8  # batch size for extraction

DOMAIN_LABELS = {
    "electricity": "Electricity Load",
    "solar": "Solar Energy",
    "LOOP_SEATTLE": "Traffic (Seattle)",
    "SZ_TAXI": "Taxi (Shenzhen)",
    "temperature_rain_with_missing": "Temperature/Rain",
    "us_births": "US Births",
    "saugeenday": "River Flow (Saugeen)",
    "kdd_cup_2018_with_missing": "Air Quality (KDD)",
    "covid_deaths": "COVID Deaths",
    "hospital": "Hospital Admissions",
    "restaurant": "Restaurant Sales",
    "hierarchical_sales": "Retail Sales",
}


def load_domain_windows(hf_token, target_domains):
    """Load one representative TS window per domain."""
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
        longest = max(series_list, key=len)
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

    # ── 1. Load domain targets ──
    print("Loading domain-specific TS windows...", flush=True)
    domain_windows = load_domain_windows(hf_token, list(DOMAIN_LABELS.keys()))
    print(f"  Got {len(domain_windows)} domain windows\n")

    # ── 2. Load ALL WikiText sequences ──
    print(f"Loading {N_SEQS} WikiText sequences...", flush=True)
    wiki_seqs = load_wikitext_sequences(max_sequences=N_SEQS, seq_len=T, hf_token=hf_token)
    actual_n = len(wiki_seqs)
    print(f"  Got {actual_n} sequences\n")

    # ── 3. Extract + project in streaming fashion ──
    pred_path = f"{OUT_DIR}/pred_all_{actual_n}.pt"
    if os.path.exists(pred_path):
        print(f"Loading cached predictions from {pred_path}...", flush=True)
        pred = torch.load(pred_path, map_location="cpu", weights_only=True)
        print(f"  Loaded: {pred.shape}")
    else:
        print("Loading model and mapper...", flush=True)
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

        print(f"Extracting + projecting {actual_n} sequences (batch={EXTRACT_BS})...", flush=True)
        all_pred = []
        t0 = time.time()
        with torch.no_grad():
            for start in range(0, actual_n, EXTRACT_BS):
                batch = wiki_seqs[start:start + EXTRACT_BS]
                ids = torch.tensor(np.stack([s["input_ids"] for s in batch]),
                                   dtype=torch.long, device=DEVICE)
                model(input_ids=ids, use_cache=False)
                layers = [captured[i].float()[:, :T, :] for i in range(28)]
                h = torch.cat(layers, dim=-1)  # (B, T, D) on GPU
                yr = mapper(h).squeeze(-1)  # (B, T)
                ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
                pred_batch = ((yr - yr.mean(-1, keepdim=True)) / ys).cpu()
                all_pred.append(pred_batch)
                captured.clear()

                if start % 200 == 0 and start > 0:
                    elapsed = time.time() - t0
                    rate = start / elapsed
                    eta = (actual_n - start) / rate
                    print(f"  {start}/{actual_n} ({elapsed:.0f}s, {rate:.1f} seq/s, ETA {eta:.0f}s)",
                          flush=True)

        for h in handles:
            h.remove()
        del model, mapper; gc.collect(); torch.cuda.empty_cache()

        pred = torch.cat(all_pred, dim=0)
        torch.save(pred, pred_path)
        print(f"  Done: {pred.shape} saved to {pred_path} ({time.time()-t0:.0f}s)\n")

    # ── 4. Decode text for matched sequences only (lazy — decode after matching) ──
    print("Finding top-K matches per domain...", flush=True)
    all_results = []
    matched_indices = set()

    for domain, target_ts in domain_windows:
        d2 = ((pred - target_ts.unsqueeze(0)) ** 2).mean(-1)
        top_idx = d2.argsort()[:TOP_K].numpy()
        top_dist = d2[top_idx].numpy()
        matched_indices.update(top_idx.tolist())

        label = DOMAIN_LABELS.get(domain, domain)
        print(f"\n  [{label}]")
        matches = []
        for rank, (idx, dist) in enumerate(zip(top_idx, top_dist)):
            matches.append({"rank": rank + 1, "seq_idx": int(idx), "distance": float(dist)})
            print(f"    #{rank+1}: seq {idx}, d={dist:.4f}")

        all_results.append({"domain": domain, "label": label, "matches": matches})

    # Decode only the matched texts
    print(f"\nDecoding text for {len(matched_indices)} matched sequences...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", token=hf_token)
    decoded_texts = {}
    for idx in sorted(matched_indices):
        text = tokenizer.decode(wiki_seqs[idx]["input_ids"], skip_special_tokens=True)
        decoded_texts[idx] = text

    # Attach text to results
    for result in all_results:
        for m in result["matches"]:
            idx = m["seq_idx"]
            m["text_preview"] = decoded_texts[idx][:500]
            m["full_text"] = decoded_texts[idx]

    # ── 5. Save results ──
    results_json = []
    for r in all_results:
        entry = {"domain": r["domain"], "label": r["label"], "matches": []}
        for m in r["matches"]:
            entry["matches"].append({
                "rank": m["rank"], "seq_idx": m["seq_idx"],
                "distance": m["distance"], "text_preview": m["text_preview"],
            })
        results_json.append(entry)
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    with open(f"{OUT_DIR}/full_texts.json", "w") as f:
        json.dump([{
            "domain": r["domain"], "label": r["label"],
            "matches": [{"rank": m["rank"], "seq_idx": m["seq_idx"],
                         "distance": m["distance"], "full_text": m["full_text"]}
                        for m in r["matches"]]
        } for r in all_results], f, indent=2)

    # ── 6. Plots ──
    print("\nGenerating plots...", flush=True)
    domain_dict = dict(domain_windows)

    # Per-domain plots
    for result in all_results:
        domain = result["domain"]
        label = result["label"]
        target_ts = domain_dict[domain]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle(f"Reverse Mapping (100K corpus): {label}\nTarget TS (green) vs decoded WikiText (blue)",
                     fontsize=13, fontweight='bold')

        axes[0].plot(target_ts.numpy(), color="green", linewidth=1.5)
        axes[0].set_ylabel("Target TS", fontsize=10)
        axes[0].set_ylim(-4, 4)
        axes[0].set_title(f"Target: {label}", fontsize=11)

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

    # Summary grid
    n_domains = len(all_results)
    n_cols = 3
    n_rows = (n_domains + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Reverse Mapping (100K WikiText): Best Match per Domain",
                 fontsize=15, fontweight='bold', y=1.01)

    for i, result in enumerate(all_results):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        domain = result["domain"]
        target_ts = domain_dict[domain]
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

    for i in range(n_domains, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/summary_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 7. Print summary ──
    print(f"\n{'='*70}")
    print(f"REVERSE MAPPING RESULTS (100K WikiText corpus)")
    print(f"{'='*70}")
    for result in all_results:
        m = result["matches"][0]
        print(f"\n{result['label']} (d={m['distance']:.4f}):")
        print(f"  Seq #{m['seq_idx']}: \"{m['text_preview'][:150]}...\"")

    print(f"\n\nComparison: 1920 vs 100K corpus")
    print(f"{'Domain':<25} {'1920 best':>10} {'100K best':>10} {'Improvement':>12}")
    print("-" * 60)
    # Load old results for comparison
    old_path = "mapping_results/reverse_mapping/results.json"
    if os.path.exists(old_path):
        with open(old_path) as f:
            old_results = {r["domain"]: r["matches"][0]["distance"] for r in json.load(f)}
        for result in all_results:
            d = result["domain"]
            new_dist = result["matches"][0]["distance"]
            old_dist = old_results.get(d)
            if old_dist:
                imp = (old_dist - new_dist) / old_dist * 100
                print(f"{result['label']:<25} {old_dist:>10.4f} {new_dist:>10.4f} {imp:>10.1f}%")
            else:
                print(f"{result['label']:<25} {'n/a':>10} {new_dist:>10.4f}")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
