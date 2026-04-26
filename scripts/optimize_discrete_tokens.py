"""
Discrete token optimization for reverse mapping.
Given target TS and the continuous-optimized embeddings, find actual token
sequences that minimize decoded loss via coordinate descent.

Approach:
  1. Initialize from best WikiText match (already found by reverse_mapping_100k)
  2. For each position, evaluate top-K candidate tokens (nearest to continuous optimum)
  3. Keep the token that minimizes full forward-pass loss
  4. Repeat for multiple passes

Usage:
    /usr/bin/python3 scripts/optimize_discrete_tokens.py
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

T = 512
D_CONCAT = 28 * 1024
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/optimized_text"

N_CANDIDATES = 100   # top-K nearest tokens to try per position
N_PASSES = 3         # number of full sweeps over all positions
EVAL_BATCH = 50      # how many candidates to evaluate in one forward pass

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


def forward_and_loss(model, mapper, token_ids, target, captured):
    """Run forward pass, compute decoded TS, return MSE loss."""
    captured.clear()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=token_ids, use_cache=False)
        layers = [captured[i] for i in range(28)]
        concat = torch.cat(layers, dim=-1)
        y = mapper(concat).squeeze(-1).float()
        y_std = y.std(dim=-1, keepdim=True).clamp(min=1e-4)
        y = (y - y.mean(dim=-1, keepdim=True)) / y_std
        loss = F.mse_loss(y, target.unsqueeze(0).expand_as(y))
    return loss.item(), y


def get_gradient_candidates(model, mapper, embed_layer, captured, current_ids, target, n_candidates):
    """Use GCG-style gradient to find promising token candidates per position.
    Returns (T, n_candidates) tensor of token indices."""
    # Need gradients through embedding
    one_hot = F.one_hot(current_ids, embed_layer.weight.shape[0]).to(dtype=embed_layer.weight.dtype)
    one_hot.requires_grad_(True)

    # Forward with differentiable embedding lookup
    embeds = one_hot @ embed_layer.weight  # (T, d) — differentiable
    captured.clear()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(inputs_embeds=embeds.unsqueeze(0), use_cache=False)
        layers = [captured[i] for i in range(28)]
        concat = torch.cat(layers, dim=-1)
        y = mapper(concat).squeeze(-1).float()
        y_std = y.std(dim=-1, keepdim=True).clamp(min=1e-4)
        y = (y - y.mean(dim=-1, keepdim=True)) / y_std
        loss = F.mse_loss(y, target.unsqueeze(0))

    loss.backward()

    # Gradient w.r.t. one_hot: (T, V) — negative gradient = tokens that reduce loss
    grad = one_hot.grad  # (T, V)
    # Top candidates = most negative gradient (steepest descent)
    topk = (-grad).topk(n_candidates, dim=-1).indices  # (T, n_candidates)
    return topk.detach()


def coordinate_descent(model, mapper, embed_layer, tokenizer, captured,
                       init_ids, target_ts,
                       n_candidates=N_CANDIDATES, n_passes=N_PASSES):
    """
    Optimize discrete tokens via GCG-style coordinate descent.
    1. Compute gradient to find top-K candidates per position
    2. For each position, evaluate candidates and keep the best
    3. Repeat for multiple passes
    """
    target = target_ts.to(DEVICE)
    current_ids = init_ids.clone().to(DEVICE)  # (T,)

    # Initial loss
    init_loss, init_pred = forward_and_loss(
        model, mapper, current_ids.unsqueeze(0), target, captured)
    print(f"    Initial loss: {init_loss:.6f}", flush=True)

    best_loss = init_loss
    best_ids = current_ids.clone()
    loss_history = [init_loss]

    for pass_idx in range(n_passes):
        n_improved = 0

        # Get gradient-based candidates for ALL positions at once
        print(f"    Computing gradient candidates...", flush=True)
        candidates = get_gradient_candidates(
            model, mapper, embed_layer, captured, current_ids, target, n_candidates)
        # candidates: (T, n_candidates)

        # Sweep positions in random order
        positions = torch.randperm(T).tolist()

        for pi, pos in enumerate(positions):
            cands = candidates[pos]  # (n_candidates,)
            original_token = current_ids[pos].item()

            # Evaluate candidates in batches
            best_cand_loss = best_loss
            best_cand_token = original_token

            for batch_start in range(0, len(cands), EVAL_BATCH):
                batch_cands = cands[batch_start:batch_start + EVAL_BATCH]
                B = len(batch_cands)

                batch_ids = current_ids.unsqueeze(0).expand(B, -1).clone()
                batch_ids[:, pos] = batch_cands

                captured.clear()
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    model(input_ids=batch_ids, use_cache=False)
                    layers = [captured[i] for i in range(28)]
                    concat = torch.cat(layers, dim=-1)
                    y = mapper(concat).squeeze(-1).float()
                    y_std = y.std(dim=-1, keepdim=True).clamp(min=1e-4)
                    y = (y - y.mean(dim=-1, keepdim=True)) / y_std
                    per_sample_loss = ((y - target.unsqueeze(0)) ** 2).mean(-1)

                min_idx = per_sample_loss.argmin().item()
                min_loss = per_sample_loss[min_idx].item()
                if min_loss < best_cand_loss:
                    best_cand_loss = min_loss
                    best_cand_token = batch_cands[min_idx].item()

            if best_cand_token != original_token:
                current_ids[pos] = best_cand_token
                best_loss = best_cand_loss
                best_ids = current_ids.clone()
                n_improved += 1

            if pi % 100 == 0 and pi > 0:
                print(f"      pass {pass_idx+1}, pos {pi}/{T}: "
                      f"loss={best_loss:.6f}, improved={n_improved}", flush=True)

        loss_history.append(best_loss)
        print(f"    Pass {pass_idx+1}/{n_passes}: loss={best_loss:.6f}, "
              f"tokens changed={n_improved}/{T}", flush=True)

    # Final prediction
    final_loss, final_pred = forward_and_loss(
        model, mapper, best_ids.unsqueeze(0), target, captured)
    decoded_text = tokenizer.decode(best_ids.cpu(), skip_special_tokens=True)

    return {
        "loss": final_loss,
        "pred": final_pred.squeeze(0).cpu(),
        "token_ids": best_ids.cpu(),
        "text": decoded_text,
        "loss_history": loss_history,
        "init_loss": init_loss,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    print("Loading domain targets...", flush=True)
    domain_windows = load_domain_windows(hf_token, list(DOMAIN_LABELS.keys()))
    print(f"  Got {len(domain_windows)} domains\n")

    # Load model (bfloat16)
    print("Loading PT model...", flush=True)
    full_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(DEVICE).eval()
    transformer = full_model.model

    # Register hooks
    captured = {}
    handles = []
    for li in range(28):
        def make_hook(idx):
            def hook(m, inp, out):
                captured[idx] = (out[0] if isinstance(out, tuple) else out)
            return hook
        handles.append(transformer.layers[li].register_forward_hook(make_hook(li)))

    # Load mapper
    mapper = nn.Linear(D_CONCAT, 1).to(DEVICE)
    mapper.load_state_dict(torch.load(
        "mapping_results/ablation/mapper_text_PT.pt", map_location="cpu", weights_only=True))
    mapper.eval()

    embed_layer = transformer.embed_tokens
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", token=hf_token)

    # Load WikiText best matches as initialization
    wiki_results_path = "mapping_results/reverse_mapping_100k/results.json"
    wiki_results = {}
    if os.path.exists(wiki_results_path):
        with open(wiki_results_path) as f:
            for r in json.load(f):
                wiki_results[r["domain"]] = r["matches"][0]

    # Load WikiText sequences for init token IDs
    from src.data.wikitext import load_wikitext_sequences
    wiki_seqs = load_wikitext_sequences(max_sequences=100000, seq_len=T, hf_token=hf_token)

    # Load continuous optimized embeddings if available
    # (from the parallel continuous optimization run)
    cont_embeds_available = os.path.exists(f"{OUT_DIR}/results.json")

    all_results = []
    print(f"\n{'='*60}")
    print("DISCRETE TOKEN OPTIMIZATION")
    print(f"{'='*60}\n")

    for domain, target_ts in domain_windows:
        label = DOMAIN_LABELS.get(domain, domain)
        print(f"\n--- {label} ---")

        # Initialize from best WikiText match
        wiki_match = wiki_results.get(domain)
        if wiki_match:
            init_idx = wiki_match["seq_idx"]
            init_ids = torch.tensor(wiki_seqs[init_idx]["input_ids"], dtype=torch.long)
            wiki_dist = wiki_match["distance"]
            print(f"  Init from WikiText seq {init_idx} (d={wiki_dist:.4f})")
        else:
            init_ids = torch.randint(0, embed_layer.weight.shape[0], (T,))
            wiki_dist = float('nan')
            print(f"  Init from random tokens")

        # Run coordinate descent with GCG-style gradient candidates
        result = coordinate_descent(
            transformer, mapper, embed_layer, tokenizer, captured,
            init_ids, target_ts,
            n_candidates=N_CANDIDATES, n_passes=N_PASSES)

        print(f"  Result: {wiki_dist:.4f} (wiki) → {result['loss']:.4f} (optimized)")
        print(f"  Text: \"{result['text'][:150]}...\"")

        all_results.append({
            "domain": domain,
            "label": label,
            "wiki_dist": wiki_dist,
            "opt_loss": result["loss"],
            "init_loss": result["init_loss"],
            "text": result["text"][:500],
            "full_text": result["text"],
            "loss_history": result["loss_history"],
        })

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f"Discrete Token Optimization: {label}", fontsize=13, fontweight='bold')

        axes[0].plot(target_ts.numpy(), color="green", linewidth=1.5, label="Target TS")
        axes[0].set_ylabel("Target"); axes[0].set_ylim(-4, 4); axes[0].legend()

        # WikiText best
        if wiki_match:
            # Recompute wiki prediction
            wiki_ids = torch.tensor(wiki_seqs[wiki_match["seq_idx"]]["input_ids"],
                                    dtype=torch.long, device=DEVICE).unsqueeze(0)
            _, wiki_pred = forward_and_loss(transformer, mapper, wiki_ids, target_ts.to(DEVICE), captured)
            axes[1].plot(target_ts.numpy(), color="green", linewidth=1.2, alpha=0.5, label="Target")
            axes[1].plot(wiki_pred.squeeze(0).cpu().numpy(), color="blue", linewidth=1.2, alpha=0.8,
                         label=f"WikiText best (d={wiki_dist:.4f})")
            axes[1].set_ylabel("WikiText"); axes[1].set_ylim(-4, 4); axes[1].legend(fontsize=9)

        axes[2].plot(target_ts.numpy(), color="green", linewidth=1.2, alpha=0.5, label="Target")
        axes[2].plot(result["pred"].numpy(), color="red", linewidth=1.2, alpha=0.8,
                     label=f"Optimized tokens (d={result['loss']:.4f})")
        axes[2].set_ylabel("Optimized"); axes[2].set_ylim(-4, 4); axes[2].legend(fontsize=9)
        text_short = result["text"][:120].replace('\n', ' ')
        axes[2].set_title(f'"{text_short}..."', fontsize=8, style='italic', color='gray')
        axes[2].set_xlabel("Timestep")

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/{domain}_discrete.png", dpi=150, bbox_inches="tight")
        plt.close()

        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("DISCRETE OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Domain':<25} {'Wiki 100K':>10} {'Optimized':>10} {'Improv':>8}")
    print("-" * 55)
    for r in all_results:
        imp = (r['wiki_dist'] - r['opt_loss']) / r['wiki_dist'] * 100 if r['wiki_dist'] > 0 else 0
        print(f"{r['label']:<25} {r['wiki_dist']:>10.4f} {r['opt_loss']:>10.4f} {imp:>7.1f}%")

    with open(f"{OUT_DIR}/discrete_results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "loss_history"} for r in all_results], f, indent=2)

    for h in handles:
        h.remove()
    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
