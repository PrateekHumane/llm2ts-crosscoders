"""
Optimize continuous token embeddings to find text that decodes into
a target time series via the trained linear mapper.

Approach:
  - Learnable soft embeddings (1, T, d_embed), frozen model + mapper
  - Minimize MSE(decoded_output, target_TS)
  - Project to nearest discrete tokens after optimization
  - Compare: random init vs warm-start from best WikiText match

Usage:
    /usr/bin/python3 scripts/optimize_text_for_ts.py
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

# ── Config ──
T = 512
D_CONCAT = 28 * 1024
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/optimized_text"

# Optimization params
N_STEPS = 2000
LR = 0.05
LOG_EVERY = 200

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


class DifferentiableDecoder(nn.Module):
    """Frozen PT model + mapper, differentiable w.r.t. input embeddings.
    Uses the model's own forward with inputs_embeds and hooks to capture layers."""
    def __init__(self, model, mapper):
        super().__init__()
        self.model = model  # Qwen3Model (the .model part of the CausalLM)
        self.mapper = mapper  # frozen linear mapper
        self.captured = {}
        self.handles = []

        # Register hooks on all 28 layers
        for li in range(28):
            def make_hook(idx):
                def hook(m, inp, out):
                    self.captured[idx] = out[0] if isinstance(out, tuple) else out
                return hook
            self.handles.append(model.layers[li].register_forward_hook(make_hook(li)))

    def forward(self, soft_embeds):
        """
        soft_embeds: (1, T, d_model) — learnable continuous embeddings
        Returns: (1, T) predicted time series
        """
        self.captured.clear()

        # Use the model's own forward — handles rotary embeddings, causal mask, etc.
        self.model(inputs_embeds=soft_embeds, use_cache=False)

        # Concatenate all 28 layers
        all_layers = [self.captured[i] for i in range(28)]
        concat = torch.cat(all_layers, dim=-1)  # (1, T, D_CONCAT)

        # Project through mapper
        y = self.mapper(concat).squeeze(-1)  # (1, T)

        # Normalize
        y_std = y.std(dim=-1, keepdim=True).clamp(min=1e-4)
        y = (y - y.mean(dim=-1, keepdim=True)) / y_std

        return y

    def cleanup(self):
        for h in self.handles:
            h.remove()


def optimize_for_target(decoder, embed_layer, target_ts, tokenizer,
                        init_ids=None, n_steps=N_STEPS, lr=LR):
    """
    Optimize soft embeddings to match target_ts.
    Returns: (best_prediction, best_loss, optimized_token_ids, decoded_text, loss_history)
    """
    target = target_ts.unsqueeze(0).to(DEVICE)  # (1, T)
    d_model = embed_layer.weight.shape[1]
    vocab_size = embed_layer.weight.shape[0]

    # Initialize soft embeddings
    if init_ids is not None:
        # Warm start from existing tokens
        with torch.no_grad():
            init_embeds = embed_layer(init_ids.to(DEVICE)).clone()
        soft_embeds = nn.Parameter(init_embeds)
    else:
        # Random init: sample from embedding distribution
        with torch.no_grad():
            mean = embed_layer.weight.mean(0)
            std = embed_layer.weight.std(0)
        soft_embeds = nn.Parameter(
            mean.unsqueeze(0).unsqueeze(0).expand(1, T, -1) +
            std.unsqueeze(0).unsqueeze(0).expand(1, T, -1) * torch.randn(1, T, d_model, device=DEVICE) * 0.1
        )

    optimizer = torch.optim.Adam([soft_embeds], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps, eta_min=lr * 0.01)

    best_loss = float('inf')
    best_embeds = None
    loss_history = []

    for step in range(n_steps):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred = decoder(soft_embeds)
            loss = F.mse_loss(pred.float(), target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_embeds = soft_embeds.data.clone()

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"      step {step}/{n_steps}: loss={loss_val:.6f} (best={best_loss:.6f})", flush=True)

    # Project to nearest discrete tokens (on CPU to avoid OOM with large vocab)
    with torch.no_grad():
        emb_cpu = best_embeds.squeeze(0).cpu().float()  # (T, d)
        vocab_cpu = embed_layer.weight.cpu().float()  # (V, d)
        # Normalize for cosine similarity
        emb_norm = F.normalize(emb_cpu, dim=-1)  # (T, d)
        vocab_norm = F.normalize(vocab_cpu, dim=-1)  # (V, d)
        # Compute in chunks over T to avoid large intermediate
        token_ids_list = []
        for t in range(emb_norm.shape[0]):
            sims = emb_norm[t] @ vocab_norm.T  # (V,)
            token_ids_list.append(sims.argmax().item())
        token_ids = torch.tensor(token_ids_list, dtype=torch.long, device=DEVICE)

        # Get prediction from discrete tokens
        discrete_embeds = embed_layer(token_ids.unsqueeze(0))
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            discrete_pred = decoder(discrete_embeds)
        discrete_loss = F.mse_loss(discrete_pred.float(), target).item()

    decoded_text = tokenizer.decode(token_ids.cpu(), skip_special_tokens=True)

    # Also get the continuous prediction
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            cont_pred = decoder(best_embeds)

    return {
        "cont_pred": cont_pred.float().squeeze(0).cpu(),
        "cont_loss": best_loss,
        "discrete_pred": discrete_pred.float().squeeze(0).cpu(),
        "discrete_loss": discrete_loss,
        "token_ids": token_ids.cpu(),
        "text": decoded_text,
        "loss_history": loss_history,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    # Load targets
    print("Loading domain targets...", flush=True)
    domain_windows = load_domain_windows(hf_token, list(DOMAIN_LABELS.keys()))
    print(f"  Got {len(domain_windows)} domains\n")

    # Load model in bfloat16 with gradient checkpointing to save memory
    print("Loading PT model...", flush=True)
    full_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
    ).to(DEVICE).eval()
    transformer = full_model.model
    transformer.gradient_checkpointing_enable()

    # Freeze everything
    for p in full_model.parameters():
        p.requires_grad = False

    # Load mapper
    mapper = nn.Linear(D_CONCAT, 1).to(DEVICE)
    mapper.load_state_dict(torch.load(
        "mapping_results/ablation/mapper_text_PT.pt", map_location="cpu", weights_only=True))
    mapper.eval()
    for p in mapper.parameters():
        p.requires_grad = False

    decoder = DifferentiableDecoder(transformer, mapper)
    embed_layer = transformer.embed_tokens

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", token=hf_token)

    # Load 100K results for comparison
    wiki_comparison = {}
    wiki_results_path = "mapping_results/reverse_mapping_100k/results.json"
    if os.path.exists(wiki_results_path):
        with open(wiki_results_path) as f:
            for r in json.load(f):
                wiki_comparison[r["domain"]] = r["matches"][0]["distance"]

    # Optimize for each domain
    all_results = []
    print(f"\n{'='*60}")
    print("OPTIMIZING TEXT EMBEDDINGS FOR EACH DOMAIN")
    print(f"{'='*60}\n")

    for domain, target_ts in domain_windows:
        label = DOMAIN_LABELS.get(domain, domain)
        print(f"\n--- {label} ---")

        # Random initialization
        print(f"  [Random init]", flush=True)
        result_rand = optimize_for_target(
            decoder, embed_layer, target_ts, tokenizer,
            init_ids=None, n_steps=N_STEPS, lr=LR)

        print(f"  Random: cont={result_rand['cont_loss']:.4f} discrete={result_rand['discrete_loss']:.4f}")
        print(f"  Text: \"{result_rand['text'][:150]}...\"")

        wiki_dist = wiki_comparison.get(domain, float('nan'))
        all_results.append({
            "domain": domain,
            "label": label,
            "wiki_best": wiki_dist,
            "opt_cont_loss": result_rand["cont_loss"],
            "opt_discrete_loss": result_rand["discrete_loss"],
            "opt_text": result_rand["text"][:500],
            "loss_history": result_rand["loss_history"],
        })

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f"Optimized Text for {label}", fontsize=13, fontweight='bold')

        # Target
        axes[0].plot(target_ts.numpy(), color="green", linewidth=1.5, label="Target TS")
        axes[0].set_ylabel("Target"); axes[0].set_ylim(-4, 4); axes[0].legend()

        # Continuous optimized
        axes[1].plot(target_ts.numpy(), color="green", linewidth=1.2, alpha=0.5, label="Target TS")
        axes[1].plot(result_rand["cont_pred"].numpy(), color="red", linewidth=1.2, alpha=0.8,
                     label=f"Continuous opt (d={result_rand['cont_loss']:.4f})")
        axes[1].set_ylabel("Continuous"); axes[1].set_ylim(-4, 4); axes[1].legend(fontsize=9)

        # Discrete (projected to tokens)
        axes[2].plot(target_ts.numpy(), color="green", linewidth=1.2, alpha=0.5, label="Target TS")
        axes[2].plot(result_rand["discrete_pred"].numpy(), color="blue", linewidth=1.2, alpha=0.8,
                     label=f"Discrete tokens (d={result_rand['discrete_loss']:.4f})")
        axes[2].set_ylabel("Discrete"); axes[2].set_ylim(-4, 4); axes[2].legend(fontsize=9)
        text_short = result_rand["text"][:120].replace('\n', ' ')
        axes[2].set_title(f'"{text_short}..."', fontsize=8, style='italic', color='gray')
        axes[2].set_xlabel("Timestep")

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/{domain}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Loss curve
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result_rand["loss_history"], color="red", linewidth=1)
        ax.axhline(wiki_dist, color="blue", linestyle="--", label=f"WikiText best ({wiki_dist:.4f})")
        ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss"); ax.set_title(f"Optimization: {label}")
        ax.legend(); ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/{domain}_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Domain':<25} {'Wiki 100K':>10} {'Opt Cont':>10} {'Opt Disc':>10} {'Improv':>8}")
    print("-" * 65)
    for r in all_results:
        imp = (r['wiki_best'] - r['opt_cont_loss']) / r['wiki_best'] * 100 if r['wiki_best'] > 0 else 0
        print(f"{r['label']:<25} {r['wiki_best']:>10.4f} {r['opt_cont_loss']:>10.4f} "
              f"{r['opt_discrete_loss']:>10.4f} {imp:>7.1f}%")

    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "loss_history"} for r in all_results], f, indent=2)

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
