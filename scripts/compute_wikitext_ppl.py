"""
Compute WikiText-103 perplexity for FT (LangInit) and RI (RandInit) checkpoints.

Uses the same tokenizer (Qwen/Qwen3-0.6B) and dataset (WikiText-103) as the
erank computation in reproduction/effective_rank/run_checkpoints.py.

Saves results to results/wikitext_perplexity.json for use by plot_paper.py.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.wikitext import load_wikitext_sequences

CHECKPOINTS = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
}
BASE_MODEL = "Qwen/Qwen3-0.6B"
STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SEQ_LEN = 512
N_SEQUENCES = 200
OUT_PATH = "results/wikitext_perplexity.json"


def compute_perplexity(model, input_ids_list, device):
    """Compute perplexity over a list of (seq_len,) token ID tensors."""
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for ids in input_ids_list:
            ids = ids.unsqueeze(0).to(device)
            outputs = model(ids, labels=ids)
            seq_len = ids.size(1) - 1
            total_nll += outputs.loss.float().item() * seq_len
            total_tokens += seq_len

    return math.exp(total_nll / total_tokens)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")
    sequences = load_wikitext_sequences(
        max_sequences=N_SEQUENCES, seq_len=SEQ_LEN, hf_token=hf_token
    )
    input_ids_list = [torch.from_numpy(s["input_ids"]) for s in sequences]
    print(f"Loaded {len(input_ids_list)} sequences of {SEQ_LEN} tokens")

    device = torch.device("cuda:0")
    results = {}

    print(f"Computing PT baseline ({BASE_MODEL})...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    ppl = compute_perplexity(model, input_ids_list, device)
    results["PT"] = {"perplexity": ppl}
    print(f"  PT: {ppl:.2f}")
    del model
    torch.cuda.empty_cache()

    for model_name, base_path in CHECKPOINTS.items():
        results[model_name] = {}
        for step in STEPS:
            ckpt_path = os.path.join(base_path, f"checkpoint-{step}")
            print(f"Computing {model_name} step {step}...", end=" ", flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path, torch_dtype=torch.bfloat16
            ).to(device)
            model.eval()
            ppl = compute_perplexity(model, input_ids_list, device)
            results[model_name][step] = {"perplexity": ppl}
            print(f"ppl={ppl:.2f}")
            del model
            torch.cuda.empty_cache()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
