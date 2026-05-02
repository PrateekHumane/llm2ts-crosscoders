"""
Sharpness measurement for a single model.
Run with: CUDA_VISIBLE_DEVICES=X python scripts/loss_landscape_sharpness.py --model FT|RI|PT
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, glob, argparse

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = 512; SEED = 42
OUT_DIR = "mapping_results/loss_landscape"
MODEL_PATHS = {"PT": "Qwen/Qwen3-0.6B", "FT": "models/ft", "RI": "models/ri"}


def load_ts_data(hf_token, n_windows=200):
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    windows = []
    for f in sorted(glob.glob(os.path.join(path, "**/*.arrow"), recursive=True)):
        with open(f, "rb") as fp:
            tbl = ipc.open_stream(fp).read_all()
        for t in tbl['target'].to_pylist():
            arr = np.array(t, dtype=np.float32)
            if len(arr) >= T:
                start = int(len(arr)*0.7)
                if start+T > len(arr): start = len(arr)-T
                w = arr[start:start+T]; mu, sigma = w.mean(), w.std()
                if sigma < 1e-6: continue
                w = (w-mu)/sigma
                bins = ((np.clip(w,-5,5)+5)/10*512).astype(np.int64).clip(0,511)
                windows.append(bins)
                if len(windows) >= n_windows: return windows
    return windows


def compute_ts_loss(model, token_ids_list, device, batch_size=5):
    model.eval()
    total = 0; n = 0
    for i in range(0, len(token_ids_list), batch_size):
        batch = token_ids_list[i:i+batch_size]
        ids = torch.tensor(np.stack(batch), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               ids[:, 1:].contiguous().view(-1))
        total += loss.item(); n += 1
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["FT", "RI", "PT"])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    hf_token = os.environ.get("HF_TOKEN")
    os.makedirs(OUT_DIR, exist_ok=True)

    ts_windows = load_ts_data(hf_token, 200)
    eval_windows = ts_windows[:50]

    print(f"{args.model}: Loading...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[args.model], dtype=torch.float32).to(device).eval()

    base_loss = compute_ts_loss(model, eval_windows, device)
    total_norm = sum((p.data**2).sum().item() for p in model.parameters())
    n_params = sum(p.numel() for p in model.parameters())
    rms_weight = np.sqrt(total_norm / n_params)
    print(f"  Base loss: {base_loss:.4f}, RMS weight: {rms_weight:.4f}")

    sigmas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = {"base_loss": base_loss, "rms_weight": rms_weight, "sigmas": {}}

    for sigma in sigmas:
        abs_sigma = sigma * rms_weight
        losses = []
        for sample in range(5):
            orig = {n: p.data.clone() for n, p in model.named_parameters()}
            torch.manual_seed(SEED + sample*1000 + int(sigma*10000))
            for p in model.parameters():
                p.data.add_(abs_sigma * torch.randn_like(p))
            losses.append(compute_ts_loss(model, eval_windows, device))
            for n, p in model.named_parameters():
                p.data.copy_(orig[n])

        results["sigmas"][sigma] = {
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "delta": float(np.mean(losses)) - base_loss,
            "relative_delta": (float(np.mean(losses)) - base_loss) / base_loss,
        }
        print(f"  σ={sigma:.4f}: loss={np.mean(losses):.4f} ({(np.mean(losses)-base_loss)/base_loss*100:+.1f}%)")

    with open(f"{OUT_DIR}/sharpness_{args.model}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/sharpness_{args.model}.json")


if __name__ == "__main__":
    main()
