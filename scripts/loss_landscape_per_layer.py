"""
Per-layer Hessian eigenvalues: compute Lanczos on each layer separately.
Gives a layer-by-layer curvature profile for PT vs RandomInit.

Run 4 instances on 4 GPUs:
  GPU 0: layers 0-6   (PT + RandomInit)
  GPU 1: layers 7-13  (PT + RandomInit)
  GPU 2: layers 14-20 (PT + RandomInit)
  GPU 3: layers 21-27 (PT + RandomInit)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/loss_landscape_per_layer.py --layers 0-6
    CUDA_VISIBLE_DEVICES=1 python scripts/loss_landscape_per_layer.py --layers 7-13
    CUDA_VISIBLE_DEVICES=2 python scripts/loss_landscape_per_layer.py --layers 14-20
    CUDA_VISIBLE_DEVICES=3 python scripts/loss_landscape_per_layer.py --layers 21-27
"""
import torch
import torch.nn.functional as F
import numpy as np
import json, os, sys, gc, glob, time, argparse

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = 512; SEED = 42
OUT_DIR = "mapping_results/loss_landscape"
N_LANCZOS = 30
N_RANDOM_DIRS = 30
N_DATA = 50  # sequences for HVP
DATA_BS = 2  # batch size — float32 + create_graph is very memory hungry


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


def compute_loss(model, batch_ids):
    logits = model(input_ids=batch_ids).logits.float()
    return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                           batch_ids[:, 1:].contiguous().view(-1))


def exact_hvp(model, batch_ids, vector_list, params):
    loss = compute_loss(model, batch_ids)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gv = sum((g * v).sum() for g, v in zip(grads, vector_list))
    hvp = torch.autograd.grad(gv, params, retain_graph=False)
    return [h.detach() for h in hvp]


def batched_hvp(model, data_batches, vector_list, params):
    hvp_sum = None
    for batch_ids in data_batches:
        hvp = exact_hvp(model, batch_ids, vector_list, params)
        if hvp_sum is None:
            hvp_sum = [h.clone() for h in hvp]
        else:
            for i in range(len(hvp_sum)):
                hvp_sum[i] += hvp[i]
        torch.cuda.empty_cache()
    return [h / len(data_batches) for h in hvp_sum]


def lanczos(model, data_batches, params, n_iterations=30):
    model.train()
    torch.manual_seed(SEED)
    q = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum((qi**2).sum() for qi in q))
    q = [qi / norm for qi in q]

    Q_vectors = [q]; alphas = []; betas = [0.0]

    for j in range(n_iterations):
        w = batched_hvp(model, data_batches, q, params)
        alpha = sum((qi * wi).sum().item() for qi, wi in zip(q, w))
        alphas.append(alpha)

        for i in range(len(w)):
            w[i] = w[i] - alpha * q[i]
            if j > 0:
                w[i] = w[i] - betas[j] * Q_vectors[j-1][i]

        for k in range(j + 1):
            dot = sum((wi * Q_vectors[k][i]).sum().item() for i, wi in enumerate(w))
            for i in range(len(w)):
                w[i] = w[i] - dot * Q_vectors[k][i]

        beta = torch.sqrt(sum((wi**2).sum() for wi in w)).item()
        betas.append(beta)
        if beta < 1e-10: break

        q = [wi / beta for wi in w]
        Q_vectors.append(q)

        for k in range(len(Q_vectors)-1):
            Q_vectors[k] = [qi.detach() for qi in Q_vectors[k]]

    m = len(alphas)
    T_mat = np.zeros((m, m))
    for i in range(m): T_mat[i, i] = alphas[i]
    for i in range(m-1): T_mat[i, i+1] = betas[i+1]; T_mat[i+1, i] = betas[i+1]
    return np.sort(np.linalg.eigvalsh(T_mat))[::-1].tolist()


def random_direction_curvature(model, data_batches, params, n_dirs=30):
    model.train()
    curvatures = []
    for di in range(n_dirs):
        torch.manual_seed(SEED + di * 7)
        v = [torch.randn_like(p) for p in params]
        norm = torch.sqrt(sum((vi**2).sum() for vi in v))
        v = [vi / norm for vi in v]
        hvp = batched_hvp(model, data_batches, v, params)
        curv = sum((vi * hi).sum().item() for vi, hi in zip(v, hvp))
        curvatures.append(curv)
        torch.cuda.empty_cache()
    return curvatures


def process_layer(model, layer_idx, data_batches, device):
    """Compute Hessian eigenvalues and random curvature for one layer."""
    # Freeze all, unfreeze target layer
    for p in model.parameters():
        p.requires_grad = False
    params = []
    for p in model.model.layers[layer_idx].parameters():
        p.requires_grad = True
        params.append(p)

    n_params = sum(p.numel() for p in params)

    # Lanczos
    eigs = lanczos(model, data_batches, params, n_iterations=N_LANCZOS)

    # Random direction curvature
    rand_curvs = random_direction_curvature(model, data_batches, params, n_dirs=N_RANDOM_DIRS)

    # Implied trace from random directions
    mean_curv = np.mean(rand_curvs)
    implied_trace = mean_curv * n_params

    return {
        "layer": layer_idx,
        "n_params": n_params,
        "eigenvalues": eigs[:10],
        "all_lanczos_eigs": eigs,
        "top1": eigs[0] if eigs else 0,
        "top5_sum": sum(eigs[:5]) if len(eigs) >= 5 else sum(eigs),
        "condition_ratio": eigs[0] / eigs[min(9, len(eigs)-1)] if len(eigs) > 1 and eigs[min(9,len(eigs)-1)] > 0 else float('inf'),
        "random_curv_mean": float(np.mean(rand_curvs)),
        "random_curv_std": float(np.std(rand_curvs)),
        "implied_trace": float(implied_trace),
        "random_curvatures": rand_curvs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", required=True, help="e.g. 0-6")
    args = parser.parse_args()

    start_layer, end_layer = map(int, args.layers.split("-"))
    layer_range = list(range(start_layer, end_layer + 1))

    device = torch.device("cuda:0")
    hf_token = os.environ.get("HF_TOKEN")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print(f"Loading TS data ({N_DATA} sequences)...", flush=True)
    ts_windows = load_ts_data(hf_token, N_DATA)
    print(f"  Got {len(ts_windows)} sequences")

    # Prepare batches
    data_batches = [
        torch.tensor(np.stack(ts_windows[i:i+DATA_BS]), dtype=torch.long, device=device)
        for i in range(0, min(N_DATA, len(ts_windows)), DATA_BS)
    ]
    print(f"  {len(data_batches)} batches of {DATA_BS}")

    # Process each init × layer
    for init_name in ["PT", "RandomInit"]:
        print(f"\n{'='*60}")
        print(f"MODEL: {init_name} | Layers {start_layer}-{end_layer}")
        print(f"{'='*60}")

        if init_name == "PT":
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(device)
        else:
            config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
            model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32).to(device)

        results = {}
        for li in layer_range:
            t0 = time.time()
            print(f"\n  Layer {li}:", flush=True)
            r = process_layer(model, li, data_batches, device)
            results[li] = r
            print(f"    params={r['n_params']/1e6:.1f}M  λ₁={r['top1']:.1f}  "
                  f"cond={r['condition_ratio']:.1f}  "
                  f"rand_curv={r['random_curv_mean']:.6f}  "
                  f"trace≈{r['implied_trace']:.1f}  ({time.time()-t0:.0f}s)")

        # Save
        out_file = f"{OUT_DIR}/per_layer_{init_name}_L{start_layer}-{end_layer}.json"
        with open(out_file, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
        print(f"\n  Saved to {out_file}")

        del model; gc.collect(); torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    main()
