"""
Hessian eigenvalue computation for a single initialization.
Run with: CUDA_VISIBLE_DEVICES=X python scripts/loss_landscape_hessian.py --init PT|RandomInit

Uses exact autograd HVPs + Lanczos algorithm.
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
    total_loss = 0; n = 0
    for i in range(0, len(token_ids_list), batch_size):
        batch = token_ids_list[i:i+batch_size]
        ids = torch.tensor(np.stack(batch), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               ids[:, 1:].contiguous().view(-1))
        total_loss += loss.item(); n += 1
    return total_loss / n


def exact_hvp(model, batch_ids, vector_list, params):
    """Exact HVP via Pearlmutter trick, only for specified params."""
    logits = model(input_ids=batch_ids).logits.float()
    loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                           batch_ids[:, 1:].contiguous().view(-1))
    grads = torch.autograd.grad(loss, params, create_graph=True)
    gv = sum((g * v).sum() for g, v in zip(grads, vector_list) if g is not None)
    hvp = torch.autograd.grad(gv, params, retain_graph=False)
    return [h.detach() for h in hvp]


def batched_hvp(model, data_batches, vector_list, params):
    n = len(data_batches)
    hvp_sum = None
    for batch_ids in data_batches:
        hvp = exact_hvp(model, batch_ids, vector_list, params)
        if hvp_sum is None:
            hvp_sum = [h.clone() for h in hvp]
        else:
            for i in range(len(hvp_sum)):
                hvp_sum[i] += hvp[i]
        # Free computation graph memory
        torch.cuda.empty_cache()
    return [h / n for h in hvp_sum]


def lanczos(model, data_batches, n_eigenvalues=10, n_iterations=50, layer_indices=None):
    """Lanczos on Hessian restricted to specific layers' parameters.
    If layer_indices is None, uses layers [6, 7, 8, 9, 10] (mid layers)."""
    # Freeze all params, then unfreeze target layers only
    for p in model.parameters():
        p.requires_grad = False

    if layer_indices is None:
        layer_indices = [6, 7, 8, 9, 10]

    target_params = []
    for li in layer_indices:
        for p in model.model.layers[li].parameters():
            p.requires_grad = True
            target_params.append(p)

    n_params = sum(p.numel() for p in target_params)
    print(f"    Hessian over {len(target_params)} param tensors "
          f"({n_params/1e6:.1f}M params) from layers {layer_indices}", flush=True)

    model.train()

    torch.manual_seed(SEED)
    q = [torch.randn_like(p) for p in target_params]
    norm = torch.sqrt(sum((qi**2).sum() for qi in q))
    q = [qi / norm for qi in q]

    Q_vectors = [q]
    alphas = []; betas = [0.0]

    print(f"    Lanczos: {n_iterations} iterations...", flush=True)
    t0 = time.time()

    for j in range(n_iterations):
        w = batched_hvp(model, data_batches, q, target_params)
        alpha = sum((qi * wi).sum().item() for qi, wi in zip(q, w))
        alphas.append(alpha)

        for i in range(len(w)):
            w[i] = w[i] - alpha * q[i]
            if j > 0:
                w[i] = w[i] - betas[j] * Q_vectors[j-1][i]

        # Full reorthogonalization
        for k in range(j + 1):
            dot = sum((wi * Q_vectors[k][i]).sum().item() for i, wi in enumerate(w))
            for i in range(len(w)):
                w[i] = w[i] - dot * Q_vectors[k][i]

        beta = torch.sqrt(sum((wi**2).sum() for wi in w)).item()
        betas.append(beta)
        if beta < 1e-10:
            print(f"    Converged at iter {j+1}", flush=True); break

        q = [wi / beta for wi in w]
        Q_vectors.append(q)

        if (j+1) % 5 == 0:
            T_mat = np.diag(alphas) + np.diag(betas[1:len(alphas)], 1) + np.diag(betas[1:len(alphas)], -1)
            eigs = np.sort(np.linalg.eigvalsh(T_mat))[::-1]
            top = eigs[:min(5, len(eigs))]
            print(f"      iter {j+1}: top eigs = {[f'{e:.2f}' for e in top]} ({time.time()-t0:.0f}s)", flush=True)

        for k in range(len(Q_vectors)-1):
            Q_vectors[k] = [qi.detach() for qi in Q_vectors[k]]

    m = len(alphas)
    T_mat = np.zeros((m, m))
    for i in range(m): T_mat[i, i] = alphas[i]
    for i in range(m-1): T_mat[i, i+1] = betas[i+1]; T_mat[i+1, i] = betas[i+1]
    eigs = np.sort(np.linalg.eigvalsh(T_mat))[::-1]
    print(f"    Done ({time.time()-t0:.0f}s): top-{n_eigenvalues} = {[f'{e:.2f}' for e in eigs[:n_eigenvalues]]}")
    return eigs[:n_eigenvalues].tolist(), eigs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True, choices=["PT", "RandomInit"])
    args = parser.parse_args()

    device = torch.device("cuda:0")
    hf_token = os.environ.get("HF_TOKEN")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading TS data...", flush=True)
    ts_windows = load_ts_data(hf_token, 200)
    hessian_windows = ts_windows[:100]
    # Batch size 1 — exact HVP with create_graph=True is very memory intensive
    # Use 10 single-sequence batches for averaging
    HESSIAN_BS = 1
    batches = [torch.tensor(hessian_windows[i:i+HESSIAN_BS], dtype=torch.long, device=device)
               for i in range(10)]

    print(f"\n{args.init}: Loading model...", flush=True)
    if args.init == "PT":
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to(device)
        model.gradient_checkpointing_enable()
    else:
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32).to(device)
        model.gradient_checkpointing_enable()

    base_loss = compute_ts_loss(model, hessian_windows[:50], device)
    print(f"  Base loss: {base_loss:.4f}")

    model.train()
    top_eigs, all_eigs = lanczos(model, batches, n_eigenvalues=10, n_iterations=30)

    result = {"base_loss": base_loss, "top_eigenvalues": top_eigs, "all_eigenvalues": all_eigs}
    with open(f"{OUT_DIR}/hessian_{args.init}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/hessian_{args.init}.json")


if __name__ == "__main__":
    main()
