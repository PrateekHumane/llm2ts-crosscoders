"""Plot held-out predictions — run after held-out eval completes."""
import torch, torch.nn as nn, numpy as np, os, sys, gc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.data.wikitext import load_wikitext_sequences
from scripts.mapping_experiment import load_ts_windows, compute_acf
from sklearn.cluster import KMeans

device = torch.device("cuda:0")
cfg = Config()
d_concat = 28 * 1024; T = 512

# Load held-out wiki seqs
wiki_seqs = load_wikitext_sequences(max_sequences=5000, seq_len=512, hf_token=os.environ.get("HF_TOKEN"))
held_out = wiki_seqs[2000:2500]  # just 500 for plotting speed

ts = load_ts_windows(cfg, 10000, os.environ.get("HF_TOKEN"))

# Extract
print("Extracting held-out (500 seqs)...", flush=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).model.to(device).eval()
captured = {}
handles = []
for li in range(28):
    def make_hook(idx):
        def hook(m, i, o):
            captured[idx] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    handles.append(model.layers[li].register_forward_hook(make_hook(li)))
all_hs = []
with torch.no_grad():
    for start in range(0, len(held_out), 8):
        batch = held_out[start:start+8]
        ids = torch.tensor(np.stack([s["input_ids"] for s in batch]), dtype=torch.long, device=device)
        model(input_ids=ids, use_cache=False)
        layers = [captured[i].float().cpu()[:, :T, :] for i in range(28)]
        all_hs.append(torch.cat(layers, dim=-1))
        captured.clear()
for h in handles: h.remove()
del model; gc.collect(); torch.cuda.empty_cache()
h_held = torch.cat(all_hs, dim=0)

# Predict
mapper = nn.Linear(d_concat, 1)
mapper.load_state_dict(torch.load("mapping_results/concat_layers/mapper_pt_concat.pt", map_location="cpu", weights_only=True))
mapper = mapper.to(device).eval()
all_pred = []
with torch.no_grad():
    for start in range(0, h_held.shape[0], 32):
        h_b = h_held[start:start+32].to(device)
        yr = mapper(h_b).squeeze(-1)
        ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
        all_pred.append(((yr - yr.mean(-1, keepdim=True)) / ys).cpu())
pred = torch.cat(all_pred, dim=0)
del h_held; gc.collect(); torch.cuda.empty_cache()

# NN
nn_indices = []; nn_dists = []
for i in range(pred.shape[0]):
    d2 = ((pred[i:i+1] - ts)**2).mean(-1).squeeze(0)
    nn_indices.append(d2.argmin().item()); nn_dists.append(d2.min().item())
nn_dists = np.array(nn_dists); nn_indices = np.array(nn_indices)
print(f"Held-out: Unique={len(set(nn_indices))}, NN mean={nn_dists.mean():.4f}")

out = "mapping_results/concat_layers/plots_held_out"
os.makedirs(out, exist_ok=True)

# 1. Overlay
fig, ax = plt.subplots(figsize=(14, 6))
for i in np.random.choice(pred.shape[0], min(100, pred.shape[0]), replace=False):
    ax.plot(pred[i].numpy(), alpha=0.1, linewidth=0.5, color="blue")
ax.set_title(f"HELD-OUT WikiText — 100 predictions ({len(set(nn_indices))} unique)", fontsize=13)
ax.set_ylim(-4, 4); ax.set_xlabel("Timestep")
plt.savefig(f"{out}/overlay.png", dpi=150, bbox_inches="tight"); plt.close()

# 2. Best 6 overlaid with matches
best_idx = np.argsort(nn_dists)
fig, axes = plt.subplots(6, 1, figsize=(14, 18))
fig.suptitle("HELD-OUT: Best 6 predictions vs nearest real TS", fontsize=13, fontweight='bold')
for row, idx in enumerate(best_idx[:6]):
    ti = nn_indices[idx]
    axes[row].plot(ts[ti].numpy(), color="green", linewidth=1.2, alpha=0.6, label="Real TS")
    axes[row].plot(pred[idx].numpy(), color="blue", linewidth=1.2, alpha=0.7, label="Predicted (held-out)")
    axes[row].set_ylim(-4, 4); axes[row].set_ylabel(f"d={nn_dists[idx]:.3f}", fontsize=9)
    if row == 0: axes[row].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{out}/best_6.png", dpi=150, bbox_inches="tight"); plt.close()

# 3. Random 6
fig, axes = plt.subplots(6, 1, figsize=(14, 18))
fig.suptitle("HELD-OUT: 6 random predictions vs nearest real TS", fontsize=13, fontweight='bold')
for row, idx in enumerate(np.random.choice(pred.shape[0], 6, replace=False)):
    ti = nn_indices[idx]
    axes[row].plot(ts[ti].numpy(), color="green", linewidth=1.2, alpha=0.6, label="Real TS")
    axes[row].plot(pred[idx].numpy(), color="blue", linewidth=1.2, alpha=0.7, label="Predicted (held-out)")
    axes[row].set_ylim(-4, 4); axes[row].set_ylabel(f"d={nn_dists[idx]:.3f}", fontsize=9)
    if row == 0: axes[row].legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{out}/random_6.png", dpi=150, bbox_inches="tight"); plt.close()

# 4. Quality grid: best/mid/worst
fig, axes = plt.subplots(12, 2, figsize=(16, 36))
fig.suptitle("HELD-OUT: Best / Mid / Worst predictions", fontsize=14, fontweight='bold')
axes[0,0].set_title("Predicted (held-out text)", fontsize=11)
axes[0,1].set_title("Nearest Real TS", fontsize=11)
pick = np.concatenate([best_idx[:4], best_idx[len(best_idx)//3:len(best_idx)//3+4], best_idx[-4:]])
for row, idx in enumerate(pick):
    ti = nn_indices[idx]
    axes[row,0].plot(pred[idx].numpy(), color="blue", linewidth=1); axes[row,0].set_ylim(-4, 4)
    label = "BEST" if row < 4 else ("MID" if row < 8 else "WORST")
    axes[row,0].set_ylabel(f"{label}\nd={nn_dists[idx]:.3f}", fontsize=8)
    axes[row,1].plot(ts[ti].numpy(), color="green", linewidth=1); axes[row,1].set_ylim(-4, 4)
plt.tight_layout(); plt.savefig(f"{out}/quality_grid.png", dpi=150, bbox_inches="tight"); plt.close()

# 5. Side-by-side: training vs held-out overlay
mm = np.memmap("mapping_results/concat_layers/pt_concat_hs.bin", dtype='float16', mode='r', shape=(2000, T, d_concat))
train_pred = []
with torch.no_grad():
    for start in range(0, 500, 32):
        h_b = torch.from_numpy(mm[start:start+32].copy()).float().to(device)
        yr = mapper(h_b).squeeze(-1)
        ys = yr.std(-1, keepdim=True).clamp(min=1e-4)
        train_pred.append(((yr - yr.mean(-1, keepdim=True)) / ys).cpu())
train_pred = torch.cat(train_pred, dim=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
for i in np.random.choice(train_pred.shape[0], 50, replace=False):
    axes[0].plot(train_pred[i].numpy(), alpha=0.1, linewidth=0.5, color="blue")
axes[0].set_title("Training WikiText (50 preds)", fontsize=12); axes[0].set_ylim(-4, 4)
for i in np.random.choice(pred.shape[0], 50, replace=False):
    axes[1].plot(pred[i].numpy(), alpha=0.1, linewidth=0.5, color="red")
axes[1].set_title("Held-Out WikiText (50 preds)", fontsize=12); axes[1].set_ylim(-4, 4)
plt.suptitle("Training vs Held-Out prediction diversity", fontsize=13)
plt.tight_layout(); plt.savefig(f"{out}/train_vs_held.png", dpi=150, bbox_inches="tight"); plt.close()

print(f"All plots saved to {out}/")
