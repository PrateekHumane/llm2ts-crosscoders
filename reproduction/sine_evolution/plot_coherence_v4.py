"""
Refined B2a: side-by-side LangInit/RandInit t-SNE pairs above the coherence
curve at steps 1, 512, 8192. Arrows point to the specific phase-coherence
data points (enlarged). Titles above boxes. Legend at bottom, no box.

Usage:
    python3 reproduction/sine_evolution/plot_coherence_v4.py
"""
import json
import threading

import numpy as np
import torch
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/sine_evolution"

# ── Load data ────────────────────────────────────────────────────────────────

with open(f"{OUTPUT_DIR}/phase_coherence_checkpoints.json") as f:
    pc_data = json.load(f)
pc_checkpoints = pc_data["checkpoints"]
pt_pc = pc_data["PT"]

pc_families = {}
for key, val in pc_checkpoints.items():
    m = val["model"]
    if m not in pc_families:
        pc_families[m] = []
    pc_families[m].append(val)
for m in pc_families:
    pc_families[m].sort(key=lambda x: x["step"])

grad_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
grad_ri = [0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0010, 0.0042, 0.0020, -0.0011, 0.2279, 0.2533, 0.2014, 0.2263, 0.2283]
grad_ft = [0.4598, 0.4594, 0.4551, 0.4346, 0.2410, 0.2115, 0.6195, 0.5878, 0.1222, 0.1102, 0.1651, 0.1593, 0.1170, 0.0611, 0.0593]

# ── Extract t-SNE ────────────────────────────────────────────────────────────

T = 512; PERIOD = 64; SKIP = 5; LAYER = 13
N_BINS = 1024; BIN_LOW = -5.0; BIN_HIGH = 5.0

t_arr = np.arange(T, dtype=np.float32)
sine = np.sin(2 * np.pi * t_arr / PERIOD)
mean, std = float(np.mean(sine)), float(np.std(sine))
normed = (sine - mean) / std
clipped = np.clip(normed, BIN_LOW, BIN_HIGH)
bins = ((clipped - BIN_LOW) / (BIN_HIGH - BIN_LOW) * N_BINS).astype(np.int64)
bins = np.clip(bins, 0, N_BINS - 1)
input_ids = torch.from_numpy(bins).unsqueeze(0)
phase = ((t_arr % PERIOD) / PERIOD)[SKIP:]

SNAPSHOT_STEPS = [1, 512, 8192]
CKPT_BASE = {
    "FT": "/workspace/NanoTS_v2/checkpoints/0.6B_pretrained_seed420",
    "RI": "/workspace/NanoTS_v2/checkpoints/random_quantile_loss",
}

all_jobs = []
for model in ["FT", "RI"]:
    for step in SNAPSHOT_STEPS:
        all_jobs.append((f"{model}_s{step}", f"{CKPT_BASE[model]}/checkpoint-{step}"))

@torch.no_grad()
def extract_tsne(name, model_path, device):
    full_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device)
    model = full_model.model; model.eval()
    captured = {}
    def hook(module, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured["hs"] = o.detach().float().cpu()
    handle = model.layers[LAYER].register_forward_hook(hook)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=input_ids.to(device), use_cache=False)
    handle.remove()
    hs = captured["hs"].squeeze(0).numpy()[SKIP:]
    del full_model, model; torch.cuda.empty_cache()
    return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(hs)

print("Extracting t-SNE snapshots...", flush=True)
tsne_projs = {}
lock = threading.Lock()
gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
for round_idx in range(0, len(all_jobs), 4):
    batch = all_jobs[round_idx:round_idx + 4]
    threads = []
    for i, (name, path) in enumerate(batch):
        def worker(n, p, d):
            proj = extract_tsne(n, p, d)
            with lock: tsne_projs[n] = proj
        t = threading.Thread(target=worker, args=(name, path, gpus[i]))
        threads.append(t); t.start()
    for t in threads: t.join()
    print(f"  Round {round_idx // 4 + 1} done", flush=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 6.5, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
})

C_FT = "#D55E00"; C_RI = "#0072B2"
input_name = "Sine p=64"
ft_steps_pc = [e["step"] for e in pc_families["FT"]]
ft_pc = [1.0 - e["coherence"][input_name] for e in pc_families["FT"]]
ri_steps_pc = [e["step"] for e in pc_families["RI"]]
ri_pc = [1.0 - e["coherence"][input_name] for e in pc_families["RI"]]


def draw_tsne_ax(ax, proj, phase, border_color=None, label=None):
    p = proj.copy()
    mx = np.abs(p).max()
    if mx > 0: p = p / mx
    for j in range(len(p) - 1):
        c = plt.cm.hsv(phase[j])
        ax.plot(p[j:j+2, 0], p[j:j+2, 1], color=c,
                linewidth=0.6, alpha=0.85, solid_capstyle="round")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#f8fafc")
    if border_color:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.5)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)
    if label:
        ax.text(0.5, 0.92, label, transform=ax.transAxes, fontsize=4.5,
                ha="center", va="top", color=border_color, fontweight="bold")


def get_pc_at_step(model, step):
    for e in pc_families[model]:
        if e["step"] == step:
            return 1.0 - e["coherence"][input_name]
    return None


# ── Plot ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(6.0, 4.2))

ax1 = fig.add_axes([0.10, 0.14, 0.78, 0.40])

l1, = ax1.plot(ft_steps_pc, ft_pc, "-o", color=C_FT, markersize=3,
               linewidth=1.5, label="LangInit — phase coh.", zorder=5)
l2, = ax1.plot(ri_steps_pc, ri_pc, "-o", color=C_RI, markersize=3,
               linewidth=1.5, label="RandInit — phase coh.", zorder=5)
pt_val = 1.0 - pt_pc[input_name]
ax1.axhline(pt_val, color="#999", linestyle=":", linewidth=0.8, alpha=0.7)
ax1.text(1.3, pt_val + 0.02, "PT baseline", fontsize=5.5, color="#999")
ax1.set_xscale("log")
ax1.set_xlabel("Training Step")
ax1.set_ylabel("Phase Coherence\n(higher = more periodic)", fontsize=8)
ax1.set_ylim(-0.05, 0.95)
ax1.set_xlim(0.7, 15000)

ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)
l3, = ax2.plot(grad_steps, grad_ft, "--s", color=C_FT, markersize=2, linewidth=1,
               alpha=0.35, label="LangInit — grad. coh.")
l4, = ax2.plot(grad_steps, grad_ri, "--s", color=C_RI, markersize=2, linewidth=1,
               alpha=0.35, label="RandInit — grad. coh.")
ax2.set_ylabel("Gradient Coherence", fontsize=8)
ax2.set_ylim(-0.05, 0.7)

# Enlarged markers at snapshot steps
for step in SNAPSHOT_STEPS:
    ft_val = get_pc_at_step("FT", step)
    ri_val = get_pc_at_step("RI", step)
    if ft_val is not None:
        ax1.plot(step, ft_val, "o", color=C_FT, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.8, zorder=6)
    if ri_val is not None:
        ax1.plot(step, ri_val, "o", color=C_RI, markersize=8,
                 markeredgecolor="white", markeredgewidth=1.8, zorder=6)

# Legend at bottom, no box, with a bit of padding above
lines = [l1, l2, l3, l4]
ax1.legend(lines, [l.get_label() for l in lines],
           loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=4, fontsize=6, frameon=False, columnspacing=1.2)

# ── Filmstrip above: side-by-side pairs with arrows to data points ──────────

pair_w = 0.10
pair_h = 0.17
pair_gap = 0.012

# Spread pairs a bit more horizontally
pair_centers_fig = [0.20, 0.50, 0.78]
# Reduce vertical gap: filmstrip closer to the plot
film_y = 0.62

for step, fx_center in zip(SNAPSHOT_STEPS, pair_centers_fig):
    fx_ft = fx_center - pair_w - pair_gap / 2
    fx_ri = fx_center + pair_gap / 2

    ax_ft = fig.add_axes([fx_ft, film_y, pair_w, pair_h])
    draw_tsne_ax(ax_ft, tsne_projs[f"FT_s{step}"], phase, C_FT, label="LangInit")

    ax_ri = fig.add_axes([fx_ri, film_y, pair_w, pair_h])
    draw_tsne_ax(ax_ri, tsne_projs[f"RI_s{step}"], phase, C_RI, label="RandInit")

    # Arrows from bottom-center of each box to its data point
    ft_val = get_pc_at_step("FT", step)
    ri_val = get_pc_at_step("RI", step)

    if ft_val is not None:
        ax1.annotate("",
                     xy=(step, ft_val), xycoords="data",
                     xytext=(fx_ft + pair_w / 2, film_y - 0.005),
                     textcoords="figure fraction",
                     arrowprops=dict(arrowstyle="-|>", color=C_FT,
                                     lw=0.9, alpha=0.4,
                                     connectionstyle="arc3,rad=0.0"))

    if ri_val is not None:
        ax1.annotate("",
                     xy=(step, ri_val), xycoords="data",
                     xytext=(fx_ri + pair_w / 2, film_y - 0.005),
                     textcoords="figure fraction",
                     arrowprops=dict(arrowstyle="-|>", color=C_RI,
                                     lw=0.9, alpha=0.4,
                                     connectionstyle="arc3,rad=0.0"))

fig.patch.set_facecolor("white")
out = f"{OUTPUT_DIR}/fig_coherence_B2a_v2.png"
fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
plt.close()
print(f"Saved {out}")
