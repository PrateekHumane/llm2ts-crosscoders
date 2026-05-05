"""
Figure: Hidden-state trajectories for synthetic inputs at Layer 8.
Shows input waveform (colored by phase) and 2D PCA trajectories through PT, FT, RI.
First 5 positions skipped to avoid attention-sink artifact.
PCA fitted independently per model-input pair.

Output: NeurIPS26/figures/sec5/fig_manifold_synthetic.png
"""
import torch
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys; sys.path.insert(0, '../..')
from src.config import Config
from src.data.tokenize import normalize_window, uniform_bin_tokenize
from transformers import AutoModelForCausalLM

cfg = Config()
device = torch.device('cuda:1')
SKIP = 5; T = 512; LAYER = 8; period = 64

# ── Synthetic inputs ──
def make_sine(): return np.sin(2 * np.pi * np.arange(T, dtype=np.float32) / period)
def make_square(): return np.sign(np.sin(2 * np.pi * np.arange(T, dtype=np.float32) / period))
def make_sawtooth(): return 2 * ((np.arange(T, dtype=np.float32) % period) / period) - 1
def make_two_freq():
    t = np.arange(T, dtype=np.float32)
    return np.sin(2 * np.pi * t / 64) + 0.5 * np.sin(2 * np.pi * t / 17)
def make_trend():
    t = np.arange(T, dtype=np.float32)
    return t / T + 0.3 * np.sin(2 * np.pi * t / 80)

inputs = [
    ('Sine', make_sine()),
    ('Square wave', make_square()),
    ('Sawtooth', make_sawtooth()),
    ('Two frequencies', make_two_freq()),
    ('Trend + oscillation', make_trend()),
]
n_inputs = len(inputs)
models_info = [('PT', cfg.model_pt), ('FT', cfg.model_ft), ('RI', cfg.model_ri)]
n_models = len(models_info)

phase = ((np.arange(SKIP, T) % period) / period)

fig, axes = plt.subplots(n_inputs, n_models + 1, figsize=(5.5, 5.5),
                          gridspec_kw={'hspace': 0.15, 'wspace': 0.08,
                                       'width_ratios': [0.8] + [1]*n_models})

# ── Input column (colored by phase) ──
for ri, (name, signal) in enumerate(inputs):
    ax = axes[ri, 0]
    norm_sig, _, _ = normalize_window(signal)
    t = np.arange(T)
    for j in range(SKIP, T - 1):
        c = plt.cm.hsv(phase[j - SKIP])
        ax.plot([t[j], t[j+1]], [norm_sig[j], norm_sig[j+1]],
                color=c, linewidth=0.8, solid_capstyle='round')
    ax.set_xlim(0, T)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor('#f8fafc')
    ax.patch.set_alpha(0.5)
    ax.set_ylabel(name, fontsize=5.5, fontweight='600', rotation=90, labelpad=3, color='#1e293b')
    if ri == 0:
        ax.set_title('Input', fontsize=6.5, fontweight='600', pad=4, color='#1e293b')

# ── PCA trajectories per model ──
for mi, (model_name, model_path) in enumerate(models_info):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()

    captured = {}
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured['hs'] = out.detach()
    handle = model.model.layers[LAYER].register_forward_hook(hook)

    for ri, (name, signal) in enumerate(inputs):
        norm_sig, _, _ = normalize_window(signal)
        bin_tokens = uniform_bin_tokenize(norm_sig, cfg.n_bins, cfg.bin_low, cfg.bin_high)
        input_ids = torch.tensor(bin_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)

        hs = captured['hs'].squeeze(0).float().cpu().numpy()[SKIP:]

        pca = PCA(n_components=2)
        proj = pca.fit_transform(hs)
        var_exp = pca.explained_variance_ratio_.sum() * 100

        mx = np.abs(proj).max()
        if mx > 0:
            proj = proj / mx

        ax = axes[ri, mi + 1]
        for j in range(len(proj) - 1):
            c = plt.cm.hsv(phase[j])
            ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c, linewidth=0.7,
                    alpha=0.85, solid_capstyle='round')

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor('#f8fafc')
        ax.patch.set_alpha(0.5)
        ax.text(0.97, 0.03, f'{var_exp:.0f}%', transform=ax.transAxes,
                fontsize=4.5, color='#64748b', ha='right', va='bottom', fontfamily='monospace')
        if ri == 0:
            ax.set_title(model_name, fontsize=6.5, fontweight='600', pad=4, color='#1e293b')

    handle.remove()
    del model; torch.cuda.empty_cache()

fig.patch.set_facecolor('white')
fig.savefig('../figures/sec5/fig_manifold_synthetic.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved fig_manifold_synthetic.png")
