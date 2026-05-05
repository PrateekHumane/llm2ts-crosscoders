"""
Figure: Sine wave hidden-state trajectories across layers for FT vs RI.
Shows 2D PCA of hidden states at 5 selected layers, colored by input phase.
First 5 positions skipped to avoid attention-sink artifact.

Output: NeurIPS26/figures/sec5/fig_sine_evolution.png
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

T = 512; period = 64; SKIP = 5
t_arr = np.arange(T, dtype=np.float32)
sine = np.sin(2 * np.pi * t_arr / period)
norm_sine, _, _ = normalize_window(sine)
bin_tokens = uniform_bin_tokenize(norm_sine, cfg.n_bins, cfg.bin_low, cfg.bin_high)
input_ids = torch.tensor(bin_tokens, dtype=torch.long, device=device).unsqueeze(0)

phase = (t_arr % period) / period

layers_to_show = [2, 6, 8, 16, 24]
n_layers = len(layers_to_show)

fig, axes = plt.subplots(2, n_layers, figsize=(5.0, 2.6),
                          gridspec_kw={'hspace': 0.12, 'wspace': 0.08})

for mi, (model_name, model_path) in enumerate([('FT', cfg.model_ft), ('RI', cfg.model_ri)]):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device).eval()

    captured = {}
    handles = []
    for li in layers_to_show:
        def make_hook(l):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[l] = out.detach()
            return hook
        handles.append(model.model.layers[li].register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()

    for ci, li in enumerate(layers_to_show):
        ax = axes[mi, ci]
        hs = captured[li].squeeze(0).float().cpu().numpy()[SKIP:]
        phase_clean = phase[SKIP:]

        pca = PCA(n_components=2)
        proj = pca.fit_transform(hs)
        var_exp = pca.explained_variance_ratio_.sum() * 100

        mx = np.abs(proj).max()
        if mx > 0:
            proj = proj / mx

        for j in range(len(proj) - 1):
            c = plt.cm.hsv(phase_clean[j])
            ax.plot(proj[j:j+2, 0], proj[j:j+2, 1], color=c, linewidth=0.8,
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
                fontsize=4.5, color='#64748b', ha='right', va='bottom',
                fontfamily='monospace')

        if mi == 0:
            ax.set_title(f'Layer {li}', fontsize=6.5, fontweight='600', pad=4, color='#1e293b')
        if ci == 0:
            ax.set_ylabel(model_name, fontsize=8, fontweight='bold', labelpad=3, color='#1e293b')

    del model; torch.cuda.empty_cache()

fig.patch.set_facecolor('white')
fig.savefig('../figures/sec5/fig_sine_evolution.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("Saved fig_sine_evolution.png")
