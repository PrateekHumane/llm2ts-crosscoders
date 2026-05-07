"""
Plot crosscoder feature figures for NeurIPS appendix.
Each feature: top-3 diverse time-series windows with activation overlay.

Output: NeurIPS26/figures/appendix/fig_crosscoder_*.png
"""
import json
import math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent.parent.parent / 'analysis'
OUT = Path(__file__).resolve().parent.parent / 'figures' / 'appendix'
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    {'layer': 10, 'fid': 1712, 'name': 'Quantitative magnitude transitions',
     'slug': 'magnitude'},
    {'layer': 9, 'fid': 2469, 'name': 'Tropical weather systems',
     'slug': 'weather'},
    {'layer': 7, 'fid': 3888, 'name': 'Naval battle events',
     'slug': 'naval', 'window_indices': [0, 4, 6]},
]

N_WINDOWS = 3

act_cmap = LinearSegmentedColormap.from_list(
    'act', [(1, 1, 1, 0), (0.91, 0.30, 0.14, 0.50)])

TS_COLOR = '#1a3550'
PEAK_COLOR = '#d93025'
SPINE_COLOR = '#d1d5db'
TITLE_COLOR = '#111827'
SUB_COLOR = '#4b5563'
BG_COLOR = '#fafbfc'


def dedup_windows(windows, n):
    """Pick n windows with distinct series_idx and varied peak regions."""
    seen_series = set()
    seen_peak_bin = set()
    out = []
    # First pass: unique series AND different peak region (bins of 100)
    for w in windows:
        sid = w['series_idx']
        pbin = w['peak_timestep'] // 100
        if sid not in seen_series and pbin not in seen_peak_bin:
            seen_series.add(sid)
            seen_peak_bin.add(pbin)
            out.append(w)
            if len(out) == n:
                return out
    # Second pass: just unique series
    for w in windows:
        sid = w['series_idx']
        if sid not in seen_series:
            seen_series.add(sid)
            out.append(w)
            if len(out) == n:
                return out
    # Fallback
    for w in windows:
        if w not in out:
            out.append(w)
            if len(out) == n:
                return out
    return out


def plot_feature(feat):
    layer = feat['layer']
    fid = feat['fid']
    base = ANALYSIS / f'layer_{layer}' / 'PT_FT' / f'feature_{fid}'
    all_windows = json.load(open(base / 'windows.json'))
    if 'window_indices' in feat:
        windows = [all_windows[i] for i in feat['window_indices']]
    else:
        windows = dedup_windows(all_windows, N_WINDOWS)

    fig, axes = plt.subplots(1, N_WINDOWS, figsize=(5.5, 1.55),
                             gridspec_kw={'wspace': 0.32})

    for i, (ax, win) in enumerate(zip(axes, windows)):
        raw = np.array(win['raw_values'], dtype=np.float64)
        act_pt = np.array(win['activations_pt'], dtype=np.float64)
        peak = win['peak_timestep']
        act_val = win['activation_value']
        T = len(raw)
        t = np.arange(T)

        ax.set_facecolor(BG_COLOR)

        act_max = act_pt.max()
        if act_max > 0:
            act_norm = act_pt / act_max
        else:
            act_norm = act_pt

        for j in range(T - 1):
            ax.axvspan(j, j + 1, color=act_cmap(act_norm[j]),
                       lw=0, zorder=0)

        ax.plot(t, raw, color=TS_COLOR, lw=0.55, zorder=2)

        ax.axvline(peak, color=PEAK_COLOR, lw=0.8, ls='--',
                   alpha=0.75, zorder=3)

        rng = np.nanmax(raw) - np.nanmin(raw)
        pad = max(rng * 0.08, 1e-6)
        ax.set_ylim(np.nanmin(raw) - pad, np.nanmax(raw) + pad)
        ax.set_xlim(0, T)

        ax.text(0.98, 0.95, f'act={act_val:.1f}',
                transform=ax.transAxes, fontsize=5.5, fontweight='500',
                color=PEAK_COLOR, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.8))

        if i == 0:
            ax.set_ylabel('value', labelpad=2, fontsize=6.5)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel('timestep', labelpad=1, fontsize=6.5)
        ax.tick_params(length=2, pad=2, labelsize=5.5)

        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(0.4)

    fig.suptitle(
        f'Feature {fid} (Layer {layer}) — {feat["name"]}',
        fontsize=8, fontweight='600', color=TITLE_COLOR, y=1.06)

    outpath = OUT / f'fig_crosscoder_{feat["slug"]}.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved {outpath.name}')


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 7,
    'axes.linewidth': 0.4,
})

for f in FEATURES:
    plot_feature(f)
print('Done.')
