"""
Manifold Structure: Does the geometry of neural representations
mirror the geometry of the input time series?

Inspired by Moisescu-Pareja et al. (2025) who showed modular arithmetic's
circular structure appears as a torus/disc in neural representations.

We create synthetic TS with known geometric structure and check whether
the hidden state trajectories reflect that structure:
  1. Pure sine wave → expect closed loop
  2. Sine + trend → expect helix
  3. Linear trend → expect line
  4. Step function → expect two clusters
  5. Two frequencies → expect 2-torus or figure-8
  6. Real periodic TS (electricity) → expect loops at natural period

Each synthetic TS is bin-tokenized and passed through FT, RI, PT.

Usage:
    /usr/bin/python3 scripts/manifold_structure.py
"""
import torch
import numpy as np
import os, sys, gc, glob, json

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

import pyarrow.ipc as ipc
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T = 512
DEVICE = torch.device("cuda:0")
OUT_DIR = "mapping_results/manifold_structure"
MODEL_COLORS = {"PT": "#2196F3", "FT": "#4CAF50", "RI": "#E91E63"}
MODEL_PATHS = {"PT": "Qwen/Qwen3-0.6B", "FT": "models/ft", "RI": "models/ri"}


# ═══════════════════════════════════════════════════════════════
# Synthetic time series with known structure
# ═══════════════════════════════════════════════════════════════

def make_synthetic_ts():
    """Create synthetic TS with known geometric properties."""
    t = np.linspace(0, 1, T, endpoint=False)
    synthetics = {}

    # 1. Pure sine (period 64 → 8 full cycles)
    period = 64
    synthetics["sine_p64"] = {
        "values": np.sin(2 * np.pi * t * (T / period)),
        "label": f"Sine (period {period}, 8 cycles)",
        "expected": "Closed loop traced 8 times",
        "period": period,
    }

    # 2. Pure sine (period 128 → 4 cycles)
    period = 128
    synthetics["sine_p128"] = {
        "values": np.sin(2 * np.pi * t * (T / period)),
        "label": f"Sine (period {period}, 4 cycles)",
        "expected": "Closed loop traced 4 times",
        "period": period,
    }

    # 3. Sine + linear trend
    synthetics["sine_trend"] = {
        "values": np.sin(2 * np.pi * t * 8) + 3 * t,
        "label": "Sine + linear trend",
        "expected": "Helix (loop + drift)",
        "period": 64,
    }

    # 4. Linear trend (monotonic)
    synthetics["linear"] = {
        "values": t * 4 - 2,
        "label": "Linear trend",
        "expected": "Line or gentle curve",
        "period": None,
    }

    # 5. Step function
    step = np.zeros(T)
    step[T//3:2*T//3] = 2.0
    synthetics["step"] = {
        "values": step,
        "label": "Step function",
        "expected": "Two clusters with transition",
        "period": None,
    }

    # 6. Two frequencies (64 + 128)
    synthetics["two_freq"] = {
        "values": np.sin(2 * np.pi * t * 8) + 0.5 * np.sin(2 * np.pi * t * 4),
        "label": "Two frequencies (p=64 + p=128)",
        "expected": "2-torus or figure-8",
        "period": 128,  # LCM
    }

    # 7. Square wave (period 64)
    sq = np.sign(np.sin(2 * np.pi * t * 8))
    synthetics["square"] = {
        "values": sq,
        "label": "Square wave (period 64)",
        "expected": "Rectangle or two-lobed loop",
        "period": 64,
    }

    # 8. Random walk
    rng = np.random.default_rng(42)
    rw = np.cumsum(rng.standard_normal(T))
    synthetics["random_walk"] = {
        "values": rw,
        "label": "Random walk",
        "expected": "Irregular, non-repeating path",
        "period": None,
    }

    # Z-score normalize and bin-tokenize each
    for key in synthetics:
        v = synthetics[key]["values"].astype(np.float32)
        mu, sigma = v.mean(), v.std()
        if sigma < 1e-6:
            sigma = 1.0
        v_norm = (v - mu) / sigma
        v_clipped = np.clip(v_norm, -5, 5)
        bins = ((v_clipped + 5) / 10 * 512).astype(np.int64).clip(0, 511)
        synthetics[key]["input_ids"] = bins
        synthetics[key]["values_norm"] = v_norm

    return synthetics


def load_real_periodic_ts(hf_token):
    """Load a real periodic TS (electricity — has daily cycle)."""
    path = snapshot_download("Salesforce/GiftEval", repo_type="dataset", token=hf_token)
    arrow_files = sorted(glob.glob(os.path.join(path, "**/*.arrow"), recursive=True))
    for f in arrow_files:
        rel = os.path.relpath(f, path)
        if rel.startswith("electricity/H"):  # hourly → daily period = 24
            tbl = ipc.open_stream(open(f, "rb")).read_all()
            longest = np.array(max(tbl['target'].to_pylist(), key=len), dtype=np.float32)
            # Take 512 hours = ~21 days → ~21 daily cycles
            start = int(len(longest) * 0.5)
            w = longest[start:start + T]
            mu, sigma = w.mean(), w.std()
            w_norm = (w - mu) / sigma
            bins = ((np.clip(w_norm, -5, 5) + 5) / 10 * 512).astype(np.int64).clip(0, 511)
            return {
                "input_ids": bins,
                "values_norm": w_norm,
                "label": "Real electricity (hourly, ~21 daily cycles)",
                "expected": "Loop with period ~24",
                "period": 24,
            }
    return None


def extract_layer(model, token_ids, layer_idx, device):
    captured = {}
    def hook(m, i, o):
        captured['h'] = (o[0] if isinstance(o, tuple) else o).detach()
    handle = model.layers[layer_idx].register_forward_hook(hook)
    ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        model(input_ids=ids, use_cache=False)
    h = captured['h'].squeeze(0)[1:, :].float().cpu()  # skip pos 0
    handle.remove()
    return h


def plot_2d(ax, proj, period, n_total, label, color_by_phase=True):
    """Plot 2D trajectory, optionally colored by phase within period."""
    n = len(proj)
    if color_by_phase and period is not None:
        cmap = plt.cm.hsv
        for t in range(n - 1):
            phase = (t % period) / period
            ax.plot(proj[t:t+2, 0], proj[t:t+2, 1],
                    color=cmap(phase), alpha=0.6, linewidth=0.8)
    else:
        for t in range(n - 1):
            frac = t / n
            ax.plot(proj[t:t+2, 0], proj[t:t+2, 1],
                    color='steelblue', alpha=0.15 + 0.6 * frac, linewidth=0.8)
    ax.scatter(proj[0, 0], proj[0, 1], color='green', s=80, marker='^',
               zorder=10, edgecolors='black', linewidth=0.8, label='Start')
    ax.scatter(proj[-1, 0], proj[-1, 1], color='red', s=80, marker='v',
               zorder=10, edgecolors='black', linewidth=0.8, label='End')


def plot_3d(ax, proj, period, n_total, color_by_phase=True):
    """Plot 3D trajectory."""
    n = len(proj)
    if color_by_phase and period is not None:
        cmap = plt.cm.hsv
        for t in range(n - 1):
            phase = (t % period) / period
            ax.plot(proj[t:t+2, 0], proj[t:t+2, 1], proj[t:t+2, 2],
                    color=cmap(phase), alpha=0.5, linewidth=0.7)
    else:
        for t in range(n - 1):
            frac = t / n
            ax.plot(proj[t:t+2, 0], proj[t:t+2, 1], proj[t:t+2, 2],
                    color='steelblue', alpha=0.15 + 0.5 * frac, linewidth=0.7)
    ax.scatter(proj[0, 0], proj[0, 1], proj[0, 2], color='green', s=60, marker='^',
               zorder=10, edgecolors='black', linewidth=0.5)
    ax.scatter(proj[-1, 0], proj[-1, 1], proj[-1, 2], color='red', s=60, marker='v',
               zorder=10, edgecolors='black', linewidth=0.5)


def main():
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN")

    # Create synthetic TS
    print("Creating synthetic time series...", flush=True)
    synthetics = make_synthetic_ts()

    # Load real periodic TS
    real_periodic = load_real_periodic_ts(hf_token)
    if real_periodic:
        synthetics["real_electricity"] = real_periodic

    print(f"  Created {len(synthetics)} synthetic/real TS:")
    for key, s in synthetics.items():
        print(f"    {key}: {s['label']} → expected: {s['expected']}")

    # Target layers
    target_layers = [4, 8, 16]

    # Process each model
    all_data = {}  # (model, ts_key, layer) -> (T-1, D)

    for mname, mpath in MODEL_PATHS.items():
        print(f"\nLoading {mname}...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            mpath, dtype=torch.bfloat16).model.to(DEVICE).eval()

        for ts_key, ts_info in synthetics.items():
            for li in target_layers:
                h = extract_layer(model, ts_info["input_ids"], li, DEVICE)
                all_data[(mname, ts_key, li)] = h

        del model; gc.collect(); torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════
    print("\nGenerating plots...", flush=True)

    # ── Main figure: key synthetics through FT at layer 8 ──
    key_ts = ["sine_p64", "sine_trend", "linear", "step", "two_freq", "random_walk"]
    if "real_electricity" in synthetics:
        key_ts.append("real_electricity")

    # 2D grid: each row = one TS, columns = input waveform + 3 models
    n_rows = len(key_ts)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.5 * n_rows))
    fig.suptitle("Neural Representation Manifolds: Do Representations Mirror Input Structure?\n"
                 "Left: input TS. Right 3: hidden state trajectory at Layer 8 (colored by phase within period)",
                 fontsize=14, fontweight='bold')

    for ri, ts_key in enumerate(key_ts):
        ts_info = synthetics[ts_key]
        period = ts_info.get("period")

        # Column 0: input waveform
        ax = axes[ri, 0]
        v = ts_info["values_norm"]
        if period is not None:
            cmap = plt.cm.hsv
            for t in range(len(v) - 1):
                phase = (t % period) / period
                ax.plot([t, t+1], [v[t], v[t+1]], color=cmap(phase), linewidth=1)
        else:
            ax.plot(v, color='steelblue', linewidth=1)
        ax.set_ylabel(ts_info["label"], fontsize=9, fontweight='bold')
        ax.set_xlim(0, T)
        if ri == 0:
            ax.set_title("Input TS", fontsize=12, fontweight='bold')
        if ri == n_rows - 1:
            ax.set_xlabel("Timestep")

        # Columns 1-3: PT, FT, RI
        for mi, mname in enumerate(["PT", "FT", "RI"]):
            ax = axes[ri, mi + 1]
            h = all_data[(mname, ts_key, 8)]
            h_np = h.numpy()
            h_c = h_np - h_np.mean(axis=0)
            pca = PCA(n_components=2).fit(h_c)
            proj = pca.transform(h_c)

            plot_2d(ax, proj, period, len(proj), ts_info["label"])

            ev = pca.explained_variance_ratio_
            if ri == 0:
                ax.set_title(f"{mname}\nPC1+2={sum(ev[:2])*100:.0f}%",
                             fontsize=12, fontweight='bold', color=MODEL_COLORS[mname])
            else:
                ax.set_title(f"PC1+2={sum(ev[:2])*100:.0f}%", fontsize=9, color=MODEL_COLORS[mname])
            ax.grid(alpha=0.2)
            if ri == 0 and mi == 2:
                ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT_DIR}/plots/manifold_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved manifold_grid.png")

    # ── 3D versions for the most interesting cases ──
    interesting = ["sine_p64", "sine_trend", "two_freq"]
    if "real_electricity" in synthetics:
        interesting.append("real_electricity")

    for ts_key in interesting:
        ts_info = synthetics[ts_key]
        period = ts_info.get("period")

        fig = plt.figure(figsize=(20, 6))
        fig.suptitle(f"3D Representation Manifold: {ts_info['label']}\n"
                     f"Expected: {ts_info['expected']}  |  Colored by phase within period",
                     fontsize=13, fontweight='bold')

        for mi, mname in enumerate(["PT", "FT", "RI"]):
            ax = fig.add_subplot(1, 3, mi + 1, projection='3d')
            h = all_data[(mname, ts_key, 8)]
            h_np = h.numpy()
            h_c = h_np - h_np.mean(axis=0)
            pca = PCA(n_components=3).fit(h_c)
            proj = pca.transform(h_c)

            plot_3d(ax, proj, period, len(proj))

            ev = pca.explained_variance_ratio_
            ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)", fontsize=8)
            ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)", fontsize=8)
            ax.set_zlabel(f"PC3 ({ev[2]*100:.0f}%)", fontsize=8)
            ax.set_title(f"{mname} (PC1-3={sum(ev[:3])*100:.0f}%)",
                         fontsize=12, fontweight='bold', color=MODEL_COLORS[mname])
            ax.view_init(elev=25, azim=45)
            ax.tick_params(labelsize=6)

        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/{ts_key}_3d.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {ts_key}_3d.png")

    # ── Layer evolution for sine wave through FT ──
    ts_key = "sine_p64"
    ts_info = synthetics[ts_key]
    period = ts_info["period"]
    layers_to_show = [0, 2, 4, 8, 12, 16, 20, 24, 27]

    # Need to extract all layers for FT
    print("Extracting all layers for sine through FT...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["FT"], dtype=torch.bfloat16).model.to(DEVICE).eval()
    sine_layers = {}
    for li in layers_to_show:
        sine_layers[li] = extract_layer(model, ts_info["input_ids"], li, DEVICE)
    del model; gc.collect(); torch.cuda.empty_cache()

    n_cols = 3
    n_rows_grid = (len(layers_to_show) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows_grid, n_cols, figsize=(6 * n_cols, 5 * n_rows_grid))
    fig.suptitle(f"Sine Wave (period 64) Through FT: Layer-by-Layer Evolution\n"
                 f"Colored by phase — does a loop emerge?", fontsize=14, fontweight='bold')

    for idx, li in enumerate(layers_to_show):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        h = sine_layers[li].numpy()
        h_c = h - h.mean(axis=0)
        pca = PCA(n_components=2).fit(h_c)
        proj = pca.transform(h_c)

        plot_2d(ax, proj, period, len(proj), f"Layer {li}")

        ev = pca.explained_variance_ratio_
        ax.set_title(f"Layer {li}  (PC1+2={sum(ev[:2])*100:.0f}%)",
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.2)

    # Hide unused axes
    for idx in range(len(layers_to_show), n_rows_grid * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{OUT_DIR}/plots/sine_layer_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved sine_layer_evolution.png")

    # ── Phase coherence analysis ──
    # For periodic inputs, check if same-phase points cluster together
    print("\nPhase coherence analysis...", flush=True)
    results = {}
    for ts_key in ["sine_p64", "sine_p128", "two_freq"]:
        if ts_key not in synthetics:
            continue
        ts_info = synthetics[ts_key]
        period = ts_info["period"]
        if period is None:
            continue

        results[ts_key] = {}
        for mname in ["PT", "FT", "RI"]:
            h = all_data[(mname, ts_key, 8)].numpy()
            h_c = h - h.mean(axis=0)
            n = len(h_c)

            # For each pair of points at the same phase (t, t+period),
            # compute their distance. Compare to average distance.
            same_phase_dists = []
            all_dists = []
            for t in range(n - period):
                d = np.linalg.norm(h_c[t] - h_c[t + period])
                same_phase_dists.append(d)
            for t in range(0, n, 10):
                for s in range(t + 1, min(t + 100, n)):
                    all_dists.append(np.linalg.norm(h_c[t] - h_c[s]))

            same_mean = np.mean(same_phase_dists)
            all_mean = np.mean(all_dists)
            ratio = same_mean / all_mean  # < 1 means same-phase points are closer

            results[ts_key][mname] = {
                "same_phase_dist": float(same_mean),
                "all_dist": float(all_mean),
                "ratio": float(ratio),
            }
            print(f"  {ts_key} {mname}: same_phase={same_mean:.2f} all={all_mean:.2f} ratio={ratio:.3f}")

    with open(f"{OUT_DIR}/phase_coherence.json", "w") as f:
        json.dump(results, f, indent=2)

    # Phase coherence bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ts_keys_plot = [k for k in ["sine_p64", "sine_p128", "two_freq"] if k in results]
    x = np.arange(len(ts_keys_plot))
    width = 0.25
    for mi, mname in enumerate(["PT", "FT", "RI"]):
        vals = [results[k][mname]["ratio"] for k in ts_keys_plot]
        bars = ax.bar(x + mi * width, vals, width, color=MODEL_COLORS[mname], label=mname)
        for i, v in enumerate(vals):
            ax.text(x[i] + mi * width, v + 0.01, f"{v:.2f}", ha='center', fontsize=8)

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='No phase structure')
    ax.set_xticks(x + width)
    ax.set_xticklabels([synthetics[k]["label"][:25] for k in ts_keys_plot], fontsize=9)
    ax.set_ylabel("Same-phase dist / All-pairs dist")
    ax.set_title("Phase Coherence: Do same-phase points cluster?\n(< 1 means the model represents periodicity)",
                 fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/plots/phase_coherence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved phase_coherence.png")

    print(f"\nAll saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
