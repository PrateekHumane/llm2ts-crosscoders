"""
Phase 1 validation script.

Checks:
1. GiftEval loads correctly and produces train/val/test WindowDatasets
2. FT/RI tokenization produces correct shapes
3. PT text serialization + mean pooling produces correct shapes
4. All three model extractors run a forward pass and return (num_layers, B*T, hidden_size)
5. Activation shapes match expectations

Run with:
    source ~/.bashrc && python scripts/validate_phase1.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import Config
from src.data.dataset import build_datasets
from src.data.tokenize import (
    normalize_window, uniform_bin_tokenize, window_to_text,
    get_timestep_token_spans, mean_pool_hidden_states,
)
from src.models.extractor import ModelExtractor

HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = torch.device("cuda:0")
CFG = Config()

# ── 1. Dataset ───────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: GiftEval loading + windowing")
print("=" * 60)

train_ds, val_ds, test_ds = build_datasets(
    context_length=CFG.context_length,
    train_frac=CFG.train_frac,
    val_frac=CFG.val_frac,
    hf_token=HF_TOKEN,
)

sample = train_ds[0]
assert sample["values"].shape == (CFG.context_length,), \
    f"Expected ({CFG.context_length},), got {sample['values'].shape}"
print(f"  Sample window shape: {sample['values'].shape}  ✓")
print(f"  series_idx={sample['series_idx']}, offset={sample['offset']}")
print()

# ── 2. FT/RI tokenization ────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2: FT/RI uniform-bin tokenization")
print("=" * 60)

values = sample["values"]
norm, mean, std = normalize_window(values)
print(f"  Raw   mean={mean:.4f}  std={std:.4f}")
print(f"  Norm  mean={norm.mean():.4f}  std={norm.std():.4f}")

tokens = uniform_bin_tokenize(norm, CFG.n_bins, CFG.bin_low, CFG.bin_high)
assert tokens.shape == (CFG.context_length,), f"Bad token shape: {tokens.shape}"
assert tokens.min() >= 0 and tokens.max() < CFG.n_bins, \
    f"Tokens out of range: [{tokens.min()}, {tokens.max()}]"
print(f"  Token shape: {tokens.shape}  ✓")
print(f"  Token range: [{tokens.min()}, {tokens.max()}]  (expected [0, {CFG.n_bins-1}])")
print()

# ── 3. PT text serialization ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 3: PT text serialization")
print("=" * 60)

text = window_to_text(norm)
parts = text.split()
assert len(parts) == CFG.context_length, \
    f"Expected {CFG.context_length} parts, got {len(parts)}"
print(f"  Text length (chars): {len(text)}")
print(f"  Text parts (timesteps): {len(parts)}  ✓")
print(f"  First 5 values: {' '.join(parts[:5])}")
print()

# ── 4. Full model extraction (small batch) ───────────────────────────────────
print("=" * 60)
print("STEP 4: Model extraction (batch_size=2)")
print("=" * 60)

extractor = ModelExtractor(CFG, DEVICE, hf_token=HF_TOKEN)

# Use a small batch of 2 windows for speed
batch = [train_ds[i] for i in range(2)]
B = len(batch)
T = CFG.context_length

print(f"\nExtracting FT + RI activations...")
ft_acts, ri_acts = extractor.extract_ft_ri(batch)

assert len(ft_acts) == CFG.num_layers, f"FT: expected {CFG.num_layers} layers, got {len(ft_acts)}"
assert len(ri_acts) == CFG.num_layers, f"RI: expected {CFG.num_layers} layers, got {len(ri_acts)}"
for i, (ft, ri) in enumerate(zip(ft_acts, ri_acts)):
    assert ft.shape == (B * T, CFG.hidden_size), \
        f"FT layer {i}: expected ({B*T}, {CFG.hidden_size}), got {ft.shape}"
    assert ri.shape == (B * T, CFG.hidden_size), \
        f"RI layer {i}: expected ({B*T}, {CFG.hidden_size}), got {ri.shape}"

print(f"  FT: {len(ft_acts)} layers, each {ft_acts[0].shape}  ✓")
print(f"  RI: {len(ri_acts)} layers, each {ri_acts[0].shape}  ✓")

print(f"\nExtracting PT activations (text pipeline)...")
pt_acts = extractor.extract_pt(batch)

assert len(pt_acts) == CFG.num_layers, f"PT: expected {CFG.num_layers} layers, got {len(pt_acts)}"
for i, pt in enumerate(pt_acts):
    assert pt.shape == (B * T, CFG.hidden_size), \
        f"PT layer {i}: expected ({B*T}, {CFG.hidden_size}), got {pt.shape}"

print(f"  PT: {len(pt_acts)} layers, each {pt_acts[0].shape}  ✓")

# ── 5. Sanity checks ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 5: Activation sanity checks")
print("=" * 60)

for name, acts in [("PT", pt_acts), ("FT", ft_acts), ("RI", ri_acts)]:
    l8 = acts[8]
    print(f"  {name} layer 8 — mean={l8.mean():.4f}  std={l8.std():.4f}  "
          f"min={l8.min():.4f}  max={l8.max():.4f}")
    assert not torch.isnan(l8).any(), f"{name} layer 8 has NaNs!"
    assert not torch.isinf(l8).any(), f"{name} layer 8 has Infs!"

print()
print("=" * 60)
print("ALL PHASE 1 CHECKS PASSED ✓")
print("=" * 60)
