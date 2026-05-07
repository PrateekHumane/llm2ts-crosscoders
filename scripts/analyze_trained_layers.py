"""
Run analysis pipeline on all trained layers.
Sequentially: precompute val activations → categorize → extract top features → plot.

Usage:
    python3 scripts/analyze_trained_layers.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_layer import analyze_single_layer, N_WIKI_SEQS, WIKI_SEQ_LEN
from src.config import Config
from src.data.dataset import (
    build_datasets, WindowDataset, load_gifteval_series, temporal_split,
)
from src.data.wikitext import load_wikitext_sequences

LAYERS = [13, 0, 6, 20]


def main():
    cfg = Config()
    cfg.linear_crosscoder = True
    cfg.latent_dim = 4096
    cfg.top_k = 64
    cfg.checkpoint_dir = "checkpoints/linear_d4096"

    hf_token = os.environ.get("HF_TOKEN")

    os.makedirs("analysis", exist_ok=True)
    os.makedirs(cfg.precompute_dir, exist_ok=True)

    print("Loading GiftEval validation set (non-overlapping)...", flush=True)
    series_list = load_gifteval_series(hf_token)
    val_splits = []
    for s in series_list:
        _, va, _ = temporal_split(s, cfg.train_frac, cfg.val_frac)
        if len(va) >= cfg.context_length:
            val_splits.append(va)
    val_ds = WindowDataset(val_splits, cfg.context_length, stride=cfg.context_length)
    print(f"  Val windows (non-overlapping): {len(val_ds):,}", flush=True)

    print(f"Loading WikiText ({N_WIKI_SEQS:,} sequences)...", flush=True)
    wiki_sequences = load_wikitext_sequences(
        max_sequences=N_WIKI_SEQS, seq_len=WIKI_SEQ_LEN, hf_token=hf_token
    )
    print(f"  WikiText sequences: {len(wiki_sequences):,}", flush=True)

    for layer_idx in LAYERS:
        analyze_single_layer(layer_idx, cfg, val_ds, wiki_sequences, hf_token)

    print("\nAll layers analyzed!")


if __name__ == "__main__":
    main()
