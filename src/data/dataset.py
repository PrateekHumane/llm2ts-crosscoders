"""
GiftEval dataset loader with temporal train/val/test split and windowing.
"""
import glob
import os

import numpy as np
import pyarrow as pa
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset


def load_gifteval_series(hf_token: str | None = None) -> list[np.ndarray]:
    """
    Load GiftEval and return list of 1-D float arrays (one per series).

    GiftEval has heterogeneous schemas across sub-datasets (some have
    'past_feat_dynamic_real'), so we bypass the datasets library and read
    arrow files directly via pyarrow.
    """
    cache_dir = snapshot_download(
        "Salesforce/GiftEval",
        repo_type="dataset",
        token=hf_token,
    )

    arrow_files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.arrow"), recursive=True))
    if not arrow_files:
        raise RuntimeError(f"No arrow files found in {cache_dir}")

    print(f"  Found {len(arrow_files)} arrow files")

    series_list = []
    for fpath in arrow_files:
        try:
            reader = pa.ipc.open_stream(fpath)
            tbl = reader.read_all()
        except Exception:
            continue

        if "target" not in tbl.schema.names:
            continue

        for i in range(len(tbl)):
            target = tbl["target"][i]
            if hasattr(target, "as_py"):
                target = target.as_py()
            arr = np.array(target, dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.flatten()
            if len(arr) > 0:
                series_list.append(arr)

    return series_list


def temporal_split(series: np.ndarray, train_frac: float, val_frac: float):
    """Split a single series into (train, val, test) by time."""
    n = len(series)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return series[:n_train], series[n_train:n_train + n_val], series[n_train + n_val:]


class WindowDataset(Dataset):
    """
    Yields non-overlapping windows (for training) or sliding windows (for val/test).
    Each item is a dict with:
        'values': float32 array of shape (context_length,)
        'series_idx': int
        'offset': int  (start position in the original split)
    """
    def __init__(
        self,
        series_splits: list[np.ndarray],
        context_length: int,
        stride: int | None = None,   # None = non-overlapping (stride=context_length)
    ):
        self.context_length = context_length
        stride = stride if stride is not None else context_length

        self.windows = []  # list of (series_idx, offset)
        for sid, series in enumerate(series_splits):
            n = len(series)
            for start in range(0, n - context_length + 1, stride):
                self.windows.append((sid, start))

        self._series = series_splits

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sid, start = self.windows[idx]
        values = self._series[sid][start:start + self.context_length].copy()
        return {
            "values": values,
            "series_idx": sid,
            "offset": start,
        }


def build_datasets(
    context_length: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    hf_token: str | None = None,
) -> tuple[WindowDataset, WindowDataset, WindowDataset]:
    """Load GiftEval, split, and return (train_ds, val_ds, test_ds)."""
    print("Loading GiftEval...")
    series_list = load_gifteval_series(hf_token)
    print(f"  Loaded {len(series_list)} series")
    total_timesteps = sum(len(s) for s in series_list)
    print(f"  Total timesteps: {total_timesteps:,}")

    train_splits, val_splits, test_splits = [], [], []
    for s in series_list:
        tr, va, te = temporal_split(s, train_frac, val_frac)
        if len(tr) >= context_length:
            train_splits.append(tr)
        if len(va) >= context_length:
            val_splits.append(va)
        if len(te) >= context_length:
            test_splits.append(te)

    train_ds = WindowDataset(train_splits, context_length, stride=context_length)
    val_ds = WindowDataset(val_splits, context_length, stride=1)
    test_ds = WindowDataset(test_splits, context_length, stride=1)

    print(f"  Train windows (non-overlap): {len(train_ds):,}")
    print(f"  Val windows   (sliding):     {len(val_ds):,}")
    print(f"  Test windows  (sliding):     {len(test_ds):,}")

    return train_ds, val_ds, test_ds
