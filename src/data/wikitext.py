"""
WikiText-103 data preparation for PT activation extraction.

Downloads WikiText-103 from HuggingFace, tokenizes with the Qwen3-0.6B
tokenizer, and chunks into fixed-length sequences.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def load_wikitext_sequences(
    max_sequences: int = 30000,
    seq_len: int = 512,
    hf_token: str | None = None,
) -> list[dict]:
    """
    Download WikiText-103, tokenize with Qwen3-0.6B, and chunk into
    fixed-length sequences.

    Parameters
    ----------
    max_sequences : int
        Maximum number of sequences to return.
    seq_len : int
        Number of tokens per sequence (remainder is dropped).
    hf_token : str or None
        HuggingFace token for gated repos (not needed for WikiText, but
        passed through to the tokenizer download).

    Returns
    -------
    list[dict]
        Each dict has:
        - "input_ids": numpy int64 array of shape (seq_len,)
        - "text": the decoded text string for that chunk
    """
    # ------------------------------------------------------------------
    # 1. Download WikiText-103 parquet files via huggingface_hub
    # ------------------------------------------------------------------
    print("Downloading WikiText-103...")
    cache_dir = snapshot_download(
        "wikitext",
        repo_type="dataset",
        token=hf_token,
        allow_patterns=["wikitext-103-v1/*"],
    )
    data_dir = os.path.join(cache_dir, "wikitext-103-v1")

    # Prefer train split (largest), fall back to validation then test
    split_files = _find_split_files(data_dir, "train")
    if not split_files:
        print("  Train split not found, trying validation...")
        split_files = _find_split_files(data_dir, "validation")
    if not split_files:
        print("  Validation split not found, trying test...")
        split_files = _find_split_files(data_dir, "test")
    if not split_files:
        raise RuntimeError(
            f"No parquet files found for any split in {data_dir}. "
            f"Available files: {os.listdir(data_dir)}"
        )
    print(f"  Using {len(split_files)} parquet file(s)")

    # ------------------------------------------------------------------
    # 2. Read text from parquet files
    # ------------------------------------------------------------------
    print("Reading text from parquet files...")
    all_text = _read_parquet_text(split_files)
    print(f"  Read {len(all_text)} lines of text")

    # ------------------------------------------------------------------
    # 3. Load tokenizer (no special tokens added during encoding)
    # ------------------------------------------------------------------
    print("Loading Qwen3-0.6B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", token=hf_token
    )

    # ------------------------------------------------------------------
    # 4. Tokenize all text into one flat token stream
    # ------------------------------------------------------------------
    print("Tokenizing...")
    all_token_ids = _tokenize_texts(tokenizer, all_text, max_sequences, seq_len)
    print(f"  Total tokens: {len(all_token_ids):,}")

    # ------------------------------------------------------------------
    # 5. Chunk into sequences of exactly seq_len tokens
    # ------------------------------------------------------------------
    n_sequences = min(len(all_token_ids) // seq_len, max_sequences)
    if n_sequences == 0:
        raise RuntimeError(
            f"Not enough tokens ({len(all_token_ids)}) to form even one "
            f"sequence of length {seq_len}."
        )

    print(f"  Chunking into {n_sequences} sequences of {seq_len} tokens...")
    sequences = []
    for i in range(n_sequences):
        start = i * seq_len
        end = start + seq_len
        ids = np.array(all_token_ids[start:end], dtype=np.int64)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        sequences.append({
            "input_ids": ids,
            "text": text,
        })

    print(f"  Done: {len(sequences)} sequences")
    return sequences


def _find_split_files(data_dir: str, split: str) -> list[str]:
    """Find parquet files for a given split in the data directory."""
    pattern = os.path.join(data_dir, f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def _read_parquet_text(parquet_files: list[str]) -> list[str]:
    """
    Read the 'text' column from a list of parquet files.

    Returns a list of non-empty text strings.
    """
    texts = []
    for fpath in parquet_files:
        table = pq.read_table(fpath, columns=["text"])
        col = table.column("text")
        for val in col:
            s = val.as_py()
            if s is not None and s.strip():
                texts.append(s)
    return texts


def _tokenize_texts(
    tokenizer,
    texts: list[str],
    max_sequences: int,
    seq_len: int,
) -> list[int]:
    """
    Tokenize a list of text strings into a single flat list of token IDs.

    Stops early once enough tokens have been collected to fill max_sequences
    chunks.
    """
    target_tokens = max_sequences * seq_len
    all_ids: list[int] = []

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= target_tokens:
            break

    return all_ids
