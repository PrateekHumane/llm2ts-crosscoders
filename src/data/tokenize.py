"""
Tokenization helpers for FT/RI (uniform binning) and PT (text serialization).
"""
import numpy as np
import torch


# ---------------------------------------------------------------------------
# FT / RI  — uniform-bin tokenizer
# ---------------------------------------------------------------------------

def normalize_window(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score a window. Returns (normalized, mean, std)."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        std = 1.0
    return (values - mean) / std, mean, std


def uniform_bin_tokenize(values: np.ndarray, n_bins: int = 512,
                          low: float = -5.0, high: float = 5.0) -> np.ndarray:
    """
    Convert normalized values to integer token IDs via uniform binning.
    Values outside [low, high] are clipped.
    Returns int64 array of shape (len(values),).
    """
    safe   = np.nan_to_num(values, nan=0.0, posinf=high, neginf=low)
    clipped = np.clip(safe, low, high)
    # Map [low, high] → [0, n_bins-1]
    bins = ((clipped - low) / (high - low) * n_bins).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


# ---------------------------------------------------------------------------
# PT — text serialization
# ---------------------------------------------------------------------------

def window_to_text(normalized_values: np.ndarray, decimals: int = 3) -> str:
    """Convert normalized window to space-separated decimal strings."""
    fmt = f"{{:.{decimals}f}}"
    return " ".join(fmt.format(v) for v in normalized_values)


def get_timestep_token_spans(
    text: str,
    normalized_values: np.ndarray,
    tokenizer,
    decimals: int = 3,
) -> list[tuple[int, int]]:
    """
    For each timestep, return the (start, end) token indices in the tokenized
    sequence that correspond to that timestep's text representation.

    Returns list of (tok_start, tok_end) pairs, length = len(normalized_values).
    tok_end is exclusive.
    """
    fmt = f"{{:.{decimals}f}}"
    value_strings = [fmt.format(v) for v in normalized_values]

    # Tokenize the full text with char offsets
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]  # list of (char_start, char_end)

    spans = []
    char_cursor = 0
    for vs in value_strings:
        char_start = char_cursor
        char_end = char_cursor + len(vs)
        char_cursor = char_end + 1  # +1 for the space separator

        # Find tokens that overlap with [char_start, char_end)
        tok_start = None
        tok_end = None
        for i, (cs, ce) in enumerate(offsets):
            if ce <= char_start:
                continue
            if cs >= char_end:
                break
            if tok_start is None:
                tok_start = i
            tok_end = i + 1

        if tok_start is None:
            # Fallback: single token at current position (shouldn't happen)
            tok_start = len(spans)
            tok_end = tok_start + 1

        spans.append((tok_start, tok_end))

    return spans


def mean_pool_hidden_states(
    hidden_states: torch.Tensor,  # (seq_len, hidden_size)
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """
    For each timestep span, mean-pool hidden states across sub-tokens.
    Returns (num_timesteps, hidden_size).
    """
    results = []
    for (start, end) in spans:
        if end <= start:
            end = start + 1
        end = min(end, hidden_states.size(0))
        vec = hidden_states[start:end].mean(dim=0)
        results.append(vec)
    return torch.stack(results, dim=0)
