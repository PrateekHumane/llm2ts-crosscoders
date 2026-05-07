"""Circuit matching: cosine similarity between synthetic TS and WikiText activations."""
import torch
import numpy as np
import heapq
from dataclasses import dataclass


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TokenMatch:
    similarity: float
    text_context: str
    token_text: str
    position: int
    passage_idx: int
    layer: int
    ts_type: str
    act_type: str

    def __lt__(self, other):
        return self.similarity < other.similarity

    def to_dict(self):
        return vars(self)


@dataclass
class WindowMatch:
    similarity: float
    text: str
    start_pos: int
    passage_idx: int
    layer: int
    ts_type: str
    act_type: str

    def __lt__(self, other):
        return self.similarity < other.similarity

    def to_dict(self):
        return vars(self)


# ── top-K tracker ────────────────────────────────────────────────────────────

class TopKTracker:
    """Min-heap that keeps the top-K highest-similarity items."""

    def __init__(self, k: int):
        self.k = k
        self.heap: list = []

    def push(self, item):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def get_sorted(self) -> list:
        return sorted(self.heap, key=lambda x: -x.similarity)


# ── profile computation ─────────────────────────────────────────────────────

ACT_TYPES = ("mlp", "attn", "res")


def compute_type_profiles(
    all_activations: dict[tuple, torch.Tensor],
    type_labels: torch.Tensor,
    type_names: list[str],
    layer_indices: list[int],
) -> dict[tuple, torch.Tensor]:
    """L2-normalised mean activation per (act_type, layer, ts_type)."""
    profiles: dict[tuple, torch.Tensor] = {}
    for at in ACT_TYPES:
        for li in layer_indices:
            key = (at, li)
            if key not in all_activations:
                continue
            acts = all_activations[key]
            for i, name in enumerate(type_names):
                mask = type_labels == i
                if mask.sum() == 0:
                    continue
                p = acts[mask].mean(dim=0)
                p = p / (p.norm() + 1e-8)
                profiles[(at, li, name)] = p
    return profiles


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalise along last dim."""
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _gather_profiles(profiles, at, li, type_names):
    """Stack profiles for all types into (n_valid, D), return valid type names."""
    vecs, names = [], []
    for ts in type_names:
        p = profiles.get((at, li, ts))
        if p is not None:
            vecs.append(p)
            names.append(ts)
    if not vecs:
        return None, []
    return torch.stack(vecs), names  # (n_valid, D)


# ── per-batch matching (batched across types) ────────────────────────────────

def match_batch_tokens(
    batch_acts: dict[tuple, torch.Tensor],
    profiles: dict[tuple, torch.Tensor],
    tokenizer,
    input_ids: torch.Tensor,
    passage_indices: list[int],
    trackers: dict[tuple, TopKTracker],
    layer_indices: list[int],
    type_names: list[str],
    per_passage_top: int = 3,
):
    B, T = input_ids.shape
    for at in ACT_TYPES:
        for li in layer_indices:
            acts = batch_acts.get((at, li))
            if acts is None:
                continue
            prof_mat, valid_types = _gather_profiles(profiles, at, li, type_names)
            if prof_mat is None:
                continue
            acts_n = _normalize(acts)
            all_sims = (acts_n @ prof_mat.T).cpu()  # (B, T, n_valid)

            for ti, ts_type in enumerate(valid_types):
                sims = all_sims[:, :, ti]
                tracker = trackers[(at, li, ts_type, "token")]
                for b in range(B):
                    top_vals, top_idxs = sims[b].topk(min(per_passage_top, T))
                    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
                        cs = max(0, idx - 5)
                        ce = min(T, idx + 6)
                        tracker.push(TokenMatch(
                            similarity=val,
                            text_context=tokenizer.decode(input_ids[b, cs:ce].tolist()),
                            token_text=tokenizer.decode([input_ids[b, idx].item()]),
                            position=idx,
                            passage_idx=passage_indices[b],
                            layer=li,
                            ts_type=ts_type,
                            act_type=at,
                        ))


def match_batch_windows(
    batch_acts: dict[tuple, torch.Tensor],
    profiles: dict[tuple, torch.Tensor],
    tokenizer,
    input_ids: torch.Tensor,
    passage_indices: list[int],
    trackers: dict[tuple, TopKTracker],
    layer_indices: list[int],
    type_names: list[str],
    window_size: int = 50,
    stride: int = 25,
    per_passage_top: int = 2,
):
    B, T = input_ids.shape
    if T < window_size:
        return
    for at in ACT_TYPES:
        for li in layer_indices:
            acts = batch_acts.get((at, li))
            if acts is None:
                continue
            prof_mat, valid_types = _gather_profiles(profiles, at, li, type_names)
            if prof_mat is None:
                continue
            pooled = torch.nn.functional.avg_pool1d(
                acts.transpose(1, 2), kernel_size=window_size, stride=stride,
            ).transpose(1, 2)
            pooled_n = _normalize(pooled)
            all_wsims = (pooled_n @ prof_mat.T).cpu()  # (B, n_win, n_valid)

            for ti, ts_type in enumerate(valid_types):
                wsims = all_wsims[:, :, ti]
                tracker = trackers[(at, li, ts_type, "window")]
                for b in range(B):
                    top_vals, top_idxs = wsims[b].topk(min(per_passage_top, wsims.shape[1]))
                    for val, widx in zip(top_vals.tolist(), top_idxs.tolist()):
                        start = widx * stride
                        end = min(start + window_size, T)
                        tracker.push(WindowMatch(
                            similarity=val,
                            text=tokenizer.decode(input_ids[b, start:end].tolist()),
                            start_pos=start,
                            passage_idx=passage_indices[b],
                            layer=li,
                            ts_type=ts_type,
                            act_type=at,
                        ))
