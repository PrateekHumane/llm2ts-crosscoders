"""Component ablation utilities for causal circuit identification."""
import torch
import torch.nn.functional as F
from contextlib import contextmanager


@contextmanager
def ablate_head(model, layer_idx: int, head_idx: int,
                num_heads: int = 16, head_dim: int = 128):
    """Zero-ablate one attention head via o_proj pre-hook."""
    o_proj = model.model.layers[layer_idx].self_attn.o_proj

    def hook(module, args):
        x = args[0].clone()                        # (B, T, num_heads*head_dim)
        B, T, _ = x.shape
        x = x.view(B, T, num_heads, head_dim)
        x[:, :, head_idx] = 0
        return (x.view(B, T, -1),)

    handle = o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_mlp(model, layer_idx: int):
    """Zero-ablate an MLP layer (residual passes through unchanged)."""
    mlp = model.model.layers[layer_idx].mlp

    def hook(module, inp, out):
        return torch.zeros_like(out)

    handle = mlp.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def compute_loss(model, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Next-token CE loss per sequence. Returns (B,) tensor on CPU."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1]              # (B, T-1, V)
    targets = input_ids[:, 1:]               # (B, T-1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view(logits.size(0), -1)               # (B, T-1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        return ((loss * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu()
    return loss.mean(1).cpu()
