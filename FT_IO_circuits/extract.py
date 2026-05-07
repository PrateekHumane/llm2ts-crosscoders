"""Hook-based activation extraction for Qwen3 models."""
import torch
import torch.nn as nn
from contextlib import contextmanager
from transformers import AutoModelForCausalLM


def load_model(path: str, device: str = "cuda:0") -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    return model.to(device).eval()


@contextmanager
def activation_hooks(model, layer_indices: list[int]):
    """Hook MLP intermediate, attention output, and residual stream.

    Captures:
        ("mlp", layer):  down_proj input — (B, T, 3072)
        ("attn", layer): o_proj input   — (B, T, 2048)
        ("res", layer):  layer output   — (B, T, 1024)
    """
    captured: dict[tuple, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    for idx in layer_indices:
        layer = model.model.layers[idx]

        def _mlp(mod, inp, out, li=idx):
            captured[("mlp", li)] = inp[0].detach().float()

        def _attn(mod, inp, out, li=idx):
            captured[("attn", li)] = inp[0].detach().float()

        def _res(mod, inp, out, li=idx):
            h = out[0] if isinstance(out, tuple) else out
            captured[("res", li)] = h.detach().float()

        handles.append(layer.mlp.down_proj.register_forward_hook(_mlp))
        handles.append(layer.self_attn.o_proj.register_forward_hook(_attn))
        handles.append(layer.register_forward_hook(_res))

    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def extract_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    layer_indices: list[int],
    attention_mask: torch.Tensor | None = None,
) -> dict[tuple, torch.Tensor]:
    """Forward pass → captured activations {(type, layer): (B, T, D)}."""
    with activation_hooks(model, layer_indices) as captured:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return dict(captured)
