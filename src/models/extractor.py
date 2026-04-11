"""
Load all three models and extract per-layer hidden states for a batch of windows.

Key design decisions:
- Uses forward hooks to capture ONLY the needed layers (not output_hidden_states=True
  which materializes all 28 layers simultaneously).
- PT extraction sub-batches windows since PT sequences are ~3x longer than FT/RI
  (~1536 tokens/window vs 512 tokens/window).
- Token spans (which token indices correspond to each timestep) are cached by
  (series_idx, offset) so tokenization only runs once per unique window.
- Mean-pooling over sub-token spans is vectorized with scatter_add.
- All 3 models loaded as base transformers (no lm_head) to avoid OOM.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.tokenize import (
    normalize_window,
    uniform_bin_tokenize,
    window_to_text,
    get_timestep_token_spans,
)
from src.config import Config


def _make_layer_hook(storage: dict, layer_idx: int):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        storage[layer_idx] = hidden.detach()
    return hook


def _extract_with_hooks(
    model,
    input_ids: torch.Tensor,
    layer_indices: list[int],
    attention_mask: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    """
    Run a forward pass capturing hidden states at specific layers.
    Truncates the model to only run through the deepest needed layer,
    saving compute for early-layer assignments (e.g. GPU 0 needs only 7/28 layers).
    """
    max_layer = max(layer_indices)
    captured: dict[int, torch.Tensor] = {}
    handles = [
        model.layers[idx].register_forward_hook(_make_layer_hook(captured, idx))
        for idx in layer_indices
    ]
    # Temporarily truncate to max_layer+1 layers to skip unnecessary computation
    original_layers = model.layers
    model.layers = original_layers[: max_layer + 1]
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        model.layers = original_layers
        for h in handles:
            h.remove()
    return captured


def _vectorized_pool(
    hs: torch.Tensor,          # (B, L, H)
    idx_map: torch.Tensor,     # (B, L) — timestep index per token, -1 for padding
    T: int,
) -> torch.Tensor:             # (B, T, H)
    """Mean-pool hidden states to timestep vectors via scatter_add."""
    B, L, H = hs.shape
    valid   = idx_map >= 0                                              # (B, L)
    safe_idx = idx_map.clamp(min=0).unsqueeze(-1).expand(B, L, H)      # (B, L, H)

    result = torch.zeros(B, T, H, dtype=hs.dtype, device=hs.device)
    count  = torch.zeros(B, T, 1, dtype=hs.dtype, device=hs.device)

    result.scatter_add_(1, safe_idx, hs * valid.unsqueeze(-1))
    count.scatter_add_(1, idx_map.clamp(min=0).unsqueeze(-1),
                       valid.unsqueeze(-1).to(hs.dtype))

    return result / count.clamp(min=1.0)


def _build_idx_map(all_spans: list[list[tuple[int, int]]], L: int,
                   T: int, device: torch.device) -> torch.Tensor:
    """Build (B, L) token-to-timestep index tensor from a list of span lists."""
    B = len(all_spans)
    idx = torch.full((B, L), -1, dtype=torch.long, device=device)
    for b, spans in enumerate(all_spans):
        for t, (start, end) in enumerate(spans):
            end = min(end, L)
            if start < end:
                idx[b, start:end] = t
    return idx


class ModelExtractor:
    """
    Loads PT, FT, RI base transformer models and extracts per-layer hidden states.
    """

    def __init__(
        self,
        config: Config,
        device: torch.device,
        hf_token: str | None = None,
        pt_sub_batch: int = 16,
    ):
        self.config       = config
        self.device       = device
        self.pt_sub_batch = pt_sub_batch
        self._span_cache: dict[tuple, list] = {}   # (series_idx, offset) → spans
        dtype = torch.bfloat16

        print(f"[{device}] Loading PT model: {config.model_pt}")
        self.pt_model = AutoModelForCausalLM.from_pretrained(
            config.model_pt, dtype=dtype, token=hf_token,
        ).model.to(device).eval()

        print(f"[{device}] Loading FT model: {config.model_ft}")
        self.ft_model = AutoModelForCausalLM.from_pretrained(
            config.model_ft, dtype=dtype, token=hf_token,
        ).model.to(device).eval()

        print(f"[{device}] Loading RI model: {config.model_ri}")
        self.ri_model = AutoModelForCausalLM.from_pretrained(
            config.model_ri, dtype=dtype, token=hf_token,
        ).model.to(device).eval()

        print(f"[{device}] Loading PT tokenizer")
        self.pt_tokenizer = AutoTokenizer.from_pretrained(
            config.model_pt, token=hf_token
        )
        if self.pt_tokenizer.pad_token is None:
            self.pt_tokenizer.pad_token = self.pt_tokenizer.eos_token

    def _get_spans(self, window: dict) -> list[tuple[int, int]]:
        """Return cached token spans for this window, computing if needed."""
        key = (window["series_idx"], window["offset"])
        if key not in self._span_cache:
            norm, _, _ = normalize_window(window["values"])
            text  = window_to_text(norm)
            spans = get_timestep_token_spans(text, norm, self.pt_tokenizer)
            self._span_cache[key] = spans
        return self._span_cache[key]

    @torch.no_grad()
    def extract_ft_ri(
        self,
        windows: list[dict],
        layer_indices: list[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        cfg = self.config
        B, T = len(windows), cfg.context_length

        all_tokens = []
        for w in windows:
            norm, _, _ = normalize_window(w["values"])
            all_tokens.append(uniform_bin_tokenize(norm, cfg.n_bins, cfg.bin_low, cfg.bin_high))
        input_ids = torch.from_numpy(np.stack(all_tokens)).to(dtype=torch.long, device=self.device)

        ft_raw = _extract_with_hooks(self.ft_model, input_ids, layer_indices)
        ri_raw = _extract_with_hooks(self.ri_model, input_ids, layer_indices)

        ft_acts = {i: ft_raw[i].reshape(B * T, -1).float() for i in layer_indices}
        ri_acts = {i: ri_raw[i].reshape(B * T, -1).float() for i in layer_indices}
        return ft_acts, ri_acts

    @torch.no_grad()
    def extract_pt(
        self,
        windows: list[dict],
        layer_indices: list[int],
    ) -> dict[int, torch.Tensor]:
        cfg = self.config
        B, T = len(windows), cfg.context_length

        layer_chunks: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}

        for sub_start in range(0, B, self.pt_sub_batch):
            sub = windows[sub_start : sub_start + self.pt_sub_batch]
            sub_B = len(sub)

            texts, all_spans = [], []
            for w in sub:
                norm, _, _ = normalize_window(w["values"])
                texts.append(window_to_text(norm))
                all_spans.append(self._get_spans(w))

            enc = self.pt_tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            L = input_ids.size(1)

            # Build token→timestep index once per sub-batch (shared across layers)
            idx_map = _build_idx_map(all_spans, L, T, self.device)

            captured = _extract_with_hooks(
                self.pt_model, input_ids, layer_indices, attention_mask
            )

            for idx in layer_indices:
                hs     = captured[idx].float()                       # (sub_B, L, H)
                pooled = _vectorized_pool(hs, idx_map, T)            # (sub_B, T, H)
                layer_chunks[idx].append(pooled)

        pt_acts = {
            i: torch.cat(layer_chunks[i], dim=0).reshape(B * T, -1)
            for i in layer_indices
        }
        return pt_acts

    @torch.no_grad()
    def extract_all(
        self,
        windows: list[dict],
        layer_indices: list[int] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        if layer_indices is None:
            layer_indices = list(range(self.config.num_layers))
        ft_acts, ri_acts = self.extract_ft_ri(windows, layer_indices)
        pt_acts = self.extract_pt(windows, layer_indices)
        return pt_acts, ft_acts, ri_acts
