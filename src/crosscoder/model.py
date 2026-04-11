"""
Crosscoder: shared encoder + per-domain decoders.

Architecture:
  Encoder: Linear(1024→2048) → ReLU → Linear(2048→4096) → TopK(k=64)
  Decoder (×3): Linear(4096→2048) → ReLU → Linear(2048→1024)

Includes AuxK loss for dead-feature recovery (OpenAI, 2024).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.latent_dim),
        )
        self.k = cfg.top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (N, hidden_size) → (z_sparse, pre_activations)."""
        pre = self.net(x)              # (N, latent_dim)
        z = self._topk(pre)            # (N, latent_dim), sparse
        return z, pre

    def _topk(self, x: torch.Tensor) -> torch.Tensor:
        """Zero out all but top-k activations per sample."""
        vals, idx = torch.topk(x, self.k, dim=-1)
        z = torch.zeros_like(x)
        # out-of-place scatter to guarantee gradient flow through vals
        z = z.scatter(-1, idx, F.relu(vals))
        return z


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Linear(cfg.mlp_hidden, cfg.hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    @torch.no_grad()
    def normalize_columns(self):
        """Unit-normalize each column of the final linear layer's weight matrix."""
        w = self.net[-1].weight  # (hidden_size, mlp_hidden)
        norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        w.div_(norms)


class Crosscoder(nn.Module):
    """
    Shared encoder + three domain-specific decoders.
    Maintains per-domain running normalization stats (not learned parameters).
    """
    DOMAINS = ("PT", "FT", "RI")

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(cfg)
        self.decoders = nn.ModuleDict({d: Decoder(cfg) for d in self.DOMAINS})

        # Per-domain running stats — updated via EMA, not trained
        for d in self.DOMAINS:
            self.register_buffer(f"{d}_mean", torch.zeros(cfg.hidden_size))
            self.register_buffer(f"{d}_std",  torch.ones(cfg.hidden_size))
            self.register_buffer(f"{d}_initialized", torch.tensor(False))

    # ── Normalization ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def update_stats(self, x: torch.Tensor, domain: str, momentum: float = 0.01):
        """EMA update of per-domain mean/std from a batch of activations."""
        batch_mean = x.mean(dim=0)
        batch_std  = x.std(dim=0).clamp(min=1e-6)
        initialized = getattr(self, f"{domain}_initialized")
        if not initialized.item():
            getattr(self, f"{domain}_mean").copy_(batch_mean)
            getattr(self, f"{domain}_std").copy_(batch_std)
            initialized.fill_(True)
        else:
            getattr(self, f"{domain}_mean").lerp_(batch_mean, momentum)
            getattr(self, f"{domain}_std").lerp_(batch_std, momentum)

    def normalize(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        mean = getattr(self, f"{domain}_mean")
        std  = getattr(self, f"{domain}_std")
        return (x - mean) / (std + 1e-8)

    def denormalize(self, x: torch.Tensor, domain: str) -> torch.Tensor:
        mean = getattr(self, f"{domain}_mean")
        std  = getattr(self, f"{domain}_std")
        return x * std + mean

    # ── Forward ───────────────────────────────────────────────────────────────

    def encode_single(self, x: torch.Tensor, domain: str
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize and encode a single domain. Returns (z_sparse, pre_acts, normed)."""
        n = self.normalize(x, domain)
        z, pre = self.encoder(n)
        return z, pre, n

    def forward(
        self,
        x_pt: torch.Tensor,
        x_ft: torch.Tensor,
        x_ri: torch.Tensor,
        update_stats: bool = True,
        dead_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Full forward pass — each domain encoded independently through
        the shared encoder.

        Returns (loss, (z_pt, z_ft, z_ri)).
        dead_mask: (latent_dim,) bool tensor — True for dead features.
        """
        if update_stats:
            self.update_stats(x_pt, "PT")
            self.update_stats(x_ft, "FT")
            self.update_stats(x_ri, "RI")

        z_pt, pre_pt, n_pt = self.encode_single(x_pt, "PT")
        z_ft, pre_ft, n_ft = self.encode_single(x_ft, "FT")
        z_ri, pre_ri, n_ri = self.encode_single(x_ri, "RI")

        # Per-domain reconstruction loss
        recon_pt = self.decoders["PT"](z_pt)
        recon_ft = self.decoders["FT"](z_ft)
        recon_ri = self.decoders["RI"](z_ri)

        loss = (
            F.mse_loss(recon_pt, n_pt)
            + F.mse_loss(recon_ft, n_ft)
            + F.mse_loss(recon_ri, n_ri)
        )

        # AuxK: auxiliary loss on dead features (shared dead mask, per-domain forward)
        if dead_mask is not None and dead_mask.any():
            aux_loss = self._auxk_loss(
                pre_pt, pre_ft, pre_ri,
                n_pt, n_ft, n_ri,
                recon_pt, recon_ft, recon_ri,
                dead_mask,
            )
            loss = loss + self.cfg.auxk_coeff * aux_loss

        return loss, (z_pt, z_ft, z_ri)

    def _auxk_loss(
        self,
        pre_pt: torch.Tensor, pre_ft: torch.Tensor, pre_ri: torch.Tensor,
        n_pt: torch.Tensor, n_ft: torch.Tensor, n_ri: torch.Tensor,
        recon_pt: torch.Tensor, recon_ft: torch.Tensor, recon_ri: torch.Tensor,
        dead_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        AuxK loss (OpenAI, 2024): per-domain top-k among dead features,
        decode, MSE on residual. Shared dead mask across domains.
        """
        k_aux = min(self.cfg.auxk_k, int(dead_mask.sum().item()))
        if k_aux == 0:
            return torch.tensor(0.0, device=pre_pt.device)

        total = torch.tensor(0.0, device=pre_pt.device)
        for pre, n, recon, decoder in [
            (pre_pt, n_pt, recon_pt, self.decoders["PT"]),
            (pre_ft, n_ft, recon_ft, self.decoders["FT"]),
            (pre_ri, n_ri, recon_ri, self.decoders["RI"]),
        ]:
            res = (n - recon).detach()
            pre_dead = pre.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
            vals, idx = torch.topk(pre_dead, k_aux, dim=-1)
            z_aux = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
            total = total + F.mse_loss(decoder(z_aux), res)

        return total

    # ── Post-step constraint ──────────────────────────────────────────────────

    @torch.no_grad()
    def normalize_decoder_columns(self):
        for decoder in self.decoders.values():
            decoder.normalize_columns()

    # ── Dead neuron tracking ──────────────────────────────────────────────────

    def count_dead_neurons(self, activation_counts: torch.Tensor,
                           total_batches: int) -> int:
        threshold = self.cfg.dead_neuron_threshold * total_batches
        return int((activation_counts < threshold).sum().item())
