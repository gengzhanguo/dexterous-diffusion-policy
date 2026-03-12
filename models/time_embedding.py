"""
Sinusoidal positional embedding for diffusion timestep conditioning.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """
    Maps a scalar timestep t ∈ [0, T] to a fixed sinusoidal embedding,
    then projects to `emb_dim` via a two-layer MLP.

    Architecture (standard from DDPM / Ho et al.):
        t  →  [sin(ω_i · t), cos(ω_i · t)]_i  →  Linear → SiLU → Linear
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        assert emb_dim % 2 == 0, "emb_dim must be even"
        self.half_dim = emb_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) long or float tensor of timestep indices.
        Returns:
            emb: (B, emb_dim)
        """
        device = t.device
        half = self.half_dim
        # log-spaced frequencies
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / (half - 1)
        ).float()
        t = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)  # (B, half)
        emb = torch.cat([t.sin(), t.cos()], dim=-1)             # (B, emb_dim)
        return self.mlp(emb)
