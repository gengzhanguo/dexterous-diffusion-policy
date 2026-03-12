"""
MLP-based conditional noise prediction network for diffusion policy.

Architecture
────────────
  obs  ──► obs_encoder (MLP)   ──►─────────────┐
  t    ──► time_emb (Sinusoidal)──►─────────────┤
  noisy_actions ──►────────────────────────────►├──► cat ──► denoiser (MLP) ──► noise_pred
                                                └─────────────────────────────────────────┘

The denoiser is a deep residual MLP:
  input → Linear → [Block × N] → Linear → output
Each block: LayerNorm → Linear → SiLU → Linear (+residual)
Conditioning is injected additively via a projection of (obs_emb + time_emb).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from .time_embedding import SinusoidalPosEmb


# ─────────────────────────────────────────────────────────────────────────── #
# Building blocks                                                              #
# ─────────────────────────────────────────────────────────────────────────── #

class ResidualBlock(nn.Module):
    """
    LayerNorm → Linear(dim, dim) → SiLU → Linear(dim, dim) + skip
    Optionally adds a conditioning signal of shape (B, cond_dim).
    """

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.act = nn.SiLU()
        self.cond_proj = nn.Linear(cond_dim, dim * 2)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, dim)
            cond: (B, cond_dim) — conditioning signal (obs + time)
        """
        h = self.norm(x)
        h = self.fc1(h)
        h = h + self.cond_proj(cond)          # additive conditioning
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        return x + h                           # residual


# ─────────────────────────────────────────────────────────────────────────── #
# Main network                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

class ConditionalMLP(nn.Module):
    """
    Predicts the noise added to a (flattened) action chunk, conditioned on:
      - a stacked observation vector  (obs_dim,)
      - a diffusion timestep          scalar

    Inputs
    ──────
    noisy_actions : (B, act_dim * action_horizon)   — flattened noisy chunk
    obs           : (B, obs_dim * obs_horizon)       — stacked observations
    t             : (B,)                             — integer timestep

    Output
    ──────
    noise_pred    : (B, act_dim * action_horizon)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_horizon: int,
        obs_horizon: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.act_flat_dim = act_dim * action_horizon
        self.obs_flat_dim = obs_dim * obs_horizon
        cond_dim = hidden_dim + time_emb_dim

        # Timestep embedding
        self.time_emb = SinusoidalPosEmb(time_emb_dim)

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.obs_flat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection for noisy actions
        self.input_proj = nn.Linear(self.act_flat_dim, hidden_dim)

        # Residual denoiser blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, cond_dim, dropout) for _ in range(num_layers)]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, self.act_flat_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, act_dim * H)
        obs: torch.Tensor,             # (B, obs_dim * obs_horizon)
        t: torch.Tensor,               # (B,) int64
    ) -> torch.Tensor:
        # Build conditioning vector: [obs_emb || time_emb]
        obs_emb = self.obs_encoder(obs)          # (B, hidden_dim)
        t_emb = self.time_emb(t)                 # (B, time_emb_dim)
        cond = torch.cat([obs_emb, t_emb], dim=-1)  # (B, cond_dim)

        # Denoiser
        h = self.input_proj(noisy_actions)       # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, cond)

        return self.output_proj(h)               # (B, act_flat_dim)
