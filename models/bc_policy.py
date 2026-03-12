"""
Behavioral Cloning (BC) policy for visual observations.

Architecture:
  image (B, obs_h, 3, H, W)  ──► ImageEncoder → img_emb  (B, img_emb_dim)
  state (B, obs_h * state_dim)──► Linear → state_emb                        (B, state_emb_dim)
  concat  ──► obs_emb (B, img_emb_dim + state_emb_dim)
             ↓
  MLP (obs_emb → hidden → action)

This policy directly predicts actions (no diffusion) with MSE loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import ConditionalMLP # Reusing blocks from here
from models.image_encoder import build_image_encoder
from utils.normalizer import RunningNormalizer


class BCPolicy(nn.Module):
    """
    Behavioral Cloning policy with image + proprioception observations.
    Directly predicts actions using an MLP.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        action_horizon: int = 16,
        obs_horizon: int = 1,
        # Image encoder
        encoder_type: str = "resnet18",
        img_emb_dim: int = 256,
        encoder_frozen: bool = True,
        encoder_pretrained: bool = True,
        # State encoder
        state_emb_dim: int = 64,
        # Text encoder (optional)
        text_emb_dim: int = 0,
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        clip_frozen: bool = True,
        # Action prediction MLP
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.act_flat_dim = act_dim * action_horizon
        self.obs_emb_dim = img_emb_dim + state_emb_dim + text_emb_dim   # combined obs embedding dim

        # ── Image encoder ──────────────────────────────────────────────── #
        self.image_encoder = build_image_encoder(
            encoder_type=encoder_type,
            emb_dim=img_emb_dim,
            frozen=encoder_frozen,
            pretrained=encoder_pretrained,
        )

        # ── State encoder ──────────────────────────────────────────────── #
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim * obs_horizon, state_emb_dim),
            nn.SiLU(),
            nn.Linear(state_emb_dim, state_emb_dim),
        )

        # ── Text encoder (optional) ────────────────────────────────────── #
        self.text_encoder: nn.Module | None = None
        if text_emb_dim > 0:
            from models.text_encoder import CLIPTextEncoder
            self.text_encoder = CLIPTextEncoder(
                clip_model_name=clip_model_name,
                pretrained=clip_pretrained,
                text_emb_dim=text_emb_dim,
                freeze_clip=clip_frozen,
            )

        # ── Action prediction MLP ──────────────────────────────────────── #
        # Reusing ConditionalMLP blocks for hidden layers
        layers = [
            nn.Linear(self.obs_emb_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, self.act_flat_dim))
        self.action_pred_mlp = nn.Sequential(*layers)

        # Normalizers (set externally)
        self.state_normalizer: RunningNormalizer | None = None
        self.act_normalizer:   RunningNormalizer | None = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Observation encoding                                                #
    # ------------------------------------------------------------------ #

    def encode_obs(
        self,
        images: torch.Tensor,   # (B, obs_horizon, 3, H, W)
        states: torch.Tensor,   # (B, obs_horizon * state_dim)
        texts: Optional[list[str]] = None, # (B,) list of strings
    ) -> torch.Tensor:
        """Encode image + state (+ text) → single obs embedding (B, obs_emb_dim)."""
        B, T, C, H, W = images.shape

        imgs_flat = images.view(B * T, C, H, W)
        img_embs  = self.image_encoder(imgs_flat)
        img_embs  = img_embs.view(B, T, -1).mean(dim=1)

        state_emb = self.state_encoder(states)

        parts = [img_embs, state_emb]
        if self.text_encoder and texts:
            text_emb = self.text_encoder(texts)  # (B, text_emb_dim)
            parts.append(text_emb)

        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        images:  torch.Tensor,  # (B, obs_horizon, 3, H, W)
        states:  torch.Tensor,  # (B, obs_horizon * state_dim), normalized
        actions: torch.Tensor,  # (B, act_flat_dim), normalized
        texts: Optional[list[str]] = None,
    ) -> torch.Tensor:
        obs_emb = self.encode_obs(images, states, texts)       # (B, obs_emb_dim)
        pred_actions = self.action_pred_mlp(obs_emb)    # (B, act_flat_dim)
        return F.mse_loss(pred_actions, actions)

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,   # (B, obs_horizon, 3, H, W)
        states: torch.Tensor,   # (B, obs_horizon * state_dim), raw (will normalize)
        texts: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """
        Returns (B, action_horizon, act_dim) — denormalized actions.
        """
        B = images.shape[0]
        device = images.device

        # Normalize state
        if self.state_normalizer is not None:
            states = self.state_normalizer.normalize(states)

        obs_emb = self.encode_obs(images, states, texts)       # (B, obs_emb_dim)
        pred_actions_flat = self.action_pred_mlp(obs_emb) # (B, act_flat_dim)

        # Denormalize
        if self.act_normalizer is not None:
            pred_actions_flat = self.act_normalizer.denormalize(pred_actions_flat)

        return pred_actions_flat.view(B, self.action_horizon, self.act_dim)

    # ------------------------------------------------------------------ #
    # Misc                                                                #
    # ------------------------------------------------------------------ #

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_total(self) -> int:
        return sum(p.numel() for p in self.parameters())
