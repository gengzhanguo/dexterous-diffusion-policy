"""
VisualDiffusionPolicy: image + state conditioned diffusion policy.

Architecture
────────────
  image (B, obs_h, 3, H, W)  ──► ImageEncoder per frame → mean → img_emb  (B, img_emb_dim)
  state (B, obs_h * state_dim)──► Linear → state_emb                        (B, state_emb_dim)
  concat  ──► obs_emb (B, img_emb_dim + state_emb_dim)
             ↓
  Same ConditionalMLP denoiser as DiffusionPolicy (core unchanged)
  Same DDPM training loss, DDIM inference

The diffusion core (noise scheduler, DDIM sampler) is IDENTICAL to the
low-dim version — only the observation encoder changes.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import ConditionalMLP
from models.image_encoder import build_image_encoder
from diffusion.noise_scheduler import DDPMScheduler
from diffusion.ddim import DDIMSampler
from utils.normalizer import RunningNormalizer


class VisualDiffusionPolicy(nn.Module):
    """
    Diffusion policy with image + proprioception observations.

    Inputs at training:
        images : (B, obs_horizon, 3, H, W)    float32 in [0, 1]
        states : (B, obs_horizon * state_dim)  float32, normalized
        actions: (B, act_dim * action_horizon) float32, normalized

    Inputs at inference:
        Same images/states; outputs (B, action_horizon, act_dim) denormalized.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        action_horizon: int = 16,
        obs_horizon: int = 1,
        img_fusion: str = "mean",          # "mean" | "concat"
        # Image encoder
        encoder_type: str = "resnet18",    # "resnet18" | "resnet50" | "small_cnn"
        img_emb_dim: int = 256,
        encoder_frozen: bool = True,
        encoder_pretrained: bool = True,
        # State encoder
        state_emb_dim: int = 64,
        # Text encoder (optional)
        text_emb_dim: int = 0,             # 0 = no text conditioning
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "openai",
        clip_frozen: bool = True,
        # Denoiser MLP
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
        # Diffusion
        num_train_timesteps: int = 100,
        beta_schedule: str = "cosine",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        device: str = "cuda", # Add device parameter
    ):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.act_flat_dim = act_dim * action_horizon
        self.img_fusion = img_fusion
        # concat fusion: img_emb_dim * obs_horizon; mean fusion: img_emb_dim
        fused_img_dim = img_emb_dim * obs_horizon if img_fusion == "concat" else img_emb_dim
        self.obs_dim = fused_img_dim + state_emb_dim + text_emb_dim

        # ── Image encoder ──────────────────────────────────────────────── #
        self.image_encoder = build_image_encoder(
            encoder_type=encoder_type,
            emb_dim=img_emb_dim,
            frozen=encoder_frozen,
            pretrained=encoder_pretrained,
            dropout=dropout,
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

        # ── Denoiser (identical to low-dim version) ────────────────────── #
        self.net = ConditionalMLP(
            obs_dim=self.obs_dim,
            act_dim=act_dim,
            action_horizon=action_horizon,
            obs_horizon=1,               # obs already embedded to single vector
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )

        # ── Noise scheduler ────────────────────────────────────────────── #
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        ).to(torch.device(device)) # Move scheduler buffers to device


        # Normalizers (set externally)
        self.state_normalizer: RunningNormalizer | None = None
        self.act_normalizer:   RunningNormalizer | None = None

    # ------------------------------------------------------------------ #
    # Observation encoding                                                #
    # ------------------------------------------------------------------ #

    def encode_obs(
        self,
        images: torch.Tensor,   # (B, obs_horizon, 3, H, W)
        states: torch.Tensor,   # (B, obs_horizon * state_dim)
        texts: Optional[list[str]] = None, # (B,) list of strings
    ) -> torch.Tensor:
        """Encode image + state (+ text) → single obs embedding (B, obs_dim)."""
        B, T, C, H, W = images.shape

        # Encode each frame independently, then fuse over obs_horizon
        imgs_flat = images.view(B * T, C, H, W)
        img_embs  = self.image_encoder(imgs_flat)          # (B*T, img_emb_dim)
        img_embs  = img_embs.view(B, T, -1)
        if self.img_fusion == "concat":
            img_embs = img_embs.reshape(B, -1)            # (B, T*img_emb_dim)
        else:
            img_embs = img_embs.mean(dim=1)               # (B, img_emb_dim)

        # Encode state
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
        B = actions.shape[0]
        device = actions.device

        obs_emb = self.encode_obs(images, states, texts)       # (B, obs_dim)

        t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(actions)
        x_t = self.scheduler.add_noise(actions, noise, t)
        noise_pred = self.net(x_t, obs_emb, t)

        target = noise if self.scheduler.prediction_type == "epsilon" else actions
        return F.mse_loss(noise_pred, target)

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(
        self,
        images: torch.Tensor,   # (B, obs_horizon, 3, H, W)
        states: torch.Tensor,   # (B, obs_horizon * state_dim), raw (will normalize)
        texts: Optional[list[str]] = None,
        num_ddim_steps: int = 20,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Returns (B, action_horizon, act_dim) — denormalized actions.
        """
        B = images.shape[0]
        device = images.device

        # Normalize state
        # state_normalizer was fitted on single-step states (state_dim,)
        # but states is (B, obs_horizon * state_dim) — normalize per timestep
        if self.state_normalizer is not None:
            B_s = states.shape[0]
            states = states.view(B_s, self.obs_horizon, self.state_dim)
            states = self.state_normalizer.normalize(states)
            states = states.view(B_s, self.obs_horizon * self.state_dim)

        obs_emb = self.encode_obs(images, states, texts)       # (B, obs_dim)

        sampler = DDIMSampler(self.scheduler, num_inference_steps=num_ddim_steps, eta=eta)

        def noise_pred_fn(x_t, obs_, t_):
            return self.net(x_t, obs_, t_)

        x0 = sampler.sample(noise_pred_fn, obs=obs_emb,
                            shape=(B, self.act_flat_dim), device=device)

        if self.act_normalizer is not None:
            # act_normalizer was fitted on single-step actions (act_dim,)
            # x0 is flat (B, action_horizon * act_dim) — reshape, denorm, flatten back
            x0 = x0.view(B, self.action_horizon, self.act_dim)
            x0 = self.act_normalizer.denormalize(x0)
            return x0

        return x0.view(B, self.action_horizon, self.act_dim)

    # ------------------------------------------------------------------ #
    # Misc                                                                #
    # ------------------------------------------------------------------ #

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_parameters_total(self) -> int:
        return sum(p.numel() for p in self.parameters())
