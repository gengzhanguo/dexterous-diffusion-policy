"""
DiffusionPolicy: wraps the ConditionalMLP + DDPMScheduler into a single
training/inference module.

Training:
  1. Sample random timestep t ~ Uniform[0, T]
  2. Add noise:  x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
  3. Predict ε̂ = net(x_t, obs, t)
  4. Loss:       MSE(ε, ε̂)

Inference:
  - DDIM denoising from x_T ~ N(0,I) using the trained net
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import ConditionalMLP
from diffusion.noise_scheduler import DDPMScheduler
from diffusion.ddim import DDIMSampler
from utils.normalizer import RunningNormalizer


class DiffusionPolicy(nn.Module):
    """
    Full diffusion policy: noise prediction network + scheduler + inference.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_horizon: int = 16,
        obs_horizon: int = 1,
        hidden_dim: int = 512,
        num_layers: int = 4,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
        num_train_timesteps: int = 100,
        beta_schedule: str = "cosine",
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.act_flat_dim = act_dim * action_horizon
        self.obs_flat_dim = obs_dim * obs_horizon

        # Noise prediction network
        self.net = ConditionalMLP(
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_horizon=action_horizon,
            obs_horizon=obs_horizon,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_emb_dim=time_emb_dim,
            dropout=dropout,
        )

        # Scheduler (buffers kept on CPU; moved when needed)
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
        )

        # Normalizers (set externally after fitting to dataset)
        self.obs_normalizer: RunningNormalizer | None = None
        self.act_normalizer: RunningNormalizer | None = None

    # ------------------------------------------------------------------ #
    # Training forward                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        obs: torch.Tensor,     # (B, obs_flat_dim)
        actions: torch.Tensor, # (B, act_flat_dim) — clean, normalized
    ) -> torch.Tensor:
        """Compute DDPM training loss (MSE on noise prediction)."""
        B = obs.shape[0]
        device = obs.device

        # 1. Sample timesteps
        t = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=device)

        # 2. Add noise
        noise = torch.randn_like(actions)
        x_t = self.scheduler.add_noise(actions, noise, t)

        # 3. Predict noise
        noise_pred = self.net(x_t, obs, t)

        # 4. MSE loss
        if self.scheduler.prediction_type == "epsilon":
            target = noise
        else:
            target = actions

        return F.mse_loss(noise_pred, target)

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict_action(
        self,
        obs: torch.Tensor,           # (B, obs_flat_dim) — already normalized
        num_ddim_steps: int = 20,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        Run DDIM denoising to predict a (normalized) action chunk.

        Returns:
            actions: (B, action_horizon, act_dim) — **denormalized**
        """
        B = obs.shape[0]
        device = obs.device

        sampler = DDIMSampler(self.scheduler, num_inference_steps=num_ddim_steps, eta=eta)

        def noise_pred_fn(x_t, obs_, t_):
            return self.net(x_t, obs_, t_)

        x0 = sampler.sample(
            noise_pred_fn,
            obs=obs,
            shape=(B, self.act_flat_dim),
            device=device,
        )   # (B, act_flat_dim)

        # Denormalize
        if self.act_normalizer is not None:
            x0 = self.act_normalizer.denormalize(x0)

        return x0.view(B, self.action_horizon, self.act_dim)

    # ------------------------------------------------------------------ #
    # Parameter count                                                     #
    # ------------------------------------------------------------------ #

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
