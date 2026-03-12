"""
DDPM noise scheduler with cosine or linear beta schedule.
Used for both training (add_noise) and DDPM sampling (step).
DDIM sampling is in ddim.py and shares this scheduler's buffers.
"""
from __future__ import annotations

import math
import torch
import numpy as np
from typing import Literal


class DDPMScheduler:
    """
    Scheduler implementing:
      q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t)·I)   [forward process]
      p(x_{t-1}|x_t)                                [reverse process]

    Reference: Ho et al. 2020, Nichol & Dhariwal 2021 (cosine schedule).
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: Literal["linear", "cosine"] = "cosine",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: Literal["epsilon", "sample"] = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range
        self.prediction_type = prediction_type

        # ── beta schedule ────────────────────────────────────────────── #
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float64)
        elif beta_schedule == "cosine":
            betas = self._cosine_betas(num_train_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {beta_schedule}")

        self.betas = betas.float()
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])

        # ᾱ_t convenience quantities
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).sqrt()
        self.log_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod).log()
        self.sqrt_recip_alphas_cumprod = (1.0 / self.alphas_cumprod).sqrt()
        self.sqrt_recipm1_alphas_cumprod = (1.0 / self.alphas_cumprod - 1.0).sqrt()

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * self.alphas_cumprod_prev.sqrt() / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * alphas.sqrt() / (1.0 - self.alphas_cumprod)
        )

    # ------------------------------------------------------------------ #
    # Static helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from Nichol & Dhariwal 2021."""
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(1e-4, 0.9999)

    # ------------------------------------------------------------------ #
    # Forward process                                                      #
    # ------------------------------------------------------------------ #

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε.

        Args:
            x_start:   (B, D)   clean action chunk
            noise:     (B, D)   standard Gaussian
            timesteps: (B,)     int64 timestep indices
        """
        device = timesteps.device
        # view(-1, 1, ..., 1) with (ndim-1) trailing ones so shapes broadcast correctly.
        # e.g. x_start (B, D) → s_a (B, 1); x_start (B, T, D) → s_a (B, 1, 1)
        trail = [1] * (x_start.ndim - 1)
        s_a = self.sqrt_alphas_cumprod[timesteps].to(device).view(-1, *trail).float()
        s_b = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device).view(-1, *trail).float()
        return s_a * x_start + s_b * noise

    # ------------------------------------------------------------------ #
    # Reverse process (DDPM step)                                         #
    # ------------------------------------------------------------------ #

    def step(
        self,
        model_output: torch.Tensor,   # noise or x_0 prediction (B, D)
        t: int,
        x_t: torch.Tensor,            # (B, D)
    ) -> torch.Tensor:
        """
        Single DDPM reverse step: sample x_{t-1} from p(x_{t-1} | x_t).
        """
        device = x_t.device

        # Reconstruct x_0
        if self.prediction_type == "epsilon":
            sqrt_recip = self.sqrt_recip_alphas_cumprod[t].to(device)
            sqrt_recip_m1 = self.sqrt_recipm1_alphas_cumprod[t].to(device)
            pred_x0 = sqrt_recip * x_t - sqrt_recip_m1 * model_output
        else:
            pred_x0 = model_output

        if self.clip_sample:
            pred_x0 = pred_x0.clamp(-self.clip_sample_range, self.clip_sample_range)

        # Posterior mean
        coef1 = self.posterior_mean_coef1[t].to(device)
        coef2 = self.posterior_mean_coef2[t].to(device)
        pred_mean = coef1 * pred_x0 + coef2 * x_t

        # Add noise (skip at t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            log_var = self.posterior_log_variance_clipped[t].to(device)
            pred_sample = pred_mean + (0.5 * log_var).exp() * noise
        else:
            pred_sample = pred_mean

        return pred_sample

    # ------------------------------------------------------------------ #
    # Inference timestep sequence                                         #
    # ------------------------------------------------------------------ #

    def get_timesteps(self, num_inference_steps: int | None = None) -> list[int]:
        """Return decreasing timestep sequence for inference."""
        n = num_inference_steps or self.num_train_timesteps
        step = self.num_train_timesteps // n
        return list(range(self.num_train_timesteps - 1, -1, -step))[:n]

    def to(self, device: torch.device | str) -> "DDPMScheduler":
        """Move all internal tensors to device."""
        for attr in dir(self):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self

