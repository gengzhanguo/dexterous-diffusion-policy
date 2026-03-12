"""
DDIM (Denoising Diffusion Implicit Models) sampler.

Reference: Song et al. 2020 - https://arxiv.org/abs/2010.02502
Key idea: skip intermediate timesteps while maintaining the reverse mapping.
At eta=0 the process is fully deterministic.
"""
from __future__ import annotations

import torch
import numpy as np
from .noise_scheduler import DDPMScheduler


class DDIMSampler:
    """
    Deterministic/stochastic sampler using the DDIM update rule.

    Usage:
        sampler = DDIMSampler(scheduler, num_inference_steps=20)
        x = sampler.sample(noise_pred_fn, obs, shape=(B, act_flat_dim))
    """

    def __init__(
        self,
        scheduler: DDPMScheduler,
        num_inference_steps: int = 20,
        eta: float = 0.0,          # 0 = deterministic, 1 = DDPM-equivalent
    ):
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.timesteps = self._get_timesteps()

    # ------------------------------------------------------------------ #
    # Timestep selection                                                   #
    # ------------------------------------------------------------------ #

    def _get_timesteps(self) -> list[int]:
        """Uniformly spaced timesteps from T-1 → 0."""
        T = self.scheduler.num_train_timesteps
        step_ratio = T // self.num_inference_steps
        timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()
        timesteps = (T - 1 - timesteps).astype(int).tolist()
        return sorted(timesteps, reverse=True)

    # ------------------------------------------------------------------ #
    # DDIM update rule                                                    #
    # ------------------------------------------------------------------ #

    def step(
        self,
        model_output: torch.Tensor,   # (B, D) — predicted noise or x_0
        t: int,
        t_prev: int,
        x_t: torch.Tensor,            # (B, D) — current noisy sample
    ) -> torch.Tensor:
        """
        Single DDIM step: x_t → x_{t_prev}.
        """
        device = x_t.device
        sched = self.scheduler

        alpha_prod_t      = sched.alphas_cumprod[t].to(device)
        alpha_prod_t_prev = sched.alphas_cumprod[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0, device=device)

        beta_prod_t = 1.0 - alpha_prod_t

        # Predict x_0
        if sched.prediction_type == "epsilon":
            pred_x0 = (x_t - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        else:
            pred_x0 = model_output

        if sched.clip_sample:
            pred_x0 = pred_x0.clamp(-sched.clip_sample_range, sched.clip_sample_range)

        # Direction pointing to x_t
        sigma = (
            self.eta
            * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)).sqrt()
            * (1 - alpha_prod_t / alpha_prod_t_prev).sqrt()
        )

        dir_xt = (1 - alpha_prod_t_prev - sigma ** 2).clamp(min=0).sqrt() * model_output

        # DDIM update
        noise = torch.randn_like(x_t) if sigma > 0 else torch.zeros_like(x_t)
        x_prev = alpha_prod_t_prev.sqrt() * pred_x0 + dir_xt + sigma * noise
        return x_prev

    # ------------------------------------------------------------------ #
    # Full denoising loop                                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def sample(
        self,
        noise_pred_fn,            # callable(x_t, obs, t_tensor) → noise_pred
        obs: torch.Tensor,        # (B, obs_flat_dim)
        shape: tuple[int, ...],   # (B, act_flat_dim)
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """
        Run full DDIM reverse chain from pure Gaussian noise → clean actions.

        Args:
            noise_pred_fn: the network's forward method, signature:
                           (noisy_actions, obs, t) → noise_pred
            obs:           conditioning observation, already on device
            shape:         desired output shape, typically (B, act_flat_dim)
            device:        target device
        Returns:
            x0: (B, act_flat_dim) — denoised (normalized) action chunk
        """
        x = torch.randn(shape, device=device)
        timesteps = self.timesteps

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)
            noise_pred = noise_pred_fn(x, obs, t_tensor)

            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.step(noise_pred, t, t_prev, x)

        return x
