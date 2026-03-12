"""
Evaluation module: rolls out DiffusionPolicy in the environment,
computes success metrics, and optionally records video frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusion.ddpm import DiffusionPolicy
from diffusion.visual_ddpm import VisualDiffusionPolicy
# env.adroit_wrapper requires gymnasium — import lazily to avoid hard dep
from env.video_recorder import VideoRecorder
from utils.normalizer import RunningNormalizer


# ─────────────────────────────────────────────────────────────────────────── #
# Evaluator                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class Evaluator:
    """
    Runs policy rollouts and computes evaluation metrics.

    The policy predicts an *action chunk* of shape (action_horizon, act_dim).
    We execute the entire chunk before re-querying the policy (receding-horizon).
    """

    def __init__(
        self,
        policy: DiffusionPolicy,
        env: AdroitWrapper,
        num_ddim_steps: int = 20,
        eta: float = 0.0,
        device: str = "cuda",
    ):
        self.policy = policy.to(device)
        self.policy.eval()
        self.env = env
        self.num_ddim_steps = num_ddim_steps
        self.eta = eta
        self.device = torch.device(device)

    # ------------------------------------------------------------------ #
    # Single rollout                                                      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def rollout(
        self,
        max_steps: int = 200,
        seed: int = 0,
        record_frames: bool = False,
    ) -> dict:
        """
        Execute one episode.

        Returns:
            dict with keys: total_reward, success, length, frames (list | None)
        """
        obs, info = self.env.reset(seed=seed)
        total_reward = 0.0
        success = False
        frames = [] if record_frames else None
        step = 0

        while step < max_steps:
            # Prepare tensors for predict_action
            images_t = torch.from_numpy(obs["images"]).to(self.device)   # (1, oh, 3, H, W)
            states_t = torch.from_numpy(obs["states"]).to(self.device)   # (1, oh*state_dim)

            # Predict action chunk
            action_chunk = self.policy.predict_action(
                images_t,
                states_t,
                num_ddim_steps=self.num_ddim_steps,
                eta=self.eta,
            )  # (1, action_horizon, act_dim)
            action_chunk = action_chunk[0].cpu().numpy()  # (action_horizon, act_dim)

            # Execute chunk in env (receding-horizon)
            for t in range(self.policy.action_horizon):
                if step >= max_steps:
                    break
                action = action_chunk[t]
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                step += 1

                if record_frames:
                    frame = self.env.render()
                    if frame is not None:
                        frames.append(frame)

                if self.env.is_success(info):
                    success = True

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

        return {
            "total_reward": total_reward,
            "success": success,
            "length": step,
            "frames": frames,
        }

    # ------------------------------------------------------------------ #
    # Batch evaluation                                                    #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        num_episodes: int = 50,
        max_steps: int = 200,
        seed_offset: int = 0,
        record_video: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Run `num_episodes` rollouts and aggregate metrics.

        Args:
            record_video: if a path is given, saves the *first* rollout as mp4.
        """
        rewards, lengths, successes = [], [], []
        saved_video = False

        pbar = tqdm(range(num_episodes), desc="Evaluating", disable=not verbose)
        for i in pbar:
            record = record_video is not None and not saved_video
            result = self.rollout(max_steps=max_steps, seed=seed_offset + i, record_frames=record)

            rewards.append(result["total_reward"])
            lengths.append(result["length"])
            successes.append(result["success"])

            if record and result["frames"]:
                recorder = VideoRecorder(record_video, fps=30)
                for frame in result["frames"]:
                    recorder.add_frame(frame)
                recorder.save()
                saved_video = True

            pbar.set_postfix(
                reward=f"{result['total_reward']:.2f}",
                success=result["success"],
                sr=f"{np.mean(successes):.2%}",
            )

        metrics = {
            "success_rate":   float(np.mean(successes)),
            "mean_reward":    float(np.mean(rewards)),
            "std_reward":     float(np.std(rewards)),
            "max_reward":     float(np.max(rewards)),
            "mean_length":    float(np.mean(lengths)),
            "num_episodes":   num_episodes,
        }
        return metrics

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _encode_obs(self, obs: dict) -> torch.Tensor:
        """Encode image and state observations using the policy's encoder."""
        images_t = torch.from_numpy(obs["images"]).to(self.device)
        states_t = torch.from_numpy(obs["states"]).to(self.device)

        # Normalization (state only)
        if self.policy.state_normalizer is not None:
            states_t = self.policy.state_normalizer.normalize(states_t)

        return self.policy.encode_obs(images_t, states_t)


# ─────────────────────────────────────────────────────────────────────────── #
# Checkpoint loader                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

def load_policy(
    checkpoint_path: str | Path,
    cfg: OmegaConf, # Pass the entire config
    device: str = "cuda",
) -> nn.Module:
    """
    Reconstruct a DiffusionPolicy (or VisualDiffusionPolicy) from a saved checkpoint.
    Restores normalizer states automatically. Automatically detects policy type.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine policy type from checkpoint or config
    policy_type = ckpt["policy_type"] if "policy_type" in ckpt else "DiffusionPolicy" # Assuming VisualDiffusionPolicy has 'image_encoder'
    if hasattr(cfg.model, "encoder_type") and cfg.model.encoder_type is not None:
        policy_type = "VisualDiffusionPolicy"

    if policy_type == "VisualDiffusionPolicy":
        policy = VisualDiffusionPolicy(
            state_dim=cfg.dataset.state_dim,
            act_dim=cfg.dataset.act_dim,
            action_horizon=cfg.dataset.action_horizon,
            obs_horizon=cfg.dataset.obs_horizon,
            encoder_type=cfg.model.encoder_type,
            img_emb_dim=cfg.model.img_emb_dim,
            encoder_frozen=cfg.model.encoder_frozen,
            encoder_pretrained=cfg.model.encoder_pretrained,
            state_emb_dim=cfg.model.state_emb_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            time_emb_dim=cfg.model.time_emb_dim,
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=cfg.diffusion.beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=cfg.diffusion.clip_sample,
            clip_sample_range=cfg.diffusion.clip_sample_range,
            device=device,
        )
    else:
        policy = DiffusionPolicy(
            obs_dim=cfg.dataset.obs_dim,
            act_dim=cfg.dataset.act_dim,
            action_horizon=cfg.dataset.action_horizon,
            obs_horizon=cfg.dataset.obs_horizon,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            time_emb_dim=cfg.model.time_emb_dim,
            num_train_timesteps=cfg.diffusion.num_train_timesteps,
            beta_schedule=cfg.diffusion.beta_schedule,
            prediction_type=cfg.diffusion.prediction_type,
            clip_sample=cfg.diffusion.clip_sample,
            clip_sample_range=cfg.diffusion.clip_sample_range, # Add missing parameter
        )

    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    # Restore normalizers
    if ckpt.get("state_normalizer") is not None: # Use state_normalizer
        policy.state_normalizer = RunningNormalizer()
        policy.state_normalizer.load_state_dict(ckpt["state_normalizer"])

    if ckpt.get("act_normalizer") is not None:
        policy.act_normalizer = RunningNormalizer()
        policy.act_normalizer.load_state_dict(ckpt["act_normalizer"])

    print(f"[load_policy] Loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return policy
