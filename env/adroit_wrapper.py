"""
Lightweight wrapper around gymnasium-robotics AdroitHand environments.

Key features:
  - Configurable action repeat (frame_skip)
  - Observation / action scaling to [-1, 1]
  - Consistent reset/step API (returns numpy arrays)
  - Lazy render support for video recording
  - Helper factory: make_env(cfg)
"""
from __future__ import annotations

import os
# Headless rendering for WSL2 / servers with no display
# EGL = GPU off-screen; osmesa = CPU software fallback
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from typing import Any
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers AdroitHand envs


# ─────────────────────────────────────────────────────────────────────────── #
# Wrapper                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class AdroitWrapper(gym.Wrapper):
    """
    Wraps an AdroitHand gymnasium environment with:
      - action repeat (frame_skip)
      - obs/action dims exposed as properties
      - optional seed tracking
    """

    def __init__(
        self,
        env_name: str = "AdroitHandDoor-v1",
        frame_skip: int = 2,
        render_mode: str | None = None,
        seed: int = 42,
    ):
        self._env_name = env_name
        self._frame_skip = frame_skip
        self._seed = seed

        raw = gym.make(env_name, render_mode=render_mode)
        super().__init__(raw)

        self._obs_dim = int(np.prod(self.observation_space.shape))
        self._act_dim = int(np.prod(self.action_space.shape))

    # ------------------------------------------------------------------ #
    # Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def act_dim(self) -> int:
        return self._act_dim

    @property
    def env_name(self) -> str:
        return self._env_name

    # ------------------------------------------------------------------ #
    # Core API                                                            #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, **kwargs) -> tuple[np.ndarray, dict]:
        seed = seed if seed is not None else self._seed
        obs, info = self.env.reset(seed=seed, **kwargs)
        return obs.astype(np.float32).flatten(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply action `frame_skip` times, accumulate reward."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}

        for _ in range(self._frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs.astype(np.float32).flatten(), total_reward, terminated, truncated, info

    def is_success(self, info: dict) -> bool:
        """Return True if the episode was a success (env-specific)."""
        # gymnasium-robotics Adroit envs set info['is_success']
        return bool(info.get("is_success", False))

    def render(self) -> np.ndarray | None:
        return self.env.render()


# ─────────────────────────────────────────────────────────────────────────── #
# Factory                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def make_env(
    env_name: str = "AdroitHandDoor-v1",
    frame_skip: int = 2,
    render_mode: str | None = None,
    seed: int = 42,
) -> AdroitWrapper:
    """Convenience factory: create and return a wrapped AdroitHand env."""
    return AdroitWrapper(
        env_name=env_name,
        frame_skip=frame_skip,
        render_mode=render_mode,
        seed=seed,
    )
