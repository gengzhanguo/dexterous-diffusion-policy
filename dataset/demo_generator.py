"""
Demo dataset generator for AdroitHandDoor-v1.

Two policy modes:
  1. ScriptedPolicy  — time-parameterized heuristic with noise
  2. RandomPolicy    — random exploration with reward filtering

Both save to HDF5. Script mode is ~10× faster at generating usable demos.

File format (HDF5):
  /demo_{i}/
    observations   (T, obs_dim)  float32
    actions        (T, act_dim)  float32
    rewards        (T,)          float32
    dones          (T,)          bool
    success        ()            bool
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from tqdm import tqdm

from env.adroit_wrapper import make_env, AdroitWrapper


# ─────────────────────────────────────────────────────────────────────────── #
# Policy implementations                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class ScriptedPolicy:
    """
    Time-parameterized heuristic for AdroitHandDoor-v1.

    The door task requires:
      Phase 0 (0–30%):  Reach — move hand forward, slightly close fingers
      Phase 1 (30–60%): Grasp + turn — close fingers, rotate wrist
      Phase 2 (60–100%): Push — extend arm, maintain grip

    Joint ordering for AdroitHandDoor-v1 (28-dim):
      [0]    : wrist_pro   (+ = toward door)
      [1]    : wrist_flex
      [2]    : wrist_dev
      [3-6]  : index       (finger MCP/PIP/DIP/ABD)
      [7-10] : middle
      [11-14]: ring
      [15-18]: little
      [19-22]: thumb
      [23-27]: forearm (shoulder-like DOFs)

    Note: Exact ordering may vary between mujoco-py and mujoco versions.
    This heuristic applies generic "close-and-push" motions that tend to
    produce reward signal; noise is added for trajectory diversity.
    """

    def __init__(
        self,
        act_dim: int,
        noise_scale: float = 0.15,
        seed: int = 0,
    ):
        self.act_dim = act_dim
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

        # Waypoint joint targets (act_dim values); overridden for known dims
        self._waypoints = self._build_waypoints(act_dim)

    def _build_waypoints(self, D: int) -> list[np.ndarray]:
        """Build 3 waypoints; broadcast to act_dim."""
        # phase 0: reach (forward wrist, fingers half-open)
        w0 = np.zeros(D)
        w0[0] = 0.3           # wrist forward
        if D > 3:
            w0[3:7] = 0.2     # index slightly closed

        # phase 1: grasp + turn (close fingers, rotate wrist)
        w1 = np.zeros(D)
        w1[0] = 0.5
        w1[1] = 0.4           # wrist flex (turn handle)
        if D > 3:
            w1[3:7]   = 0.7   # index closed
            w1[7:11]  = 0.7   # middle closed
            w1[11:15] = 0.5   # ring closed
            w1[19:22] = 0.6   # thumb

        # phase 2: push open
        w2 = np.zeros(D)
        w2[0] = 0.8
        w2[1] = 0.4
        if D > 3:
            w2[3:7]   = 0.7
            w2[7:11]  = 0.7
            w2[11:15] = 0.5
            w2[19:22] = 0.6
        if D > 23:
            w2[23:] = 0.3     # arm extension

        return [w0, w1, w2]

    def act(self, obs: np.ndarray, t: int, T: int) -> np.ndarray:
        progress = t / max(T - 1, 1)

        if progress < 0.3:
            alpha = progress / 0.3
            base = (1 - alpha) * self._waypoints[0] + alpha * self._waypoints[1]
        elif progress < 0.6:
            alpha = (progress - 0.3) / 0.3
            base = (1 - alpha) * self._waypoints[1] + alpha * self._waypoints[2]
        else:
            base = self._waypoints[2].copy()

        noise = self.rng.normal(0, self.noise_scale, self.act_dim)
        return np.clip(base + noise, -1.0, 1.0)


class RandomPolicy:
    """Uniform random action with optional Ornstein-Uhlenbeck noise for smoother motion."""

    def __init__(self, act_dim: int, ou_theta: float = 0.15, seed: int = 0):
        self.act_dim = act_dim
        self.ou_theta = ou_theta
        self.rng = np.random.default_rng(seed)
        self._state = np.zeros(act_dim)

    def reset(self):
        self._state = np.zeros(self.act_dim)

    def act(self, obs: np.ndarray, t: int, T: int) -> np.ndarray:
        # OU process: smoother than pure random
        dw = self.rng.normal(0, 1, self.act_dim)
        self._state += -self.ou_theta * self._state + 0.3 * dw
        return np.clip(self._state, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────── #
# Generator                                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

class DemoGenerator:
    """
    Rollout a policy in the environment and save demonstrations to HDF5.
    """

    def __init__(
        self,
        env_name: str = "AdroitHandDoor-v1",
        frame_skip: int = 2,
        max_episode_steps: int = 200,
        min_reward_threshold: float = 0.0,
        seed: int = 42,
    ):
        self.env_name = env_name
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.min_reward_threshold = min_reward_threshold
        self.seed = seed

        self.env = make_env(env_name=env_name, frame_skip=frame_skip, seed=seed)

    def generate(
        self,
        num_demos: int,
        output_path: str | Path,
        policy_type: Literal["scripted", "random"] = "scripted",
        max_attempts: int | None = None,
        verbose: bool = True,
    ) -> dict:
        """
        Collect `num_demos` demonstrations and save to HDF5.

        Returns summary statistics.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if max_attempts is None:
            max_attempts = num_demos * 10

        # Build policy
        if policy_type == "scripted":
            policy = ScriptedPolicy(self.env.act_dim, seed=self.seed)
        else:
            policy = RandomPolicy(self.env.act_dim, seed=self.seed)

        saved = 0
        attempts = 0
        rewards_all = []
        successes = 0

        t0 = time.time()

        with h5py.File(output_path, "w") as f:
            pbar = tqdm(total=num_demos, desc=f"Generating demos ({policy_type})", disable=not verbose)

            while saved < num_demos and attempts < max_attempts:
                obs_list, act_list, rew_list, done_list = [], [], [], []
                obs, info = self.env.reset(seed=self.seed + attempts)
                attempts += 1

                if hasattr(policy, "reset"):
                    policy.reset()

                total_reward = 0.0
                success = False

                for t in range(self.max_episode_steps):
                    action = policy.act(obs, t, self.max_episode_steps)
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                    obs_list.append(obs.copy())
                    act_list.append(action.copy())
                    rew_list.append(reward)
                    done_list.append(done)

                    total_reward += reward
                    if self.env.is_success(info):
                        success = True

                    obs = next_obs
                    if done:
                        break

                # Filter by minimum total reward
                if total_reward < self.min_reward_threshold:
                    continue

                # Save demo
                grp = f.create_group(f"demo_{saved}")
                grp.create_dataset("observations", data=np.array(obs_list, dtype=np.float32))
                grp.create_dataset("actions",      data=np.array(act_list, dtype=np.float32))
                grp.create_dataset("rewards",      data=np.array(rew_list, dtype=np.float32))
                grp.create_dataset("dones",        data=np.array(done_list, dtype=bool))
                grp.attrs["total_reward"] = total_reward
                grp.attrs["success"] = success
                grp.attrs["length"] = len(obs_list)

                saved += 1
                rewards_all.append(total_reward)
                if success:
                    successes += 1

                pbar.update(1)
                pbar.set_postfix(reward=f"{total_reward:.2f}", success=success)

            pbar.close()

        elapsed = time.time() - t0
        stats = {
            "saved": saved,
            "attempts": attempts,
            "acceptance_rate": saved / max(attempts, 1),
            "success_rate": successes / max(saved, 1),
            "mean_reward": float(np.mean(rewards_all)) if rewards_all else 0.0,
            "max_reward": float(np.max(rewards_all)) if rewards_all else 0.0,
            "elapsed_s": elapsed,
        }

        if verbose:
            print(f"\n✓ Saved {saved} demos to {output_path}")
            print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")
            print(f"  Success rate:    {stats['success_rate']:.1%}")
            print(f"  Mean reward:     {stats['mean_reward']:.3f}")
            print(f"  Time:            {elapsed:.1f}s")

        return stats
