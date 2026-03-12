#!/usr/bin/env python3
"""
Record multiple rollout videos from a trained checkpoint.

Options:
  --grid      Save a single grid video (N rows × M cols of episodes)
  --episodes  Save individual episode videos

Usage:
    python scripts/record_video.py --checkpoint checkpoints/best.pt --n 8
    python scripts/record_video.py --checkpoint checkpoints/best.pt --n 4 --grid
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import imageio
import numpy as np
from omegaconf import OmegaConf

from env.adroit_wrapper import make_env
from evaluation.evaluator import Evaluator, load_policy
from env.video_recorder import VideoRecorder, make_grid


def parse_args():
    p = argparse.ArgumentParser(description="Record policy rollout video(s)")
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--config",         type=str, default="configs/default.yaml")
    p.add_argument("--n",              type=int, default=4,
                   help="Number of episodes to record")
    p.add_argument("--output_dir",     type=str, default="videos")
    p.add_argument("--grid",           action="store_true",
                   help="Save episodes as a single grid video")
    p.add_argument("--fps",            type=int, default=30)
    p.add_argument("--num_ddim_steps", type=int, default=None)
    p.add_argument("--max_steps",      type=int, default=200)
    p.add_argument("--device",         type=str, default=None)
    p.add_argument("--seed",           type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.num_ddim_steps:
        cfg.inference.num_ddim_steps = args.num_ddim_steps

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        env_name=cfg.env.name,
        frame_skip=cfg.env.frame_skip,
        render_mode="rgb_array",
        seed=args.seed,
    )

    policy = load_policy(
        checkpoint_path=args.checkpoint,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        action_horizon=cfg.dataset.action_horizon,
        obs_horizon=cfg.dataset.obs_horizon,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        time_emb_dim=cfg.model.time_emb_dim,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        device=device,
    )

    evaluator = Evaluator(
        policy=policy,
        env=env,
        num_ddim_steps=cfg.inference.num_ddim_steps,
        eta=cfg.inference.eta,
        device=device,
    )

    all_frame_seqs = []
    rewards, successes = [], []

    for i in range(args.n):
        print(f"Recording episode {i+1}/{args.n} …")
        result = evaluator.rollout(max_steps=args.max_steps, seed=args.seed + i, record_frames=True)
        frames = result["frames"]
        all_frame_seqs.append(frames)
        rewards.append(result["total_reward"])
        successes.append(result["success"])
        print(f"  reward={result['total_reward']:.2f}  success={result['success']}  steps={result['length']}")

        if not args.grid:
            path = output_dir / f"episode_{i:03d}.mp4"
            rec = VideoRecorder(path, fps=args.fps)
            for frame in frames:
                rec.add_frame(frame)
            rec.save()

    if args.grid:
        grid_frames = make_grid(all_frame_seqs, ncols=min(4, args.n))
        path = output_dir / "grid.mp4"
        imageio.mimwrite(str(path), grid_frames, fps=args.fps, quality=8)
        print(f"\nGrid video saved to: {path}")

    print(f"\nSummary:")
    print(f"  Success rate: {np.mean(successes):.1%}")
    print(f"  Mean reward:  {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()
