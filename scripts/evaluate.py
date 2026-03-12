#!/usr/bin/env python3
"""
Evaluate a trained diffusion policy checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --num_episodes 100
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --video videos/eval.mp4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from env.adroit_wrapper import make_env
from evaluation.evaluator import Evaluator, load_policy


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate diffusion policy")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--config",        type=str, default="configs/default.yaml")
    p.add_argument("--env",           type=str, default=None)
    p.add_argument("--num_episodes",  type=int, default=None)
    p.add_argument("--max_steps",     type=int, default=None)
    p.add_argument("--num_ddim_steps",type=int, default=None)
    p.add_argument("--video",         type=str, default=None,
                   help="Save first rollout as mp4 (requires render_mode=rgb_array)")
    p.add_argument("--device",        type=str, default=None)
    p.add_argument("--seed",          type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # CLI overrides
    if args.env:           cfg.env.name = args.env
    if args.num_episodes:  cfg.evaluation.num_episodes = args.num_episodes
    if args.max_steps:     cfg.evaluation.max_episode_steps = args.max_steps
    if args.num_ddim_steps:cfg.inference.num_ddim_steps = args.num_ddim_steps

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    render_mode = "rgb_array" if args.video else None

    print(f"\n[Env] {cfg.env.name}")
    env = make_env(
        env_name=cfg.env.name,
        frame_skip=cfg.env.frame_skip,
        render_mode=render_mode,
        seed=args.seed,
    )
    obs_dim = env.obs_dim
    act_dim = env.act_dim

    print(f"[Policy] Loading {args.checkpoint} …")
    policy = load_policy(
        checkpoint_path=args.checkpoint,
        obs_dim=obs_dim,
        act_dim=act_dim,
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

    print(f"\n[Evaluate] {cfg.evaluation.num_episodes} episodes …\n")
    metrics = evaluator.evaluate(
        num_episodes=cfg.evaluation.num_episodes,
        max_steps=cfg.evaluation.max_episode_steps,
        seed_offset=args.seed,
        record_video=args.video,
        verbose=True,
    )

    print("\n" + "═" * 50)
    print(" Evaluation Results")
    print("═" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")
    print("═" * 50)

    if args.video:
        print(f"\n  Video saved to: {args.video}")


if __name__ == "__main__":
    main()
