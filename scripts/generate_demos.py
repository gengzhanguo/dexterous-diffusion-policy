#!/usr/bin/env python3
"""
Generate a demonstration dataset for AdroitHandDoor-v1.

Usage:
    python scripts/generate_demos.py --num_demos 1000 --policy scripted
    python scripts/generate_demos.py --num_demos 500  --policy random --min_reward 0.1
"""
import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.demo_generator import DemoGenerator


def parse_args():
    p = argparse.ArgumentParser(description="Generate demonstration dataset")
    p.add_argument("--env",           type=str, default="AdroitHandDoor-v1")
    p.add_argument("--num_demos",     type=int, default=1000,
                   help="Number of demonstrations to collect")
    p.add_argument("--policy",        type=str, default="scripted",
                   choices=["scripted", "random"],
                   help="Demo collection policy")
    p.add_argument("--output",        type=str, default="data/demos/demos.hdf5")
    p.add_argument("--frame_skip",    type=int, default=2)
    p.add_argument("--max_steps",     type=int, default=200,
                   help="Max steps per episode")
    p.add_argument("--min_reward",    type=float, default=0.0,
                   help="Minimum cumulative reward to keep a trajectory")
    p.add_argument("--max_attempts",  type=int, default=None,
                   help="Max rollout attempts (default: num_demos × 10)")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'═'*60}")
    print(f" Generating {args.num_demos} demos | policy={args.policy}")
    print(f" env={args.env} | output={args.output}")
    print(f"{'═'*60}\n")

    generator = DemoGenerator(
        env_name=args.env,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
        min_reward_threshold=args.min_reward,
        seed=args.seed,
    )

    stats = generator.generate(
        num_demos=args.num_demos,
        output_path=args.output,
        policy_type=args.policy,
        max_attempts=args.max_attempts,
        verbose=True,
    )

    print(f"\nDataset saved to: {args.output}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
