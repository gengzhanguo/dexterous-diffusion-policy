#!/usr/bin/env python3
"""
Train the diffusion policy.

Usage:
    # Default config (~4-6h, 200 epochs)
    python scripts/train.py

    # Fast config (~1-2h, 100 epochs)
    python scripts/train.py --config configs/fast.yaml

    # Custom overrides
    python scripts/train.py --epochs 50 --batch_size 512 --lr 3e-4

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/epoch_0050.pt
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf

from env.adroit_wrapper import make_env
from dataset.dataset_loader import load_dataset
from diffusion.ddpm import DiffusionPolicy
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion policy")
    p.add_argument("--config",      type=str, default="configs/default.yaml")
    p.add_argument("--data",        type=str, default=None, help="Path to HDF5 dataset (overrides config)")
    p.add_argument("--env",         type=str, default=None)
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--hidden_dim",  type=int, default=None)
    p.add_argument("--num_layers",  type=int, default=None)
    p.add_argument("--amp",         action="store_true", default=None)
    p.add_argument("--no_amp",      dest="amp", action="store_false")
    p.add_argument("--device",      type=str, default=None,
                   help="cuda | cpu (auto-detects if not set)")
    p.add_argument("--resume",      type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--run_name",    type=str, default="diffusion_policy")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # CLI overrides
    if args.data:       cfg.dataset.path = args.data
    if args.env:        cfg.env.name = args.env
    if args.epochs:     cfg.training.epochs = args.epochs
    if args.batch_size: cfg.training.batch_size = args.batch_size
    if args.lr:         cfg.training.lr = args.lr
    if args.hidden_dim: cfg.model.hidden_dim = args.hidden_dim
    if args.num_layers: cfg.model.num_layers = args.num_layers
    if args.amp is not None: cfg.training.amp = args.amp

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}  ({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")
    else:
        print("[Device] CPU (no CUDA detected)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Probe env for dims ─────────────────────────────────────────────
    print(f"\n[Env] Probing {cfg.env.name} …")
    env = make_env(cfg.env.name, frame_skip=cfg.env.frame_skip, seed=args.seed)
    obs_dim = env.obs_dim
    act_dim = env.act_dim
    env.env.close()
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}")

    # ── Dataset ────────────────────────────────────────────────────────
    print(f"\n[Dataset] Loading {cfg.dataset.path} …")
    train_loader, val_loader, dataset = load_dataset(
        hdf5_path=cfg.dataset.path,
        obs_horizon=cfg.dataset.obs_horizon,
        action_horizon=cfg.dataset.action_horizon,
        val_split=cfg.dataset.val_split,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        normalize=cfg.dataset.normalize,
        max_demos=cfg.dataset.num_demos,
        seed=args.seed,
    )

    # ── Policy ─────────────────────────────────────────────────────────
    print("\n[Policy] Building DiffusionPolicy …")
    policy = DiffusionPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_horizon=cfg.dataset.action_horizon,
        obs_horizon=cfg.dataset.obs_horizon,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        time_emb_dim=cfg.model.time_emb_dim,
        dropout=cfg.model.dropout,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=cfg.diffusion.clip_sample,
        clip_sample_range=cfg.diffusion.clip_sample_range,
    )

    # Attach normalizers
    policy.obs_normalizer = dataset.obs_normalizer
    policy.act_normalizer = dataset.act_normalizer

    # ── Trainer ────────────────────────────────────────────────────────
    trainer = Trainer(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        grad_accumulation=cfg.training.grad_accumulation,
        amp=cfg.training.amp,
        warmup_steps=cfg.training.warmup_steps,
        lr_scheduler=cfg.training.lr_scheduler,
        epochs=cfg.training.epochs,
        checkpoint_dir=cfg.paths.checkpoint_dir,
        log_dir=cfg.paths.log_dir,
        log_every=cfg.training.log_every,
        checkpoint_every=cfg.training.checkpoint_every,
        run_name=args.run_name,
        device=device,
        seed=args.seed,
    )

    # Optional resume
    if args.resume:
        trainer.load_checkpoint(args.resume)

    print(f"\n[Training] Starting … ({cfg.training.epochs} epochs)\n")
    trainer.train()


if __name__ == "__main__":
    main()
