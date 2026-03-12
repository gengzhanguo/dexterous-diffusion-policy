#!/usr/bin/env python3
"""
Train the Visual Diffusion Policy on Robomimic image datasets.

Usage:
    # Download dataset first
    python scripts/download_dataset.py --task lift --type ph --img_size 84

    # Train with default config
    python scripts/train_visual.py

    # Custom overrides
    python scripts/train_visual.py \
        --data data/robomimic/lift_ph_84px.hdf5 \
        --img_size 64 \
        --encoder small_cnn \
        --epochs 100 \
        --batch_size 128

    # Unfreeze backbone (fine-tune ResNet too)
    python scripts/train_visual.py --unfreeze_encoder
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf

from dataset.robomimic_loader import load_robomimic_dataset
from diffusion.visual_ddpm import VisualDiffusionPolicy
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(
        description="Train visual diffusion policy",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Fresh run (epoch 1 → 300)
  python scripts/train_visual.py --config configs/robomimic_can_small.yaml

  # Continue to epoch 1000 from a checkpoint (LR auto-aligned)
  python scripts/train_visual.py --config configs/robomimic_can_small.yaml \\
      --resume checkpoints/can_small/final.pt \\
      --end_epoch 1000

  # Re-run a specific slice, e.g. epoch 401-600
  python scripts/train_visual.py --config configs/robomimic_can_small.yaml \\
      --resume checkpoints/can_small/epoch_0400.pt \\
      --end_epoch 600
""",
    )
    p.add_argument("--config",    type=str, default="configs/robomimic_lift.yaml")
    p.add_argument("--data",      type=str, default=None)
    p.add_argument("--img_size",  type=int, default=None)
    p.add_argument("--encoder",   type=str, default=None,
                   choices=["resnet18", "resnet50", "small_cnn"])
    p.add_argument("--end_epoch", type=int, default=None,
                   help="Last epoch to train (= total epochs on the LR schedule).\n"
                        "Overrides config training.epochs.")
    p.add_argument("--epochs",    type=int, default=None,
                   help="Alias for --end_epoch (deprecated, use --end_epoch).")
    p.add_argument("--batch_size",type=int, default=None)
    p.add_argument("--lr",        type=float, default=None)
    p.add_argument("--unfreeze_encoder", action="store_true",
                   help="Fine-tune the image encoder backbone (uses more VRAM)")
    p.add_argument("--device",    type=str, default=None)
    p.add_argument("--resume",    type=str, default=None,
                   help="Path to checkpoint to resume from.\n"
                        "start_epoch is auto-detected (saved_epoch + 1).")
    p.add_argument("--fresh_lr",  action="store_true",
                   help="When resuming, start a brand-new LR schedule from --lr\n"
                        "instead of fast-forwarding the old schedule.\n"
                        "Use this when fine-tuning with a much smaller LR.")
    p.add_argument("--run_name",  type=str, default="visual_diffusion_policy")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    # CLI overrides
    if args.data:       cfg.dataset.path = args.data
    if args.img_size:   cfg.dataset.img_size = args.img_size
    if args.encoder:    cfg.model.encoder_type = args.encoder
    # --end_epoch takes priority; --epochs is the legacy alias
    end_epoch = args.end_epoch or args.epochs
    if end_epoch:       cfg.training.epochs = end_epoch
    if args.batch_size: cfg.training.batch_size = args.batch_size
    if args.lr:         cfg.training.lr = args.lr
    if args.unfreeze_encoder:
        cfg.model.encoder_frozen = False

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**3} GB)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataset ────────────────────────────────────────────────────────
    print(f"\n[Dataset] Loading {cfg.dataset.path} …")
    train_loader, val_loader, dataset = load_robomimic_dataset(
        hdf5_path=cfg.dataset.path,
        task_name=cfg.dataset.task,
        obs_horizon=cfg.dataset.obs_horizon,
        action_horizon=cfg.dataset.action_horizon,
        img_size=cfg.dataset.img_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        max_demos=cfg.dataset.max_demos,
        seed=args.seed,
    )

    state_dim = dataset.state_dim
    act_dim   = dataset.act_dim
    print(f"  state_dim={state_dim}  act_dim={act_dim}")

    # ── Policy ─────────────────────────────────────────────────────────
    print("\n[Policy] Building VisualDiffusionPolicy …")
    policy = VisualDiffusionPolicy(
        state_dim=state_dim,
        act_dim=act_dim,
        action_horizon=cfg.dataset.action_horizon,
        obs_horizon=cfg.dataset.obs_horizon,
        img_fusion=getattr(cfg.model, "img_fusion", "mean"),
        encoder_type=cfg.model.encoder_type,
        img_emb_dim=cfg.model.img_emb_dim,
        encoder_frozen=cfg.model.encoder_frozen,
        encoder_pretrained=cfg.model.encoder_pretrained,
        state_emb_dim=cfg.model.state_emb_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        time_emb_dim=cfg.model.time_emb_dim,
        dropout=cfg.model.dropout,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        clip_sample=cfg.diffusion.clip_sample,
        clip_sample_range=cfg.diffusion.clip_sample_range,
        device=device, # Pass device to policy
    )

    # Attach normalizers
    policy.state_normalizer = dataset.state_normalizer
    policy.act_normalizer   = dataset.act_normalizer

    trainable = policy.num_parameters()
    total     = policy.num_parameters_total()
    print(f"  Trainable params: {trainable:,}  /  Total: {total:,}")
    print(f"  Encoder frozen: {cfg.model.encoder_frozen}")

    # ── Trainer ────────────────────────────────────────────────────────
    # Patch trainer's compute_loss to handle image+state batches
    # We subclass Trainer's _train/_val epoch via monkey-patching the batch format

    class VisualTrainer(Trainer):
        """Overrides batch unpacking to handle {image, state, action} dicts."""

        def _unpack(self, batch):
            images  = batch["image"].to(self.device, non_blocking=True)
            states  = batch["state"].to(self.device, non_blocking=True)
            actions = batch["action"].to(self.device, non_blocking=True)
            return images, states, actions

        def _train_epoch(self, epoch):
            import time
            from tqdm import tqdm
            from torch.cuda.amp import autocast
            import torch.nn as nn

            self.policy.train()
            total_loss = 0.0
            n_batches  = 0
            self.optimizer.zero_grad()

            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)):
                images, states, actions = self._unpack(batch)

                with autocast(enabled=self.amp):
                    loss = self.policy.compute_loss(images, states, actions)
                    loss = loss / self.grad_accumulation

                self.scaler.scale(loss).backward()

                if (step + 1) % self.grad_accumulation == 0 or (step + 1 == len(self.train_loader)):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.global_step += 1

                total_loss += loss.item() * self.grad_accumulation
                n_batches  += 1

                if self.global_step % self.log_every == 0:
                    self.logger.log_scalars(
                        {"loss/train_step": loss.item() * self.grad_accumulation,
                         "lr": self._current_lr()},
                        step=self.global_step,
                    )

            return total_loss / max(n_batches, 1)

        @torch.no_grad()
        def _val_epoch(self):
            from torch.cuda.amp import autocast
            self.policy.eval()
            total_loss = 0.0
            n_batches  = 0

            for batch in self.val_loader:
                images, states, actions = self._unpack(batch)
                with autocast(enabled=self.amp):
                    loss = self.policy.compute_loss(images, states, actions)
                total_loss += loss.item()
                n_batches  += 1

            return total_loss / max(n_batches, 1)

    # Support per-layer LR: encoder_lr (backbone) vs lr (MLP/projection)
    encoder_lr = getattr(cfg.training, 'encoder_lr', None)
    if encoder_lr is not None and hasattr(policy, 'image_encoder'):
        encoder_ids = set(id(p) for p in policy.image_encoder.parameters())
        param_groups = [
            {'params': [p for p in policy.parameters() if id(p) not in encoder_ids],
             'lr': cfg.training.lr},
            {'params': [p for p in policy.image_encoder.parameters()],
             'lr': encoder_lr, 'name': 'encoder'},
        ]
        print(f"[LR] MLP/proj={cfg.training.lr:.1e}  encoder={encoder_lr:.1e}")
    else:
        param_groups = None

    trainer = VisualTrainer(
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
        param_groups=param_groups,
    )
    trainer.early_stopping_patience = getattr(cfg.training, 'early_stopping_patience', 0)
    trainer.overfit_ratio_threshold = getattr(cfg.training, 'overfit_ratio_threshold', 0.0)

    if args.resume:
        resumed_epoch = trainer.load_checkpoint(args.resume, fresh_lr=args.fresh_lr)
        remaining = cfg.training.epochs - resumed_epoch
        lr_note = f"fresh LR={cfg.training.lr:.1e}" if args.fresh_lr else "LR fast-forwarded"
        print(f"\n[Training] epoch {trainer.start_epoch} → {cfg.training.epochs}  "
              f"({remaining} remaining, {lr_note})\n")
    else:
        print(f"\n[Training] Starting fresh — epoch 1 → {cfg.training.epochs}\n")
    trainer.train()


if __name__ == "__main__":
    main()
