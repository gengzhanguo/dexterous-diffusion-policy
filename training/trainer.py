"""
Training loop for DiffusionPolicy.

Features:
  - AMP (fp16 mixed precision)
  - Gradient accumulation
  - Cosine LR schedule with linear warmup
  - Periodic checkpointing
  - TensorBoard logging
  - Validation loss tracking + best model saving
"""
from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.ddpm import DiffusionPolicy
from utils.logger import Logger
from utils.normalizer import RunningNormalizer


class Trainer:
    """Manages the full training lifecycle for DiffusionPolicy."""

    def __init__(
        self,
        policy: DiffusionPolicy,
        train_loader: DataLoader,
        val_loader: DataLoader,
        # Optimisation
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        grad_clip: float = 1.0,
        grad_accumulation: int = 1,
        amp: bool = True,
        warmup_steps: int = 500,
        lr_scheduler: str = "cosine",      # "cosine" | "constant"
        epochs: int = 200,
        # Checkpointing
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        log_every: int = 50,
        checkpoint_every: int = 10,        # epochs
        run_name: str = "diffusion_policy",
        device: str = "cuda",
        seed: int = 42,
        param_groups: list | None = None,   # per-layer LR groups
    ):
        self.policy = policy.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.grad_accumulation = grad_accumulation
        self.amp = amp and (device == "cuda")
        self.log_every = log_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)

        # Optimizer (supports per-layer LR via param_groups)
        opt_params = param_groups if param_groups is not None else policy.parameters()
        self.optimizer = AdamW(
            opt_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # LR scheduler
        total_steps = epochs * len(train_loader)
        warmup = LinearLR(self.optimizer, start_factor=1e-3, total_iters=warmup_steps)
        if lr_scheduler == "cosine":
            main = CosineAnnealingLR(
                self.optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=lr * 0.01,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, main],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = warmup  # constant after warmup (only warms up)

        # Total steps for this scheduler (built over full `epochs` horizon)
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps

        # Early stopping
        self.early_stopping_patience = 0   # 0 = disabled
        self._no_improve_count = 0

        # Overfit ratio monitoring (val / train threshold, 0 = disabled)
        self.overfit_ratio_threshold = 0.0

        # AMP scaler
        self.scaler = GradScaler(enabled=self.amp)

        # Logger
        self.logger = Logger(log_dir=log_dir, run_name=run_name)
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.start_epoch = 1  # may be updated by load_checkpoint

        n_params = policy.num_parameters()
        print(f"[Trainer] Parameters: {n_params:,}")
        print(f"[Trainer] AMP: {self.amp} | grad_accum: {grad_accumulation} | device: {device}")

    # ------------------------------------------------------------------ #
    # Main entry                                                          #
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        """Run the full training loop."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch()
            epoch_time = time.time() - t0

            # Log epoch-level scalars
            self.logger.log_scalars(
                {"loss/train_epoch": train_loss, "loss/val": val_loss},
                step=epoch,
            )

            # Overfit ratio check
            ratio = val_loss / (train_loss + 1e-8)
            ratio_str = f"  gap={ratio:.2f}x"
            if self.overfit_ratio_threshold > 0 and ratio > self.overfit_ratio_threshold:
                ratio_str += f" ⚠️  OVERFIT (>{self.overfit_ratio_threshold:.1f}x)"

            print(
                f"Epoch {epoch:>4d}/{self.epochs}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"lr={self._current_lr():.2e}  t={epoch_time:.1f}s"
                + ratio_str
            )

            # Log gap ratio to TensorBoard
            self.logger.log_scalars({"loss/val_train_ratio": ratio}, step=epoch)

            # Save best + early stopping counter
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best.pt", epoch, val_loss)
                print(f"  ✓ New best val loss: {val_loss:.5f}")
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
                if self.early_stopping_patience > 0:
                    remaining = self.early_stopping_patience - self._no_improve_count
                    if remaining <= 10:
                        print(f"  [EarlyStopping] No improvement for {self._no_improve_count} epochs (patience={self.early_stopping_patience}, {remaining} left)")

            # Periodic checkpoint
            if epoch % self.checkpoint_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, val_loss)

            # Early stopping
            if self.early_stopping_patience > 0 and self._no_improve_count >= self.early_stopping_patience:
                print(f"\n[EarlyStopping] Triggered at epoch {epoch} — no improvement for {self.early_stopping_patience} epochs.")
                break

        # Final checkpoint
        self._save_checkpoint("final.pt", self.epochs, self.best_val_loss)
        self.logger.close()
        print(f"\n[Trainer] Done. Best val loss: {self.best_val_loss:.5f}")

    # ------------------------------------------------------------------ #
    # Epoch loops                                                         #
    # ------------------------------------------------------------------ #

    def _train_epoch(self, epoch: int) -> float:
        self.policy.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)):
            obs    = batch["obs"].to(self.device, non_blocking=True)
            action = batch["action"].to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                loss = self.policy.compute_loss(obs, action)
                loss = loss / self.grad_accumulation

            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (step + 1) % self.grad_accumulation == 0 or (step + 1 == len(self.train_loader)):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * self.grad_accumulation
            n_batches += 1

            if self.global_step % self.log_every == 0:
                self.logger.log_scalars(
                    {
                        "loss/train_step": loss.item() * self.grad_accumulation,
                        "lr": self._current_lr(),
                    },
                    step=self.global_step,
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.policy.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            obs    = batch["obs"].to(self.device, non_blocking=True)
            action = batch["action"].to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                loss = self.policy.compute_loss(obs, action)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ #
    # Checkpoint I/O                                                      #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "state_normalizer": self.policy.state_normalizer.state_dict() if self.policy.state_normalizer else None,
            "act_normalizer": self.policy.act_normalizer.state_dict() if self.policy.act_normalizer else None,
        }
        path = self.checkpoint_dir / filename
        torch.save(ckpt, path)

    def fast_forward_scheduler(self, n_steps: int) -> None:
        """Advance the scheduler by n_steps to restore the correct LR position.

        Call this after load_checkpoint when you want the LR to be exactly where
        it would have been on the full-horizon cosine curve at step n_steps.
        This is O(n_steps) but typically fast (pure Python, no GPU).
        """
        print(f"[Trainer] Fast-forwarding scheduler by {n_steps:,} steps …", end=" ", flush=True)
        for _ in range(n_steps):
            self.scheduler.step()
        print(f"done  (lr={self._current_lr():.2e})")

    def load_checkpoint(self, path: str | Path, start_epoch: Optional[int] = None,
                        fresh_lr: bool = False) -> int:
        """Load model/optimizer/scaler state from a checkpoint.

        Args:
            path: Path to checkpoint file.
            start_epoch: Override the epoch to resume from.  If None, the
                saved epoch + 1 is used automatically.
            fresh_lr: If True, do NOT fast-forward the scheduler — start a
                brand-new LR schedule from the LR set at Trainer init.
                Use this when fine-tuning with a much smaller learning rate.

        Returns:
            The epoch number stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model_state_dict"])
        if not fresh_lr:
            # Restore optimizer state only when continuing the same LR schedule
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        saved_epoch = ckpt["epoch"]
        saved_global_step = ckpt.get("global_step", 0)
        self.global_step = saved_global_step if not fresh_lr else 0
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        self.start_epoch = (start_epoch if start_epoch is not None else saved_epoch + 1)

        if fresh_lr:
            print(f"[Trainer] Fresh LR schedule — starting from lr={self._current_lr():.2e}")
        else:
            # Fast-forward the scheduler so LR is correct for the resumed position
            self.fast_forward_scheduler(saved_global_step)

        # Restore normalizers
        if hasattr(self.policy, "state_normalizer") and ckpt.get("state_normalizer") is not None:
            self.policy.state_normalizer = RunningNormalizer()
            self.policy.state_normalizer.load_state_dict(ckpt["state_normalizer"])
        if hasattr(self.policy, "act_normalizer") and ckpt.get("act_normalizer") is not None:
            self.policy.act_normalizer = RunningNormalizer()
            self.policy.act_normalizer.load_state_dict(ckpt["act_normalizer"])

        print(
            f"[Trainer] Resumed from {path}\n"
            f"          saved_epoch={saved_epoch}  global_step={saved_global_step}\n"
            f"          will train epoch {self.start_epoch} → {self.epochs}"
        )
        return saved_epoch

    # ------------------------------------------------------------------ #
    # Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
