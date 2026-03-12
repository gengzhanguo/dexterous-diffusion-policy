"""
Multi-HDF5 dataset loader: merge several Robomimic datasets into one DataLoader.

Supports mixing e.g. can_ph + can_mh + can_mg by:
  1. Loading each HDF5 independently (respecting its train/valid mask).
  2. Fitting a single shared normalizer over ALL training data.
  3. Re-applying that shared normalizer to every sub-dataset.
  4. ConcatDataset → single DataLoader.

Usage:
    from dataset.multi_dataset import load_multi_robomimic_dataset

    train_loader, val_loader, ref = load_multi_robomimic_dataset(
        datasets=[
            {"path": "data/robomimic/can_ph_84px.hdf5",  "task": "can"},
            {"path": "data/robomimic/can_mh_84px.hdf5",  "task": "can"},
            {"path": "data/robomimic/can_mg_84px.hdf5",  "task": "can"},
        ],
        obs_horizon=1,
        action_horizon=16,
        img_size=84,
        batch_size=64,
    )
    # ref exposes .state_dim, .act_dim, .state_normalizer, .act_normalizer
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from dataset.robomimic_loader import RobomimicImageDataset
from utils.normalizer import RunningNormalizer


# ─────────────────────────────────────────────────────────────────────────── #
# Thin reference wrapper so train_visual.py can read .state_dim etc.          #
# ─────────────────────────────────────────────────────────────────────────── #

class MultiDatasetRef:
    """Carries shared metadata for the merged dataset (dim + normalizers)."""
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        state_normalizer: RunningNormalizer,
        act_normalizer: RunningNormalizer,
    ):
        self.state_dim = state_dim
        self.act_dim   = act_dim
        self.state_normalizer = state_normalizer
        self.act_normalizer   = act_normalizer


# ─────────────────────────────────────────────────────────────────────────── #
# Main factory                                                                  #
# ─────────────────────────────────────────────────────────────────────────── #

def load_multi_robomimic_dataset(
    datasets: list[dict],        # [{"path": ..., "task": ...}, ...]
    obs_horizon: int   = 1,
    action_horizon: int = 16,
    img_size: int      = 84,
    batch_size: int    = 64,
    num_workers: int   = 4,
    max_demos: Optional[int] = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, MultiDatasetRef]:
    """
    Load and merge multiple Robomimic HDF5 datasets.

    Args:
        datasets:  list of dicts with keys "path" and "task".
                   All tasks must share the same state_dim and act_dim.
        ...        (same as load_robomimic_dataset)

    Returns:
        (train_loader, val_loader, ref)
        ref has .state_dim, .act_dim, .state_normalizer, .act_normalizer
    """
    if not datasets:
        raise ValueError("datasets list is empty")

    print(f"\n[MultiDataset] Loading {len(datasets)} dataset(s) …")

    # ── Pass 1: collect raw states & actions to fit shared normalizers ── #
    all_states_train:  list[np.ndarray] = []
    all_actions_train: list[np.ndarray] = []

    # We temporarily load with normalize=False to gather raw values
    probe_ds_list: list[RobomimicImageDataset] = []
    for entry in datasets:
        path, task = entry["path"], entry["task"]
        print(f"  → probing  {Path(path).name}  (task={task})")
        try:
            ds = RobomimicImageDataset(
                hdf5_path=path,
                task_name=task,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
                img_size=img_size,
                split="train",
                max_demos=max_demos,
                normalize_state=False,
                normalize_action=False,
                augment=False,
            )
        except Exception:
            # Fallback: no train/valid mask in this HDF5
            ds = RobomimicImageDataset(
                hdf5_path=path,
                task_name=task,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
                img_size=img_size,
                split="all",
                max_demos=max_demos,
                normalize_state=False,
                normalize_action=False,
                augment=False,
            )
        probe_ds_list.append(ds)

        # Extract raw state/action arrays from sliding-window samples
        for img_seq, state_seq, act_chunk in ds._samples:
            all_states_train.append(state_seq.reshape(-1, ds.state_dim))
            all_actions_train.append(act_chunk)

    # ── Fit shared normalizers ────────────────────────────────────────── #
    all_states  = np.concatenate(all_states_train,  axis=0)
    all_actions = np.concatenate(all_actions_train, axis=0)

    shared_state_norm = RunningNormalizer()
    shared_state_norm.fit(all_states)
    shared_act_norm = RunningNormalizer()
    shared_act_norm.fit(all_actions)

    state_dim = probe_ds_list[0].state_dim
    act_dim   = probe_ds_list[0].act_dim

    # Sanity check: all datasets must have same dims
    for ds in probe_ds_list:
        if ds.state_dim != state_dim or ds.act_dim != act_dim:
            raise ValueError(
                f"Dimension mismatch: expected state_dim={state_dim}, act_dim={act_dim} "
                f"but got state_dim={ds.state_dim}, act_dim={ds.act_dim} "
                f"for a dataset."
            )

    print(f"\n[MultiDataset] Shared normalizer fitted on "
          f"{len(all_states):,} state rows, {len(all_actions):,} action rows")

    # ── Pass 2: build final datasets with shared normalizers ──────────── #
    train_datasets: list[RobomimicImageDataset] = []
    val_datasets:   list[RobomimicImageDataset] = []

    for entry in datasets:
        path, task = entry["path"], entry["task"]
        print(f"  → loading  {Path(path).name}  (task={task})")
        try:
            train_ds = RobomimicImageDataset(
                hdf5_path=path,
                task_name=task,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
                img_size=img_size,
                split="train",
                max_demos=max_demos,
                normalize_state=True,
                normalize_action=True,
                state_normalizer=shared_state_norm,
                act_normalizer=shared_act_norm,
                augment=True,
            )
            val_ds = RobomimicImageDataset(
                hdf5_path=path,
                task_name=task,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
                img_size=img_size,
                split="valid",
                max_demos=max_demos,
                normalize_state=True,
                normalize_action=True,
                state_normalizer=shared_state_norm,
                act_normalizer=shared_act_norm,
                augment=False,
            )
        except Exception:
            # Fallback: random 90/10 split
            full_ds = RobomimicImageDataset(
                hdf5_path=path,
                task_name=task,
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
                img_size=img_size,
                split="all",
                max_demos=max_demos,
                normalize_state=True,
                normalize_action=True,
                state_normalizer=shared_state_norm,
                act_normalizer=shared_act_norm,
                augment=False,
            )
            n_val   = max(1, int(len(full_ds) * 0.1))
            n_train = len(full_ds) - n_val
            gen     = torch.Generator().manual_seed(seed)
            train_ds, val_ds = torch.utils.data.random_split(
                full_ds, [n_train, n_val], generator=gen)

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    # ── Merge & DataLoaders ───────────────────────────────────────────── #
    merged_train = ConcatDataset(train_datasets)
    merged_val   = ConcatDataset(val_datasets)

    print(f"\n[MultiDataset] train={len(merged_train):,}  "
          f"val={len(merged_val):,}  "
          f"state_dim={state_dim}  act_dim={act_dim}  "
          f"batch={batch_size}")

    train_loader = DataLoader(
        merged_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        merged_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    ref = MultiDatasetRef(
        state_dim=state_dim,
        act_dim=act_dim,
        state_normalizer=shared_state_norm,
        act_normalizer=shared_act_norm,
    )
    return train_loader, val_loader, ref
