"""
PyTorch Dataset for diffusion policy training.

Reads HDF5 demos and creates sliding-window samples:
  - obs_seq:  (obs_horizon, obs_dim)       ← current + past observations
  - act_chunk:(action_horizon, act_dim)    ← future action chunk to predict

Both are normalized to [-1, 1] using RunningNormalizer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
from tqdm import tqdm

from utils.normalizer import RunningNormalizer


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

class DemoDataset(Dataset):
    """
    Sliding-window dataset from HDF5 demo file.

    Each item is a dict:
        obs   : (obs_dim * obs_horizon,)     float32, normalized
        action: (act_dim * action_horizon,)  float32, normalized
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        obs_horizon: int = 1,
        action_horizon: int = 16,
        obs_normalizer: Optional[RunningNormalizer] = None,
        act_normalizer: Optional[RunningNormalizer] = None,
        normalize: bool = True,
        max_demos: Optional[int] = None,
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.normalize = normalize

        # Load all demos into memory
        all_obs, all_acts = self._load_hdf5(hdf5_path, max_demos)

        # Fit normalizers if not provided
        if normalize:
            flat_obs = np.concatenate(all_obs, axis=0)
            flat_act = np.concatenate(all_acts, axis=0)

            if obs_normalizer is None:
                obs_normalizer = RunningNormalizer()
                obs_normalizer.fit(flat_obs)
            if act_normalizer is None:
                act_normalizer = RunningNormalizer()
                act_normalizer.fit(flat_act)

        self.obs_normalizer = obs_normalizer
        self.act_normalizer = act_normalizer

        # Build sliding-window index
        self._samples: list[tuple[np.ndarray, np.ndarray]] = []
        for obs_traj, act_traj in zip(all_obs, all_acts):
            T = len(obs_traj)
            if T < obs_horizon + action_horizon:
                continue
            # Pad front with first obs for obs_horizon context
            pad_obs = np.tile(obs_traj[0], (obs_horizon - 1, 1))
            padded_obs = np.concatenate([pad_obs, obs_traj], axis=0)  # (T+obs_h-1, D)

            for i in range(T - action_horizon + 1):
                obs_seq = padded_obs[i: i + obs_horizon]      # (obs_h, obs_dim)
                act_chunk = act_traj[i: i + action_horizon]   # (act_h, act_dim)
                self._samples.append((obs_seq, act_chunk))

        print(f"[DemoDataset] {len(self._samples):,} samples from {len(all_obs)} demos")
        self.obs_dim = all_obs[0].shape[-1]
        self.act_dim = all_acts[0].shape[-1]

    # ------------------------------------------------------------------ #
    # HDF5 I/O                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_hdf5(
        path: str | Path,
        max_demos: int | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Load all demos from an HDF5 file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        all_obs, all_acts = [], []
        with h5py.File(path, "r") as f:
            keys = sorted(f.keys())
            if max_demos is not None:
                keys = keys[:max_demos]
            for key in tqdm(keys, desc="Loading demos", leave=False):
                grp = f[key]
                all_obs.append(grp["observations"][:])   # (T, obs_dim)
                all_acts.append(grp["actions"][:])        # (T, act_dim)
        return all_obs, all_acts

    # ------------------------------------------------------------------ #
    # Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs_seq, act_chunk = self._samples[idx]

        # Normalize
        if self.normalize and self.obs_normalizer is not None:
            obs_seq = np.stack([self.obs_normalizer.normalize(o) for o in obs_seq])
        if self.normalize and self.act_normalizer is not None:
            act_chunk = np.stack([self.act_normalizer.normalize(a) for a in act_chunk])

        return {
            "obs":    torch.from_numpy(obs_seq.flatten().astype(np.float32)),
            "action": torch.from_numpy(act_chunk.flatten().astype(np.float32)),
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Factory                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def load_dataset(
    hdf5_path: str | Path,
    obs_horizon: int = 1,
    action_horizon: int = 16,
    val_split: float = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    normalize: bool = True,
    max_demos: int | None = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DemoDataset]:
    """
    Load dataset and return (train_loader, val_loader, dataset).
    The dataset contains fitted normalizers accessible as:
        dataset.obs_normalizer
        dataset.act_normalizer
    """
    dataset = DemoDataset(
        hdf5_path=hdf5_path,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        normalize=normalize,
        max_demos=max_demos,
    )

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(f"[DataLoader] train={n_train:,}  val={n_val:,}  batch={batch_size}")
    return train_loader, val_loader, dataset
