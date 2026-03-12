"""
Dataset loader for Robomimic HDF5 datasets (image observations).

Robomimic HDF5 schema:
  data/
    demo_0/
      obs/
        agentview_image        : (T, H, W, 3) uint8   ← main camera
        robot0_eye_in_hand_image: (T, H, W, 3) uint8  ← wrist camera (optional)
        robot0_eef_pos         : (T, 3)
        robot0_eef_quat        : (T, 4)
        robot0_gripper_qpos    : (T, 2)
      actions : (T, act_dim)
      rewards : (T,)
      dones   : (T,)
    demo_1/ ...
  mask/
    train : list of demo keys
    valid : list of demo keys

Each __getitem__ returns a sliding-window sample:
  {
    "image"  : (obs_horizon, 3, H, W)  float32 in [0,1]
    "state"  : (obs_horizon, state_dim) float32, normalized
    "action" : (action_horizon, act_dim) float32, normalized
  }

All tensors are flattened along the first axis before being fed to the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from utils.normalizer import RunningNormalizer


# ─────────────────────────────────────────────────────────────────────────── #
# State key configuration per task                                              #
# ─────────────────────────────────────────────────────────────────────────── #

    # State key configuration per task                                              #
# ─────────────────────────────────────────────────────────────────────────── #

DEFAULT_STATE_KEYS = {
    "lift": [
        "robot0_eef_pos",      # (3,)  end-effector position
        "robot0_eef_quat",     # (4,)  end-effector orientation
        "robot0_gripper_qpos", # (2,)  gripper joint positions
    ],  # total state_dim = 9
    "can": [
        "robot0_eef_pos",      # (3,)
        "robot0_eef_quat",     # (4,)
        "robot0_gripper_qpos", # (2,)
    ], # total state_dim = 9
}

DEFAULT_IMAGE_KEY = "agentview_image"

# Default task descriptions for each Robomimic task
DEFAULT_TASK_TEXTS = {
    "lift": "pick up the red cube and place it in the bin",
    "can":  "pick up the can and place it on the plate",
    "square": "pick up the red block and place it in the green bin",
    "transport": "pick up the green block and move it to the other side",
}


# ─────────────────────────────────────────────────────────────────────────── #
# Dataset                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

class RobomimicImageDataset(Dataset):
    """
    Sliding-window dataset from a Robomimic HDF5 file with image observations.

    Each sample contains:
      image  : (obs_horizon, 3, H, W)       — RGB frames, float32 ∈ [0,1]
      state  : (obs_horizon * state_dim,)   — proprioception, normalized
      action : (action_horizon * act_dim,)  — action chunk, normalized
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        task_name: str,
        obs_horizon: int = 1,
        action_horizon: int = 16,
        image_key: str = DEFAULT_IMAGE_KEY,
        state_keys: Optional[list[str]] = None,
        img_size: int = 84,
        normalize_state: bool = True,
        normalize_action: bool = True,
        state_normalizer: Optional[RunningNormalizer] = None,
        act_normalizer: Optional[RunningNormalizer] = None,
        split: str = "train",             # "train" | "valid" | "all"
        max_demos: Optional[int] = None,
        augment: bool = False,            # enable image augmentation (train only)
    ):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.image_key = image_key
        self.task_name = task_name
        self.state_keys = state_keys or DEFAULT_STATE_KEYS.get(task_name, DEFAULT_STATE_KEYS["lift"])
        self.img_size = img_size
        self.augment = augment

        # Build augmentation transforms (applied per-frame at __getitem__ time)
        if augment:
            import torchvision.transforms as T
            self._aug = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.84, 1.0), ratio=(0.9, 1.1)),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                T.RandomGrayscale(p=0.02),
            ])
        else:
            self._aug = None

        # Load data into memory
        images, states, actions = self._load(hdf5_path, split, max_demos)

        # Fit normalizers
        if normalize_state:
            all_states = np.concatenate(states, axis=0)
            if state_normalizer is None:
                state_normalizer = RunningNormalizer()
                state_normalizer.fit(all_states)
        if normalize_action:
            all_actions = np.concatenate(actions, axis=0)
            if act_normalizer is None:
                act_normalizer = RunningNormalizer()
                act_normalizer.fit(all_actions)

        self.state_normalizer = state_normalizer
        self.act_normalizer = act_normalizer

        # State/action dims
        self.state_dim = states[0].shape[-1]
        self.act_dim = actions[0].shape[-1]

        # Build sliding-window index
        self._samples: list[tuple] = []  # (img_seq, state_seq, act_chunk)
        for img_traj, state_traj, act_traj in zip(images, states, actions):
            T = len(act_traj)
            if T < obs_horizon + action_horizon:
                continue
            # Pad front
            pad_img   = np.tile(img_traj[0:1], (obs_horizon - 1, 1, 1, 1))
            pad_state = np.tile(state_traj[0:1], (obs_horizon - 1, 1))
            padded_img   = np.concatenate([pad_img, img_traj], axis=0)
            padded_state = np.concatenate([pad_state, state_traj], axis=0)

            for i in range(T - action_horizon + 1):
                img_seq   = padded_img[i: i + obs_horizon]      # (oh, H, W, 3)
                state_seq = padded_state[i: i + obs_horizon]    # (oh, state_dim)
                act_chunk = act_traj[i: i + action_horizon]     # (ah, act_dim)
                self._samples.append((img_seq, state_seq, act_chunk))

        print(f"[RobomimicDataset] {len(self._samples):,} samples  "
              f"state_dim={self.state_dim}  act_dim={self.act_dim}  "
              f"img={img_size}px")

    # ------------------------------------------------------------------ #
    # HDF5 loading                                                        #
    # ------------------------------------------------------------------ #

    def _load(
        self,
        path: str | Path,
        split: str,
        max_demos: Optional[int],
    ) -> tuple[list, list, list]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}\n"
                f"Run: python scripts/download_dataset.py --task lift --type ph"
            )

        images, states, actions = [], [], []

        with h5py.File(path, "r") as f:
            # Select demo keys by split
            if split != "all" and f"mask/{split}" in f:
                keys = [k.decode() if isinstance(k, bytes) else k
                        for k in f[f"mask/{split}"][:]]
            else:
                keys = sorted(f["data"].keys())

            if max_demos:
                keys = keys[:max_demos]

            for dk in tqdm(keys, desc=f"Loading {split}", leave=False):
                demo = f["data"][dk]
                obs  = demo["obs"]

                # ── images ──────────────────────────────────────────── #
                if self.image_key not in obs:
                    available = list(obs.keys())
                    raise KeyError(
                        f"Image key '{self.image_key}' not found. "
                        f"Available: {available}"
                    )
                img = obs[self.image_key][:]    # (T, H, W, 3) uint8

                # Resize if needed
                if img.shape[1] != self.img_size:
                    img = self._resize_images(img, self.img_size)

                # HWC uint8 → CHW float32 in [0,1]
                img = img.astype(np.float32) / 255.0      # (T, H, W, 3)
                img = img.transpose(0, 3, 1, 2)            # (T, 3, H, W)
                images.append(img)

                # ── state ───────────────────────────────────────────── #
                state_parts = []
                for sk in self.state_keys:
                    if sk in obs:
                        state_parts.append(obs[sk][:])
                state = np.concatenate(state_parts, axis=-1)  # (T, state_dim)
                states.append(state.astype(np.float32))

                # ── actions ─────────────────────────────────────────── #
                actions.append(demo["actions"][:].astype(np.float32))

        return images, states, actions

    @staticmethod
    def _resize_images(imgs: np.ndarray, size: int) -> np.ndarray:
        try:
            import cv2
        except ImportError:
            raise ImportError("pip install opencv-python  (needed for image resizing)")
        T, H, W, C = imgs.shape
        out = np.empty((T, size, size, C), dtype=imgs.dtype)
        for t in range(T):
            out[t] = cv2.resize(imgs[t], (size, size), interpolation=cv2.INTER_AREA)
        return out

    # ------------------------------------------------------------------ #
    # Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_seq, state_seq, act_chunk = self._samples[idx]

        # Normalize state and action
        if self.state_normalizer:
            state_seq = np.stack([self.state_normalizer.normalize(s) for s in state_seq])
        if self.act_normalizer:
            act_chunk = np.stack([self.act_normalizer.normalize(a) for a in act_chunk])

        img_tensor = torch.from_numpy(img_seq)   # (obs_h, 3, H, W)

        # Apply augmentation per frame (consistent spatial aug across obs_horizon)
        if self._aug is not None:
            # Use the same random seed for all frames in a window so spatial
            # transforms are consistent (same crop region for all frames)
            import torchvision.transforms.functional as TF
            frames = []
            # Sample a single crop for the whole window
            i_crop, j_crop, h_crop, w_crop = \
                __import__('torchvision').transforms.RandomResizedCrop.get_params(
                    img_tensor[0], scale=(0.84, 1.0), ratio=(0.9, 1.1))
            for frame in img_tensor:
                frame = TF.resized_crop(frame, i_crop, j_crop, h_crop, w_crop,
                                        [self.img_size, self.img_size])
                # Color jitter independently per frame (temporal variation is OK)
                frame = self._aug.transforms[1](frame)   # ColorJitter
                frame = self._aug.transforms[2](frame)   # RandomGrayscale
                frames.append(frame)
            img_tensor = torch.stack(frames)

        return {
            "image":  img_tensor,
            "state":  torch.from_numpy(state_seq.flatten().astype(np.float32)),
            "action": torch.from_numpy(act_chunk.flatten().astype(np.float32)),
            "text":   DEFAULT_TASK_TEXTS.get(self.task_name, "robot manipulation task"),
        }


# ─────────────────────────────────────────────────────────────────────────── #
# Factory                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def load_robomimic_dataset(
    hdf5_path: str | Path,
    task_name: str,
    obs_horizon: int = 1,
    action_horizon: int = 16,
    img_size: int = 84,
    batch_size: int = 64,
    num_workers: int = 4,
    max_demos: Optional[int] = None,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, RobomimicImageDataset]:
    """
    Load Robomimic dataset and return (train_loader, val_loader, dataset).
    Uses the built-in train/valid split from the HDF5 mask if available,
    otherwise does an 90/10 random split.
    """
    # Try to use built-in train split
    try:
        train_ds = RobomimicImageDataset(
            hdf5_path, task_name=task_name,
            obs_horizon=obs_horizon, action_horizon=action_horizon,
            img_size=img_size, split="train", max_demos=max_demos,
            augment=True,   # data augmentation for training
        )
        val_ds = RobomimicImageDataset(
            hdf5_path, task_name=task_name,
            obs_horizon=obs_horizon, action_horizon=action_horizon,
            img_size=img_size, split="valid", max_demos=max_demos,
            augment=False,  # no augmentation for validation
            # Reuse normalizers fitted on train
            state_normalizer=train_ds.state_normalizer,
            act_normalizer=train_ds.act_normalizer,
        )
    except Exception:
        # Fallback: random split
        full_ds = RobomimicImageDataset(
            hdf5_path, task_name=task_name,
            obs_horizon=obs_horizon, action_horizon=action_horizon,
            img_size=img_size, split="all", max_demos=max_demos,
        )
        n_val = max(1, int(len(full_ds) * 0.1))
        n_train = len(full_ds) - n_val
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
        # For property access
        train_ds = train_ds
        val_ds   = val_ds

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    ref = train_ds if isinstance(train_ds, RobomimicImageDataset) \
          else train_ds.dataset
    print(f"[DataLoader] train={len(train_ds):,}  val={len(val_ds):,}  "
          f"batch={batch_size}")
    return train_loader, val_loader, ref
