"""
Running normalizer: fits mean/std from a dataset and normalizes to [-1, 1].
"""
from __future__ import annotations

import numpy as np
import torch


class RunningNormalizer:
    """
    Normalizes tensors to approximately [-1, 1] using dataset statistics.
    Supports both numpy arrays and PyTorch tensors.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Fitting                                                              #
    # ------------------------------------------------------------------ #

    def fit(self, data: np.ndarray) -> "RunningNormalizer":
        """Fit normalizer from a 2-D array (N, D)."""
        assert data.ndim == 2, "Expected shape (N, D)"
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Transform                                                            #
    # ------------------------------------------------------------------ #

    def normalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Scale to mean=0, std=1 and clip to [-1, 1]."""
        self._check_fitted()
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            device = x.device
            mean = torch.tensor(self.mean, dtype=x.dtype, device=device)
            std = torch.tensor(self.std, dtype=x.dtype, device=device)
            out = (x - mean) / (std + self.eps)
            return out.clamp(-1.0, 1.0)
        out = (x - self.mean) / (self.std + self.eps)
        return np.clip(out, -1.0, 1.0)

    def denormalize(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Inverse of normalize."""
        self._check_fitted()
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            device = x.device
            mean = torch.tensor(self.mean, dtype=x.dtype, device=device)
            std = torch.tensor(self.std, dtype=x.dtype, device=device)
            return x * (std + self.eps) + mean
        return x * (self.std + self.eps) + self.mean

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.std = d["std"]
        self._fitted = True

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Normalizer has not been fitted. Call fit() first.")
