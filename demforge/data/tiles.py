"""Torch dataset for DEMForge tile files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DEMTileDataset(Dataset):
    """Load `.npz` DEM tile samples.

    Each file must contain `x`, `y`, `target`, and `coarse`.
    """

    def __init__(self, root: str | Path, augment: bool = False) -> None:
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.npz"))
        self.augment = augment
        if not self.files:
            raise FileNotFoundError(f"No .npz tiles found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def _augment(self, arrays: list[np.ndarray]) -> list[np.ndarray]:
        """Apply matched random flips/rotations."""

        if not self.augment:
            return arrays

        k = np.random.randint(0, 4)
        if k:
            arrays = [np.rot90(arr, k=k, axes=(-2, -1)).copy() for arr in arrays]

        if np.random.rand() < 0.5:
            arrays = [np.flip(arr, axis=-1).copy() for arr in arrays]

        if np.random.rand() < 0.5:
            arrays = [np.flip(arr, axis=-2).copy() for arr in arrays]

        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            x = data["x"].astype(np.float32)
            y = data["y"].astype(np.float32)
            target = data["target"].astype(np.float32)
            coarse = data["coarse"].astype(np.float32)

        x, y, target, coarse = self._augment([x, y, target, coarse])

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "target": torch.from_numpy(target),
            "coarse": torch.from_numpy(coarse),
            "path": str(path),
        }
