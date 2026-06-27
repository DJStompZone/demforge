"""Heightmap rendering helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_height_array(path: str | Path) -> np.ndarray:
    """Load a height array from .npy or .npz."""

    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = "pred" if "pred" in data.files else "target" if "target" in data.files else data.files[0]
            arr = data[key]
    else:
        raise ValueError(f"Unsupported input format: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def normalize_image(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1]."""

    lo = float(np.nanpercentile(arr, 1.0))
    hi = float(np.nanpercentile(arr, 99.0))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def hillshade(height: np.ndarray, azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """Create a simple hillshade image."""

    dy, dx = np.gradient(height * z_factor)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    az = np.deg2rad(azimuth)
    alt = np.deg2rad(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return normalize_image(shade)


def slope_map(height: np.ndarray) -> np.ndarray:
    """Compute normalized slope magnitude."""

    dy, dx = np.gradient(height)
    return normalize_image(np.sqrt(dx * dx + dy * dy))


def render_triptych(height: np.ndarray, out: str | Path, title: str | None = None) -> None:
    """Render height, hillshade, and slope views."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    images = [
        ("Height", normalize_image(height), "terrain"),
        ("Hillshade", hillshade(height), "gray"),
        ("Slope", slope_map(height), "magma"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    if title:
        fig.suptitle(title)

    for axis, (label, image, cmap) in zip(axes, images, strict=True):
        axis.imshow(image, cmap=cmap)
        axis.set_title(label)
        axis.set_axis_off()

    fig.savefig(out, dpi=160)
    plt.close(fig)
