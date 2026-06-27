"""Heightmap derivative and normalization operations."""

from __future__ import annotations

import numpy as np


def robust_normalize(height: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize a heightmap using robust percentile range.

    Args:
        height: Heightmap as a 2D float array.
        eps: Numerical stability term.

    Returns:
        Tuple of normalized height in roughly [-1, 1] and metadata.
    """

    clean = height.astype(np.float32)
    p02 = float(np.nanpercentile(clean, 2.0))
    p98 = float(np.nanpercentile(clean, 98.0))
    median = float(np.nanmedian(clean))
    scale = max((p98 - p02) / 2.0, eps)
    normalized = (clean - median) / scale
    normalized = np.clip(normalized, -2.0, 2.0) / 2.0
    return normalized.astype(np.float32), {"median": median, "p02": p02, "p98": p98, "scale": scale}


def resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D array with pure NumPy bilinear interpolation."""

    in_h, in_w = arr.shape
    if in_h == out_h and in_w == out_w:
        return arr.astype(np.float32)

    y = np.linspace(0, in_h - 1, out_h)
    x = np.linspace(0, in_w - 1, out_w)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    top = arr[y0[:, None], x0[None, :]] * (1.0 - wx)[None, :] + arr[y0[:, None], x1[None, :]] * wx[None, :]
    bottom = arr[y1[:, None], x0[None, :]] * (1.0 - wx)[None, :] + arr[y1[:, None], x1[None, :]] * wx[None, :]
    return (top * (1.0 - wy)[:, None] + bottom * wy[:, None]).astype(np.float32)


def make_coarse(height: np.ndarray, downscale: int) -> np.ndarray:
    """Downsample then upsample a normalized heightmap."""

    if downscale <= 1:
        return height.astype(np.float32)

    h, w = height.shape
    small_h = max(1, h // downscale)
    small_w = max(1, w // downscale)
    small = resize_bilinear(height, small_h, small_w)
    return resize_bilinear(small, h, w)


def derivatives(height: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dx, dz, and Laplacian derivative maps."""

    dz, dx = np.gradient(height.astype(np.float32))
    lap = np.gradient(dx, axis=1) + np.gradient(dz, axis=0)
    return dx.astype(np.float32), dz.astype(np.float32), lap.astype(np.float32)


def make_model_sample(height: np.ndarray, downscale: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create model input, residual target, target, and coarse arrays."""

    target = height.astype(np.float32)
    coarse = make_coarse(target, downscale)
    dx, dz, lap = derivatives(coarse)
    x = np.stack([coarse, dx, dz, lap], axis=0).astype(np.float32)
    y = (target - coarse)[None, :, :].astype(np.float32)
    target = target[None, :, :].astype(np.float32)
    coarse = coarse[None, :, :].astype(np.float32)
    return x, y, target, coarse
