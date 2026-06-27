"""Terrain metrics for QA."""

from __future__ import annotations

import numpy as np


def terrain_metrics(height: np.ndarray) -> dict[str, float]:
    """Compute basic terrain metrics."""

    arr = np.asarray(height, dtype=np.float32)
    dy, dx = np.gradient(arr)
    slope = np.sqrt(dx * dx + dy * dy)
    lap = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)

    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "relief": float(np.nanmax(arr) - np.nanmin(arr)),
        "slope_mean": float(np.nanmean(slope)),
        "slope_p95": float(np.nanpercentile(slope, 95)),
        "slope_p99": float(np.nanpercentile(slope, 99)),
        "laplacian_abs_mean": float(np.nanmean(np.abs(lap))),
    }
