#!/usr/bin/env python3
"""Generate synthetic DEM-ish tiles for smoke-testing the workflow.

This is not the product. This is just a cheap way to test dataset loading,
training, checkpointing, sampling, and rendering before touching real DEMs.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demforge.terrain_ops import make_model_sample, robust_normalize


def value_noise(seed: int, size: int, scale: float) -> np.ndarray:
    """Create smooth-ish value noise using FFT low-pass filtering."""

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(size, size)).astype(np.float32)
    fft = np.fft.rfft2(noise)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    lowpass = np.exp(-(radius * scale) ** 2)
    result = np.fft.irfft2(fft * lowpass, s=(size, size)).astype(np.float32)
    result -= result.min()
    result /= max(float(result.max()), 1e-6)
    return result


def synthetic_height(seed: int, size: int) -> np.ndarray:
    """Make a fake ridgy terrain heightmap."""

    base = value_noise(seed, size, 12.0)
    broad = value_noise(seed + 11, size, 6.0)
    ridge_noise = value_noise(seed + 37, size, 18.0)
    ridges = 1.0 - np.abs(ridge_noise * 2.0 - 1.0)
    ridges = np.power(np.clip(ridges, 0.0, 1.0), 2.2)

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (x - size / 2) / (size / 2)
    y = (y - size / 2) / (size / 2)
    falloff = np.clip(np.sqrt(x * x + y * y), 0.0, 1.0)

    height = broad * 0.45 + base * 0.25 + ridges * 0.45
    height = height * (1.0 - falloff * 0.18)
    height -= height.min()
    height /= max(float(height.max()), 1e-6)
    height = height * 260.0 + 40.0
    return height.astype(np.float32)


def write_dataset(out: Path, count: int, size: int, downscale: int, val_fraction: float) -> None:
    """Write synthetic train/val tiles."""

    if out.exists():
        shutil.rmtree(out)

    train_dir = out / "train"
    val_dir = out / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    val_count = max(1, int(count * val_fraction))

    for index in range(count):
        height = synthetic_height(10_000 + index, size)
        normalized, norm_meta = robust_normalize(height)
        x, y, target, coarse = make_model_sample(normalized, downscale=downscale)

        split = "val" if index < val_count else "train"
        path = (val_dir if split == "val" else train_dir) / f"synthetic_{index:05d}.npz"
        meta = {
            "source": "synthetic",
            "index": index,
            "size": size,
            "downscale": downscale,
            "normalization": norm_meta,
        }
        np.savez_compressed(path, x=x, y=y, target=target, coarse=coarse, meta=json.dumps(meta))

    print(f"wrote {count - val_count} train tiles to {train_dir}")
    print(f"wrote {val_count} val tiles to {val_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    write_dataset(Path(args.out), args.count, args.size, args.downscale, args.val_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
