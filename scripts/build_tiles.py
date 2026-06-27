#!/usr/bin/env python3
"""Build paired DEM training tiles from GeoTIFF rasters."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demforge.terrain_ops import make_model_sample, robust_normalize


def valid_tile(tile: np.ndarray, nodata: float | None, min_valid_fraction: float) -> bool:
    """Check whether a raster tile contains enough valid data."""

    if nodata is None:
        mask = np.isfinite(tile)
    else:
        mask = np.isfinite(tile) & (tile != nodata)

    if float(mask.mean()) < min_valid_fraction:
        return False

    clean = tile[mask]
    if clean.size == 0:
        return False

    relief = float(np.nanpercentile(clean, 98) - np.nanpercentile(clean, 2))
    return relief > 2.0


def fill_invalid(tile: np.ndarray, nodata: float | None) -> np.ndarray:
    """Fill invalid cells with tile median."""

    arr = tile.astype(np.float32)
    if nodata is None:
        mask = np.isfinite(arr)
    else:
        mask = np.isfinite(arr) & (arr != nodata)

    median = float(np.nanmedian(arr[mask])) if mask.any() else 0.0
    arr[~mask] = median
    return arr


def split_for_file(path: Path, val_sources: set[str], val_fraction: float) -> str:
    """Assign split at source-file granularity."""

    if path.stem in val_sources:
        return "val"
    return "train"


def build_tiles(
    src: Path,
    out: Path,
    tile_size: int,
    stride: int,
    downscale: int,
    val_fraction: float,
    min_valid_fraction: float,
) -> None:
    """Build DEMForge training tiles."""

    try:
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:
        raise SystemExit("rasterio is required for real DEM tile building. Install requirements.txt.") from exc

    files = sorted([*src.rglob("*.tif"), *src.rglob("*.tiff")])
    if not files:
        raise SystemExit(f"No GeoTIFF files found under {src}")

    random.seed(1337)
    stems = [path.stem for path in files]
    random.shuffle(stems)
    val_count = max(1, int(len(stems) * val_fraction))
    val_sources = set(stems[:val_count])

    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)

    written = {"train": 0, "val": 0}

    for raster_path in files:
        split = split_for_file(raster_path, val_sources, val_fraction)
        with rasterio.open(raster_path) as ds:
            nodata = ds.nodata
            width = ds.width
            height = ds.height

            positions = [
                (x, y)
                for y in range(0, max(1, height - tile_size + 1), stride)
                for x in range(0, max(1, width - tile_size + 1), stride)
            ]

            for x0, y0 in tqdm(positions, desc=raster_path.name):
                if x0 + tile_size > width or y0 + tile_size > height:
                    continue

                window = Window(x0, y0, tile_size, tile_size)
                tile = ds.read(1, window=window).astype(np.float32)

                if not valid_tile(tile, nodata, min_valid_fraction):
                    continue

                tile = fill_invalid(tile, nodata)
                normalized, norm_meta = robust_normalize(tile)
                x, y, target, coarse = make_model_sample(normalized, downscale=downscale)

                meta = {
                    "source_file": str(raster_path),
                    "source_stem": raster_path.stem,
                    "split": split,
                    "x0": x0,
                    "y0": y0,
                    "tile_size": tile_size,
                    "stride": stride,
                    "downscale": downscale,
                    "crs": str(ds.crs),
                    "transform": list(ds.window_transform(window))[:6],
                    "normalization": norm_meta,
                }

                name = f"{raster_path.stem}_x{x0:06d}_y{y0:06d}.npz"
                np.savez_compressed(out / split / name, x=x, y=y, target=target, coarse=coarse, meta=json.dumps(meta))
                written[split] += 1

    print(json.dumps(written, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw")
    parser.add_argument("--out", default="data/tiles")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.98)
    args = parser.parse_args()

    build_tiles(Path(args.src), Path(args.out), args.tile_size, args.stride, args.downscale, args.val_fraction, args.min_valid_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
