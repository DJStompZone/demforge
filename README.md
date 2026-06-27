# DEMForge POC

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Economy-Plus/demforge/blob/main/usgs_training.ipynb)

A practical proof-of-concept for DEM-trained procedural terrain generation.

Goal:

```text
real DEM corpus
  -> paired DEM tiles
  -> train residual terrain detailer
  -> generate DEM-like fictional terrain
  -> render / QA / export
```

This POC intentionally starts with a conditional residual U-Net, not a giant diffusion model. Diffusion is the next spicy step after the boring model beats bicubic upscaling plus fractal nonsense.

## Why this architecture first?

The first useful model should answer:

> Given a coarse macro terrain heightmap, can we add believable real-world DEM-style detail?

Training target:

```text
input  = coarse/upscaled terrain + derivative channels
target = high-res terrain residual
output = predicted residual
final  = coarse + predicted residual
```

This gives us control, low VRAM use, objective metrics, and a direct Minecraft terrain pipeline.

## Quick smoke test

This does not need real DEM data. It generates fake DEM-ish tiles so you can test the training loop.

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic --count 256 --size 256
python scripts/train_residual_unet.py --config configs/residual_unet_smoke.yaml
python scripts/sample.py --checkpoint outputs/checkpoints/best.pt --data data/synthetic/val --out outputs/samples
python scripts/render_heightmap.py --input outputs/samples/sample_000_pred.npy --out outputs/renders/sample_000_pred.png
```

## Real DEM tile build

Put GeoTIFF DEM files under:

```text
data/raw/
```

Then:

```bash
python scripts/build_tiles.py --src data/raw --out data/tiles --tile-size 512 --stride 256 --downscale 8
```

Expected split:

```text
data/tiles/train/*.npz
data/tiles/val/*.npz
```

## Output file format

Each tile `.npz` contains:

- `x`: model input, shape `[C,H,W]`
- `y`: residual target, shape `[1,H,W]`
- `target`: normalized high-res height, shape `[1,H,W]`
- `coarse`: normalized coarse/upscaled height, shape `[1,H,W]`
- `meta`: JSON string metadata

## Notes

- Use real region-level validation splits. Do not randomly split neighboring tiles from the same mountain into train and val unless you want your validation score to lie like a used car salesman.
- Keep absolute elevation metadata, but train mostly on normalized local shape.
- Start 256x256 for smoke tests, then 512x512.
