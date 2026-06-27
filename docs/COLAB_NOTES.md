# Colab Notes

Recommended first run:

```bash
pip install -r requirements.txt
python scripts/make_synthetic_dataset.py --out data/synthetic --count 512 --size 256
python scripts/train_residual_unet.py --config configs/residual_unet_smoke.yaml
```

For real DEMs:

```bash
python scripts/build_tiles.py --src /content/drive/MyDrive/dem/raw --out data/tiles --tile-size 512 --stride 256 --downscale 8
python scripts/train_residual_unet.py --config configs/residual_unet_512.yaml
```

Use Drive or another mounted volume for:

- raw DEMs
- processed tiles
- checkpoints
- output renders

Do not rely on the Colab runtime disk for anything you want to keep. That thing has the permanence of a bar napkin in a hurricane.
