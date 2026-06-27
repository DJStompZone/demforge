#!/usr/bin/env python3
"""Sample trained terrain predictions from validation tiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demforge.data.tiles import DEMTileDataset
from demforge.models.unet import ResidualUNet
from demforge.render import render_triptych
from demforge.utils import pick_device


def load_model(checkpoint_path: Path, device: torch.device) -> ResidualUNet:
    """Load a trained residual U-Net."""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]["model"]
    model = ResidualUNet(
        in_channels=int(config["in_channels"]),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 48)),
        channel_mults=list(config.get("channel_mults", [1, 2, 4, 4])),
        attention_at_bottleneck=bool(config.get("attention_at_bottleneck", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def sample(checkpoint: Path, data: Path, out: Path, count: int, device_name: str) -> None:
    """Run inference on sample tiles."""

    device = pick_device(device_name)
    model = load_model(checkpoint, device)
    dataset = DEMTileDataset(data, augment=False)
    out.mkdir(parents=True, exist_ok=True)

    for index in range(min(count, len(dataset))):
        item = dataset[index]
        x = item["x"].unsqueeze(0).to(device)
        coarse = item["coarse"].unsqueeze(0).to(device)
        residual = model(x)
        pred = coarse + residual

        pred_np = pred[0, 0].float().cpu().numpy()
        coarse_np = item["coarse"][0].numpy()
        target_np = item["target"][0].numpy()

        np.save(out / f"sample_{index:03d}_pred.npy", pred_np)
        np.save(out / f"sample_{index:03d}_coarse.npy", coarse_np)
        np.save(out / f"sample_{index:03d}_target.npy", target_np)
        render_triptych(pred_np, out / f"sample_{index:03d}_pred.png", title=f"Prediction {index}")
        render_triptych(target_np, out / f"sample_{index:03d}_target.png", title=f"Target {index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="outputs/samples")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    sample(Path(args.checkpoint), Path(args.data), Path(args.out), args.count, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
