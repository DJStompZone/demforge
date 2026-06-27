#!/usr/bin/env python3
"""Train a residual U-Net terrain detail model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demforge.data.tiles import DEMTileDataset
from demforge.losses import TerrainLoss
from demforge.models.unet import ResidualUNet
from demforge.render import render_triptych
from demforge.utils import count_parameters, pick_device, read_yaml, seed_everything, write_jsonl


def make_model(config: dict[str, Any]) -> ResidualUNet:
    """Construct the configured model."""

    return ResidualUNet(
        in_channels=int(config["in_channels"]),
        out_channels=int(config.get("out_channels", 1)),
        base_channels=int(config.get("base_channels", 48)),
        channel_mults=list(config.get("channel_mults", [1, 2, 4, 4])),
        attention_at_bottleneck=bool(config.get("attention_at_bottleneck", True)),
    )


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, epoch: int, step: int, best_val: float, config: dict[str, Any]) -> None:
    """Save a resumable checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: torch.device) -> tuple[int, int, float]:
    """Load a checkpoint."""

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]), int(checkpoint["step"]), float(checkpoint.get("best_val", math.inf))


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, criterion: TerrainLoss, device: torch.device, amp: bool) -> dict[str, float]:
    """Run validation."""

    model.eval()
    totals: dict[str, float] = {}
    count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        coarse = batch["coarse"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            pred = model(x)
            _, metrics = criterion(pred, y, coarse)

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1

    return {key: value / max(1, count) for key, value in totals.items()}


@torch.no_grad()
def render_sample(model: torch.nn.Module, loader: DataLoader, device: torch.device, out: Path, step: int, amp: bool) -> None:
    """Render one validation sample."""

    model.eval()
    batch = next(iter(loader))
    x = batch["x"].to(device)
    coarse = batch["coarse"].to(device)

    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        residual = model(x)
        pred = coarse + residual

    pred_np = pred[0, 0].float().cpu().numpy()
    target_np = batch["target"][0, 0].numpy()
    coarse_np = batch["coarse"][0, 0].numpy()

    render_dir = out / "renders"
    render_triptych(coarse_np, render_dir / f"step_{step:08d}_coarse.png", title="Coarse input")
    render_triptych(pred_np, render_dir / f"step_{step:08d}_pred.png", title="Predicted terrain")
    render_triptych(target_np, render_dir / f"step_{step:08d}_target.png", title="Target terrain")


def train(config_path: Path, resume: Path | None) -> None:
    """Train the terrain model."""

    config = read_yaml(config_path)
    seed_everything(int(config.get("seed", 1337)))
    device = pick_device(str(config.get("device", "auto")))

    train_cfg = config["train"]
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    ckpt_dir = output_dir / "checkpoints"
    log_path = output_dir / "train_log.jsonl"

    model = make_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]), weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(train_cfg["epochs"])))
    amp = bool(train_cfg.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    criterion = TerrainLoss(**config.get("loss", {}))

    train_dataset = DEMTileDataset(config["data"]["train_dir"], augment=True)
    val_dataset = DEMTileDataset(config["data"]["val_dir"], augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 2)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(train_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    start_epoch = 0
    global_step = 0
    best_val = math.inf

    if resume:
        start_epoch, global_step, best_val = load_checkpoint(resume, model, optimizer, scaler, device)

    print(f"device: {device}")
    print(f"train tiles: {len(train_dataset)}")
    print(f"val tiles: {len(val_dataset)}")
    print(f"parameters: {count_parameters(model):,}")

    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    clip_grad_norm = float(train_cfg.get("clip_grad_norm", 1.0))
    save_every = int(train_cfg.get("save_every_steps", 1000))
    render_every = int(train_cfg.get("render_every_steps", 1000))

    for epoch in range(start_epoch, int(train_cfg["epochs"])):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{train_cfg['epochs']}")
        for batch_index, batch in enumerate(progress):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            coarse = batch["coarse"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss, metrics = criterion(pred, y, coarse)
                scaled_loss = loss / grad_accum

            scaler.scale(scaled_loss).backward()

            if (batch_index + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                running = 0.98 * running + 0.02 * metrics["total"] if running else metrics["total"]
                progress.set_postfix(loss=f"{running:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                if global_step % save_every == 0:
                    save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scaler, epoch, global_step, best_val, config)

                if global_step % render_every == 0:
                    render_sample(model, val_loader, device, output_dir, global_step, amp)

                write_jsonl(log_path, {"type": "train", "epoch": epoch, "step": global_step, **metrics})

        val_metrics = validate(model, val_loader, criterion, device, amp)
        scheduler.step()

        write_jsonl(log_path, {"type": "val", "epoch": epoch, "step": global_step, **val_metrics})
        print(f"val total: {val_metrics['total']:.4f}")

        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scaler, epoch + 1, global_step, best_val, config)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scaler, epoch + 1, global_step, best_val, config)
            render_sample(model, val_loader, device, output_dir, global_step, amp)
            print(f"new best: {best_val:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/residual_unet_512.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(Path(args.config), Path(args.resume) if args.resume else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
