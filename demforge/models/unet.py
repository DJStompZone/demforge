"""Compact residual U-Net for DEM detail prediction."""

from __future__ import annotations

import torch
from torch import nn


def group_count(channels: int) -> int:
    """Choose a safe GroupNorm group count."""

    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    """Residual convolution block with GroupNorm and SiLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.GroupNorm(group_count(in_channels), in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class SelfAttention2d(nn.Module):
    """Small spatial self-attention block for bottleneck context."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(batch, channels, height * width).transpose(1, 2)
        k = k.reshape(batch, channels, height * width)
        v = v.reshape(batch, channels, height * width).transpose(1, 2)

        attention = torch.softmax(torch.bmm(q, k) * (channels ** -0.5), dim=-1)
        out = torch.bmm(attention, v).transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.proj(out)


class Down(nn.Module):
    """Downsample block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResBlock(in_channels, out_channels)
        self.down = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x)
        return self.down(skip), skip


class Up(nn.Module):
    """Upsample block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block = ResBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ResidualUNet(nn.Module):
    """Predict DEM height residuals from coarse terrain controls."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 48,
        channel_mults: list[int] | tuple[int, ...] = (1, 2, 4, 4),
        attention_at_bottleneck: bool = True,
    ) -> None:
        super().__init__()
        channels = [base_channels * multiplier for multiplier in channel_mults]

        self.input = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        current = channels[0]
        for next_channels in channels:
            self.downs.append(Down(current, next_channels))
            current = next_channels

        bottleneck_layers: list[nn.Module] = [ResBlock(current, current)]
        if attention_at_bottleneck:
            bottleneck_layers.append(SelfAttention2d(current))
        bottleneck_layers.append(ResBlock(current, current))
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        self.ups = nn.ModuleList()
        for skip_channels in reversed(channels):
            self.ups.append(Up(current, skip_channels, skip_channels))
            current = skip_channels

        self.output = nn.Sequential(
            nn.GroupNorm(group_count(current), current),
            nn.SiLU(inplace=True),
            nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input(x)
        skips: list[torch.Tensor] = []
        for down in self.downs:
            x, skip = down(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for up, skip in zip(self.ups, reversed(skips), strict=True):
            x = up(x, skip)

        return self.output(x)
