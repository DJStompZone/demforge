"""Terrain-aware loss functions."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def gradient_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forward finite differences."""

    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """Compute a 2D Laplacian."""

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    channels = x.shape[1]
    return F.conv2d(x, kernel.repeat(channels, 1, 1, 1), padding=1, groups=channels)


def spectral_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Return log FFT magnitude for spectral similarity."""

    fft = torch.fft.rfft2(x.float(), norm="ortho")
    return torch.log1p(torch.abs(fft))


def seam_edges(x: torch.Tensor) -> torch.Tensor:
    """Return boundary samples used by seam loss."""

    top = x[..., 0, :]
    bottom = x[..., -1, :]
    left = x[..., :, 0]
    right = x[..., :, -1]
    return torch.cat([top.flatten(1), bottom.flatten(1), left.flatten(1), right.flatten(1)], dim=1)


class TerrainLoss(nn.Module):
    """Weighted terrain reconstruction loss.

    The model predicts residual height, but losses are applied to reconstructed
    terrain: `coarse + predicted_residual`.
    """

    def __init__(
        self,
        height_l1: float = 1.0,
        gradient_l1: float = 0.75,
        laplacian_l1: float = 0.45,
        spectral_l1: float = 0.15,
        seam_l1: float = 0.20,
    ) -> None:
        super().__init__()
        self.weights = {
            "height_l1": height_l1,
            "gradient_l1": gradient_l1,
            "laplacian_l1": laplacian_l1,
            "spectral_l1": spectral_l1,
            "seam_l1": seam_l1,
        }

    def forward(self, predicted_residual: torch.Tensor, target_residual: torch.Tensor, coarse: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        predicted = coarse + predicted_residual
        target = coarse + target_residual

        terms: dict[str, torch.Tensor] = {}
        terms["height_l1"] = F.l1_loss(predicted, target)

        pred_dx, pred_dy = gradient_xy(predicted)
        tgt_dx, tgt_dy = gradient_xy(target)
        terms["gradient_l1"] = F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)

        terms["laplacian_l1"] = F.l1_loss(laplacian(predicted), laplacian(target))
        terms["spectral_l1"] = F.l1_loss(spectral_magnitude(predicted), spectral_magnitude(target))
        terms["seam_l1"] = F.l1_loss(seam_edges(predicted), seam_edges(target))

        total = predicted.new_tensor(0.0)
        for name, value in terms.items():
            total = total + value * self.weights[name]

        metrics = {name: float(value.detach().cpu()) for name, value in terms.items()}
        metrics["total"] = float(total.detach().cpu())
        return total, metrics
