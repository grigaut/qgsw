"""Loss-updating methods."""

from __future__ import annotations

import torch


def rmse(f: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
    """RMSE."""
    return ((f - f_ref).square().mean() / f_ref.square().mean()).sqrt()


def mse(
    f: torch.Tensor, f_ref: torch.Tensor, *, variance: float | torch.Tensor = 1
) -> torch.Tensor:
    """MSE."""
    return (f - f_ref).square().sum() / variance


def update_loss(
    loss: torch.Tensor,
    f: torch.Tensor,
    f_ref: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    variance: float | torch.Tensor = 1,
) -> torch.Tensor:
    """Update loss."""
    if mask is None:
        mask = torch.ones_like(f_ref, dtype=torch.bool)
    if not mask.any():
        return loss
    f_sliced = f.flatten()[mask.flatten()]
    f_ref_sliced = f_ref.flatten()[mask.flatten()]
    return loss + mse(f_sliced, f_ref_sliced, variance=variance)
