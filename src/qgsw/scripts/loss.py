"""Loss-updating methods."""

from __future__ import annotations

import torch

from qgsw import specs


def eval_loss(
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
        return torch.tensor(0, **specs.from_tensor(f))
    f_sliced = f.flatten()[mask.flatten()]
    f_ref_sliced = f_ref.flatten()[mask.flatten()]
    return (f_sliced - f_ref_sliced).square().sum() / variance


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
    return loss + (f_sliced - f_ref_sliced).square().sum() / variance
