"""Comparison functions."""

import torch
from torch.nn import MSELoss


def RMSE(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: N802
    """RMSE.

    Args:
        x (torch.Tensor): First tensor to compare.
        y (torch.Tensor): Second tensor to compare.

    Returns:
        torch.Tensor: Scalar: sqrt(mean((x-y)**2)).
    """
    return torch.sqrt(MSELoss()(x, y))


def absolute_difference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Absolute difference.

    Args:
        x (torch.Tensor): First tensor to compare.
        y (torch.Tensor): Second tensor to compare.

    Returns:
        torch.Tensor: The output has the size of the inputs and is |x-y|.
    """
    return torch.abs(x - y)
