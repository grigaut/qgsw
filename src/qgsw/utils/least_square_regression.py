"""Linear least square regression."""

from __future__ import annotations

import torch


def perform_linear_least_squares_regression(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Perform linear least squares regression.

    Solving for A such that A = argmin ||Y-Xβ||².

    Args:
        x (torch.Tensor): Regressors (e,n,l)-shaped.
        y (torch.Tensor): Target tensor, (e,n)-shaped.

    Returns:
        torch.Tensor: β, (p,m)-shaped
    """
    xT = x.transpose(-2, -1)  # noqa: N806
    xTx = torch.einsum("...ln,...nL->...lL", xT, x)  # noqa: N806
    xTxinv_xT: torch.Tensor = torch.linalg.solve(xTx, xT)  # noqa: N806
    return torch.einsum("...ln,...n->...l", xTxinv_xT, y)
