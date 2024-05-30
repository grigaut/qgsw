# ruff: noqa
"""Finite difference operators in pytorch,
Louis Thiry, 6 march 2023."""

import torch
import torch.nn.functional as F


def comp_ke(
    u: torch.Tensor,
    U: torch.Tensor,
    v: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Compute Kinetic Energy.

    Args:
        u (torch.Tensor): Prognostic zonal velocity.
        U (torch.Tensor): Diagnostic zonal velocity.
        v (torch.Tensor): Prognostic meridional velocity.
        V (torch.Tensor): Diagnostic meridional velocity.

    Returns:
        torch.Tensor: Kinetic Energy
    """
    u_sq = u * U
    v_sq = v * V
    return 0.25 * (
        u_sq[..., 1:, :] + u_sq[..., :-1, :] + v_sq[..., 1:] + v_sq[..., :-1]
    )


def laplacian(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    return (
        f[..., 2:, 1:-1] - 2 * f[..., 1:-1, 1:-1] + f[..., :-2, 1:-1]
    ) / dx**2 + (
        f[..., 1:-1, 2:] - 2 * f[..., 1:-1, 1:-1] + f[..., 1:-1, :-2]
    ) / dy**2


def grad_perp(f: torch.Tensor) -> torch.Tensor:
    """Orthogonal gradient"""
    return f[..., :-1] - f[..., 1:], f[..., 1:, :] - f[..., :-1, :]


def div_nofluxbc(flux_x: torch.Tensor, flux_y: torch.Tensor) -> torch.Tensor:
    return torch.diff(F.pad(flux_y, (1, 1)), dim=-1) + torch.diff(
        F.pad(flux_x, (0, 0, 1, 1)), dim=-2
    )
