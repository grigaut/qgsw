"""Potential vorticity computation."""

import torch

from qgsw.solver.finite_diff import laplacian
from qgsw.spatial.core.grid_conversion import interpolate


def compute_q1_interior(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    A11: torch.Tensor,
    A12: torch.Tensor,
    dx: float,
    dy: float,
    f0: float,
    beta_effect: torch.Tensor,
) -> torch.Tensor:
    """Compute potential vorticity in the top layer interior.

    WARNING: This function only compute potential vorticity in
        the **interior**.

    Args:
        psi1 (torch.Tensor): Top layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        psi2 (torch.Tensor): Second layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        A11 (torch.Tensor): 1st row, 1st column component of the
            stretching matrix
        A12 (torch.Tensor): 1st row, 2nd column component of the
            stretching matrix
        dx (float): Horizontal distance step in the X direction.
        dy (float): Horizontal distance step in the Y direction.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta effect.
            └── (1, ny+1)-shaped

    Returns:
        torch.Tensor: Δѱ₁ - f₀² / H₁ (1/g₁ + 1/g₂) ѱ₁ + (f₀² / H₁ /  g₂) ѱ₂
            └── (n_ens, 1, nx-2, ny-2)-shaped
    """
    return (
        interpolate(
            laplacian(psi1, dx, dy)
            - f0**2
            * (A11 * psi1[..., 1:-1, 1:-1] + A12 * psi2[..., 1:-1, 1:-1])
        )
        + beta_effect
    )


def compute_q2_2l_interior(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    A21: torch.Tensor,
    A22: torch.Tensor,
    dx: float,
    dy: float,
    f0: float,
    beta_effect: torch.Tensor,
) -> torch.Tensor:
    """Compute potential vorticity in the second layer interior.

    WARNING: This function considers ѱ₃ = 0.
    WARNING: This function only compute potential vorticity in
        the **interior**.

    Args:
        psi1 (torch.Tensor): Top layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        psi2 (torch.Tensor): Second layer stream function.
            └── (n_ens, 1, nx+1, ny+1)-shaped
        A21 (torch.Tensor): 2nd row, 1st column component of the
            stretching matrix
        A22 (torch.Tensor): 2nd row, 2nd column component of the
            stretching matrix
        dx (float): Horizontal distance step in the X direction.
        dy (float): Horizontal distance step in the Y direction.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta effect.
            └── (1, ny+1)-shaped

    Returns:
        torch.Tensor: Δѱ₂ - (f₀² / H₂ /g₂) (ѱ₂ - ѱ₁)
            └── (n_ens, 1, nx-2, ny-2)-shaped
    """
    return (
        interpolate(
            laplacian(psi2, dx, dy)
            - f0**2
            * (A22 * psi2[..., 1:-1, 1:-1] + A21 * psi1[..., 1:-1, 1:-1])
        )
        + beta_effect
    )
