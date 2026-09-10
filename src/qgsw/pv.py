"""Potential vorticity computation."""

import torch

from qgsw.solver.boundary_conditions.base import Boundaries
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


def compute_q1_interior_bcs(
    psi1: Boundaries,
    psi2: Boundaries,
    A11: torch.Tensor,
    A12: torch.Tensor,
    dx: float,
    dy: float,
    f0: float,
    beta_effect: Boundaries,
) -> Boundaries:
    """Compute potential vorticity boundaries in the top layer interior.

    WARNING: In order to compute PV, the stream function boundary must be wide
        enough. The width of the boundary will be reduced by 4
        (two points reduction on each side of each boundary tensor). Hence
        a 6-points wide stream function boundary will lead to a 2-point wide
        PV boundary.

    Args:
        psi1 (Boundaries): Top layer stream function boundaries.
        psi2 (Boundaries): Second layer stream function boundaries.
        A11 (torch.Tensor): 1st row, 1st column component of the
            stretching matrix
        A12 (torch.Tensor): 1st row, 2nd column component of the
            stretching matrix
        dx (float): Horizontal distance step in the X direction.
        dy (float): Horizontal distance step in the Y direction.
        f0 (float): Coriolis parameter.
        beta_effect (Boundaries): Beta effect.

    Returns:
        Boundaries: Δѱ₁ - f₀² / H₁ (1/g₁ + 1/g₂) ѱ₁ + (f₀² / H₁ /  g₂) ѱ₂
    """
    f = lambda psi1, psi2, beta: compute_q1_interior(
        psi1, psi2, A11, A12, dx, dy, f0, beta
    )
    return Boundaries(
        top=f(psi1.top, psi2.top, beta_effect.top),
        bottom=f(psi1.bottom, psi2.bottom, beta_effect.bottom),
        left=f(psi1.left, psi2.left, beta_effect.left),
        right=f(psi1.right, psi2.right, beta_effect.right),
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
