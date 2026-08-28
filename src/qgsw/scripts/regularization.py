"""Regularization-related functions."""

from collections.abc import Callable

import torch

from qgsw import specs
from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.models.qg.stretching_matrix import compute_A_tilde
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.solver.finite_diff import grad
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.utils.reshaping import crop


def compute_regularization_func(
    psi2_basis: SpaceTimeDecomposition[
        SpaceSupportFunction, TimeSupportFunction
    ],
    H: torch.Tensor,
    g_prime: torch.Tensor,
    alpha: torch.Tensor,
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    scale: float,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build regularization function.

    Args:
        psi2_basis (SpaceTimeDecomposition[ SpaceSupportFunction, TimeSupportFunction ]):
            Basis
        H (torch.Tensor): Layers thickness.
        g_prime (torch.Tensor): Reduced gravity.
        alpha (torch.Tensor): Baroclinic radius perturbation.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta-plane
        scale (float): Regularization scaling value.

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
            Regularization function.
    """  # noqa: E501
    A_tilde = compute_A_tilde(
        H[:2],
        g_prime[:2],
        alpha,
        **specs.from_tensor(H),
    )
    A_21 = A_tilde[1:2, :1]
    A_22 = A_tilde[1:2, 1:2]

    q = space.q.xy
    x = crop(q.x, 1)
    y = crop(q.y, 1)

    fpsi2 = psi2_basis.localize(x, y)
    fdx_psi2 = psi2_basis.localize_dx(x, y)
    fdy_psi2 = psi2_basis.localize_dy(x, y)
    flap_psi2 = psi2_basis.localize_laplacian(x, y)
    fdx_lap_psi2 = psi2_basis.localize_dx_laplacian(x, y)
    fdy_lap_psi2 = psi2_basis.localize_dy_laplacian(x, y)

    def compute_reg(
        psi1: torch.Tensor,
        dpsi1: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regularization term.

        Args:
            psi1 (torch.Tensor): Top layer stream function.
            dpsi1 (torch.Tensor): Top layer stream function derivative.
            time (torch.Tensor): Time.

        Returns:
            torch.Tensor: ∂ₜq₂ + J(ѱ₂, q₂)
        """
        dt_lap_psi2 = flap_psi2.dt(time)
        dt_psi2 = fpsi2.dt(time)

        dt_q2 = dt_lap_psi2 - beta_plane.f0**2 * (
            A_22 * dt_psi2 + A_21 * interpolate(crop(dpsi1, 1))
        )

        dx_psi1, dy_psi1 = grad(psi1)
        dx_psi1 /= space.dx
        dy_psi1 /= space.dy

        dx_psi1_i = (dx_psi1[..., 1:] + dx_psi1[..., :-1]) / 2
        dy_psi1_i = (dy_psi1[..., 1:, :] + dy_psi1[..., :-1, :]) / 2

        dx_psi2 = fdx_psi2(time)
        dy_psi2 = fdy_psi2(time)

        dy_q2 = (
            fdy_lap_psi2(time)
            - beta_plane.f0**2 * (A_22 * dy_psi2 + A_21 * crop(dy_psi1_i, 1))
        ) + beta_plane.beta

        dx_q2 = fdx_lap_psi2(time) - beta_plane.f0**2 * (
            A_22 * dx_psi2 + A_21 * crop(dx_psi1_i, 1)
        )

        adv_q2 = -dy_psi2 * dx_q2 + dx_psi2 * dy_q2
        return ((dt_q2 + adv_q2) / scale).square().sum()

    return compute_reg
