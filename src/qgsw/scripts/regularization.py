"""Regularization-related functions."""

from collections.abc import Callable

import torch

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.models.core.flux import div_flux_5pts_only, div_flux_5pts_with_bc
from qgsw.solver.finite_diff import grad_perp, laplacian
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.utils.reshaping import crop


def compute_dynamical_regularization_func(
    psi2_basis: SpaceTimeDecomposition[
        SpaceSupportFunction, TimeSupportFunction
    ],
    A: torch.Tensor,
    space: SpaceDiscretization2D,
    f0: float,
    beta_effect: torch.Tensor,
    *,
    wide: bool = False,
    scale: float = 1,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build regularization function.

    Args:
        psi2_basis (SpaceTimeDecomposition[ SpaceSupportFunction, TimeSupportFunction ]):
            Basis
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        f0 (float): Coriolis parameter.
        beta_effect (torch.Tensor): Beta-effect.
        y0 (float): Reference
        wide (bool): Whether to use wide stencil flux computation method or not.
        scale (float, optional): Regularization scaling value. Defaults to 1.

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
            Regularization function.
    """  # noqa: E501
    if (A.shape[0] <= 1) or (A.shape[1] <= 1):
        msg = "A must be at least (2, 2)-shaped."
        raise ValueError(msg)

    A_21 = A[1:2, :1]
    A_22 = A[1:2, 1:2]

    dx, dy = space.dx, space.dy
    x = space.q.xy.x
    y = space.q.xy.y

    fpsi2 = psi2_basis.localize(x, y)

    if wide:
        div_flux = lambda q, u, v: div_flux_5pts_only(
            q,
            u,
            v,
            dx,
            dy,
        )
    else:
        div_flux = lambda q, u, v: div_flux_5pts_with_bc(
            q,
            u,
            v,
            dx,
            dy,
        )

    def compute_reg_(
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
        # (Indicative sizes are given for wide=False in comment)
        dt_psi2 = crop(fpsi2.dt(time), 1 + 2 * wide)  # (nx-2, ny-2)
        dt_psi2_ = crop(dt_psi2, 1)  # (nx-4, ny-4))
        lap_dt_psi2 = laplacian(dt_psi2, dx, dy)  # (nx-4, ny-4))

        dt_psi1 = interpolate(crop(dpsi1, 2 + 2 * wide))  # (nx-4, ny-4))

        dt_q2 = lap_dt_psi2 - f0**2 * (A_21 * dt_psi1 + A_22 * dt_psi2_)
        psi2 = fpsi2(time)  #  (nx, ny)
        psi2_ = crop(psi2, 1)  #  (nx-2, ny-2)
        lap_psi2 = laplacian(psi2, dx, dy)  # (nx-2, ny-2)

        psi1_ = interpolate(crop(psi1, 1))  # (nx-2, ny-2)

        beta_ = beta_effect[..., 1:-1]  # (1, ny-2)
        q2 = lap_psi2 - f0**2 * (A_21 * psi1_ + A_22 * psi2_) + beta_
        u, v = grad_perp(crop(psi1, 2 + 2 * wide))
        u /= dy  # (nx-3, ny-4)
        v /= dx  # (nx-4, ny-3)
        adv_q2 = div_flux(q2, u, v)  # (nx-4, ny-4))
        return ((dt_q2 + adv_q2) / scale).square().sum()

    return compute_reg_
