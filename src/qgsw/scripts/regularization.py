"""Regularization-related functions."""

from collections.abc import Callable

import torch

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.models.core.flux import div_flux_5pts_only, div_flux_5pts_with_bc
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.solver.finite_diff import grad, grad_perp, laplacian
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.utils.reshaping import crop


def compute_qg_dynamical_regularization_func(
    psi2_basis: SpaceTimeDecomposition[
        SpaceSupportFunction, TimeSupportFunction
    ],
    A: torch.Tensor,
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    beta_effect: torch.Tensor,
    *,
    wide: bool = False,
    scale: float = 1,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build QG regularization function.

    Args:
        psi2_basis (SpaceTimeDecomposition[ SpaceSupportFunction, TimeSupportFunction ]):
            Basis
        A (torch.Tensor): Stretching matrix.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta plane.
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

    f0 = beta_plane.f0

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

    def compute_reg(
        psi1: torch.Tensor,
        dpsi1: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute QG regularization term.

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

    return compute_reg


def compute_div_dynamical_regularization_func(
    psi_a_basis: SpaceTimeDecomposition[
        SpaceSupportFunction, TimeSupportFunction
    ],
    chi_a_basis: SpaceTimeDecomposition[
        SpaceSupportFunction, TimeSupportFunction
    ],
    space: SpaceDiscretization2D,
    beta_plane: BetaPlane,
    beta_effect: torch.Tensor,
    *,
    wide: bool = False,
    scale: float = 1,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build divergence regularization term.

    Args:
        psi_a_basis (SpaceTimeDecomposition[ SpaceSupportFunction, TimeSupportFunction ]): Psi_a basis.
        chi_a_basis (SpaceTimeDecomposition[ SpaceSupportFunction, TimeSupportFunction ]): Chi_a basis.
        space (SpaceDiscretization2D): Space.
        beta_plane (BetaPlane): Beta plane.
        beta_effect (torch.Tensor): Beta effect.
        wide (bool, optional): Whether to use wide stencil or not. Defaults to False.
        scale (float, optional): Scaling. Defaults to 1.

    Returns:
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: _description_
    """  # noqa: E501
    f_div = chi_a_basis.localize_laplacian(*space.q.xy)
    f_chi_a = chi_a_basis.localize(*space.widen(1, 1).q.xy)
    f_psi_a = psi_a_basis.localize(*space.psi.xy)
    f_zeta_a = psi_a_basis.localize_laplacian(
        *space.crop(1 + 2 * wide, 1 + 2 * wide).q.xy
    )
    f0, beta = beta_plane
    dx, dy = space.dx, space.dy

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

    def compute_reg(psi_g: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Compute divergence regularization term.

        Args:
            psi_g (torch.Tensor): Geostrophic stream function.
            time (torch.Tensor): Time.

        Returns:
            torch.Tensor: LHS of surface divergence equation.
        """
        dt_div = crop(
            f_div.dt(time)[None, None, ...], 1 + 2 * wide
        )  # (nx-2,ny-2)
        div = f_div(time)[None, None, ...]
        psi_a = f_psi_a(time)[None, None, ...]  # (nx+1,ny+1)
        chi_a = f_chi_a(time)[None, None, ...]  # (nx+2,ny+2)
        zeta_a = f_zeta_a(time)[None, None, ...]  # (nx-2,ny-2)

        u_g, v_g = grad_perp(psi_g)
        u_g /= dy  # (nx+1, ny)
        v_g /= dx  # (ny, nx+1)

        u_sf_a, v_sf_a = grad_perp(psi_a)
        u_sf_a /= dy  # (nx+1, ny)
        v_sf_a /= dx  # (ny, nx+1)

        u_pot_a_, v_pot_a_ = grad(chi_a)
        u_pot_a_ /= dx
        v_pot_a_ /= dy
        u_pot_a = crop(u_pot_a_, 0, 1)  # (nx+1, ny)
        v_pot_a = crop(v_pot_a_, 1, 0)  # (ny, nx+1)

        u = u_g + u_sf_a + u_pot_a  # (nx+1, ny)
        v = v_g + v_sf_a + v_pot_a  # (ny, nx+1)

        div_udiv = div_flux(
            div, crop(u, 1 + 2 * wide), crop(v, 1 + 2 * wide)
        )  # (nx-2,ny-2)

        dx_u_ = (u[..., 1:, :] - u[..., :-1, :]) / dx  # (nx,ny)
        dy_u_ = (u[..., 1:] - u[..., :-1]) / dy  # (nx+1,ny-1)
        dx_v_ = (v[..., 1:, :] - v[..., :-1, :]) / dx  # (nx-1,ny+1)
        dy_v_ = (v[..., 1:] - v[..., :-1]) / dy  # (nx,ny)

        dx_u = crop(dx_u_, 1 + 2 * wide)  # (nx-2,ny-2)
        dy_u = interpolate(
            crop(dy_u_, 1 + 2 * wide, 0 + 2 * wide)
        )  # (nx-2,ny-2)
        dx_v = interpolate(
            crop(dx_v_, 0 + 2 * wide, 1 + 2 * wide)
        )  # (nx-2,ny-2)
        dy_v = crop(dy_v_, 1 + 2 * wide)  # (nx-2,ny-2)

        u_i = crop((u[..., 1:, :] + u[..., :-1, :]) / 2, 1 + 2 * wide)

        return (
            (
                (
                    dt_div
                    + div_udiv
                    + 2 * (dx_v * dy_u - dy_v * dx_u)
                    - (f0 + beta_effect[..., 1 + 2 * wide : -(1 + 2 * wide)])
                    * zeta_a
                    + beta * u_i
                )
                / scale
            )
            .square()
            .sum()
        )

    return compute_reg
