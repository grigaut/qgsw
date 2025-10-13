# ruff: noqa: PGH004
# ruff: noqa
"""Velocity-sign biased flux computations.
Louis Thiry, 2023
"""

from typing import Callable

import torch
import torch.nn.functional as F

from qgsw.logging import getLogger
from qgsw.masks import Masks
from qgsw.models.core import reconstruction
from qgsw.models.core.utils import OptimizableFunction

logger = getLogger(__name__)


def stencil_2pts(
    q: torch.Tensor,
    dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Extract 2 pts stencils from q.

    Example:
    If q is 2D and dim = -2, the output is:
    (
        q[:-1, :],
        q[1:, :],
    )
    If q is 2D and dim = -1, the output is:
    (
        q[:, :-1],
        q[:, 1:],
    )

    Args:
        q (torch.Tensor): Tensor.
        dim (int): Dimension to narrow.

    Returns:
        tuple[ torch.Tensor, torch.Tensor, ]: Narrowed tensors.
    """
    n = q.shape[dim]
    return (
        q.narrow(dim, 0, n - 1),
        q.narrow(dim, 1, n - 1),
    )


def stencil_4pts(
    q: torch.Tensor,
    dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract 4 pts stencils from q.

    Example:
    If q is 2D and dim = -2, the output is:
    (
        q[:-3, :],
        q[1:-2, :],
        q[2:-1, :],
        q[3:-2, :],
    )
    If q is 2D and dim = -1, the output is:
    (
        q[:, :-3],
        q[:, 1:-2],
        q[:, 2:-1],
        q[:, 3:-2],
    )

    Args:
        q (torch.Tensor): Tensor.
        dim (int): Dimension to narrow.

    Returns:
        tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ]: All
        4 narrowed tensors.
    """
    n = q.shape[dim]
    return (
        q.narrow(dim, 0, n - 3),
        q.narrow(dim, 1, n - 3),
        q.narrow(dim, 2, n - 3),
        q.narrow(dim, 3, n - 3),
    )


def stencil_6pts(
    q: torch.Tensor,
    dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Extract 6 pts stencils from q.

    Example:
    If q is 2D and dim = -2, the output is:
    (
        q[:-5, :],
        q[1:-4, :],
        q[2:-3, :],
        q[3:-2, :],
        q[4:-1, :],
        q[5:, :],
    )
    If q is 2D and dim = -1, the output is:
    (
        q[:, :-5],
        q[:, 1:-4],
        q[:, 2:-3],
        q[:, 3:-2],
        q[:, 4:-1],
        q[:, 5:],
    )

    Args:
        q (torch.Tensor): Tensor.
        dim (int): Dimension to narrow.


    Returns:
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]: All 6 ,narrowed tensors.
    """
    n = q.shape[dim]
    return (
        q.narrow(dim, 0, n - 5),
        q.narrow(dim, 1, n - 5),
        q.narrow(dim, 2, n - 5),
        q.narrow(dim, 3, n - 5),
        q.narrow(dim, 4, n - 5),
        q.narrow(dim, 5, n - 5),
    )


def flux(
    q: torch.Tensor,
    u: torch.Tensor,
    dim: int,
    n_points: int,
    rec_func_2: Callable,
    rec_func_4: Callable,
    rec_func_6: Callable,
    mask_2: torch.Tensor,
    mask_4: torch.Tensor,
    mask_6: torch.Tensor,
) -> torch.Tensor:
    # positive and negative velocities
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # 2-points reconstruction
    q_stencil2 = stencil_2pts(q, dim)
    # Positive velocities: use direct stencil
    qi2_pos = rec_func_2(*q_stencil2)
    # Negative velocities: use reversed stencil
    qi2_neg = rec_func_2(*q_stencil2[::-1])

    if n_points == 2:
        return u_pos * qi2_pos + u_neg * qi2_neg

    # 4-points reconstruction
    pad = (1, 1, 0, 0) if dim == -1 else (0, 0, 1, 1)
    q_stencil4 = stencil_4pts(q, dim)
    # Positive velocities: use padded direct stencil
    qi4_pos = F.pad(rec_func_4(*q_stencil4), pad)
    # Negative velocities: use padded reversed stencil
    qi4_neg = F.pad(rec_func_4(*q_stencil4[::-1]), pad)

    if n_points == 4:
        return u_pos * (mask_2 * qi2_pos + mask_4 * qi4_pos) + u_neg * (
            mask_2 * qi2_neg + mask_4 * qi4_neg
        )

    # 6-points reconstruction
    pad = (2, 2, 0, 0) if dim == -1 else (0, 0, 2, 2)
    q_stencil6 = stencil_6pts(q, dim)
    # Positive velocities: use padded direct stencil
    qi6_pos = F.pad(rec_func_6(*q_stencil6), pad)
    # Negative velocities: use padded reversed stencil
    qi6_neg = F.pad(rec_func_6(*q_stencil6[::-1]), pad)

    if n_points == 6:
        return u_pos * (
            mask_2 * qi2_pos + mask_4 * qi4_pos + mask_6 * qi6_pos
        ) + u_neg * (mask_2 * qi2_neg + mask_4 * qi4_neg + mask_6 * qi6_neg)


def flux_1pts(q, u, dim):
    n = q.shape[dim]

    qi_left = q.narrow(dim, 0, n - 1)
    qi_right = q.narrow(dim, 1, n - 1)

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def flux_3pts(q, u, dim):
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 3 points inside domain.
      - 1 point near boundaries.

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
    """
    n = q.shape[dim]

    # q-interpolation: 3-points inside domain
    qm, q0, qp = (
        q.narrow(dim, 0, n - 2),
        q.narrow(dim, 1, n - 2),
        q.narrow(dim, 2, n - 2),
    )
    qi_left_in = reconstruction.linear3_left(qm, q0, qp)
    qi_right_in = reconstruction.linear3_left(qp, q0, qm)

    # q-interpolation: 2-points on boundaries
    qi_0 = reconstruction.linear2_centered(
        q.narrow(dim, 0, 1), q.narrow(dim, 1, 1)
    )
    qi_m1 = reconstruction.linear2_centered(
        q.narrow(dim, -2, 1), q.narrow(dim, -1, 1)
    )

    qi_left = torch.cat(
        [qi_0, qi_left_in.narrow(dim, 0, n - 3), qi_m1], dim=dim
    )
    qi_right = torch.cat(
        [qi_0, qi_right_in.narrow(dim, 1, n - 3), qi_m1], dim=dim
    )

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right
    return flux


def div_flux_3pts(q, u, v, dx, dy):
    q_flux_y = F.pad(flux_3pts(q, v, dim=-1), (1, 1, 0, 0))
    q_flux_x = F.pad(flux_3pts(q, u, dim=-2), (0, 0, 1, 1))

    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def flux_5pts(q, u, dim):
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 5 points inside domain.
      - 1 or 3 points near boundaries.

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
        qi: tracer field interpolated on u points, torch.Tensor, shape[dim] = n-1
    """

    n = q.shape[dim]

    # 5-points inside domain
    qmm, qm, q0, qp, qpp = (
        q.narrow(dim, 0, n - 4),
        q.narrow(dim, 1, n - 4),
        q.narrow(dim, 2, n - 4),
        q.narrow(dim, 3, n - 4),
        q.narrow(dim, 4, n - 4),
    )
    qi_left_in = reconstruction.linear5_left(qmm, qm, q0, qp, qpp)
    qi_right_in = reconstruction.linear5_left(qpp, qp, q0, qm, qmm)
    # qi_left_in = weno5z(qmm, qm, q0, qp, qpp)
    # qi_right_in = weno5z(qpp, qp, q0, qm, qmm)

    # 3pts-2pts near boundary
    qm, q0, qp = (
        torch.cat([q.narrow(dim, 0, 1), q.narrow(dim, -3, 1)], dim=dim),
        torch.cat([q.narrow(dim, 1, 1), q.narrow(dim, -2, 1)], dim=dim),
        torch.cat([q.narrow(dim, 2, 1), q.narrow(dim, -1, 1)], dim=dim),
    )
    qi_left_b = reconstruction.weno3z(qm, q0, qp)
    qi_right_b = reconstruction.weno3z(qp, q0, qm)

    qi_0 = reconstruction.linear2_centered(
        q.narrow(dim, 0, 1), q.narrow(dim, 1, 1)
    )
    qi_m1 = reconstruction.linear2_centered(
        q.narrow(dim, -2, 1), q.narrow(dim, -1, 1)
    )

    qi_left = torch.cat(
        [
            qi_0,
            qi_left_b.narrow(dim, 0, 1),
            qi_left_in,
            qi_left_b.narrow(dim, -1, 1),
        ],
        dim=dim,
    )
    qi_right = torch.cat(
        [
            qi_right_b.narrow(dim, 0, 1),
            qi_right_in,
            qi_right_b.narrow(dim, -1, 1),
            qi_m1,
        ],
        dim=dim,
    )

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def div_flux_5pts(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the divergence [uq, vq], assuming 0 on the boundary for q.

    Args:
        q (torch.Tensor): Tracer field to compute the div flux of.
                └── (n_ens, nl, nx, ny)-shaped
        u (torch.Tensor): Velocity in the zonal direction.
                └── (n_ens, nl, nx+1, ny)-shaped
        v (torch.Tensor): Velocity in the meridional direction.
                └── (n_ens, nl, nx, ny+1)-shaped
        dx (float): Infinitesimal distance in the x direction.
        dy (float): Infinitesimal distance in the x direction.

    Returns:
        torch.Tensor: ∇ · ([u v] q)
            └── (n_ens, nl, nx, ny)-shaped
    """
    q_flux_y = F.pad(flux_5pts(q, v, dim=-1), (1, 1, 0, 0))
    q_flux_x = F.pad(flux_5pts(q, u, dim=-2), (0, 0, 1, 1))

    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def flux_5pts_only(q: torch.Tensor, u: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 5 points inside domain only

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n - 5
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n - 5
        qi: tracer field interpolated on u points, torch.Tensor, shape[dim] = n - 5
    """

    n = q.shape[dim]

    # 5-points inside domain
    qmm, qm, q0, qp, qpp = (
        q.narrow(dim, 0, n - 4),
        q.narrow(dim, 1, n - 4),
        q.narrow(dim, 2, n - 4),
        q.narrow(dim, 3, n - 4),
        q.narrow(dim, 4, n - 4),
    )
    qi_left_in = reconstruction.linear5_left(qmm, qm, q0, qp, qpp)
    qi_right_in = reconstruction.linear5_left(qpp, qp, q0, qm, qmm)

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    qi_left = qi_left_in.narrow(dim, 0, n - 5)
    qi_right = qi_right_in.narrow(dim, 1, n - 5)

    # upwind flux computation
    return u_pos * qi_left + u_neg * qi_right


def div_flux_5pts_only(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the divergence [uq, vq], using a large q field.

    The large q field allows using 5 pts linear reconstruction evry where in the domain.

    Args:
        q (torch.Tensor): Tracer field to compute the div flux of.
                └── (n_ens, nl, nx+6, ny+6)-shaped
        u (torch.Tensor): Velocity in the zonal direction.
                └── (n_ens, nl, nx+1, ny)-shaped
        v (torch.Tensor): Velocity in the meridional direction.
                └── (n_ens, nl, nx, ny+1)-shaped
        dx (float): Infinitesimal distance in the x direction.
        dy (float): Infinitesimal distance in the x direction.

    Returns:
        torch.Tensor: ∇ · ([u v] q)
            └── (n_ens, nl, nx, ny)-shaped
    """
    q_flux_y = flux_5pts_only(q[..., 3:-3, :], v, dim=-1)
    q_flux_x = flux_5pts_only(q[..., :, 3:-3], u, dim=-2)

    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def flux_5_pts_replicate_qi_boundaries(
    q: torch.Tensor,
    u: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 5 points inside domain.
      - 1 or 3 points near boundaries.

    To remove the 0-padding at the boundary, the reconstruction of q
    is replicated on the outer boundary: qi(x) = qi(x+dx).

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
        qi: tracer field interpolated on u points, torch.Tensor, shape[dim] = n-1
    """
    n = q.shape[dim]

    # 5-points inside domain
    qmm, qm, q0, qp, qpp = (
        q.narrow(dim, 0, n - 4),
        q.narrow(dim, 1, n - 4),
        q.narrow(dim, 2, n - 4),
        q.narrow(dim, 3, n - 4),
        q.narrow(dim, 4, n - 4),
    )
    qi_left_in = reconstruction.linear5_left(qmm, qm, q0, qp, qpp)
    qi_right_in = reconstruction.linear5_left(qpp, qp, q0, qm, qmm)
    # qi_left_in = weno5z(qmm, qm, q0, qp, qpp)
    # qi_right_in = weno5z(qpp, qp, q0, qm, qmm)

    # 3pts-2pts near boundary
    qm, q0, qp = (
        torch.cat([q.narrow(dim, 0, 1), q.narrow(dim, -3, 1)], dim=dim),
        torch.cat([q.narrow(dim, 1, 1), q.narrow(dim, -2, 1)], dim=dim),
        torch.cat([q.narrow(dim, 2, 1), q.narrow(dim, -1, 1)], dim=dim),
    )
    qi_left_b = reconstruction.weno3z(qm, q0, qp)
    qi_right_b = reconstruction.weno3z(qp, q0, qm)

    qi_0 = reconstruction.linear2_centered(
        q.narrow(dim, 0, 1), q.narrow(dim, 1, 1)
    )
    qi_m1 = reconstruction.linear2_centered(
        q.narrow(dim, -2, 1), q.narrow(dim, -1, 1)
    )

    qi_left = torch.cat(
        [
            qi_0,
            qi_0,
            qi_left_b.narrow(dim, 0, 1),
            qi_left_in,
            qi_left_b.narrow(dim, -1, 1),
            qi_left_b.narrow(dim, -1, 1),
        ],
        dim=dim,
    )
    qi_right = torch.cat(
        [
            qi_right_b.narrow(dim, 0, 1),
            qi_right_b.narrow(dim, 0, 1),
            qi_right_in,
            qi_right_b.narrow(dim, -1, 1),
            qi_m1,
            qi_m1,
        ],
        dim=dim,
    )

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def div_flux_5pts_replicate_qi_boundaries(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the divergence [uq, vq].

    The value of the reconstruction q is duplicated on the border of the domain
    using next order decomposition: qi(x) = qi(x+dx).

    Args:
        q (torch.Tensor): Tracer field to compute the div flux of.
                └── (n_ens, nl, nx, ny)-shaped
        u (torch.Tensor): Velocity in the zonal direction.
                └── (n_ens, nl, nx+1, ny)-shaped
        v (torch.Tensor): Velocity in the meridional direction.
                └── (n_ens, nl, nx, ny+1)-shaped
        dx (float): Infinitesimal distance in the x direction.
        dy (float): Infinitesimal distance in the x direction.

    Returns:
        torch.Tensor: ∇ · ([u v] q)
            └── (n_ens, nl, nx, ny)-shaped
    """
    q_flux_y = flux_5_pts_replicate_qi_boundaries(q, v, dim=-1)
    q_flux_x = flux_5_pts_replicate_qi_boundaries(q, u, dim=-2)

    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def flux_5_pts_replicate_qi_boundaries_next_order(
    q: torch.Tensor, u: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 5 points inside domain.
      - 1 or 3 points near boundaries.

    To remove the 0-padding at the boundary, the reconstruction of q
    is replicated on the outer boundary: qi(x) = 2qi(x+dx) - qi(x+2dx).

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
        qi: tracer field interpolated on u points, torch.Tensor, shape[dim] = n-1
    """
    n = q.shape[dim]

    # 5-points inside domain
    qmm, qm, q0, qp, qpp = (
        q.narrow(dim, 0, n - 4),
        q.narrow(dim, 1, n - 4),
        q.narrow(dim, 2, n - 4),
        q.narrow(dim, 3, n - 4),
        q.narrow(dim, 4, n - 4),
    )
    qi_left_in = reconstruction.linear5_left(qmm, qm, q0, qp, qpp)
    qi_right_in = reconstruction.linear5_left(qpp, qp, q0, qm, qmm)
    # qi_left_in = weno5z(qmm, qm, q0, qp, qpp)
    # qi_right_in = weno5z(qpp, qp, q0, qm, qmm)

    # 3pts-2pts near boundary
    qm, q0, qp = (
        torch.cat([q.narrow(dim, 0, 1), q.narrow(dim, -3, 1)], dim=dim),
        torch.cat([q.narrow(dim, 1, 1), q.narrow(dim, -2, 1)], dim=dim),
        torch.cat([q.narrow(dim, 2, 1), q.narrow(dim, -1, 1)], dim=dim),
    )
    qi_left_b = reconstruction.weno3z(qm, q0, qp)
    qi_right_b = reconstruction.weno3z(qp, q0, qm)

    qi_0 = reconstruction.linear2_centered(
        q.narrow(dim, 0, 1), q.narrow(dim, 1, 1)
    )
    qi_m1 = reconstruction.linear2_centered(
        q.narrow(dim, -2, 1), q.narrow(dim, -1, 1)
    )

    qi_left = torch.cat(
        [
            2 * qi_0 - qi_left_b.narrow(dim, 0, 1),
            qi_0,
            qi_left_b.narrow(dim, 0, 1),
            qi_left_in,
            qi_left_b.narrow(dim, -1, 1),
            2 * qi_left_b.narrow(dim, -1, 1) - qi_left_in.narrow(dim, -1, 1),
        ],
        dim=dim,
    )
    qi_right = torch.cat(
        [
            2 * qi_right_b.narrow(dim, 0, 1) - qi_right_in.narrow(dim, 0, 1),
            qi_right_b.narrow(dim, 0, 1),
            qi_right_in,
            qi_right_b.narrow(dim, -1, 1),
            qi_m1,
            2 * qi_m1 - qi_right_b.narrow(dim, -1, 1),
        ],
        dim=dim,
    )

    # positive and negative parts of velocity
    u_pos = F.relu(u)
    u_neg = u - u_pos

    # upwind flux computation
    flux = u_pos * qi_left + u_neg * qi_right

    return flux


def div_flux_5pts_replicate_qi_boundaries_next_order(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the divergence [uq, vq].

    The value of the reconstruction q is duplicated on the border of the domain
    using next order decomposition: qi(x) = 2qi(x+dx)-qi(x+2dx).

    Args:
        q (torch.Tensor): Tracer field to compute the div flux of.
                └── (n_ens, nl, nx, ny)-shaped
        u (torch.Tensor): Velocity in the zonal direction.
                └── (n_ens, nl, nx+1, ny)-shaped
        v (torch.Tensor): Velocity in the meridional direction.
                └── (n_ens, nl, nx, ny+1)-shaped
        dx (float): Infinitesimal distance in the x direction.
        dy (float): Infinitesimal distance in the x direction.

    Returns:
        torch.Tensor: ∇ · ([u v] q)
            └── (n_ens, nl, nx, ny)-shaped
    """
    q_flux_y = flux_5_pts_replicate_qi_boundaries_next_order(q, v, dim=-1)
    q_flux_x = flux_5_pts_replicate_qi_boundaries_next_order(q, u, dim=-2)

    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def flux_5_pts_replicate_q_boundaries(
    q: torch.Tensor,
    u: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 5 points inside domain.
      - 1 or 3 points near boundaries.

    To remove the 0-padding at the boundary, q is replicated on the outer
    boundary along dim: q(x) = q(x+dx).

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
        qi: tracer field interpolated on u points, torch.Tensor, shape[dim] = n-1
    """

    first_bound = q.select(dim=dim, index=0).unsqueeze(dim)
    last_bound = q.select(dim=dim, index=-1).unsqueeze(dim)

    q = torch.cat([first_bound, q, last_bound], dim=dim)

    return flux_5pts(q, u, dim)


def div_flux_5pts_replicate_q_boundaries(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the divergence [uq, vq].

    The value of q is duplicated on the border of the domain using
    q(x) = q(x+dx).

    Args:
        q (torch.Tensor): Tracer field to compute the div flux of.
                └── (n_ens, nl, nx, ny)-shaped
        u (torch.Tensor): Velocity in the zonal direction.
                └── (n_ens, nl, nx+1, ny)-shaped
        v (torch.Tensor): Velocity in the meridional direction.
                └── (n_ens, nl, nx, ny+1)-shaped
        dx (float): Infinitesimal distance in the x direction.
        dy (float): Infinitesimal distance in the x direction.

    Returns:
        torch.Tensor: ∇ · ([u v] q)
            └── (n_ens, nl, nx, ny)-shaped
    """
    q_flux_y = flux_5_pts_replicate_q_boundaries(q, v, dim=-1)
    q_flux_x = flux_5_pts_replicate_q_boundaries(q, u, dim=-2)

    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def flux_5_pts_replicate_q_boundaries_next_order(
    q: torch.Tensor, u: torch.Tensor, dim: int
) -> torch.Tensor:
    """
    Flux computation for staggerded variables q and u, with solid boundaries.
    Upwind-biased stencil:
      - 5 points inside domain.
      - 1 or 3 points near boundaries.

    To remove the 0-padding at the boundary, q is replicated on the outer
    boundary along dim: q(x) = 2q(x + 2dx) - q(x + 2dx).

    Args:
        q: tracer field to interpolate, torch.Tensor, shape[dim] = n
        u: transport velocity, torch.Tensor, shape[dim] = n-1
        dim: dimension along which computations are done

    Returns:
        flux: tracer flux computed on u points, torch.Tensor, shape[dim] = n-1
        qi: tracer field interpolated on u points, torch.Tensor, shape[dim] = n-1
    """

    q0 = q.select(dim=dim, index=0).unsqueeze(dim)
    q1 = q.select(dim=dim, index=1).unsqueeze(dim)
    q_2 = q.select(dim=dim, index=-2).unsqueeze(dim)
    q_1 = q.select(dim=dim, index=-1).unsqueeze(dim)

    q = torch.cat([2 * q0 - q1, q, 2 * q_1 - q_2], dim=dim)

    return flux_5pts(q, u, dim)


def div_flux_5pts_replicate_q_boundaries_next_order(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the divergence [uq, vq].

    The value of q is duplicated on the border of the domain using next order
    decomposition q(x) = 2q(x + dx) - q(x + 2dx).

    Args:
        q (torch.Tensor): Tracer field to compute the div flux of.
                └── (n_ens, nl, nx, ny)-shaped
        u (torch.Tensor): Velocity in the zonal direction.
                └── (n_ens, nl, nx+1, ny)-shaped
        v (torch.Tensor): Velocity in the meridional direction.
                └── (n_ens, nl, nx, ny+1)-shaped
        dx (float): Infinitesimal distance in the x direction.
        dy (float): Infinitesimal distance in the x direction.

    Returns:
        torch.Tensor: ∇ · ([u v] q)
            └── (n_ens, nl, nx, ny)-shaped
    """
    q_flux_y = flux_5_pts_replicate_q_boundaries_next_order(q, v, dim=-1)
    q_flux_x = flux_5_pts_replicate_q_boundaries_next_order(q, u, dim=-2)

    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def div_flux_5pts_with_bc(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Compute the divergence [uq, vq], with boundary conditions on q.

    q is assumed to have non-zero boundaries, thus explaining is (nx+2) x (ny+2) shape.

    Args:
        q (torch.Tensor): Tracer field to compute the div flux of.
                └── (n_ens, nl, nx+2, ny+2)-shaped
        u (torch.Tensor): Velocity in the zonal direction.
                └── (n_ens, nl, nx+1, ny)-shaped
        v (torch.Tensor): Velocity in the meridional direction.
                └── (n_ens, nl, nx, ny+1)-shaped
        dx (float): Infinitesimal distance in the x direction.
        dy (float): Infinitesimal distance in the x direction.

    Returns:
        torch.Tensor: ∇ · ([u v] q)
            └── (n_ens, nl, nx, ny)-shaped
    """
    q_flux_y = flux_5pts(q[..., 1:-1, :], v, dim=-1)
    q_flux_x = flux_5pts(q[..., :, 1:-1], u, dim=-2)
    return (
        torch.diff(q_flux_x, dim=-2) / dx + torch.diff(q_flux_y, dim=-1) / dy
    )


def flux_3pts_mask(
    q: torch.Tensor,
    u: torch.Tensor,
    dim: int,
    mask_u_d1: torch.Tensor,
    mask_u_d2plus: torch.Tensor,
) -> torch.Tensor:
    n = q.shape[dim]
    pad1 = () if dim == -1 else (0, 0)
    pad2 = (0, 0) if dim == -1 else ()
    qm, q0, qp = (
        q.narrow(dim, 0, n - 2),
        q.narrow(dim, 1, n - 2),
        q.narrow(dim, 2, n - 2),
    )
    qi3_left = F.pad(reconstruction.weno3z(qm, q0, qp), pad1 + (1, 0) + pad2)
    qi3_right = F.pad(reconstruction.weno3z(qp, q0, qm), pad1 + (0, 1) + pad2)
    # qi2 = linear2(q.narrow(dim, 0, n-1), q.narrow(dim, 1, n-1))

    u_pos = F.relu(u)
    u_neg = u - u_pos
    # flux = u * (mask_u_d1 * qi2) \
    flux = mask_u_d1 * (
        u_pos * q.narrow(dim, 0, n - 1) + u_neg * q.narrow(dim, 1, n - 1)
    ) + mask_u_d2plus * (u_pos * qi3_left + u_neg * qi3_right)

    return flux


def div_flux_3pts_mask(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    mask_u_d1: torch.Tensor,
    mask_u_d2plus: torch.Tensor,
    mask_v_d1: torch.Tensor,
    mask_v_d2plus: torch.Tensor,
) -> torch.Tensor:
    q_flux_y = flux_3pts_mask(q, v, -1, mask_v_d1, mask_v_d2plus)
    q_flux_x = flux_3pts_mask(q, u, -2, mask_u_d1, mask_u_d2plus)
    return (
        torch.diff(F.pad(q_flux_x, (0, 0, 1, 1)), dim=-2) / dx
        + torch.diff(F.pad(q_flux_y, (1, 1)), dim=-1) / dy
    )


def flux_5pts_mask(
    q: torch.Tensor,
    u: torch.Tensor,
    dim: int,
    mask_u_d1: torch.Tensor,
    mask_u_d2: torch.Tensor,
    mask_u_d3plus: torch.Tensor,
) -> torch.Tensor:
    n = q.shape[dim]
    pad1 = () if dim == -1 else (0, 0)
    pad2 = (0, 0) if dim == -1 else ()
    qmm, qm, q0, qp, qpp = (
        q.narrow(dim, 0, n - 4),
        q.narrow(dim, 1, n - 4),
        q.narrow(dim, 2, n - 4),
        q.narrow(dim, 3, n - 4),
        q.narrow(dim, 4, n - 4),
    )
    # qi5_left = F.pad(weno5z(qmm, qm, q0, qp, qpp), pad1+(2,1)+pad2)
    # qi5_right = F.pad(weno5z(qpp, qp, q0, qm, qmm), pad1+(1,2)+pad2)
    qi5_left = F.pad(
        reconstruction.linear5_left(qmm, qm, q0, qp, qpp), pad1 + (2, 1) + pad2
    )
    qi5_right = F.pad(
        reconstruction.linear5_left(qpp, qp, q0, qm, qmm), pad1 + (1, 2) + pad2
    )

    # qi4 = F.pad(
    # linear4(q.narrow(dim, 0, n-3), q.narrow(dim, 1, n-3),
    # q.narrow(dim, 2, n-3), q.narrow(dim, 3, n-3)),
    # pad1+(1,1)+pad2)

    qm, q0, qp = (
        q.narrow(dim, 0, n - 2),
        q.narrow(dim, 1, n - 2),
        q.narrow(dim, 2, n - 2),
    )
    qi3_left = F.pad(
        reconstruction.linear3_left(qm, q0, qp), pad1 + (1, 0) + pad2
    )

    qi3_right = F.pad(
        reconstruction.linear3_left(qp, q0, qm), pad1 + (0, 1) + pad2
    )

    qi2 = reconstruction.linear2_centered(
        q.narrow(dim, 0, n - 1), q.narrow(dim, 1, n - 1)
    )

    u_pos = F.relu(u)
    u_neg = u - u_pos
    flux = (
        u * mask_u_d1 * qi2
        + mask_u_d2 * (u_pos * qi3_left + u_neg * qi3_right)
        + mask_u_d3plus * (u_pos * qi5_left + u_neg * qi5_right)
    )

    return flux


def div_flux_5pts_mask(
    q: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    mask_u_d1: torch.Tensor,
    mask_u_d2: torch.Tensor,
    mask_u_d3plus: torch.Tensor,
    mask_v_d1: torch.Tensor,
    mask_v_d2: torch.Tensor,
    mask_v_d3plus: torch.Tensor,
) -> torch.Tensor:
    q_flux_y = flux_5pts_mask(q, v, -1, mask_v_d1, mask_v_d2, mask_v_d3plus)
    q_flux_x = flux_5pts_mask(q, u, -2, mask_u_d1, mask_u_d2, mask_u_d3plus)

    return (
        torch.diff(F.pad(q_flux_x, (0, 0, 1, 1)), dim=-2) / dx
        + torch.diff(F.pad(q_flux_y, (1, 1)), dim=-1) / dy
    )


class Fluxes:
    """Fluxes."""

    def __init__(self, masks: Masks, optimize: bool = True) -> None:
        """Initialize all h and w fluxes.

        Args:
            masks (Masks): Masks.
            optimize (bool, optional): Whether to optimize the flux functions. Defaults to True.
        """

        def h_flux_y(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flux(
                h,
                v,
                dim=-1,
                n_points=6,
                rec_func_2=reconstruction.linear2_centered,
                rec_func_4=reconstruction.wenoz4_left,
                rec_func_6=reconstruction.wenoz6_left,
                mask_2=masks.v_sten_hy_eq2[..., 1:-1],
                mask_4=masks.v_sten_hy_eq4[..., 1:-1],
                mask_6=masks.v_sten_hy_gt6[..., 1:-1],
            )

        def h_flux_x(h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
            return flux(
                h,
                u,
                dim=-2,
                n_points=6,
                rec_func_2=reconstruction.linear2_centered,
                rec_func_4=reconstruction.wenoz4_left,
                rec_func_6=reconstruction.wenoz6_left,
                mask_2=masks.u_sten_hx_eq2[..., 1:-1, :],
                mask_4=masks.u_sten_hx_eq4[..., 1:-1, :],
                mask_6=masks.u_sten_hx_gt6[..., 1:-1, :],
            )

        def omega_flux_y(
            w: torch.Tensor, v_ugrid: torch.Tensor
        ) -> torch.Tensor:
            return flux(
                w,
                v_ugrid,
                dim=-1,
                n_points=6,
                rec_func_2=reconstruction.linear2_centered,
                rec_func_4=reconstruction.wenoz4_left,
                rec_func_6=reconstruction.wenoz6_left,
                mask_2=masks.u_sten_wy_eq2[..., 1:-1, :],
                mask_4=masks.u_sten_wy_eq4[..., 1:-1, :],
                mask_6=masks.u_sten_wy_gt4[..., 1:-1, :],
            )

        def omega_flux_x(w: torch.Tensor, u_vgrid: torch.Tensor):
            return flux(
                w,
                u_vgrid,
                dim=-2,
                n_points=6,
                rec_func_2=reconstruction.linear2_centered,
                rec_func_4=reconstruction.wenoz4_left,
                rec_func_6=reconstruction.wenoz6_left,
                mask_2=masks.v_sten_wx_eq2[..., 1:-1],
                mask_4=masks.v_sten_wx_eq4[..., 1:-1],
                mask_6=masks.v_sten_wx_gt6[..., 1:-1],
            )

        if optimize:
            with logger.section("Compiling functions..."):
                self.h_x = OptimizableFunction(h_flux_x)
                self.h_y = OptimizableFunction(h_flux_y)
                self.w_x = OptimizableFunction(omega_flux_x)
                self.w_y = OptimizableFunction(omega_flux_y)
        else:
            self.h_x = h_flux_x
            self.h_y = h_flux_y
            self.w_x = omega_flux_x
            self.w_y = omega_flux_y
