"""Init model."""

from collections.abc import Callable

import torch

from qgsw.fields.variables.tuples import PSIZeta
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.sv.core import SurfaceVorticity
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.utils.interpolation import QuadraticInterpolation
from qgsw.utils.reshaping import crop


def init_model(
    model: QGPSIQCore | SurfaceVorticity,
    psi0: torch.Tensor,
    psi_bcs: list[Boundaries],
    psi_bcs_w: list[Boundaries],
    build_q_or_zeta: Callable[
        [torch.Tensor, torch.Tensor],
        tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[Boundaries], Boundaries],
        ],
    ],
    times: list[float],
    b: int,
) -> None:
    """Init model.

    Args:
        model (QGPSIQCore | SurfaceVorticity): Model.
        psi0 (torch.Tensor): Initial surface (geostrophic) stream function.
        psi_bcs (list[Boundaries]): Surface (geostrophic) stream function
            boundaries.
        psi_bcs_w (list[Boundaries]): Wide surface (geostrophic) stream
            function boundaries to compute pv or vorticity boundaries.
        build_q_or_zeta (Callable[ [torch.Tensor, torch.Tensor], tuple[ Callable[[torch.Tensor], torch.Tensor], Callable[[Boundaries], Boundaries], ], ]): Build Pv or vorticity computation fonctions.
        times (list[float]): Times.
        b (int): Offset.
    """  # noqa: E501
    _, A11, A12 = get_As(model)
    q_or_zetas, q_or_zetas_bc = build_q_or_zeta(A11, A12)
    q_or_zeta = crop(q_or_zetas(psi0), b - 1)
    q_bcs = [q_or_zetas_bc(p1) for p1 in psi_bcs_w]
    q_bc_interp = QuadraticInterpolation(times, q_bcs)
    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    if model.get_type() == SurfaceVorticity.get_type():
        model.psiz = PSIZeta(crop(psi0[:, :1], b), q_or_zeta)
    else:
        model.set_psiq(crop(psi0[:, :1], b), q_or_zeta)
    model.set_boundary_maps(psi_bc_interp, q_bc_interp)


def get_psi_g(
    model: QGPSIQCore | SurfaceVorticity,
) -> torch.Tensor:
    """Init model.

    Args:
        model (QGPSIQCore | SurfaceVorticity): Model.

    Returns:
        torch.Tensor: Geostrophic stream function.
    """
    if model.get_type() == SurfaceVorticity.get_type():
        return model.psi_g
    return model.psi


def get_As(  # noqa: N802
    model: QGPSIQCore | SurfaceVorticity,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retrieve stretching matrix components.

    Args:
        model (QGPSIQCore | SurfaceVorticity): Model

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A,A11,A12
    """
    try:
        A = model.A
    except AttributeError:
        A = torch.zeros((1, 1), **model.specs)

    if A.shape[1] == 1:
        A11 = A[:1, :1]
        A12 = torch.zeros_like(A11)
    else:
        A11 = A[:1, :1]
        A12 = A[:1, 1:2]
    return A, A11, A12
