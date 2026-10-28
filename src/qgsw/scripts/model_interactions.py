"""Init model."""

import torch

from qgsw.fields.variables.tuples import PSIZeta
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.sv.core import SurfaceVorticity


def init_model(
    model: QGPSIQCore | SurfaceVorticity,
    psi_g: torch.Tensor,
    q_or_zeta: torch.Tensor,
) -> None:
    """Init model.

    Args:
        model (QGPSIQCore | SurfaceVorticity): Model.
        psi_g (torch.Tensor): Geostrophic stream function.
        q_or_zeta (torch.Tensor): (Potential) vorticity.
    """
    if model.get_type() == SurfaceVorticity.get_type():
        model.psiz = PSIZeta(psi_g, q_or_zeta)
        return
    model.set_psiq(psi_g, q_or_zeta)


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
