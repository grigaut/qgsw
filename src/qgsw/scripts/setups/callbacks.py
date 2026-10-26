"""Callback setups."""

# ruff: noqa: SLF001

import torch

from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.models.sv.core import SurfaceVorticity
from qgsw.scripts.setups.types import Bases, Coefs


def pre_cycle_bases_setup(
    model: QGPSIQCore | SurfaceVorticity,
    bases: Bases,
    coefs: Coefs,
    alpha: torch.Tensor,
) -> None:
    """Sets the basis.

    Args:
        model (QGPSIQCore): Model.
        bases (Bases): Bases dict.
        coefs (Coefs): Coefs dict.
        alpha (torch.Tensor): Alpha

    Raises:
        ValueError: If the model type is not recognised.
    """
    for k, v in bases.items():
        v.set_coefs(coefs[k])

    match model.get_type():
        case QGPSIQForced._type:
            model.basis_external = bases["external"]
        case QGPSIQRGPsi2TransportDR._type:
            model.alpha = alpha
            model.basis_dynamical = bases["dynamical"]
        case QGPSIQForcedSurfML._type:
            model.alpha = alpha
            model.basis_dynamical = bases["dynamical"]
            model.basis_external = bases["external"]
        case SurfaceVorticity._type:
            model.basis_psi_a = bases["psi"]
            model.basis_chi_a = bases["chi"]
        case _:
            msg = f"Unsupported model type: {model.get_type()}."
            raise ValueError(msg)
