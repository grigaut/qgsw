"""Callback setups."""

# ruff: noqa: SLF001

import torch

from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.scripts.setups.types import Bases, Coefs


def pre_cycle_bases_setup(
    model: QGPSIQCore,
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
            model.forcing_basis = bases["external"]
        case QGPSIQRGPsi2TransportDR._type:
            model.basis = bases["dynamical"]
            model.alpha = alpha
        case QGPSIQForcedSurfML._type:
            model.basis = bases["dynamical"]
            model.external = bases["forcing_basis"]
            model.alpha = alpha
        case _:
            msg = f"Unsupported model type: {model.get_type()}."
            raise ValueError(msg)
