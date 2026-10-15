"""Optimization params setups."""

from typing import Any

import torch

from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.vardyn.core import BasisMiost
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.logging.core import getLogger
from qgsw.logging.utils import box
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.scripts.setups.types import Coefs

logger = getLogger(__name__)

COEFS_OPTIM = {
    QGPSIQForced.get_type(): {
        BasisMiost.type: 1e-13,
        WaveletBasis.type: None,
        GaussianExpBasis.type: None,
    },
    QGPSIQRGPsi2TransportDR.get_type(): {
        BasisMiost.type: None,
        WaveletBasis.type: None,
        GaussianExpBasis.type: None,
    },
    QGPSIQForcedSurfML.get_type(): {
        BasisMiost.type: None,
        WaveletBasis.type: None,
        GaussianExpBasis.type: None,
    },
}


def optim_setup(
    coefs_lr: dict[str, float],
    coefs: Coefs,
    *,
    alpha_optim: bool,
    wind_optim: bool,
) -> tuple[
    list[{dict[str, Any]}],
    Coefs,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Setup optimization.

    Args:
        coefs_lr (dict[str, float]): Learning rates for coefficients.
        coefs (Coefs): Coefficients.
        alpha_optim (bool): Whether to perform optimization on alpha.
        wind_optim (bool): Whether to perform optimization on wind.


    Returns:
        tuple[list[{dict[str, Any]}],Coefs,torch.Tensor,torch.Tensor,int]: Optim params, coefs, kappa, ub10_to_uvsurf, numel
    """  # noqa: E501
    specs = coefs[coefs.keys()[0]].specs

    numel = 0

    uv10_to_uvsurf = torch.eye(2, **specs, requires_grad=False)
    kappa = torch.tensor(0, **specs, requires_grad=False)
    coefs_grad: Coefs = {}
    for k, coef in coefs.items():
        coefs_grad[k] = coef.requires_grad_()
        params = [
            {
                "params": list(coefs_grad[k].values()),
                "lr": coefs_lr[k],
                "name": f"Basis coefs ({k})",
            }
        ]
        numel += coef.numel()
    if alpha_optim:
        kappa = kappa.requires_grad_()
        params += [{"params": [kappa], "lr": 1e-2, "name": "κ"}]
        numel += kappa.numel()
        logger.info("ɑ added to optimisable params.")  # noqa: RUF001
    if wind_optim:
        uv10_to_uvsurf = uv10_to_uvsurf.requires_grad_()
        params += [
            {
                "params": [uv10_to_uvsurf],
                "lr": 1e-2,
                "name": "Wind conversion",
            }
        ]
        numel += uv10_to_uvsurf.numel()
        logger.info("Wind conversion matrix added to optimisable params.")

    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))
    return params, coefs_grad, kappa, uv10_to_uvsurf, numel
