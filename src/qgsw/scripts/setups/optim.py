"""Optimization params setups."""

from typing import Any

import torch

from qgsw.logging.core import getLogger
from qgsw.logging.utils import box
from qgsw.scripts.setups.types import Coefs

logger = getLogger(__name__)


def optim_setup(
    coefs_lr: dict[str, float],
    coefs: Coefs,
    *,
    alpha_optim: bool,
    wind_optim: bool,
    alpha_lr: float | None = None,
    wind_lr: float | None = None,
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
        alpha_lr (float): Alpha learning rate.
        wind_lr (float): Wind learning rate.


    Returns:
        tuple[list[{dict[str, Any]}],Coefs,torch.Tensor,torch.Tensor,int]: Optim params, coefs, kappa, ub10_to_uvsurf, numel
    """  # noqa: E501
    specs = next(iter(coefs.values())).specs

    numel = 0

    uv10_to_uvsurf = torch.eye(2, **specs, requires_grad=False)
    kappa = torch.tensor(0, **specs, requires_grad=False)
    coefs_grad: Coefs = {}
    params = []
    for k, coef in coefs.items():
        coefs_grad[k] = coef.requires_grad_()
        params += [
            {
                "params": list(coefs_grad[k].values()),
                "lr": coefs_lr[k],
                "name": f"Basis coefs ({k})",
            }
        ]
        numel += coef.numel()
    if alpha_optim:
        if alpha_lr is None:
            msg = "No learning rate specified for alpha."
            raise ValueError(msg)
        kappa = kappa.requires_grad_()
        params += [{"params": [kappa], "lr": alpha_lr, "name": "κ"}]
        numel += kappa.numel()
        logger.info("ɑ added to optimisable params.")  # noqa: RUF001
    if wind_optim:
        if wind_lr is None:
            msg = "No learning rate specified for the wind."
            raise ValueError(msg)
        uv10_to_uvsurf = uv10_to_uvsurf.requires_grad_()
        params += [
            {
                "params": [uv10_to_uvsurf],
                "lr": wind_lr,
                "name": "Wind conversion",
            }
        ]
        numel += uv10_to_uvsurf.numel()
        logger.info("Wind conversion matrix added to optimisable params.")

    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))
    return params, coefs_grad, kappa, uv10_to_uvsurf, numel
