"""Optimization params setups."""

import torch

from qgsw.cli import OptimBasisParser, OptimParamParser
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.logging.core import getLogger
from qgsw.optim.utils import Param
from qgsw.scripts.setups.types import Coefs

logger = getLogger(__name__)


def init_optim_params(
    bases_parsers: dict[str, OptimBasisParser],
    coefs: Coefs,
    *,
    alpha_parser: OptimParamParser,
    wind_parser: OptimParamParser | None = None,
) -> tuple[
    dict[str, Param[DecompositionCoefs]],
    Param[torch.Tensor],
    Param[torch.Tensor],
]:
    """Setup optimization.

    Args:
        bases_parsers (dict[str, OptimParamParser]): Bases parsers.
        coefs (Coefs): Coefficients.
        alpha_parser (OptimParamParser): Whether to perform optimization on alpha.
        wind_parser (OptimParamParser): Whether to perform optimization on wind.


    Returns:
        Coefs,torch.Tensor,torch.Tensor,int]: Optim params, coefs, kappa, ub10_to_uvsurf, numel
    """  # noqa: E501
    specs = next(iter(coefs.values())).specs

    uv10_to_uvsurf = torch.eye(2, **specs, requires_grad=False)
    kappa = torch.tensor(0, **specs, requires_grad=False)
    coefs_grad: Coefs = {}
    for k, coef in coefs.items():
        if bases_parsers[k].optim:
            coefs_grad[k] = coef.requires_grad_()
        else:
            coefs_grad[k] = coef
    if alpha_parser.optim:
        kappa = kappa.requires_grad_()
    if wind_parser is not None and wind_parser.optim:
        uv10_to_uvsurf = uv10_to_uvsurf.requires_grad_()

    return (
        {
            k: Param[DecompositionCoefs](c, bases_parsers[k].learning_rate)
            for k, c in coefs_grad.items()
        },
        Param[torch.Tensor](kappa, alpha_parser.learning_rate),
        Param[torch.Tensor](
            uv10_to_uvsurf,
            1 if wind_parser is None else wind_parser.learning_rate,
        ),
    )
