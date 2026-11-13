"""Coefficient scalings."""

from collections.abc import Callable

import torch

from qgsw.cli import OptimBasisParser
from qgsw.decomposition.coefficients import DecompositionCoefs


def scale_with_cov(
    coefs: DecompositionCoefs,
    Qs: dict[int, torch.Tensor] | None,
) -> DecompositionCoefs:
    """Scale coefs using covariance matrices.

    Args:
        coefs (DecompositionCoefs): Coefs.
        Qs (dict[int, torch.Tensor] | None): Covariances.

    Returns:
        DecompositionCoefs: Scaled coefficients.
    """
    if Qs is None:
        return coefs

    Qs_sorted = [Qs[lvl] for lvl in sorted(coefs.keys())]
    return coefs.scale(*Qs_sorted)


def retrieve_scaling_func(
    bases_parsers: dict[str, OptimBasisParser],
    Qs: dict[str, dict[int, torch.Tensor] | None],
) -> dict[str, Callable[[DecompositionCoefs], DecompositionCoefs]]:
    """Build coefficient scaling function.

    Args:
        bases_parsers (dict[str, OptimBasisParser]): Basis parsers.
        Qs (dict[str, dict[int, torch.Tensor] | None]): Covariances.

    Returns:
        Callable[[Coefs], Coefs]: Coefficient scaling function.
    """
    return {
        k: lambda coef, k=k: scale_with_cov(coef, Qs[k])
        for k, b in bases_parsers.items()
    }
