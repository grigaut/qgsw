"""Coefficient scalings."""

# ruff: noqa: ARG005

from collections.abc import Callable

from qgsw.cli import OptimBasisParser
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.vardyn.core import MIOSTBasis
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.models.sv.core import SurfaceVorticity

SCALINGS = {
    QGPSIQForced.get_type(): {
        MIOSTBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        WaveletBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs.scale(
            *(U**2 / L**2 for _ in range(coefs.order))
        ),
        GaussianExpBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
    },
    QGPSIQRGPsi2TransportDR.get_type(): {
        MIOSTBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        WaveletBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        GaussianExpBasis.type: lambda coefs, L, U, psi0_mean, T_tot: (
            coefs.scale(
                *(1e-1 * psi0_mean / (T_tot) ** k for k in range(coefs.order))
            )
        ),
    },
    QGPSIQForcedSurfML.get_type(): {
        MIOSTBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        WaveletBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs.scale(
            *(U**2 / L**2 for _ in range(coefs.order))
        ),
        GaussianExpBasis.type: lambda coefs, L, U, psi0_mean, T_tot: (
            coefs.scale(
                *(1e-1 * psi0_mean / (T_tot) ** k for k in range(coefs.order))
            )
        ),
    },
    SurfaceVorticity.get_type(): {
        MIOSTBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        WaveletBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs.scale(
            *(1 * psi0_mean for _ in range(coefs.order))
        ),
        GaussianExpBasis.type: lambda coefs, L, U, psi0_mean, T_tot: (
            coefs.scale(
                *(1 * psi0_mean / (T_tot) ** k for k in range(coefs.order))
            )
        ),
    },
}


def retrieve_scaling_func(
    bases_parsers: dict[str, OptimBasisParser],
    model_type: str,
    L: float,
    U: float,
    psi0_mean: float,
    T_tot: float,
) -> dict[str, Callable[[DecompositionCoefs], DecompositionCoefs]]:
    """Build coefficient scaling function.

    Args:
        bases_parsers (dict[str, OptimBasisParser]): Basis parsers.
        basis_lrs (dict[str, float]): Basis learning rates values.
        model_type (str): Model type.
        L (float): Horizontal scale.
        U (float): Horizontal velocity scale.
        psi0_mean (float): Initial stream function mean.
        T_tot (float): Total time length.

    Returns:
        Callable[[Coefs], Coefs]: Coefficient scaling function.
    """
    return {
        k: lambda coef, t=b.type: SCALINGS[model_type][t](
            coef,
            L,
            U,
            psi0_mean,
            T_tot,
        )
        for k, b in bases_parsers.items()
    }
