"""Coefficient scalings."""

# ruff: noqa: ARG005

from collections.abc import Callable

from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.vardyn.core import BasisMiost
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.scripts.setups.types import Coefs

SCALINGS = {
    QGPSIQForced.get_type(): {
        BasisMiost.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        WaveletBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs.scale(
            *(U**2 / L**2 for _ in range(coefs.order))
        ),
        GaussianExpBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
    },
    QGPSIQRGPsi2TransportDR.get_type(): {
        BasisMiost.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        WaveletBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        GaussianExpBasis.type: lambda coefs, L, U, psi0_mean, T_tot: (
            coefs.scale(
                *(1e-1 * psi0_mean / (T_tot) ** k for k in range(coefs.order))
            )
        ),
    },
    QGPSIQForcedSurfML.get_type(): {
        BasisMiost.type: lambda coefs, L, U, psi0_mean, T_tot: coefs,
        WaveletBasis.type: lambda coefs, L, U, psi0_mean, T_tot: coefs.scale(
            *(U**2 / L**2 for _ in range(coefs.order))
        ),
        GaussianExpBasis.type: lambda coefs, L, U, psi0_mean, T_tot: (
            coefs.scale(
                *(1e-1 * psi0_mean / (T_tot) ** k for k in range(coefs.order))
            )
        ),
    },
}


def retrieve_scaling_func(
    basis_types: dict[str, str],
    model_type: str,
    L: float,
    U: float,
    psi0_mean: float,
    T_tot: float,
) -> Callable[[Coefs], Coefs]:
    """Build coefficient scaling function.

    Args:
        basis_types (dict[str, str]): Basis types.
        model_type (str): Model type.
        L (float): Horizontal scale.
        U (float): Horizontal velocity scale.
        psi0_mean (float): Initial stream function mean.
        T_tot (float): Total time length.

    Returns:
        Callable[[Coefs], Coefs]: Coefficient scaling function.
    """

    def scale(
        coefs: Coefs,
    ) -> Coefs:
        return {
            k: SCALINGS[model_type][basis](coefs[k], L, U, psi0_mean, T_tot)
            for k, basis in basis_types.items()
        }

    return scale
