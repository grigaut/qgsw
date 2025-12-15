"""Core function to build a decomposition."""

from typing import Any

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.decomposition.taylor.core import TaylorFullFieldBasis
from qgsw.decomposition.taylor_exp.core import TaylorExpBasis
from qgsw.decomposition.wavelets.core import WaveletBasis

Basis = SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]

BASES: dict[str, type[Basis]] = {
    WaveletBasis.type: WaveletBasis,
    TaylorExpBasis.type: TaylorExpBasis,
    TaylorFullFieldBasis.type: TaylorFullFieldBasis,
    GaussianExpBasis.type: GaussianExpBasis,
}


def build_basis_from_params_dict(params: dict[str, Any]) -> Basis:
    """Build the right basis given params.

    Args:
        params (dict[str, Any]): Basis parameters (output of Basis.get_params).

    Returns:
        Basis: Corresponding basis.
    """
    return BASES[params["type"]].from_params(params)
