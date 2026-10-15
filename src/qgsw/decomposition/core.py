"""Core function to build a decomposition."""

from math import sqrt
from typing import Any

import numpy as np

from qgsw.decomposition.base import SpaceTimeDecomposition
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.exp_exp.param_generator import gaussian_exp_field
from qgsw.decomposition.supports.space.base import SpaceSupportFunction
from qgsw.decomposition.supports.time.base import TimeSupportFunction
from qgsw.decomposition.taylor.core import TaylorFullFieldBasis
from qgsw.decomposition.taylor_exp.core import TaylorExpBasis
from qgsw.decomposition.vardyn.core import BasisMiost
from qgsw.decomposition.vardyn.param_generators import vardyn_like_params
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.decomposition.wavelets.param_generators import dyadic_decomposition
from qgsw.spatial.core.discretization import SpaceDiscretization2D

Basis = SpaceTimeDecomposition[SpaceSupportFunction, TimeSupportFunction]

BASES: dict[str, type[Basis]] = {
    WaveletBasis.type: WaveletBasis,
    TaylorExpBasis.type: TaylorExpBasis,
    TaylorFullFieldBasis.type: TaylorFullFieldBasis,
    GaussianExpBasis.type: GaussianExpBasis,
    BasisMiost.type: BasisMiost,
}


def build_basis_from_params_dict(params: dict[str, Any]) -> Basis:
    """Build the right basis given params.

    Args:
        params (dict[str, Any]): Basis parameters (output of Basis.get_params).

    Returns:
        Basis: Corresponding basis.
    """
    return BASES[params["type"]].from_params(params)


def build_basis_from_config(
    config: dict[str, Any],
    lon_lat: tuple[np.ndarray, np.ndarray],
    space: SpaceDiscretization2D,
    dt: float,
    n_steps: int,
    exp_cov: float = 1,
) -> tuple[Basis, dict[str, Any]]:
    """Build basis given configuration and parameters.

    Args:
        config (dict[str, Any]): Basis configuration.
        lon_lat (tuple[np.ndarray, np.ndarray]): Longitude, latitudes
        space (SpaceDiscretization2D): Cartesian Xs and Ys grids.
        dt (float): Timestep
        n_steps (int): Number of steps.
        exp_cov (float): Exponent for WaveletBasis and GaussianExpBasis
            covariance.

    Raises:
        ValueError: If the basis type is not recognised.

    Returns:
        tuple[Basis, dict[str, Any]]: Basis, kwargs
    """
    kwargs = {}
    basis_kwargs = {}
    match bt := config["type"]:
        case BasisMiost.type:
            space_params, time_params, basis_kwargs, Q = vardyn_like_params(
                config, *lon_lat, *space.q.xy, dt, n_steps
            )
            kwargs["Q"] = Q
            basis = BasisMiost(space_params, time_params, **basis_kwargs)
        case WaveletBasis.type:
            space_params, time_params = dyadic_decomposition(
                config,
                *space.psi.xy,
                Lt_max=n_steps * dt,
            )
            basis = WaveletBasis(space_params, time_params, **basis_kwargs)
            basis.n_theta = config["n_theta"]
            Q = {
                sqrt(
                    space_params[k]["sigma_x"]
                    * space_params[k]["sigma_y"]
                    / space.dx
                    / space.dy
                )
                ** (-exp_cov)
                for k in range(basis.order)
            }
            kwargs["Q"] = Q

        case GaussianExpBasis.type:
            space_params, time_params = gaussian_exp_field(
                config, *space.psi.xy, n_steps * dt
            )
            basis = GaussianExpBasis(space_params, time_params, **basis_kwargs)
            Q = {
                sqrt(
                    space_params[k]["sigma_x"]
                    * space_params[k]["sigma_y"]
                    / space.dx
                    / space.dy
                )
                ** (-exp_cov)
                for k in range(basis.order)
            }
            kwargs["Q"] = Q

        case _:
            msg = f"Unrecognised basis type: {bt}."
            raise ValueError(msg)
    return basis, kwargs
