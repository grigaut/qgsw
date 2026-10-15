"""Basis setup."""

from pathlib import Path
from typing import Any

import numpy as np

from qgsw.decomposition.config import BASIS_CONFIGS
from qgsw.decomposition.core import Basis, build_basis_from_config
from qgsw.decomposition.vardyn.core import BasisMiost
from qgsw.spatial.core.discretization import SpaceDiscretization2D


def setup_basis(
    basis_type: dict[str, str],
    lon_lat: tuple[np.ndarray, np.ndarray],
    space: SpaceDiscretization2D,
    dt: float,
    n_steps: int,
    exp_cov: float,
    file_aux: Path,
    filec_aux: Path,
) -> tuple[dict[str, Basis], dict[str, dict[str, Any]]]:
    """Setup basis.

    Args:
        basis_type (str): Bassi type.
        lon_lat (tuple[np.ndarray, np.ndarray]): Longitude, latitudes.
        space (SpaceDiscretization2D): Space.
        dt (float): Timestep.
        n_steps (int): Number of steps.
        exp_cov (float): Exponent for Wavelet and GaussianExp basis covariance.
        file_aux (Path): Path to file
        filec_aux (Path): Path to file

    Returns:
        tuple[Basis, dict[str, Any]]: Basis and additional kwargs
    """
    bases = {}
    kwargs = {}
    for k, v in basis_type.items():
        basis_config = BASIS_CONFIGS[v]
        if v == BasisMiost.type:
            basis_config["file_aux"] = file_aux
            basis_config["filec_aux"] = filec_aux
        b, kw = build_basis_from_config(
            basis_config, lon_lat, space, dt, n_steps, exp_cov
        )
        bases[k] = b
        kwargs[k] = kw
    return bases, kwargs
