"""Basis setup."""

from pathlib import Path
from typing import Any

import numpy as np

from qgsw.decomposition.config import BASIS_CONFIGS
from qgsw.decomposition.core import Basis, build_basis_from_config
from qgsw.decomposition.vardyn.core import BasisMiost
from qgsw.spatial.core.discretization import SpaceDiscretization2D


def setup_basis(
    basis_type: str,
    lon_lat: tuple[np.ndarray, np.ndarray],
    space: SpaceDiscretization2D,
    dt: float,
    n_steps: int,
    file_aux: Path,
    filec_aux: Path,
) -> tuple[Basis, dict[str, Any]]:
    """Setup basis.

    Args:
        basis_type (str): Bassi type.
        lon_lat (tuple[np.ndarray, np.ndarray]): Longitude, latitudes.
        space (SpaceDiscretization2D): Space.
        dt (float): Timestep.
        n_steps (int): Number of steps.
        file_aux (Path): Path to file
        filec_aux (Path): Path to file

    Returns:
        tuple[Basis, dict[str, Any]]: Basis and additional kwargs
    """
    basis_config = BASIS_CONFIGS[basis_type]
    if basis_type == BasisMiost.type:
        basis_config["file_aux"] = file_aux
        basis_config["filec_aux"] = filec_aux
    return build_basis_from_config(basis_config, lon_lat, space, dt, n_steps)
