"""Basis setup."""

from pathlib import Path
from typing import TypeVar

import numpy as np
import torch

from qgsw.decomposition.config import BASIS_CONFIGS
from qgsw.decomposition.core import Basis, build_basis_from_config
from qgsw.decomposition.vardyn.core import MIOSTBasis
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.models.sv.core import SurfaceVorticity
from qgsw.spatial.core.discretization import SpaceDiscretization2D

T = TypeVar("T")


def retrieve_bases_elements(
    model_type: str,
    basis_dynamical: T | None,
    basis_external: T | None,
    basis_psi_a: T | None = None,
    basis_chi_a: T | None = None,
    *,
    check: bool = False,
) -> dict[str, T]:
    """Build bases types dict.

    Args:
        model_type (str): Model type.
        basis_dynamical (T): Dynamical basis.
        basis_external (T): External basis.
        basis_psi_a (T): Ageostrophic stream function basis.
        basis_chi_a (T): Ageostrophic potential basis.
        check (bool): Whether to perform checks on basis.

    Raises:
        ValueError: If the model type is not recognised.

    Returns:
        dict[str, T]: Bases types mapping.
    """
    without_bdyn = basis_dynamical is None
    without_bext = basis_external is None
    without_bpsi = basis_psi_a is None
    without_bchi = basis_chi_a is None
    if model_type == QGPSIQRGPsi2TransportDR.get_type():
        if check and without_bdyn:
            msg = "Dynamical basis was not given to script."
            raise ValueError(msg)
        return {"dynamical": basis_dynamical}
    if model_type == QGPSIQForced.get_type():
        if check and without_bext:
            msg = "External basis was not given to script."
            raise ValueError(msg)
        return {"external": basis_external}
    if model_type == QGPSIQForcedSurfML.get_type():
        if check and (without_bdyn) and (without_bext):
            msg = "Both dynamical and external bases must be given to script."
            raise ValueError(msg)
        return {"dynamical": basis_dynamical, "external": basis_external}
    if model_type == SurfaceVorticity.get_type():
        if check and (without_bpsi) and (without_bchi):
            msg = "Both psi_a and chi_a bases must be given to script."
            raise ValueError(msg)
        return {"psi": basis_psi_a, "chi": basis_chi_a}
    msg = f"Unsupported model_type: {model_type}."
    raise ValueError(msg)


def setup_basis(
    basis_type: dict[str, str],
    lon_lat: tuple[np.ndarray, np.ndarray] | None,
    space: SpaceDiscretization2D,
    dt: float,
    n_steps: int,
    exp_cov: float,
    file_aux: Path | None,
    filec_aux: Path | None,
) -> tuple[dict[str, Basis], dict[str, dict[str, torch.Tensor]]]:
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
        tuple[Basis, dict[str, torch.Tensor]]: Basis and corresponding
        covariances.
    """
    bases = {}
    Qs = {}
    for k, v in basis_type.items():
        basis_config = BASIS_CONFIGS[v]
        if v == MIOSTBasis.type:
            basis_config["file_aux"] = file_aux
            basis_config["filec_aux"] = filec_aux
        basis, Q = build_basis_from_config(
            basis_config, lon_lat, space, dt, n_steps, exp_cov
        )
        bases[k] = basis
        Qs[k] = Q
    return bases, Qs
