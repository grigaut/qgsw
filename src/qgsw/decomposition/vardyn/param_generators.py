"""Parameters generator for the MIOST basis."""

from typing import Any

import numpy as np
import torch

from qgsw.decomposition.vardyn.basis import Basis_miost
from qgsw.specs import defaults


def vardyn_like_params(
    config: dict,
    lons: np.ndarray,
    lats: np.ndarray,
    xx: torch.Tensor,
    yy: torch.Tensor,
    dt: float,
    n_steps: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, list[torch.Tensor]],
    dict[str, Any],
    np.ndarray,
]:
    """Generate parameters for the Basis MIOSt.

    Args:
        config (dict): Configuration dict.
        lons (np.ndarray): Longitudes (in degree).
        lats (np.ndarray): Latitudes (in degree).
        xx (torch.Tensor): Cartesian x locations.
        yy (torch.Tensor): Cartesian y locations.
        dt (float): Timestep.
        n_steps (int): Number of steps.
        dtype (torch.dtype | None, optional): Data type. Defaults to None.
        device (torch.device | None, optional): Device. Defaults to None.

    Returns:
        tuple[ dict[int, torch.Tensor], dict[int, list[torch.Tensor]], dict[str, Any], np.ndarray, ]: Space_params, time_params, keywords args and correlation matrix.
    """  # noqa: E501
    specs = defaults.get(dtype=dtype, device=device)
    assert xx.shape == yy.shape
    assert xx.shape == lons.shape
    assert yy.shape == lats.shape
    basis_ref = Basis_miost(
        config,
        lons,
        lats,
        lons.shape[0],
        lons.shape[1],
        (
            lons.min().item(),
            lons.max().item(),
            lats.min().item(),
            lats.max().item(),
        ),
    )
    times_day = np.arange(n_steps + 1) * dt / (24 * 3600)
    _, Q = basis_ref.set_basis(times_day, return_q=True)
    return (
        dict(enumerate(basis_ref.Gx)),
        {
            k: [
                torch.from_numpy(basis_ref.Gt[t][k]).to(**specs)
                for t in basis_ref.Gt
            ]
            for k in range(basis_ref.nf)
        },
        {
            "xx": xx,
            "yy": yy,
            "dt": dt,
            "Nt": {
                k: [basis_ref.Nt[t][k] for t in basis_ref.Nt]
                for k in range(basis_ref.nf)
            },
            "Nx": basis_ref.Nx,
        },
        Q,
    )
