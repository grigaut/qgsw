"""Parameters generator for the MIOST basis."""

from typing import Any

import numpy as np
import torch

from qgsw.decomposition.vardyn.basis import Basis_miost
from qgsw.eNATL60.interpolation import lonlat_to_xy
from qgsw.specs import defaults


def vardyn_like_params(
    config: dict,
    lons: np.ndarray,
    lats: np.ndarray,
    dt: float,
    n_steps: int,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[
    dict[int, torch.Tensor],
    dict[int, list[torch.Tensor]],
    dict[str, Any],
    dict[int, torch.Tensor],
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
        tuple[ dict[int, torch.Tensor], dict[int, list[torch.Tensor]], dict[str, Any], dict[int, torch.Tensor], ]: Space_params, time_params, keywords args and correlation matrix.
    """  # noqa: E501
    specs = defaults.get(dtype=dtype, device=device)

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
    _, q = basis_ref.set_basis(times_day, return_q=True)
    bounds = basis_ref.iff_wavebounds
    Q = {
        k: torch.from_numpy(q[bounds[k] : bounds[k + 1]])
        .to(**specs)
        .reshape(
            (
                len(basis_ref.enst[k][0]),
                2 * basis_ref.ntheta * len(basis_ref.ENSLAT[k]),
            ),
        )
        for k in range(basis_ref.nf)
    }
    _, _, lons0, lat0 = lonlat_to_xy(
        torch.tensor(np.deg2rad(lons), **specs),
        torch.tensor(np.deg2rad(lats), **specs),
    )
    return (
        {
            lvl: {
                "centers": [
                    (
                        np.deg2rad(basis_ref.ENSLON[lvl][i]),
                        np.deg2rad(basis_ref.ENSLAT[lvl][i]),
                    )
                    for i in range(basis_ref.ENSLON[lvl].shape[0])
                ],
                "k": basis_ref.k[lvl] / 1000,  # rad.m⁻¹
                "Lx": basis_ref.DX[lvl] * 1000,  # m
            }
            for lvl in range(basis_ref.nf)
        },
        {
            lvl: {
                "centers": torch.as_tensor(basis_ref.enst[lvl], **specs)
                * 3600
                * 24,
                "tdec": torch.as_tensor(basis_ref.tdec[lvl], **specs)
                * 3600
                * 24,
                "norm": torch.as_tensor(basis_ref.norm_fact[lvl], **specs),
            }
            for lvl in range(basis_ref.nf)
        },
        {"lons0": lons0, "lat0": lat0, "n_theta": basis_ref.ntheta},
        Q,
    )
