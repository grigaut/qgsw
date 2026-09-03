"""Wind-related functions."""


## Bulk formula (from Large & Yeager 2004)

from __future__ import annotations

import torch

from qgsw import specs


def compute_drag_coef(wind_magnitude: torch.Tensor) -> torch.Tensor:
    """Compute drag coefficient.

    Based on formula from 'Diurnal to decadal global forcing for ocean and
    sea-ice models: the data sets and flux climatologies'
    by Large and Yeager (2004)

    Arbitrary threshold of 0.5 added to prevent error from null velocities.
    """
    threshold = torch.tensor(0.5, **specs.from_tensor(wind_magnitude))
    return 1e-3 * (
        0.142
        + 2.7 / torch.maximum(wind_magnitude, threshold)
        + wind_magnitude / 13.09
    )


def compute_windstress(
    uv10: torch.Tensor,
    uv10_to_uvsurf: torch.Tensor | None = None,
    *,
    rho_water: float = 1000,
    rho_air: float = 1.225,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute surface wind stress.

    Args:
        uv10 (torch.Tensor): Wind velocity at 10m.
        uv10_to_uvsurf (torch.Tensor | None, optional): 10m to surface
            conversion matrix. Defaults to None.
        rho_water (float, optional): Water density. Defaults to 1000.
        rho_air (float, optional): Air density. Defaults to 1.225.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Surface windstress.
    """
    if uv10_to_uvsurf is None:
        uv10_to_uvsurf = torch.eye(2, **specs.from_tensor(uv10))
    usurf, vsurf = torch.einsum(
        "lm,m...->l...",
        uv10_to_uvsurf,
        uv10,
    )

    u_mags = torch.sqrt(uv10.square().sum(dim=0))

    Cd = compute_drag_coef(u_mags)

    bulk_coef = Cd * rho_air / rho_water

    tauxs = bulk_coef * u_mags * usurf
    tauys = bulk_coef * u_mags * vsurf

    tauxs_i = (tauxs[..., 1:, :] + tauxs[..., :-1, :]) / 2
    tauys_i = (tauys[..., :, 1:] + tauys[..., :, :-1]) / 2

    return tauxs_i, tauys_i
