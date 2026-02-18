"""Fields computations."""

import xarray as xr

from qgsw.eNATL60.forcing import interpolate_era_da
from qgsw.eNATL60.var_keys import MEAN_SEA_LEVEL, SSH


def compute_streamfunction(
    ds: xr.Dataset,
    ds_era: xr.Dataset,
    rho0: float = 1026.0,
    g: float = 9.81,
) -> xr.DataArray:
    """Compute surface streamfunction.

    Args:
        ds (xr.Dataset): _description_
        ds_era (xr.Dataset): _description_
        rho0 (float, optional): _description_. Defaults to 1026.0.
        g (float, optional): _description_. Defaults to 9.81.

    Returns:
        xr.DataArray: _description_
    """
    msl = interpolate_era_da(ds_era[MEAN_SEA_LEVEL], ds)
    return msl / rho0 + g * ds[SSH]
