"""Functions for eNATL60 scripts."""  # noqa: N999

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

from qgsw.eNATL60.forcing import (
    interpolate_era,
    load_era_interim,
    slice_space,
    slice_time,
)
from qgsw.eNATL60.loading import load_datasets, retrieve_dates
from qgsw.eNATL60.var_keys import (
    ATMOS_PRESSURE,
    LATITUDE,
    LONGITUDE,
    MERIDIONAL_WIND_10M,
    SSH,
    SST,
    TIME,
    ZONAL_WIND_10M,
)
from qgsw.logging.core import getLogger
from qgsw.specs import defaults

if TYPE_CHECKING:
    from pathlib import Path

logger = getLogger(__name__)


def format_ds(ds: xr.Dataset) -> xr.Dataset:
    """Format Dataset."""
    # Drop useless variables
    if "axis_nbounds" in ds.dims:
        ds = ds.drop_dims("axis_nbounds")
    if "time_centered" in ds.coords:
        ds = ds.reset_coords("time_centered", drop=True)
    # Rename
    ds = ds.rename(
        {
            "time_counter": TIME,
            "nav_lon": LONGITUDE,
            "nav_lat": LATITUDE,
            "x": "i",
            "y": "j",
            "sossheig": SSH,
            "sosstsst": SST,
        }
    )
    ds = ds.transpose(TIME, "i", "j")
    return ds.set_coords([LONGITUDE, LATITUDE])


def load_netcdfs(
    files_enatl60: list[Path],
    era_folder: Path,
    *,
    load_wind: bool = True,
) -> xr.Dataset:
    """Load NetCDFs.

    Args:
        files_enatl60 (list[Path]): Files to load.
        era_folder (Path): Folder with ERA data
        load_wind (bool, optional): Whether to load wind or not.
            Defaults to True.

    Returns:
        xr.Dataset: _description_
    """
    with logger.timeit("Loading Dataset"):
        ds = load_datasets(
            *files_enatl60,
            format_func=format_ds,
            chunks="auto",
        )

    with logger.timeit("Loading ERA data"):
        dates = retrieve_dates(*files_enatl60)
        years = dates.year.unique().to_list()
        if dates.min().month == 1 and dates.min().day == 1:
            years.insert(0, dates.min().year - 1)
        msg = f"Loading data for years: {', '.join([str(y) for y in years])}"
        logger.info(msg)
        with load_era_interim(era_folder, *years) as ds_era:
            ds_era_sliced = slice_time(ds_era, ds[TIME])
            ds_era_sliced = slice_space(
                ds_era_sliced, ds[LONGITUDE], ds[LATITUDE]
            )
            interp = interpolate_era(
                ds_era_sliced[
                    [ATMOS_PRESSURE]
                    + [ZONAL_WIND_10M, MERIDIONAL_WIND_10M] * load_wind
                ],
                ds[TIME],
                ds[LONGITUDE],
                ds[LATITUDE],
            )
            ds.update(interp)
    return ds


def filter_streamfunction(
    da_psi: xr.DataArray, sigma_ic: float, sigma_bc: float
) -> tuple[xr.DataArray, xr.DataArray]:
    """Filter stream function.

    Args:
        da_psi (xr.DataArray): Stream function data array.
        sigma_ic (float): Initial condition std for filtering.
        sigma_bc (float): Boundary condition std for filtering.

    Returns:
        tuple[xr.DataArray, xr.DataArray]: _description_
    """
    with logger.timeit("Filtering stream function"):
        msg = f"Using σ={sigma_ic} for initial condition"  # noqa: RUF001
        logger.info(msg)
        psi0_filt_da = xr.DataArray(
            gaussian_filter(
                da_psi[0].data,
                sigma=(sigma_ic, sigma_ic),
                axes=(-2, -1),
            ),
            dims=da_psi[0].dims,
            coords=da_psi[0].coords,
            name="psi_filt",
        )
        msg = f"Using σ={sigma_bc} for boundary conditions"  # noqa: RUF001
        psis_filt_da = xr.DataArray(
            gaussian_filter(
                da_psi.data,
                sigma=(sigma_bc, sigma_bc),
                axes=(-2, -1),
            ),
            dims=da_psi.dims,
            coords=da_psi.coords,
            name="psi_filt",
        )
        logger.info(msg)
    return psi0_filt_da, psis_filt_da


def da_to_tensor(
    da: xr.DataArray,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert DataArray to Tensor.

    Args:
        da (xr.DataArray): DataArray.
        dtype (torch.dtype | None, optional): Data type. Defaults to None.
        device (torch.device | None, optional): Device. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    return torch.tensor(
        da.to_numpy(), **defaults.get(dtype=dtype, device=device)
    )[..., None, None, :, :]
