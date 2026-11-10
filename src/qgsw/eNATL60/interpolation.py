"""Data interpolation tools."""

from typing import overload

import numpy as np
import torch
import xarray as xr
import xesmf as xe

from qgsw.eNATL60.var_keys import LATITUDE, LONGITUDE
from qgsw.physics.constants import EARTH_RADIUS


def compute_lonlat_bounds(
    lons: xr.DataArray, lats: xr.DataArray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute longitude and latitude bounds.

    Args:
        lons (xr.DataArray): Longitudes (2D).
        lats (xr.DataArray): Latitiudes (2D).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: lon_min,
            lon_max, lat_min, lat_max
    """
    lon_min = np.deg2rad(lons[0, :].max().values)
    lon_max = np.deg2rad(lons[-1, :].min().values)
    lat_min = np.deg2rad(lats[:, 0].max().values)
    lat_max = np.deg2rad(lats[:, -1].min().values)
    return (lon_min, lon_max, lat_min, lat_max)


def compute_lonlat_from_regular_xy_grid(
    lons: xr.DataArray,
    lats: xr.DataArray,
    dx: float = 10000,
    dy: float = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute longitude / latitude grids with regular cartesian spacing.

    Args:
        lons (xr.DataArray): Longitude values.
        lats (xr.DataArray): Latitude values.
        dx (float, optional): Spacing along x in cartesian coordinates.
            Defaults to 10000.
        dy (float, optional): Spacing along y in cartesian coordinates.
            Defaults to 10000.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """
    lon_min, lon_max, lat_min, lat_max = compute_lonlat_bounds(lons, lats)

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    dlat = dy / EARTH_RADIUS

    nlat = lat_range // dlat

    lat_offset = (lat_range - nlat * dlat) / 2

    lats_ = np.arange(0, int(nlat) + 1) * dlat + lat_min + lat_offset

    dlons = dx / EARTH_RADIUS / np.cos(lats_)
    dlon_max = np.max(dlons)
    nlon = lon_range // dlon_max
    dlon_offsets = (lon_range - nlon * dlons) / 2

    lons_ = (
        np.arange(0, nlon + 1).reshape((-1, 1)) * dlons.reshape((1, -1))
        + dlon_offsets.reshape((1, -1))
        + lon_min
    )
    return lons_, np.tile(lats_.reshape(1, -1), (int(nlon) + 1, 1))


@overload
def retrieve_lat0(
    lats: np.ndarray,
) -> np.ndarray: ...
@overload
def retrieve_lat0(
    lats: torch.Tensor,
) -> torch.Tensor: ...


def retrieve_lat0(
    lats: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Retrieve reference latitude.

    Args:
        lats (np.ndarray | torch.Tensor): Latitudes.

    Returns:
        np.ndarray | torch.Tensor: Reference latitude.
    """
    return (lats.max() + lats.min()) / 2


@overload
def lonlat_to_xy(
    lons: np.ndarray, lats: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]: ...
@overload
def lonlat_to_xy(
    lons: torch.Tensor, lats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
def lonlat_to_xy(
    lons: np.ndarray | torch.Tensor,
    lats: np.ndarray | torch.Tensor,
) -> (
    tuple[np.ndarray, np.ndarray]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Convert regular longitude/latitude to cartesian x and ys.

    Args:
        lons (np.ndarray | torch.Tensor): Longitudes.
        lats (np.ndarray | torch.Tensor): Latitudes.

    Returns:
        tuple[np.ndarray, np.ndarray, float, np.ndarray] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: X,Y, reference longitudes, reference latitude.
    """  # noqa: E501
    lat0 = retrieve_lat0(lats)
    lons0 = lons[0, :1]
    ys = EARTH_RADIUS * (lats - lat0)
    cos = torch.cos if isinstance(lats, torch.Tensor) else np.cos
    xs = EARTH_RADIUS * lons * cos(lats)

    return xs, ys, lons0, lat0


@overload
def xy_to_lonlat(
    xs: np.ndarray, ys: np.ndarray, lons0: np.ndarray, lat0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def xy_to_lonlat(
    xs: torch.Tensor, ys: torch.Tensor, lons0: torch.Tensor, lat0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...


def xy_to_lonlat(
    xs: np.ndarray | torch.Tensor,
    ys: np.ndarray | torch.Tensor,
    lons0: np.ndarray | torch.Tensor,
    lat0: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """Convert cartesian coordinates to longitude latitudes.

    Args:
        xs (np.ndarray | torch.Tensor): X locations.
        ys (np.ndarray | torch.Tensor): Y locations.
        lons0 (np.ndarray | torch.Tensor): Reference longitudes.
        lat0 (np.ndarray | torch.Tensor): Reference latitudes.

    Returns:
        tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]: Longitudes, latitudes.
    """  # noqa: E501
    lats = ys / EARTH_RADIUS + lat0
    cos = torch.cos if isinstance(lats, torch.Tensor) else np.cos
    lons = xs / cos(lats) / EARTH_RADIUS + lons0
    return lons, lats


def build_regridder(
    ds: xr.Dataset,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    method: str = "bilinear",
) -> xe.Regridder:
    """Build regridder for interpolation.

    Args:
        ds (xr.Dataset): Dataset with reference lon / lat.
        lons (np.ndarray): Longitudes to interpolate onto.
        lats (np.ndarray): Latitudes to interpolate onto.
        method (str, optional): Regridding method to pass to xe.Regridder.
            Defaults to 'bilinear'.

    Returns:
        xe.Regridder: Regridder.
    """
    ds_out = xr.Dataset(
        {
            "lon": (["i", "j"], np.rad2deg(lons)),
            "lat": (["i", "j"], np.rad2deg(lats)),
        }
    )
    if (len(ds[LONGITUDE].shape) == 1) and (len(ds[LATITUDE].shape) == 1):
        lons_, lats_ = xr.broadcast(ds[LONGITUDE], ds[LATITUDE])
        lons_ref = np.ascontiguousarray(lons_)
        lats_ref = np.ascontiguousarray(lats_)
    elif (len(ds[LONGITUDE].shape) == 2) and (len(ds[LATITUDE].shape) == 2):
        lons_ref = np.ascontiguousarray(ds[LONGITUDE])
        lats_ref = np.ascontiguousarray(ds[LATITUDE])

    else:
        msg = "Incompatible lon/lat shapes."
        raise ValueError(msg)
    ds_in = xr.Dataset(
        {
            "lon": (["i", "j"], lons_ref),
            "lat": (["i", "j"], lats_ref),
        }
    )
    return xe.Regridder(
        ds_in,
        ds_out,
        method=method,
        periodic=False,
    )
