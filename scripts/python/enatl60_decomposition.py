"""Helmholtz decomposition of surface velocities from enatl60.

Decomposition functions made by Claude.
"""

import calendar
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from qgsw.eNATL60 import seasons
from qgsw.eNATL60.loading import sort_files_by_dates
from qgsw.logging.core import getLogger, setup_root_logger
from qgsw.physics.constants import EARTH_RADIUS
from qgsw.specs import defaults
from qgsw.utils.storage import get_absolute_storage_path, get_path_from_env

torch.backends.cudnn.deterministic = True

ROOT_PATH = Path.cwd()


specs = defaults.get()

logger = getLogger(__name__)
setup_root_logger(1)

domain_bounds = {"x": slice(2280, 2884.5), "y": slice(1865, 2688.5)}


season_map = {
    "summer": seasons.SUMMER,
    "autumn": seasons.AUTUMN,
    "winter": seasons.WINTER,
    "spring": seasons.SPRING,
}

dico_name_grd = {
    "gridT": {
        "x": "x_c",
        "y": "y_c",
        "nav_lon": "llon_cc",
        "nav_lat": "llat_cc",
    },
    "gridU": {
        "x": "x_r",
        "y": "y_c",
        "nav_lon": "llon_cr",
        "nav_lat": "llat_cr",
    },
    "gridV": {
        "x": "x_c",
        "y": "y_r",
        "nav_lon": "llon_rc",
        "nav_lat": "llat_rc",
    },
}
dico_name_all = {
    "sossheig": "ssh",
    "sozocrtx": "u",
    "somecrty": "v",
    "sosstsst": "sst",
}


def open_file(path_or_list: list[Path] | Path) -> xr.Dataset:
    """Open file."""
    if isinstance(path_or_list, list):
        res = xr.combine_nested(
            [open_file(f) for f in path_or_list], concat_dim="time_counter"
        ).sortby("time_counter")
    elif path_or_list.suffix == ".zarr":
        res = xr.open_zarr(path_or_list).set_coords(["nav_lat", "nav_lon"])
    elif path_or_list.suffix == ".nc":
        res = xr.open_dataset(path_or_list, chunks="auto").set_coords(
            ["nav_lat", "nav_lon"]
        )
    else:
        msg_0 = "unrecognised file type to open"
        raise ValueError(msg_0)
    if "axis_nbounds" in res.dims:
        res = res.drop_dims("axis_nbounds")
    if "time_centered" in res.coords:
        res = res.reset_coords("time_centered", drop=True)
    return res


def load(files: list[Path] | Path, var: str) -> xr.Dataset:
    """Load a dataset."""
    ds = open_file(files)
    ds = ds.rename(
        {
            v: dico_name_grd[var][v]
            for v in list(ds.dims) + list(ds.coords)
            if v in dico_name_grd[var]
        }
    )
    ds = ds.rename(
        {v: dico_name_all[v] for v in ds.data_vars if v in dico_name_all}
    )
    return ds.rename({"time_counter": "t"})


def load_season_files(
    months: list[int],
    files: list[Path],
    ufiles: list[Path],
    vfiles: list[Path],
) -> tuple[list[Path], list[Path], list[Path]]:
    """Load season files."""
    dates = pd.to_datetime([f.name[-20:-12] for f in files])
    in_season = dates.month.isin(months)
    if ((in_season[1:]) & (~in_season[:-1])).sum() + int(in_season[0]) > 1:
        msg = "Non-time-contiguous data for this season in provided dataset."
        raise ValueError(msg)
    return (
        list(np.array(files)[in_season]),
        list(np.array(ufiles)[in_season]),
        list(np.array(vfiles)[in_season]),
    )


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
class Grid:
    """Grid."""

    def __init__(self, lon_deg: np.ndarray, lat_deg: np.ndarray) -> None:
        """Instantiate grid."""
        lon = np.deg2rad(np.asarray(lon_deg, float))
        lat = np.deg2rad(np.asarray(lat_deg, float))
        if lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)  # shape (ny, nx)
        self.ny, self.nx = lon.shape
        ny, nx = self.ny, self.nx
        ids = np.arange(ny * nx).reshape(ny, nx)
        n00, n01 = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel()
        n10, n11 = ids[1:, :-1].ravel(), ids[1:, 1:].ravel()
        nc = n00.size
        rows = np.tile(np.arange(nc), 4)
        cols = np.concatenate([n00, n01, n10, n11])
        shape = (nc, ny * nx)
        # Index derivative (axes 0 = lat, axes 1 = lon)
        D0 = sp.csr_matrix(
            (np.repeat([-0.5, -0.5, 0.5, 0.5], nc), (rows, cols)), shape=shape
        )
        D1 = sp.csr_matrix(
            (np.repeat([-0.5, 0.5, -0.5, 0.5], nc), (rows, cols)), shape=shape
        )
        self.C = sp.csr_matrix(
            (np.full(4 * nc, 0.25), (rows, cols)), shape=shape
        )  # nodes -> cells
        self.Ind = sp.csr_matrix(
            (np.ones(4 * nc), (rows, cols)), shape=shape
        )  # corners

        lam, phi = lon.ravel(), lat.ravel()
        l0, l1, p0, p1 = D0 @ lam, D1 @ lam, D0 @ phi, D1 @ phi
        phic = self.C @ phi
        jac = l0 * p1 - l1 * p0
        cosp = np.cos(phic)
        # physical gradient (m^-1) : d/dx = 1/(R cos) d/dlam ; d/dy = 1/R d/dphi  # noqa: E501
        self.Gx = (
            sp.diags(p1 / jac / (EARTH_RADIUS * cosp)) @ D0
            - sp.diags(p0 / jac / (EARTH_RADIUS * cosp)) @ D1
        )
        self.Gy = (
            sp.diags(-l1 / jac / EARTH_RADIUS) @ D0
            + sp.diags(l0 / jac / EARTH_RADIUS) @ D1
        )
        self.area = EARTH_RADIUS**2 * cosp * np.abs(jac)
        self.shape = (ny, nx)


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------
def _solve(
    G: Grid,
    W: np.ndarray,
    uc: np.ndarray,
    vc: np.ndarray,
    free: np.ndarray,
    kind: str,
) -> np.ndarray:
    Gx, Gy = G.Gx[:, free], G.Gy[:, free]
    N = (Gx.T @ W @ Gx + Gy.T @ W @ Gy).tocsc()
    if kind == "grad":  # (Gx f, Gy f) ~ (u, v)
        b = Gx.T @ (W @ uc) + Gy.T @ (W @ vc)
    else:  # (-Gy f, Gx f) ~ (u, v)
        b = Gx.T @ (W @ vc) - Gy.T @ (W @ uc)
    N = N + sp.identity(N.shape[0], format="csc") * (
        1e-10 * N.diagonal().mean()
    )
    out = np.zeros(G.Gx.shape[1])
    out[free] = spsolve(N, b)
    return out


def helmholtz_2d(
    G: Grid, u: np.ndarray, v: np.ndarray, bc: str = "normal"
) -> dict[str, np.ndarray[tuple[int, int], np.dtype[Any]]]:
    """u, v : velocities."""
    uu, vv = u.ravel(), v.ravel()
    ok = np.isfinite(uu) & np.isfinite(vv)
    cell_ok = (G.Ind @ ok.astype(float)) == 4
    uc = G.C @ np.where(ok, uu, 0.0)
    vc = G.C @ np.where(ok, vv, 0.0)
    W = sp.diags(G.area * cell_ok)

    S = G.Ind.T @ sp.diags(cell_ok.astype(float))  # valid cells -> nodes
    ncell = np.asarray(S.sum(axis=1)).ravel()
    touched = ncell > 0
    interior = ncell == 4  # verify all neighbors are valid
    if bc == "normal":
        free_phi, free_psi = touched, interior
    elif bc == "tangential":
        free_phi, free_psi = interior, touched
    else:
        msg_0 = "Invalid value for bc"
        raise ValueError(msg_0)

    phi = _solve(G, W, uc, vc, free_phi, "grad")
    psi = _solve(G, W, uc, vc, free_psi, "rot")

    def to_nodes(xc: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (S @ xc) / ncell
        r[~touched] = np.nan
        return r

    ud, vd = to_nodes(G.Gx @ phi), to_nodes(G.Gy @ phi)
    ur, vr = to_nodes(-(G.Gy @ psi)), to_nodes(G.Gx @ psi)
    for a in (phi, psi):
        a[~touched] = np.nan
    sh = G.shape
    res = {
        "phi": phi,
        "psi": psi,
        "u_div": ud,
        "v_div": vd,
        "u_rot": ur,
        "v_rot": vr,
        "u_res": uu - ud - ur,
        "v_res": vv - vd - vr,
    }
    return {k: x.reshape(sh) for k, x in res.items()}


def _resolve(
    da: xr.DataArray, lon: str, lat: str
) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Return (ydim, xdim, lon2d/1d, lat2d/1d) for a variable."""
    lon = da.coords[lon]
    lat = da.coords[lat]
    if lon.ndim == 1 and lat.ndim == 1:
        ydim, xdim = lat.dims[0], lon.dims[0]
        lo, la = np.meshgrid(lon.values, lat.values)
    elif lon.ndim == 2 and lat.ndim == 2:
        ydim, xdim = lat.dims
        lo, la = lon.transpose(ydim, xdim).to_numpy(), lat.to_numpy()
    else:
        msg_0 = "lon / lat must be both in 1D or 2D."
        raise ValueError(msg_0)
    missing = {ydim, xdim} - set(da.dims)
    if missing:
        msg_1 = "Missing cooridnates."
        raise ValueError(msg_1)
    return ydim, xdim, lo, la


def helmholtz(
    u: xr.DataArray,
    v: xr.DataArray,
    lon_u: str,
    lat_u: str,
    lon_v: str,
    lat_v: str,
    bc: str = "normal",
) -> xr.Dataset:
    """Helmholtz-Hodge decomposition of (u, v) of lon/lat (degrés) grid.

    Parameters
    ----------
    u, v : xarray.DataArray. U and V DataArrays
    lon_u, lat_u: U coordintaes label.
    lon_v, lat_v: V coordinates label.
    bc : "normal" (psi=0 at the border) or "tangential" (phi=0 at the border).
    Return : xarray.Dataset on u grid with phi and psi (m2/s),
             u_div, v_div, u_rot, v_rot, u_res, v_res (m/s).
    """
    yu, xu, lo_u, la_u = _resolve(u, lon_u, lat_u)
    yv, xv, lo_v, la_v = _resolve(v, lon_v, lat_v)

    lead = [d for d in u.dims if d not in (yu, xu)]
    ut = u.transpose(*lead, yu, xu)
    vt = v.transpose(*lead, yv, xv)
    ny, nx = lo_u.shape
    u3 = ut.to_numpy().reshape(-1, ny, nx).astype(float)
    v_raw = vt.to_numpy().reshape(-1, *lo_v.shape).astype(float)

    same_grid = (
        lo_u.shape == lo_v.shape
        and np.allclose(lo_u, lo_v)
        and np.allclose(la_u, la_v)
    )
    if same_grid:
        v3 = v_raw
    else:
        tri = Delaunay(np.c_[lo_v.ravel(), la_v.ravel()])
        target = np.c_[lo_u.ravel(), la_u.ravel()]
        v3 = np.empty_like(u3)
        for t in range(u3.shape[0]):
            f = LinearNDInterpolator(tri, v_raw[t].ravel())
            v3[t] = f(target).reshape(ny, nx)  # NaN out of v envelop

    G = Grid(lo_u, la_u)
    keys = ["phi", "psi", "u_div", "v_div", "u_rot", "v_rot", "u_res", "v_res"]
    out = {k: np.full(u3.shape, np.nan) for k in keys}
    for t in range(u3.shape[0]):
        r = helmholtz_2d(G, u3[t], v3[t], bc)
        for k in keys:
            out[k][t] = r[k]

    ds = xr.Dataset()
    for k in keys:
        da = ut.copy(data=out[k].reshape(ut.shape))
        da.attrs = {
            "units": "m2 s-1" if k in ("phi", "psi") else "m s-1",
            "long_name": k,
            "bc": bc,
        }
        ds[k] = da.transpose(*u.dims)
    return ds


if __name__ == "__main__":
    output_dir = get_absolute_storage_path(Path("decomposition"))
    if not output_dir.is_dir():
        output_dir.mkdir()
        gitignore = output_dir.joinpath(".gitignore")
        with gitignore.open("w") as file:
            file.write("*")

    with logger.timeit("Retrieving files"):
        data_folder = get_path_from_env(key="eNATL60_FOLDER")
        all_files = list((data_folder / "MEANDERS" / "gridT").glob("*.nc"))

        all_files = sort_files_by_dates(*all_files)
        all_ufiles = list(
            (data_folder / "MEANDERS" / "gridU").glob("*.nc"),
        )
        all_ufiles = sort_files_by_dates(*all_ufiles)

        all_vfiles = list(
            (data_folder / "MEANDERS" / "gridV").glob("*.nc"),
        )
        all_vfiles = sort_files_by_dates(*all_vfiles)
    with logger.timeit("Loading grid"):
        dg = xr.open_zarr(data_folder / "eNATL60_rest_grid.zarr")
        dg = dg.assign_coords(hbot=dg.e3t.where(dg.tmask).sum("z_c"))
        dg["tmask"], dg["e3t"] = dg.tmask.isel(z_c=0), dg.e3t.isel(z_c=0)
        dg = dg.drop_dims(["z_c", "z_l"])
        dg = dg.chunk(dict.fromkeys(dg.dims, -1))
        dg = dg.sel({d: domain_bounds[d[0]] for d in dg.dims if d[0] in "xy"})

    for season, months in season_map.items():
        msg = f"Loading {season} files"
        s_token = logger.start_section(msg)
        for month in months:
            msg = f"Loading {calendar.month_name[month]} files"
            m_token = logger.start_section(msg)
            files, ufiles, vfiles = load_season_files(
                [month], all_files, all_ufiles, all_vfiles
            )
            if not files or not ufiles or not vfiles:
                msg = f"No files found for {calendar.month_name[month]}"
                logger.warning(msg)
                logger.end_section(m_token)
                continue
            dsh = load(files, "gridT")
            dsu = load(ufiles, "gridU")
            dsv = load(vfiles, "gridV")
            ds = xr.merge([dsh, dsu, dsv])
            ds = ds.assign_coords(
                {v: dg[v] for v in ["x_c", "y_c", "x_r", "y_r"]}
            )
            ds = ds.assign_coords(
                {c: dg[c] for c in dg.coords if c[:2] != "ll"}
            )
            with logger.timeit("Performing Helmholtz decompositon"):
                decomp = helmholtz(
                    ds["u"],
                    ds["v"],
                    lon_u="llon_cr",
                    lat_u="llat_cr",
                    lon_v="llon_rc",
                    lat_v="llat_rc",
                )
            out_file = output_dir / (f"{calendar.month_name[month]}.nc")

            decomp.to_netcdf(out_file)

            msg = f"Saved to {out_file}."
            logger.info(msg)
