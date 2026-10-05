"""Diagnostics using enatl60 data."""

import calendar
import gc
from collections.abc import Callable
from pathlib import Path

import dask.array as dar
import dask_image.ndfilters as dask_image
import gsw
import numpy as np
import pandas as pd
import torch
import xarray as xr
import xgcm

from qgsw.eNATL60 import seasons
from qgsw.eNATL60.loading import sort_files_by_dates
from qgsw.logging.core import getLogger, setup_root_logger
from qgsw.specs import defaults
from qgsw.utils.reshaping import crop_xr
from qgsw.utils.storage import get_absolute_storage_path, get_path_from_env

torch.backends.cudnn.deterministic = True

ROOT_PATH = Path.cwd()


specs = defaults.get()

logger = getLogger(__name__)
setup_root_logger(1)

domain_bounds = {"x": slice(2280, 2884.5), "y": slice(1865, 2688.5)}

vort_filename = "vorticity.npy"
div_filename = "divergence.npy"
sst_filename = "sst.npy"

data_dt = 7200
sigma = 3
bc = 10

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
    "gridf": {
        "x": "x_r",
        "y": "y_r",
        "nav_lon": "llon_rr",
        "nav_lat": "llat_rr",
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


def filt(da: xr.DataArray) -> xr.DataArray:
    """Filter data array."""
    sigmas = [sigma if d[0] in ["y", "x"] else 0 for d in da.dims]
    return xr.DataArray(
        dask_image.gaussian_filter(
            da.data,
            sigma=sigmas,
        ),
        name=da.name,
        coords=da.coords,
        dims=da.dims,
    )


def build_interpf(
    xs: xr.DataArray, ys: xr.DataArray
) -> Callable[[xr.DataArray], xr.DataArray]:
    """Build interpolation function."""

    def interp(da: xr.DataArray) -> xr.DataArray:

        interp_y = "y_r" in da.dims
        interp_x = "x_r" in da.dims

        if interp_x and interp_y:
            return da.interp({"y_r": ys, "x_r": xs})
        if interp_x:
            return da.interp({"x_r": xs})
        if interp_y:
            return da.interp({"y_r": ys})
        return da

    def interpf(da: xr.DataArray) -> xr.DataArray:
        return filt(interp(da))

    return interpf


def rms(da: xr.DataArray) -> np.ndarray:
    """Compute RMS of a data array."""
    return dar.sqrt(
        dar.square(crop_xr(da, bc, x="x_c", y="y_c")).mean(dim=("x_c", "y_c"))
    ).to_numpy()


if __name__ == "__main__":
    output_dir = get_absolute_storage_path(Path("diagnostics"))
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
        grid = xgcm.Grid(
            dg,
            metrics={
                ("X",): ["e1t", "e1u", "e1v", "e1f"],
                ("Y",): ["e2t", "e2u", "e2v", "e2f"],
            },
            padding="extend",
        )

        def dx(da: xr.DataArray) -> xr.DataArray:
            """Dx function."""
            return grid.derivative(da, "X")

        def dy(da: xr.DataArray) -> xr.DataArray:
            """Dy function."""
            return grid.derivative(da, "Y")

        def dt(da: xr.DataArray) -> xr.DataArray:
            """Dt function."""
            return da.diff(dim="t", label="lower") / data_dt

        output = {
            season: {calendar.month_name[m].lower(): [] for m in months}
            for season, months in season_map.items()
        }
        for f in [vort_filename, div_filename, sst_filename]:
            np.save(output_dir / f, output)

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
                logger.end_section()
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
            ds = ds.assign_coords(
                corio_t=gsw.f(ds.llat_cc), corio_f=gsw.f(ds.llat_rr)
            )
            f0 = ds["corio_t"].mean().compute()
            interpf = build_interpf(ds["x_c"], ds["y_c"])

            with logger.timeit("Computing velocity, vort and div"):
                u = filt(ds["u"])
                v = filt(ds["v"])
                ssh = filt(ds["ssh"])

                u_ = interpf(u)
                v_ = interpf(v)

                u_g_ = -9.81 * dy(ssh) / f0
                v_g_ = 9.81 * dx(ssh) / f0

                u_g = interpf(u_g_)
                v_g = interpf(v_g_)

                u_a = interpf(u_) - u_g
                v_a = interpf(v_) - v_g

                vort = interpf(dx(v) - dy(u))
                vort_g = interpf(dx(v_g_)) - interpf(dy(u_g_))
                vort_a = interpf(dx(v_a)) - interpf(dy(u_a))

                div = interpf(dx(u) + dy(v))

            with logger.timeit("Inferring vorticity equation terms"):
                dt_vort = filt(dt(vort))

                dt_vort_g = filt(dt(vort_g))
                dt_vort_a = filt(dt(vort_a))

                dx_vort_g = interpf(dx(vort_g))
                dy_vort_g = interpf(dy(vort_g))
                dx_vort_a = interpf(dx(vort_a))
                dy_vort_a = interpf(dy(vort_a))

                u_g_grad_vort_g = u_g * dx_vort_g + v_g * dy_vort_g
                u_a_grad_vort_g = u_a * dx_vort_g + v_a * dy_vort_g
                u_g_grad_vort_a = u_g * dx_vort_a + v_g * dy_vort_a
                u_a_grad_vort_a = u_a * dx_vort_a + v_a * dy_vort_a

                Dtg_vort_g = dt_vort_g + u_g_grad_vort_g[:-1]
                Dtg_vort_a = dt_vort_a + u_g_grad_vort_a[:-1]

                dx_f = interpf(dx(ds["corio_t"]))
                dy_f = interpf(dy(ds["corio_t"]))

                u_grad_f = u_ * dx_f + v_ * dy_f
                u_g_grad_f = u_g * dx_f + v_g * dy_f

                dx_u = interpf(dx(u))
                dy_v = interpf(dy(v))

                vort_div_u = vort * div
                vort_g_div_u = vort_g * div
                vort_a_div_u = vort_a * div

                f_div_u = div * interpf(ds["corio_t"])

                dt_vort_g_rms = rms(dt_vort_g)
                dt_vort_a_rms = rms(dt_vort_a)
                u_g_grad_vort_g_rms = rms(u_g_grad_vort_g[:-1])
                u_g_grad_vort_a_rms = rms(u_g_grad_vort_a[:-1])
                Dtg_vort_g_rms = rms(Dtg_vort_g)
                Dtg_vort_a_rms = rms(Dtg_vort_a)
                u_a_grad_vort_g_rms = rms(u_a_grad_vort_g[:-1])
                u_a_grad_vort_a_rms = rms(u_a_grad_vort_a[:-1])
                u_grad_f_rms = rms(u_grad_f[:-1])
                vort_div_u_rms = rms(vort_div_u[:-1])
                vort_g_div_u_rms = rms(vort_g_div_u[:-1])
                vort_a_div_u_rms = rms(vort_a_div_u[:-1])
                f_div_u_rms = rms(f_div_u[:-1])
            gc.collect()
            data_ = np.load(output_dir / vort_filename, allow_pickle=True)
            data = data_.item()
            data[season][calendar.month_name[month].lower()] = {
                "time": (ds["t"][:-1].to_numpy(), r"$t$"),
                "dt_vort_g": (dt_vort_g_rms, r"$\partial_t \zeta_g$"),
                "dt_vort_a": (dt_vort_a_rms, r"$\partial_t \zeta_a$"),
                "u_g_grad_vort_g": (
                    u_g_grad_vort_g_rms,
                    r"${\bf{u}}_g\cdot\nabla \zeta_g$",
                ),
                "u_g_grad_vort_a": (
                    u_g_grad_vort_a_rms,
                    r"${\bf{u}}_g\cdot\nabla \zeta_a$",
                ),
                "Dtg_vort_g": (
                    Dtg_vort_g_rms,
                    r"$\mathrm{D}_g \zeta_g / \mathrm{D}_t$",
                ),
                "Dtg_vort_a": (
                    Dtg_vort_a_rms,
                    r"$\mathrm{D}_g \zeta_a / \mathrm{D}_t$",
                ),
                "vort_div_u": (vort_div_u_rms, r"$\zeta \delta$"),
                "vort_g_div_u": (vort_g_div_u_rms, r"$\zeta_g \delta$"),
                "vort_a_div_u": (vort_a_div_u_rms, r"$\zeta_a \delta$"),
                "f_div_u": (f_div_u_rms, r"$f \delta$"),
                "u_grad_f": (u_grad_f_rms, r"${\bf{u}}\cdot\nabla f$"),
                "u_a_grad_vort_g": (
                    u_a_grad_vort_g_rms,
                    r"${\bf{u}}_a\cdot\nabla \zeta_g$",
                ),
                "u_a_grad_vort_a": (
                    u_a_grad_vort_a_rms,
                    r"${\bf{u}}_a\cdot\nabla \zeta_a$",
                ),
            }
            np.save(output_dir / vort_filename, data)
            gc.collect()
            with logger.timeit("Inferring divergence equation terms"):
                dt_div = filt(dt(div))

                dx_div = interpf(dx(div))

                dy_div = interpf(dy(div))

                u_grad_div = u_ * dx_div + v_ * dy_div
                u_g_grad_div = u_g * dx_div + v_g * dy_div
                u_a_grad_div = u_a * dx_div + v_a * dy_div

                Dt_div = dt_div + u_grad_div[:-1]

                sigma_squared = (
                    (interpf(dx(v)) + interpf(dy(u))) ** 2
                    + (interpf(dx(u)) - interpf(dy(v))) ** 2
                ) / 2
                zeta_squared = vort**2 / 2
                Q_squared = sigma_squared - zeta_squared

                d_squared = div**2 / 2

                f_zeta = -vort * filt(ds["corio_t"])

                grad_f_u = u_ * interpf(dy(ds["corio_t"])) - v_ * interpf(
                    dx(ds["corio_t"])
                )

                lap_geop = 9.81 * filt(dx(dx(ssh))) + filt(dy(dy(ssh)))

                dt_div_rms = rms(dt_div)
                Dt_div_rms = rms(Dt_div)
                u_grad_div_rms = rms(u_grad_div[:-1])
                u_g_grad_div_rms = rms(u_g_grad_div[:-1])
                u_a_grad_div_rms = rms(u_a_grad_div[:-1])
                sigma_squared_rms = rms(sigma_squared[:-1])
                zeta_squared_rms = rms(zeta_squared[:-1])
                Q_squared_rms = rms(Q_squared[:-1])
                d_squared_rms = rms(d_squared[:-1])
                f_zeta_rms = rms(f_zeta[:-1])
                grad_f_u_rms = rms(grad_f_u[:-1])
                lap_geop_rms = rms(lap_geop[:-1])
                diff_lap_geop_fz = rms(lap_geop[:-1] + f_zeta[:-1])
            gc.collect()
            data_ = np.load(output_dir / div_filename, allow_pickle=True)
            data = data_.item()
            data[season][calendar.month_name[month].lower()] = {
                "time": (ds["t"][:-1].to_numpy(), r"$t$"),
                "Dt_div": (Dt_div_rms, r"$\mathrm{D}_t \delta$"),
                "dt_div": (dt_div_rms, r"$\partial_t \delta$"),
                "u_grad_div": (
                    u_grad_div_rms,
                    r"${\bf{u}} \cdot \nabla \delta$",
                ),
                "u_g_grad_div": (
                    u_g_grad_div_rms,
                    r"${\bf{u}}_g \cdot \nabla \delta$",
                ),
                "u_a_grad_div": (
                    u_a_grad_div_rms,
                    r"${\bf{u}}_a \cdot \nabla \delta$",
                ),
                "Q_squared": (Q_squared_rms, r"$(\sigma^2 - \zeta^2)/2$"),
                "div_squared": (d_squared_rms, r"$\delta^2/2$"),
                "sigma_squared": (sigma_squared_rms, r"$\sigma^2/2$"),
                "zeta_squared": (zeta_squared_rms, r"$\zeta^2/2$"),
                "f_zeta": (f_zeta_rms, r"$-f\zeta$"),
                "grad_f_u": (
                    grad_f_u_rms,
                    r"$-\nabla^{\perp}f \cdot {\bf{u}}$",
                ),
                "lap_geop": (lap_geop_rms, r"$\Delta \Phi$"),
                "diff_lap_geo": (diff_lap_geop_fz, r"$\Delta \Phi - f\zeta$"),
            }
            np.save(output_dir / div_filename, data)
            gc.collect()
            with logger.timeit("Inferring SST advection equation terms"):
                sst = filt(ds["sst"])
                dt_sst = filt(dt(sst))

                dx_sst = interpf(dx(sst))

                dy_sst = interpf(dy(sst))

                u_grad_sst = u_ * dx_sst + v_ * dy_sst
                u_g_grad_sst = u_g * dx_sst + v_g * dy_sst
                u_a_grad_sst = u_grad_sst - u_g_grad_sst

                Dt_sst = dt_sst + u_grad_sst[:-1]

                Dtg_sst = dt_sst + u_g_grad_sst[:-1]

                dt_sst_rms = rms(dt_sst)
                Dt_sst_rms = rms(Dt_sst)
                Dtg_sst_rms = rms(Dtg_sst)
                u_grad_sst_rms = rms(u_grad_sst[:-1])
                u_g_grad_sst_rms = rms(u_g_grad_sst[:-1])
                u_a_grad_sst_rms = rms(u_a_grad_sst[:-1])
            gc.collect()
            data_ = np.load(output_dir / sst_filename, allow_pickle=True)
            data = data_.item()
            data[season][calendar.month_name[month].lower()] = {
                "time": (ds["t"][:-1].to_numpy(), r"$t$"),
                "dt_sst": (dt_sst_rms, r"$\partial_t T$"),
                "Dt_sst": (
                    Dt_sst_rms,
                    r"${\mathrm{D}} T / {\mathrm{D}} t $",
                ),
                "Dtg_sst": (
                    Dtg_sst_rms,
                    r"${\mathrm{D}}_g T / {\mathrm{D}} t $",
                ),
                "u_grad_sst": (u_grad_sst_rms, r"${\bf{u}} \cdot \nabla T$"),
                "u_g_grad_sst": (
                    u_g_grad_sst_rms,
                    r"${\bf{u}}_g \cdot \nabla T$",
                ),
                "u_a_grad_sst": (
                    u_a_grad_sst_rms,
                    r"${\bf{u}}_a \cdot \nabla T$",
                ),
            }
            np.save(output_dir / sst_filename, data)
            gc.collect()
            ds.close()
            gc.collect()
            logger.end_section(m_token, "Done")

        logger.end_section(s_token)
