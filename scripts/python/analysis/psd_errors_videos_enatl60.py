"""Videos of Power Spectral Density error."""

# ruff: noqa: B023

import gc
import sys
from collections.abc import Callable

import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from qgsw import plots
from qgsw.cli import parse_output_folder, parse_run_id
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.core import build_basis_from_params_dict
from qgsw.eNATL60 import seasons
from qgsw.eNATL60.fields_computations import (
    compute_streamfunction_with_atmospheric_pressure_xy_avg,
)
from qgsw.eNATL60.interpolation import (
    build_regridder,
    compute_lonlat_from_regular_xy_grid,
    lonlat_to_xy,
)
from qgsw.eNATL60.loading import (
    load_datasets,
    retrieve_dates,
    sort_files_by_dates,
)
from qgsw.eNATL60.var_keys import (
    ATMOS_PRESSURE,
    LATITUDE,
    LONGITUDE,
    MERIDIONAL_WIND_10M,
    SSH,
    STREAMFUNCTION,
    TIME,
    VORTICITY,
    ZONAL_WIND_10M,
)
from qgsw.eNATL60.wind import compute_windstress
from qgsw.logging.core import getLogger, setup_root_logger
from qgsw.logging.utils import box, step
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
    QGPSIQForcedSurfML,
    QGPSIQRGPsi2TransportDR,
)
from qgsw.models.sv.core import SurfaceVorticity
from qgsw.physics.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.plots.plt_wrapper import retrieve_imshow_data
from qgsw.scripts.boundaries import extract_psi_bc, extract_wide_psi_bc
from qgsw.scripts.eNATL60 import (
    da_to_tensor,
    filter_streamfunction,
    format_ds,
    load_netcdfs,
)
from qgsw.scripts.model_interactions import init_model
from qgsw.scripts.output import (
    OutputFolder,
    retrieve_output_folder,
)
from qgsw.scripts.pv import build_pv_funcs
from qgsw.scripts.setups.model import setup_model
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import laplacian
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.spatial.core.grid import Grid2D
from qgsw.specs import defaults, from_tensor
from qgsw.utils.reshaping import crop
from qgsw.utils.storage import get_path_from_env

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

logger = getLogger(__name__)

specs = defaults.get()

# Must match the training script.
data_dt = 7200
data_n_steps = 240
n_file_per_cycle = 20


b = 4


def setup(
    cycle: int,
    output: OutputFolder,
    model: QGPSIQ | SurfaceVorticity,
    psi0: torch.Tensor,
    sfs: torch.Tensor,
    times: list[torch.Tensor],
    build_pv: Callable[
        [torch.Tensor, torch.Tensor],
        tuple[
            Callable[[torch.Tensor], torch.Tensor],
            Callable[[Boundaries], Boundaries],
        ],
    ],
) -> bool:
    """Setup model.

    Args:
        cycle (int): Cycle.
        output (OutputFolder): Output.
        model (QGPSIQ | SurfaceVorticity): Model.
        psi0 (torch.Tensor): Filtered initial streamfunction.
        sfs (torch.Tensor): Filtered streamfunctions (boundaries).
        times (torch.Tensor): Times.
        build_pv (Callable): PV func.

    Returns:
        torch.Tensor | None: Wind conversion coefficient if available.
    """
    nb_cycles = len(output.optim.load())
    if cycle >= nb_cycles:
        return True
    bases_params = output.basis.load()[cycle]
    results = output.results.load()[cycle]
    if model.get_type() == QGPSIQForced.get_type():
        basis_external = build_basis_from_params_dict(bases_params["external"])
        coefs = DecompositionCoefs.from_dict(results["external"])
        basis_external.set_coefs(coefs)
        model.basis_external = basis_external
    elif model.get_type() == QGPSIQRGPsi2TransportDR.get_type():
        try:
            alpha = results["alpha"]
        except KeyError:
            alpha = torch.tensor(0, **from_tensor(model.A))
        model.alpha = alpha
        basis_dynamical = build_basis_from_params_dict(
            bases_params["dynamical"]
        )
        coefs = DecompositionCoefs.from_dict(results["dynamical"])
        basis_dynamical.set_coefs(coefs)
        model.basis_dynamical = basis_dynamical
    elif model.get_type() == QGPSIQForcedSurfML.get_type():
        try:
            alpha = results["alpha"]
        except KeyError:
            alpha = torch.tensor(0, **from_tensor(model.A))
        model.alpha = alpha
        basis_dynamical = build_basis_from_params_dict(
            bases_params["dynamical"]
        )
        coefs = DecompositionCoefs.from_dict(results["dynamical"])
        basis_dynamical.set_coefs(coefs)
        model.basis_dynamical = basis_dynamical
        basis_external = build_basis_from_params_dict(bases_params["external"])
        coefs = DecompositionCoefs.from_dict(results["external"])
        basis_external.set_coefs(coefs)
        model.basis_external = basis_external
    elif model.get_type() == SurfaceVorticity.get_type():
        basis_psi_a = build_basis_from_params_dict(bases_params["psi"])
        coefs = DecompositionCoefs.from_dict(results["psi"])
        basis_psi_a.set_coefs(coefs)
        model.basis_psi_a = basis_psi_a
        basis_chi_a = build_basis_from_params_dict(bases_params["chi"])
        coefs = DecompositionCoefs.from_dict(results["chi"])
        basis_chi_a.set_coefs(coefs)
        model.basis_chi_a = basis_chi_a
    psi_bcs = [extract_psi_bc(psi, b) for psi in sfs]
    psi_bcs_w = [extract_wide_psi_bc(psi, b, clone=True) for psi in sfs]
    init_model(model, psi0, psi_bcs, psi_bcs_w, build_pv, times, b)
    return False


if __name__ == "__main__":
    output_folder = parse_output_folder()
    run_id = parse_run_id()

    try:
        output = retrieve_output_folder(
            output_folder,
            run_id,
        )
    except FileNotFoundError:
        msg = "ERROR: Not output file available."
        logger.critical(msg)
        sys.exit(1)

    params = output.params.load()
    setup_root_logger(params["verbose"])

    config = Configuration.from_toml(output.config)
    n_cycles: int = params["cycles"]
    with_wind = not params["no_wind"]

    # Simulation parameters
    dt = params["dt"]
    if data_dt % dt != 0:
        msg = f"Timestep must divide data timestep, hence {data_dt}."
        raise ValueError(msg)
    dt_ratios = data_dt // dt
    # Same conversion as training (used as an offset in number of files).
    separation = int(params["separation"] * dt / 3600 / 24)

    ## Load eNATL60 grid

    data_folder = get_path_from_env(key="eNATL60_FOLDER")
    files = list((data_folder / "MEANDERS" / "gridT").glob("*.nc"))
    files = sort_files_by_dates(*files)

    season = {
        "summer": seasons.SUMMER,
        "autumn": seasons.AUTUMN,
        "winter": seasons.WINTER,
        "spring": seasons.SPRING,
    }

    in_season = retrieve_dates(*files.tolist()).month.isin(
        season[params["season"]]
    )
    if ((in_season[1:]) & (~in_season[:-1])).sum() + int(in_season[0]) > 1:
        msg = "Non-time-contiguous data for this season in provided dataset."
        raise ValueError(msg)
    files = files[in_season]

    ds = load_datasets(files[0], format_func=format_ds)

    lons, lats = compute_lonlat_from_regular_xy_grid(
        ds[LONGITUDE],
        ds[LATITUDE],
        dx=params["dx"],
        dy=params["dy"],
    )
    xs, ys, lons0, lat0 = lonlat_to_xy(lons, lats)

    beta_plane = BetaPlane(
        f0=2 * EARTH_ANGULAR_ROTATION * np.sin(lat0),
        beta=2 * EARTH_ANGULAR_ROTATION * np.cos(lat0) / EARTH_RADIUS,
    )
    f0 = beta_plane.f0

    psi_regridder = build_regridder(ds, lons, lats)

    def format_dsu(ds: xr.Dataset) -> xr.Dataset:
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
                "sozocrtx": "u",
            }
        )
        ds = ds.transpose(TIME, "i", "j")
        return ds.set_coords([LONGITUDE, LATITUDE])

    def format_dsv(ds: xr.Dataset) -> xr.Dataset:
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
                "somecrty": "v",
            }
        )
        ds = ds.transpose(TIME, "i", "j")
        return ds.set_coords([LONGITUDE, LATITUDE])

    ys_ = EARTH_RADIUS * (lats - lat0)
    cos = torch.cos if isinstance(lats, torch.Tensor) else np.cos
    xs_ = EARTH_RADIUS * (lons) * cos(lats)

    space_no_xoffset = SpaceDiscretization2D.from_psi_grid(
        Grid2D(
            x=torch.tensor(xs_.round(), **specs),
            y=torch.tensor(ys_.round(), **specs),
        )
    )

    ux, uy = space_no_xoffset.u.xy
    ux = ux.cpu().numpy()
    uy = uy.cpu().numpy()

    latsu = (uy / EARTH_RADIUS) + lat0
    lonsu = ux / EARTH_RADIUS / np.cos(latsu)

    vx, vy = space_no_xoffset.v.xy
    vx = vx.cpu().numpy()
    vy = vy.cpu().numpy()

    latsv = (vy / EARTH_RADIUS) + lat0
    lonsv = vx / EARTH_RADIUS / np.cos(latsv)

    ufiles = list(
        (data_folder / "MEANDERS" / "gridU").glob("*.nc"),
    )
    ufiles = sort_files_by_dates(*ufiles)

    vfiles = list(
        (data_folder / "MEANDERS" / "gridV").glob("*.nc"),
    )
    vfiles = sort_files_by_dates(*vfiles)

    dsu_init = load_datasets(ufiles[0], format_func=format_dsu)
    u_regridder = build_regridder(dsu_init, lonsu, latsu)
    dsv_init = load_datasets(vfiles[0], format_func=format_dsv)
    v_regridder = build_regridder(dsv_init, lonsv, latsv)

    def load_cycle_uv_data(
        ufiles: np.ndarray, vfiles: np.ndarray, c: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Load uv."""
        start_cycle = c * n_file_per_cycle + c * separation
        end_cycle = (c + 1) * n_file_per_cycle + c * separation

        if end_cycle > len(ufiles):
            msg = f"Not enough ufiles to perform cycle {c} and above."
            logger.warning(msg)
            return None

        ufiles_for_cycle = ufiles[start_cycle:end_cycle]

        start_cycle = c * n_file_per_cycle + c * separation
        end_cycle = (c + 1) * n_file_per_cycle + c * separation

        if end_cycle > len(vfiles):
            msg = f"Not enough vfiles to perform cycle {c} and above."
            logger.warning(msg)
            return None

        vfiles_for_cycle = vfiles[start_cycle:end_cycle]

        dsu = load_datasets(*ufiles_for_cycle, format_func=format_dsu)

        with logger.timeit("Interpolating u"):
            regridded_u: xr.DataArray = u_regridder(dsu["u"])
            dsu_interp = xr.Dataset(
                {
                    LONGITUDE: (["i", "j"], lonsu),
                    LATITUDE: (["i", "j"], latsu),
                    "u": ([TIME, "i", "j"], regridded_u.data),
                },
                regridded_u.coords,
            )
            dsu_interp = dsu_interp.set_coords([LONGITUDE, LATITUDE])
            dsu_interp = dsu_interp.load()

        dsv = load_datasets(*vfiles_for_cycle, format_func=format_dsv)

        with logger.timeit("Interpolating v"):
            regridded_v: xr.DataArray = v_regridder(dsv["v"])
            dsv_interp = xr.Dataset(
                {
                    LONGITUDE: (["i", "j"], lonsv),
                    LATITUDE: (["i", "j"], latsv),
                    "v": ([TIME, "i", "j"], regridded_v.data),
                },
                regridded_v.coords,
            )
            dsv_interp = dsv_interp.set_coords([LONGITUDE, LATITUDE])
            dsv_interp = dsv_interp.load()

        u_refs = [
            torch.tensor(u, **specs).unsqueeze(0).unsqueeze(0)
            for u in dsu_interp["u"].to_numpy()
        ]
        v_refs = [
            torch.tensor(v, **specs).unsqueeze(0).unsqueeze(0)
            for v in dsv_interp["v"].to_numpy()
        ]

        return u_refs, v_refs, dsu["u"], dsv["v"]

    ## Areas

    space_2d = SpaceDiscretization2D.from_psi_grid(
        Grid2D(
            x=torch.tensor(xs.round(), **specs),
            y=torch.tensor(ys.round(), **specs),
        )
    )

    space_interior = space_2d.slice(
        b,
        space_2d.psi.xy.x.shape[0] - b,
        b,
        space_2d.psi.xy.x.shape[1] - b,
    )

    dx = space_interior.dx
    dy = space_interior.dy

    # Parameters

    H = config.model.h
    g_prime = config.model.g_prime

    # Model
    model = setup_model(
        params["model_type"],
        space_interior,
        H,
        g_prime,
        beta_plane,
        dt,
        config.physics.slip_coef,
    )

    # PV computation
    y_w = space_2d.q.xy.y[0, :].unsqueeze(0)
    beta_effect_wide = beta_plane.beta * (y_w - model.y0)
    build_q_rg = build_pv_funcs(
        dx,
        dy,
        beta_plane.f0,
        beta_effect_wide
        * (params["model_type"] != SurfaceVorticity.get_type()),
        b,
        space_2d.nx,
    )

    from qgsw.metrics.spectral import PowerSpectralDensity

    psd_p = PowerSpectralDensity(
        space_interior.nx + 1,
        space_interior.ny + 1,
        space_interior.dx,
        space_interior.dy,
    )
    psd_v = PowerSpectralDensity(
        space_interior.nx - 1,
        space_interior.ny - 1,
        space_interior.dx,
        space_interior.dy,
    )

    def psd_err_p(f: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        """PSD Error for stream function."""
        return psd_p.iso_spec_w(f - f_ref) / psd_p.iso_spec_w(f_ref)

    def psd_err_v(f: torch.Tensor, f_ref: torch.Tensor) -> torch.Tensor:
        """PSD Error for vorticity."""
        return psd_v.iso_spec_w(f - f_ref) / psd_v.iso_spec_w(f_ref)

    from typing import Any

    from qgsw.filters.gaussian import GaussianFilter2D
    from qgsw.logging.utils import sec2text

    for c in range(n_cycles):
        start_cycle = c * n_file_per_cycle + c * separation
        end_cycle = (c + 1) * n_file_per_cycle + c * separation

        msg = f"Making videos for cycle {step(c + 1, n_cycles)}"
        logger.info(box(msg))
        torch.cuda.reset_peak_memory_stats()

        files_for_cycle = files[start_cycle:end_cycle]

        with load_netcdfs(
            files_for_cycle.tolist(),
            data_folder / "misc",
            load_wind=with_wind,
        ) as ds:
            ds[STREAMFUNCTION] = (
                compute_streamfunction_with_atmospheric_pressure_xy_avg(
                    ds[SSH],
                    ds[ATMOS_PRESSURE],
                    config.physics.rho,
                    g_prime[0].item(),
                    remove_avgs=True,
                )
            )

            psi0_filt_da, psis_filt_da = filter_streamfunction(
                ds[STREAMFUNCTION],
                params["sigma_ic"],
                params["sigma_bc"],
            )
            ds["psi_filt"] = psis_filt_da

            with logger.timeit("Interpolating dataset"):
                psi0_filt_da = psi_regridder(
                    psi0_filt_da, output_chunks=(-1, -1)
                )
                ds_interp = psi_regridder(
                    ds[
                        [STREAMFUNCTION, "psi_filt"]
                        + [ZONAL_WIND_10M, MERIDIONAL_WIND_10M] * with_wind
                    ],
                    output_chunks=(-1, -1),
                )

        with logger.timeit("Building tensors"):
            psi0 = da_to_tensor(psi0_filt_da, **specs) / f0
            psis_f = da_to_tensor(ds_interp["psi_filt"], **specs) / f0
            # Reference (unfiltered) streamfunction, shape (T, 1, 1, nx, ny)
            psis_ref = da_to_tensor(ds_interp[STREAMFUNCTION], **specs) / f0
            t0 = ds_interp[TIME][0]
            times = (ds_interp[TIME] - t0).dt.total_seconds().to_numpy()
            times = torch.tensor(times, **specs)

        if with_wind:
            from qgsw.utils.reshaping import crop_xr

            uv10 = torch.stack(
                [
                    torch.tensor(
                        crop_xr(ds_interp[ZONAL_WIND_10M], b).to_numpy(),
                        **specs,
                    ),
                    torch.tensor(
                        crop_xr(ds_interp[MERIDIONAL_WIND_10M], b).to_numpy(),
                        **specs,
                    ),
                ],
            )
        del ds_interp
        gc.collect()

        # Same order as training: reset time, then setup, then run.
        model.reset_time()
        setup(
            c,
            output,
            model,
            psi0,
            psis_f,
            times,
            build_q_rg,
        )
        del psis_f
        gc.collect()

        if with_wind:
            uv10_to_uvsurf = output.results.load()[c]["uv10_to_uvsurf"]
            tauxs, tauys = compute_windstress(
                uv10,
                uv10_to_uvsurf,
                rho_water=config.physics.rho,
                rho_air=1.225,
            )

        data_folder = get_path_from_env(key="eNATL60_FOLDER")
        vort_files = list(
            (data_folder / "MEANDERS" / "gridVort").glob("*.nc"),
        )
        vort_files = sort_files_by_dates(*vort_files)

        def format_dvort(ds: xr.Dataset) -> xr.Dataset:
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
                    "vorticity": VORTICITY,
                }
            )
            ds = ds.transpose(TIME, "i", "j")
            return ds.set_coords([LONGITUDE, LATITUDE])

        def load_cycle_vorticity_data(
            files: np.ndarray, c: int
        ) -> torch.Tensor:
            """Load vorticity."""
            start_cycle = c * n_file_per_cycle + c * separation
            end_cycle = (c + 1) * n_file_per_cycle + c * separation

            if end_cycle > len(files):
                msg = f"Not enough files to perform cycle {c} and above."
                logger.warning(msg)
                return None

            files_for_cycle = files[start_cycle:end_cycle]

            with load_datasets(
                *files_for_cycle, format_func=format_dvort
            ) as dvort:
                vorticities_regridded = (
                    torch.tensor(
                        psi_regridder(dvort[VORTICITY]).to_numpy(), **specs
                    )
                    .unsqueeze(1)
                    .unsqueeze(1)
                    / beta_plane.f0
                )

                ds_lon = dvort[LONGITUDE]
                ds_lat = dvort[LATITUDE]

                lon_slice = (
                    ds_lon
                    >= np.rad2deg(
                        lons[(b + 1) : -(b + 1), (b + 1) : -(b + 1)].min()
                    )
                ) & (
                    ds_lon
                    <= np.rad2deg(
                        lons[(b + 1) : -(b + 1), (b + 1) : -(b + 1)].max()
                    )
                )
                lat_slice = (
                    ds_lat
                    >= np.rad2deg(
                        lats[(b + 1) : -(b + 1), (b + 1) : -(b + 1)].min()
                    )
                ) & (
                    ds_lat
                    <= np.rad2deg(
                        lats[(b + 1) : -(b + 1), (b + 1) : -(b + 1)].max()
                    )
                )

                Is, Js = np.meshgrid(
                    dvort.i.values, dvort.j.values, indexing="ij"
                )

                I_slice = Is[lon_slice & lat_slice]
                J_slice = Js[lon_slice & lat_slice]

                imin, imax = I_slice.min(), I_slice.max()
                jmin, jmax = J_slice.min(), J_slice.max()

                dvort_ = dvort.sel(
                    i=slice(imin, imax + 1), j=slice(jmin, jmax + 1)
                )

                vorticities = (
                    torch.tensor(dvort_[VORTICITY].to_numpy(), **specs)
                    .unsqueeze(1)
                    .unsqueeze(1)
                    / beta_plane.f0
                )
            return vorticities, vorticities_regridded

        vorts_ref_raw, vorts_ref = load_cycle_vorticity_data(
            vort_files[in_season], c
        )

        psis_ = [model.psi.clone()]
        for n in range(1, data_n_steps):
            if with_wind:
                model.set_wind_forcing(tauxs[n - 1], tauys[n - 1])
            for _ in range(dt_ratios):
                model.step()
            psis_.append(model.psi.clone())

        filt = GaussianFilter2D(1)

        psis = torch.stack([p[0, 0] for p in psis_], dim=0)
        vorts = laplacian(psis, dx, dy) / f0

        psirs = crop(psis_ref[:, 0, 0], b)
        vortrs = crop(
            torch.stack([filt(v) for v in vorts_ref[:, 0, 0]]), b + 1
        )

        psi_diffs = psirs - psis
        vort_diffs = vortrs - vorts

        psi_psd_errs = psd_err_p(psis, psirs)
        vort_psd_errs = psd_err_v(vorts, vortrs)

        psi_vmax = psirs.abs().max()
        vort_vmax = vortrs.abs().quantile(0.98)

        psi_diff_max = psi_diffs.abs().quantile(0.98)
        vort_diff_max = vort_diffs.abs().quantile(0.98)

        psi_psd_err_max = psi_psd_errs.max().item()
        vort_psd_err_max = vort_psd_errs.max().item()

        fig = plt.figure(figsize=(15, 15), dpi=150)

        gs = fig.add_gridspec(
            4, 3, width_ratios=[1, 1, 1], height_ratios=[3, 2, 3, 2]
        )

        psi_ref = fig.add_subplot(gs[0, 0])
        psi_ref.set_title("Reference stream function")
        psi_diff = fig.add_subplot(gs[0, 1])
        psi_diff.set_title("Reference - Reconstruction")
        psi = fig.add_subplot(gs[0, 2])
        psi.set_title("Reconstructed stream function")
        psi_psd_err = fig.add_subplot(gs[1, :])
        psi_psd_err.set_title(
            "Power spectral density error on stream function"
        )

        vort_ref = fig.add_subplot(gs[2, 0])
        vort_ref.set_title("Reference vorticity")
        vort_diff = fig.add_subplot(gs[2, 1])
        vort_diff.set_title("Reference - Reconstruction")
        vort = fig.add_subplot(gs[2, 2])
        vort.set_title("Reconstructed vorticity")
        vort_psd_err = fig.add_subplot(gs[3, :])
        vort_psd_err.set_title("Power spectral density error on vorticity")

        plots.clamp_ylims(0, psi_psd_err_max, psi_psd_err)
        plots.clamp_ylims(0, vort_psd_err_max, vort_psd_err)

        psi_im_ref = plots.imshow(
            psirs[0], vmax=psi_vmax, vmin=-psi_vmax, ax=psi_ref
        )
        psi_ref.cbar.set_label("[m².s⁻¹]")
        psi_im_diff = plots.imshow(psi_diffs[0], ax=psi_diff)
        psi_diff.cbar.set_label("[m².s⁻¹]")
        psi_im_rec = plots.imshow(
            psis[0], vmax=psi_vmax, vmin=-psi_vmax, ax=psi
        )
        psi.cbar.set_label("[m².s⁻¹]")

        vort_im_ref = plots.imshow(
            vortrs[0], vmax=vort_vmax, vmin=-vort_vmax, ax=vort_ref
        )
        vort_ref.cbar.set_label("Unit of f₀")
        vort_im_diff = plots.imshow(vort_diffs[0], ax=vort_diff)
        vort_diff.cbar.set_label("Unit of f₀")
        vort_im_rec = plots.imshow(
            vorts[0], vmax=vort_vmax, vmin=-vort_vmax, ax=vort
        )
        vort.cbar.set_label("Unit of f₀")

        psi_psd_line = psi_psd_err.semilogx(
            2 * torch.pi / psd_p.kr.cpu().numpy() / 1000,
            psi_psd_errs[0].cpu().numpy(),
            color="black",
        )[0]
        vort_psd_line = vort_psd_err.semilogx(
            2 * torch.pi / psd_v.kr.cpu().numpy() / 1000,
            vort_psd_errs[0].cpu().numpy(),
            color="black",
        )[0]

        ticks = [25, 50, 100, 250, 500]
        psi_psd_err.set_xticks(ticks, minor=False)
        psi_psd_err.set_xticklabels([str(t) for t in ticks])
        psi_psd_err.xaxis.set_minor_formatter(plt.NullFormatter())
        psi_psd_err.grid(axis="x")
        psi_psd_err.xaxis.set_label_text("Wavelength [km]")

        vort_psd_err.set_xticks(ticks, minor=False)
        vort_psd_err.set_xticklabels([str(t) for t in ticks])
        vort_psd_err.xaxis.set_minor_formatter(plt.NullFormatter())
        vort_psd_err.grid(axis="x")
        vort_psd_err.xaxis.set_label_text("Wavelength [km]")

        times = torch.arange(len(psis), **specs) * data_dt

        suptitle = plots.blittable_suptitle(
            (
                f"Season: {params['season']}, "
                f"cycle {step(c + 1, params['cycles'])}"
                f"\nTime: {sec2text(times[0].item())}"
            ),
            fig,
            psi_ref,
            y=0.975,
        )

        fig.tight_layout()

        def update(frame: int) -> tuple[Any]:
            """Update function."""
            suptitle.set_text(
                f"Season: {params['season']}, "
                f"cycle {step(c + 1, params['cycles'])}"
                f"\nTime: {sec2text(times[frame].item())}"
            )

            psi_im_ref.set_array(retrieve_imshow_data(psirs[frame]))
            psi_im_diff.set_array(retrieve_imshow_data(psi_diffs[frame]))
            psi_im_rec.set_array(retrieve_imshow_data(psis[frame]))
            psi_psd_line.set_ydata(psi_psd_errs[frame].cpu().numpy())

            vort_im_ref.set_array(retrieve_imshow_data(vortrs[frame]))
            vort_im_diff.set_array(retrieve_imshow_data(vort_diffs[frame]))
            vort_im_rec.set_array(retrieve_imshow_data(vorts[frame]))
            vort_psd_line.set_ydata(vort_psd_errs[frame].cpu().numpy())

            return (
                suptitle,
                psi_im_ref,
                psi_im_diff,
                psi_im_rec,
                psi_psd_line,
                vort_im_ref,
                vort_im_diff,
                vort_im_rec,
                vort_psd_line,
            )

        videos = output.folder / "videos"

        videos.mkdir(parents=True, exist_ok=True)

        anim = FuncAnimation(fig, update, frames=len(psis), blit=True)
        anim.save(videos / f"cycle_{c + 1}.mp4", fps=15)
