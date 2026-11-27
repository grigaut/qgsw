"""Videos of Power Spectral Density error."""

# ruff: noqa: B023

import gc
import sys
from collections.abc import Callable

import numpy as np
import torch
import xarray as xr
import xgcm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist

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
    ZONAL_WIND_10M,
)
from qgsw.eNATL60.wind import compute_windstress
from qgsw.filters.gaussian import GaussianFilter2D
from qgsw.logging.core import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
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
domain_bounds = {"x": slice(2280, 2884.5), "y": slice(1865, 2688.5)}


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

    def format_decomp(ds: xr.Dataset) -> xr.Dataset:
        """Format Dataset."""
        # Drop useless variables
        ds = ds.drop_vars(["u_res", "v_res"])
        # Rename
        ds = ds.rename(
            {
                "t": TIME,
                "llon_cr": LONGITUDE,
                "llat_cr": LATITUDE,
            }
        )
        ds["i"] = ds["x_r"]
        ds["j"] = ds["y_c"]
        ds = ds.transpose(TIME, "x_r", "y_c")
        return ds.set_coords([LONGITUDE, LATITUDE])

    def load_cycle_decomp_data(
        files: np.ndarray,
    ) -> tuple[
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
    ]:
        """Load decomposition."""
        files_for_cycle = [
            data_folder
            / "decomposition"
            / ("eNATL60-BLB002X-surface-decomposition-2D_" + f.name[-20:])
            for f in files
        ]

        ds = load_datasets(
            *files_for_cycle,
            concat_dim="t",
            format_func=format_decomp,
        )
        psi, phi = ds["psi"], ds["phi"]
        u_rot, v_rot = ds["u_rot"], ds["v_rot"]
        u_div, v_div = ds["u_div"], ds["v_div"]
        return psi, phi, u_rot, v_rot, u_div, v_div

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
        space_interior.dx,
        space_interior.dy,
        beta_plane.f0,
        beta_effect_wide
        * (params["model_type"] != SurfaceVorticity.get_type()),
        b,
        space_2d.nx,
    )

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

        psi, phi, u_rot, v_rot, u_div, v_div = load_cycle_decomp_data(
            files_for_cycle
        )
        lap_psi = dx(dx(psi)) + dy(dy(psi))

        regridder = build_regridder(psi, lons, lats)

        psi_recs_ = [model.psi.clone()]
        for n in range(1, data_n_steps):
            if with_wind:
                model.set_wind_forcing(tauxs[n - 1], tauys[n - 1])
            for _ in range(dt_ratios):
                model.step()
            psi_recs_.append(model.psi.clone())

        psi_refs = regridder(psi)
        lap_psi_refs = regridder(lap_psi)

        f = GaussianFilter2D(1)

        psi_refs = crop(
            crop(torch.tensor(psi_refs.to_numpy(), **specs), b), 15
        )
        psi_refs = psi_refs - psi_refs.mean(dim=[-2, -1], keepdim=True)
        psi_recs = crop(torch.stack([p[0, 0] for p in psi_recs_], dim=0), 15)
        psi_recs = psi_recs - psi_recs.mean(dim=[-2, -1], keepdim=True)
        psi_diffs = psi_refs - psi_recs

        psi_refs_max = psi_refs.abs().max()
        psi_diffs_max = psi_diffs.abs().quantile(0.98)

        lap_psi_refs = crop(
            crop(torch.tensor(lap_psi_refs.to_numpy(), **specs), b), 5
        )
        lap_psi_refs = torch.stack([f(p) for p in lap_psi_refs], dim=0)
        lap_psi_recs = crop(
            torch.stack(
                [
                    (laplacian(p[0, 0], space_interior.dx, space_interior.dy))
                    for p in psi_recs_
                ],
                dim=0,
            ),
            5 - 1,
        )
        lap_psi_recs = lap_psi_recs - lap_psi_recs.mean(
            dim=[-2, -1], keepdim=True
        )
        lap_psi_diffs = lap_psi_refs - lap_psi_recs

        lap_psi_refs_max = lap_psi_refs.abs().quantile(0.98)
        lap_psi_diffs_max = lap_psi_diffs.abs().quantile(0.98)

        fig, axs = plots.subplots(2, 3)

        plots.set_coltitles(
            ["Helmholtz decompostion", "Difference", "Reconstruction"], axs
        )
        plots.set_rowtitles(["ѱ", "Δѱ"], axs)

        im_psi_ref = plots.imshow(
            psi_recs[0], ax=axs[0, 0], vmin=-psi_refs_max, vmax=psi_refs_max
        )
        im_psi_diff = plots.imshow(
            psi_diffs[0], ax=axs[0, 1], vmin=-psi_diffs_max, vmax=psi_diffs_max
        )
        im_psi_rec = plots.imshow(
            psi_recs[0], ax=axs[0, 2], vmin=-psi_refs_max, vmax=psi_refs_max
        )

        im_lap_psi_ref = plots.imshow(
            lap_psi_recs[0],
            ax=axs[1, 0],
            vmin=-lap_psi_refs_max,
            vmax=lap_psi_refs_max,
        )
        im_lap_psi_diff = plots.imshow(
            lap_psi_diffs[0],
            ax=axs[1, 1],
            vmin=-lap_psi_diffs_max,
            vmax=lap_psi_diffs_max,
        )
        im_lap_psi_rec = plots.imshow(
            lap_psi_recs[0],
            ax=axs[1, 2],
            vmin=-lap_psi_refs_max,
            vmax=lap_psi_refs_max,
        )

        times = torch.arange(len(psi_refs), **specs) * data_dt

        suptitle = plots.blittable_suptitle(
            (
                f"Season: {params['season']}, "
                f"cycle {step(c + 1, params['cycles'])}"
                f"\nTime: {sec2text(times[0].item())}"
            ),
            fig,
            axs[0, 1],
            y=0.975,
        )

        plt.tight_layout()

        def update(frame: int) -> tuple[Artist, ...]:
            """Update."""
            suptitle.set_text(
                f"Season: {params['season']}, "
                f"cycle {step(c + 1, params['cycles'])}"
                f"\nTime: {sec2text(times[frame].item())}"
            )
            im_psi_ref.set_array(retrieve_imshow_data(psi_refs[frame]))
            im_psi_diff.set_array(retrieve_imshow_data(psi_diffs[frame]))
            im_psi_rec.set_array(retrieve_imshow_data(psi_recs[frame]))
            im_lap_psi_ref.set_array(retrieve_imshow_data(lap_psi_refs[frame]))
            im_lap_psi_diff.set_array(
                retrieve_imshow_data(lap_psi_diffs[frame])
            )
            im_lap_psi_rec.set_array(retrieve_imshow_data(lap_psi_recs[frame]))
            return (
                suptitle,
                im_psi_ref,
                im_psi_diff,
                im_psi_rec,
                im_lap_psi_ref,
                im_lap_psi_diff,
                im_lap_psi_rec,
            )

        videos = output.folder / "videos"
        anim = FuncAnimation(fig, update, frames=len(psi_recs))
        anim.save(videos / f"decomposition_cycle_{c + 1}.mp4", fps=15)

        fig, axs = plots.subplots(2, 2)

        plots.set_coltitles(
            ["Rotational component", "Divergent component"], axs
        )
        plots.set_rowtitles(["Zonal velocity", "Meridional velocity"], axs)

        u_rot = crop(u_rot.to_numpy(), 25)
        v_rot = crop(v_rot.to_numpy(), 25)
        u_div = crop(u_div.to_numpy(), 25)
        v_div = crop(v_div.to_numpy(), 25)

        u_rot_max = np.nanmax(np.abs(u_rot)).item()
        u_div_max = np.nanmax(np.abs(u_div)).item()
        v_rot_max = np.nanmax(np.abs(v_rot)).item()
        v_div_max = np.nanmax(np.abs(v_div)).item()

        im_u_rot = plots.imshow(
            u_rot[0], ax=axs[0, 0], vmin=-u_rot_max, vmax=u_rot_max
        )
        im_u_div = plots.imshow(
            u_div[0], ax=axs[0, 1], vmin=-u_div_max, vmax=u_div_max
        )
        im_v_rot = plots.imshow(
            v_rot[0], ax=axs[1, 0], vmin=-v_rot_max, vmax=v_rot_max
        )
        im_v_div = plots.imshow(
            v_div[0], ax=axs[1, 1], vmin=-v_div_max, vmax=v_div_max
        )

        times = torch.arange(len(u_rot), **specs) * data_dt

        suptitle = plots.blittable_suptitle(
            (
                f"Season: {params['season']}, "
                f"cycle {step(c + 1, params['cycles'])}"
                f"\nTime: {sec2text(times[0].item())}"
            ),
            fig,
            axs[0, 1],
            y=0.975,
        )

        plt.tight_layout()

        def update(frame: int) -> tuple[Artist, ...]:
            """Update."""
            suptitle.set_text(
                f"Season: {params['season']}, "
                f"cycle {step(c + 1, params['cycles'])}"
                f"\nTime: {sec2text(times[frame].item())}"
            )
            im_u_rot.set_array(retrieve_imshow_data(u_rot[frame]))
            im_u_div.set_array(retrieve_imshow_data(u_div[frame]))
            im_v_rot.set_array(retrieve_imshow_data(v_rot[frame]))
            im_v_div.set_array(retrieve_imshow_data(v_div[frame]))
            return (
                suptitle,
                im_u_rot,
                im_u_div,
                im_v_rot,
                im_v_div,
            )

        videos = output.folder / "videos"
        anim = FuncAnimation(fig, update, frames=len(u_rot))
        anim.save(videos / f"decomposition_uv_cycle_{c + 1}.mp4", fps=15)
