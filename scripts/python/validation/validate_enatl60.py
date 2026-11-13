"""Validate eNATL60 results."""

import gc
import sys
from collections.abc import Callable

import numpy as np
import torch

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
    get_tracking_filename,
    retrieve_output_folder,
)
from qgsw.scripts.pv import build_pv_funcs
from qgsw.scripts.setups.model import setup_model
from qgsw.scripts.validation import Losses
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import grad_perp
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

    outputs = []

    for c in range(n_cycles):
        start_cycle = c * n_file_per_cycle + c * separation
        end_cycle = (c + 1) * n_file_per_cycle + c * separation

        msg = f"Computing loss for cycle {step(c + 1, n_cycles)}"
        logger.info(box(msg))
        losses = Losses(space_interior)
        losses.with_uv()
        losses.cycle = c
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
            psis_ref = (
                da_to_tensor(ds_interp[STREAMFUNCTION], **specs)[:, None, None]
                / f0
            )
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
        losses.no_data = setup(
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
            uv10_to_uvsurf = output.optim.load()[c]["uv10_to_uvsurf"]
            tauxs, tauys = compute_windstress(
                uv10,
                uv10_to_uvsurf,
                rho_water=config.physics.rho,
                rho_air=1.225,
            )

        u_refs, v_refs = grad_perp(psis_ref)
        u_refs /= dy
        v_refs /= dx

        losses.compute_loss(model, crop(psis_ref[0], b))
        losses.compute_uv_losses(model, crop(u_refs[0], b), crop(v_refs[0], b))
        for n in range(1, data_n_steps):
            if with_wind:
                model.set_wind_forcing(tauxs[n - 1], tauys[n - 1])
            for _ in range(dt_ratios):
                model.step()
            losses.compute_loss(model, crop(psis_ref[n], b))
            losses.compute_uv_losses(
                model, crop(u_refs[n], b), crop(v_refs[n], b)
            )

        del psis_ref, u_refs, v_refs
        torch.cuda.empty_cache()
        gc.collect()

        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        msg_mem = (
            f"Cycle {step(c + 1, n_cycles)} | "
            f"Max memory allocated: {max_mem:.1f} MB."
        )
        logger.info(box(msg_mem, style="="))

        outputs.append(losses.losses)
        output.scores.save(outputs)

    if run_id is not None:
        tracking_file = get_tracking_filename(run_id)
        tracking_file.unlink(missing_ok=True)
        msg = "Tracking file removed."
        logger.info(box(msg))
