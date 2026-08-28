"""Variational analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
import xarray as xr

from qgsw.cli import ScriptsArgsParser
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.exp_exp.param_generator import gaussian_exp_field
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
    SST,
    STREAMFUNCTION,
    TIME,
    ZONAL_WIND_10M,
)
from qgsw.eNATL60.wind import compute_windstress
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQCore
from qgsw.models.qg.psiq.mixed_layer.forced import QGPSIQSSTRGSI
from qgsw.observations import FullDomainMask, SatelliteTrackMask
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.physics.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.pv import (
    compute_q1_interior,
)
from qgsw.scripts.boundaries import (
    extract_psi_bc,
    extract_q_bc,
    extract_sst_bc,
)
from qgsw.scripts.eNATL60 import (
    da_to_tensor,
    filter_streamfunction,
    format_ds,
    load_netcdfs,
)
from qgsw.scripts.loss import update_loss
from qgsw.scripts.regularization import compute_regularization_func
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
)
from qgsw.spatial.core.grid import Grid2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils.interpolation import QuadraticInterpolation
from qgsw.utils.reshaping import crop
from qgsw.utils.storage import get_path_from_env

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)


### Boundaries offset

b = 4

if __name__ == "__main__":
    ## Config

    args = ScriptsArgsParser.va_setup(
        prefix_default="results_enatl60_atmp",
        cycles_default=4,
    )
    args.add_regularization(gamma_default=0.1)
    args.add_alpha()
    args.add_season(default="summer")
    args.retrieve()
    with_reg = not args.no_reg
    with_alpha = not args.no_alpha
    with_obs_track = args.obs_track
    with_wind = not args.no_wind

    specs = defaults.get()

    setup_root_logger(args.verbose)
    logger = getLogger(__name__)

    ROOT_PATH = Path(__file__).parent.parent
    config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))

    output_dir = config.io.output.directory

    # Simulation parameters

    dt = 3600
    optim_max_step = args.optim
    n_file_per_cycle = 20
    n_steps_per_cyle = 240 * 2 - 1
    comparison_interval = args.comparison
    n_cycles = args.cycles

    separation = int(args.separation * dt / 3600 / 24)

    sigma_bc = 10
    sigma_ic = 10

    ## Load eNATL60 grid

    ### Data folder

    data_folder = get_path_from_env(key="eNATL60_FOLDER")
    files = list((data_folder / "MEANDERS" / "gridT").glob("*.nc"))

    files = sort_files_by_dates(*files)

    season = {
        "summer": seasons.SUMMER,
        "autumn": seasons.AUTUMN,
        "winter": seasons.WINTER,
        "spring": seasons.SPRING,
    }

    in_season = retrieve_dates(*files.tolist()).month.isin(season[args.season])
    if ((in_season[1:]) & (~in_season[:-1])).sum() + int(in_season[0]) > 1:
        msg = "Non-time-contiguous data for this season in provided dataset."
        raise ValueError(msg)
    files = files[in_season]

    ### Load only one file to access grid informations

    ds = load_datasets(files[0], format_func=format_ds)

    ### Compute longitude / latitudes
    dx = dy = 10000
    lons, lats = compute_lonlat_from_regular_xy_grid(
        ds[LONGITUDE],
        ds[LATITUDE],
        dx=dx,
        dy=dy,
    )
    xs, ys = lonlat_to_xy(lons, lats)

    ### Compute β-plane parameters

    lat0 = (lats.max() + lats.min()) / 2
    beta_plane = BetaPlane(
        f0=2 * EARTH_ANGULAR_ROTATION * np.sin(lat0),
        beta=2 * EARTH_ANGULAR_ROTATION * np.cos(lat0) / EARTH_RADIUS,
    )
    f0 = beta_plane.f0

    ### Build regridder

    psi_regridder = build_regridder(ds, lons, lats)
    sst_regridder = build_regridder(ds, interpolate(lons), interpolate(lats))

    ## Areas
    nx, ny = lats.shape
    xx = torch.tensor(xs.round(), **specs)
    space_2d = SpaceDiscretization2D.from_psi_grid(
        Grid2D(
            x=xx - xx[0, :],
            y=torch.tensor(ys.round(), **specs),
        )
    )

    space_interior = space_2d.slice(
        b,
        space_2d.psi.xy.x.shape[0] - b,
        b,
        space_2d.psi.xy.x.shape[1] - b,
    )

    nx = space_interior.nx
    ny = space_interior.ny
    dx = space_interior.dx
    dy = space_interior.dy

    ## Observations

    if with_obs_track:
        obs_mask = SatelliteTrackMask(
            space_interior.psi.xy.x,
            space_interior.psi.xy.y,
            track_width=100000,
            track_interval=600000,
            theta=torch.pi / 12,
            full_coverage_time=20 * 3600 * 24,
        )
        if comparison_interval != 1:
            msg = (
                "Using Satellite track, comparison interval "
                "inferred from tracks trajectory."
            )
            logger.warning(box(msg, style="="))
        n_obs = obs_mask.compute_obs_nb(240, 7200)
        msg_obs = (
            "Surface observed along satellite tracks,"
            f" {n_obs} pixels observed."
        )
    else:
        obs_mask = FullDomainMask(
            space_interior.psi.xy.x,
            space_interior.psi.xy.y,
            dt=comparison_interval * dt,
        )
        msg_obs = (
            f"Full surface observed every {sec2text(comparison_interval * dt)}"
        )

    ## Regularization

    gamma = args.gamma / comparison_interval

    if with_reg:
        msg_reg = f"Using ɣ = {gamma:#8.3g} to weight regularization"  # noqa: RUF001
        if gamma != args.gamma:
            msg_reg += (
                f" (rescaled from ɣ = {args.gamma:#5.3g} to"  # noqa: RUF001
                " account for observations sparsity)."
            )
        else:
            msg_reg += "."
    else:
        msg_reg = "No regularization."

    ## Output
    prefix = args.complete_prefix()
    filename = f"{prefix}.pt"
    output_file = output_dir.joinpath(filename)

    ## Logs

    msg_simu = (
        f"Performing {n_cycles} cycles of {n_steps_per_cyle} "
        f"steps with up to {optim_max_step} optimization steps."
    )
    if args.separation != 0:
        msg_simu += (
            f"\nCycles are separated by {sec2text(separation * 24 * 3600)}."
        )
    msg_season = f"Season: {args.season}."
    msg_sf = "Reconstructing ψ using atmospheric pressure and ssh."
    lon_min = np.rad2deg(lons.min())
    lon_max = np.rad2deg(lons.max())
    lat_min = np.rad2deg(lats.min())
    lat_max = np.rad2deg(lats.max())
    msg_area = (
        f"Longitudes in [{lon_min:#.3g}°, {lon_max:#.3g}°],"
        f" latitudes in [{lat_min:#.3g}°, {lat_max:#.3g}°]."
    )
    if with_wind:
        msg_wind = "Using wind from ERA interim DFS5."
    else:
        msg_wind = "No wind considered."
    msg_output = f"Output will be saved to {output_file}."

    logger.info(
        box(
            msg_simu,
            msg_season,
            msg_sf,
            msg_area,
            msg_wind,
            msg_obs,
            msg_reg,
            msg_output,
            style="=",
        )
    )

    # Parameters

    H = config.model.h
    g_prime = config.model.g_prime
    bottom_drag_coef = config.physics.bottom_drag_coefficient
    slip_coef = config.physics.slip_coef

    # Model

    M = TypeVar("M", bound=QGPSIQCore)

    def set_inhomogeneous_model(model: M) -> M:
        """Set up inhomogeneous model."""
        model.masks = Masks.empty_tensor(nx, ny, device=specs["device"])
        model.bottom_drag_coef = 0
        model.wide = True
        model.slip_coef = slip_coef
        model.dt = dt
        return model

    model = QGPSIQSSTRGSI(
        space_2d=space_interior,
        H=H[:2],
        beta_plane=beta_plane,
        g_prime=g_prime[:2],
    )
    model: QGPSIQSSTRGSI = set_inhomogeneous_model(model)
    model.H_ml = 10
    model.temp_1_offset = 4

    y_w = space_2d.q.xy.y[0, :].unsqueeze(0)
    beta_effect = beta_plane.beta * (y_w - model.y0)

    build_compute_q_rg = lambda A11, A12: (
        lambda psi1: compute_q1_interior(
            psi1,
            torch.zeros_like(psi1),
            A11,
            A12,
            dx,
            dy,
            beta_plane.f0,
            beta_effect[:, 1:-1],
        )
    )

    outputs = []

    L: float = dx.item()

    for c in range(n_cycles):
        torch.cuda.reset_peak_memory_stats()

        start_cycle = c * n_file_per_cycle + c * separation
        end_cycle = (c + 1) * n_file_per_cycle + c * separation

        if end_cycle > len(files):
            msg = f"Not enough files to perform cycle {c} and above."
            logger.warning(msg)
            break

        files_for_cycle = files[start_cycle:end_cycle]

        with load_netcdfs(
            files_for_cycle.tolist(), data_folder / "misc", load_wind=with_wind
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
                sigma_ic,
                sigma_bc,
            )
            ds["psi_filt"] = psis_filt_da

            with logger.timeit("Interpolating dataset"):
                psi0_filt_da: xr.DataArray = psi_regridder(
                    psi0_filt_da, output_chunks=(-1, -1)
                )
                ds_interp: xr.Dataset = psi_regridder(
                    ds[
                        [STREAMFUNCTION, "psi_filt"]
                        + [ZONAL_WIND_10M, MERIDIONAL_WIND_10M] * with_wind
                    ],
                    output_chunks=(-1, -1),
                )
                ds_interp[LONGITUDE] = (["i", "j"], lons)
                ds_interp[LATITUDE] = (["i", "j"], lats)

            with logger.timeit("Interpolating SST"):
                regridded_sst: xr.DataArray = sst_regridder(
                    ds[SST], output_chunks=(-1, -1)
                )
                ds_sst_interp = xr.Dataset(
                    {
                        LONGITUDE: (["i", "j"], interpolate(lons)),
                        LATITUDE: (["i", "j"], interpolate(lats)),
                        SST: regridded_sst,
                    },
                    regridded_sst.coords,
                )
                ds_sst_interp = ds_sst_interp.set_coords([LONGITUDE, LATITUDE])

        with logger.timeit("Building tensors"):
            psi0 = da_to_tensor(psi0_filt_da, **specs) / f0
            psis = da_to_tensor(ds_interp[STREAMFUNCTION], **specs) / f0
            psis_f = da_to_tensor(ds_interp["psi_filt"], **specs) / f0
            ssts = da_to_tensor(ds_sst_interp[SST], **specs) + 273.15
            t0 = ds_interp[TIME][0]
            times = (ds_interp[TIME] - t0).dt.total_seconds().to_numpy()
            times = torch.tensor(times, **specs)
        with logger.timeit("Retrieving boundaries"):
            psi_bcs = [extract_psi_bc(p, b) for p in psis_f]
            sst_bcs = [extract_sst_bc(s, b) for s in ssts]

        if with_wind:
            u10 = ds_interp[ZONAL_WIND_10M].to_numpy()
            v10 = ds_interp[MERIDIONAL_WIND_10M].to_numpy()
            uv10 = torch.stack(
                [
                    crop(torch.tensor(u10, **specs), b),
                    crop(torch.tensor(v10, **specs), b),
                ],
            )
        psi0_mean = psi0.mean()
        var_ref = crop(psis[:, 0, 0], b).var()
        U: float = psi0_mean / L
        T = L / U

        s = step(c + 1, n_cycles)
        msg = f"Cycle {s}: eNATL60 data loaded and processed."
        logger.info(box(msg, style="round"))

        psi_bc_interp = QuadraticInterpolation(times, psi_bcs)
        sst_bc_interp = QuadraticInterpolation(times, sst_bcs)

        xx = space_interior.psi.xy.x
        yy = space_interior.psi.xy.y

        space_params, time_params = gaussian_exp_field(
            0, 3, xx, yy, 240 * 7200, 240 / 6 * 7200
        )
        basis = GaussianExpBasis(space_params, time_params)
        coefs = DecompositionCoefs.zeros_like(basis.generate_random_coefs())
        coefs = coefs.requires_grad_()

        if with_alpha:
            kappa = torch.tensor(0, **specs, requires_grad=True)
            numel = kappa.numel() + coefs.numel()
            params = [
                {"params": [kappa], "lr": 1e-2, "name": "κ"},
                {
                    "params": list(coefs.values()),
                    "lr": 1e0,
                    "name": "Decomposition coefs",
                },
            ]
        else:
            kappa = torch.tensor(0, **specs)
            numel = coefs.numel()
            params = [
                {
                    "params": list(coefs.values()),
                    "lr": 1e0,
                    "name": "Decomposition coefs",
                },
            ]
        uv10_to_uvsurf = torch.eye(2, **specs, requires_grad=False)

        msg = f"Control vector contains {numel} elements."
        logger.info(box(msg, style="round"))

        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        lr_callback = LRChangeCallback(optimizer)
        early_stop = EarlyStop()

        coefs_scaled = coefs.scale(
            *(
                1e-1 * psi0_mean / (n_steps_per_cyle * dt) ** k
                for k in range(basis.order)
            )
        )
        epsilon = 0.1
        register_params = RegisterParams(
            alpha=torch.exp(epsilon * kappa + kappa * kappa.abs()) - 1,
            coefs=coefs_scaled.to_dict(),
            uv10_to_uvsurf=uv10_to_uvsurf,
        )

        for o in range(optim_max_step):
            optimizer.zero_grad()
            model.reset_time()

            with torch.enable_grad():
                if with_wind:
                    tauxs_i, tauys_i = compute_windstress(
                        uv10,
                        uv10_to_uvsurf,
                        rho_water=config.physics.rho,
                        rho_air=1.225,
                    )

                alpha = torch.exp(epsilon * kappa + kappa * kappa.abs()) - 1
                coefs_scaled = coefs.scale(
                    *(
                        1e-1 * psi0_mean / (n_steps_per_cyle * dt) ** k
                        for k in range(basis.order)
                    )
                )

                basis.set_coefs(coefs_scaled)

                model.basis = basis
                model.alpha = alpha

                compute_reg = compute_regularization_func(
                    basis,
                    H,
                    g_prime,
                    alpha,
                    space_interior,
                    beta_plane,
                    scale=1 / T**2,
                )

                compute_q_rg = build_compute_q_rg(
                    model.A[:1, :1],
                    model.A[:1, 1:2],
                )
                q0 = crop(compute_q_rg(psi0), b - 1)

                qs = (compute_q_rg(p1) for p1 in psis_f)

                q_bcs = [extract_q_bc(q, b) for q in qs]
                q_bc_interp = QuadraticInterpolation(times, q_bcs)

                model.set_psiqsst(crop(psi0[:, :1], b), q0, crop(ssts[0], b))
                model.set_boundary_maps(
                    psi_bc_interp, q_bc_interp, sst_bc_interp
                )

                loss = torch.tensor(0, **specs)

                loss = update_loss(
                    loss,
                    model.psi[0, 0],
                    crop(psis[0][0, 0], b),
                    mask=obs_mask.at_time(model.time),
                    variance=var_ref,
                )

                for n in range(1, n_steps_per_cyle):
                    psi1_ = model.psi
                    time = model.time.clone()

                    if n % 2 == 1 and with_wind:
                        model.set_wind_forcing(
                            tauxs_i[(n - 1) // 2], tauys_i[(n - 1) // 2]
                        )

                    model.step()
                    if torch.isnan(model.psi).any():
                        msg = f"NaN in stream function at {o=} in step {n=}"
                        raise OverflowError(msg)
                    psi1 = model.psi

                    if with_reg:
                        dpsi1_ = (psi1 - psi1_) / dt
                        reg = gamma * (compute_reg(psi1_, dpsi1_, time))
                        loss += reg
                    if n % 2 == 0:
                        loss = update_loss(
                            loss,
                            psi1[0, 0],
                            crop(psis[n // 2][0, 0], b),
                            mask=obs_mask.at_time(model.time),
                            variance=var_ref,
                        )
                    if n % 20 == 0:
                        loss = update_loss(
                            loss,
                            model.sst[0, 0],
                            crop(ssts[n // 2], b),
                            variance=crop(ssts[n // 2], b).square().sum()
                            / 10000,
                        )

            if torch.isnan(loss.detach()):
                msg = "Loss has diverged."
                logger.warning(box(msg, style="="))
                break

            if torch.isnan(model.psi).any():
                msg = "Streamfunction has diverged."
                logger.warning(box(msg, style="="))
                break

            register_params.step(
                loss,
                alpha=alpha,
                coefs=coefs_scaled.to_dict(),
                uv10_to_uvsurf=uv10_to_uvsurf,
            )

            if early_stop.step(loss):
                msg = f"Convergence reached after {o + 1} iterations."
                logger.info(msg)
                break

            loss_ = loss.cpu().item()

            msg = (
                f"Cycle {step(c + 1, n_cycles)} | "
                f"Optimization step {step(o + 1, optim_max_step)} | "
                f"Loss: {loss_:>#10.5g} | "
                f"Best loss: {register_params.best_loss:>#10.5g}"
            )
            logger.info(msg)

            loss.backward()

            if with_alpha:
                torch.nn.utils.clip_grad_value_([kappa], clip_value=1.0)

            torch.nn.utils.clip_grad_norm_(list(coefs.values()), max_norm=1e0)

            optimizer.step()
            scheduler.step(loss)
            lr_callback.step()

        best_loss = register_params.best_loss
        msg = f"Optimization completed with loss: {best_loss:>#10.5g}"
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
        logger.info(box(msg, msg_mem, style="round"))
        output = {
            "cycle": c,
            "config": {
                "comparison_interval": comparison_interval,
                "no-wind": args.no_wind,
                "obstrack": args.obs_track,
                "gamma": args.gamma if with_reg else 0,
                "basis": basis.get_params(),
                "numel": numel,
                "sigma_bc": sigma_bc,
                "sigma_ic": sigma_ic,
                "dt": dt,
                "separation_steps": args.separation,
                "season": args.season,
            },
            "optim": {
                "max_steps": optim_max_step,
                "nb_steps": o + 1,
                "loss": best_loss,
            },
            "specs": {"max_memory_allocated": max_mem},
            "alpha": register_params.params["alpha"],
            "coefs": register_params.params["coefs"],
            "uv10_to_uvsurf": register_params.params["uv10_to_uvsurf"],
        }
        outputs.append(output)

        torch.save(outputs, output_file)
        msg = f"Outputs saved to {output_file}"
        logger.info(box(msg, style="="))
