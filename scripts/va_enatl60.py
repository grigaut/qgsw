"""Variational analysis."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from qgsw.cli import ScriptsArgsParser, parse_presets
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
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
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.observations import FullDomainMask, SatelliteTrackMask
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.physics.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.scripts.boundaries import (
    extract_psi_bc,
    extract_wide_psi_bc,
)
from qgsw.scripts.eNATL60 import (
    create_masks,
    da_to_tensor,
    filter_streamfunction,
    format_ds,
    load_netcdfs,
    retrieve_masked,
)
from qgsw.scripts.loss import mse
from qgsw.scripts.output import build_output_folder, create_output_folder
from qgsw.scripts.parameterizations import kappa_to_alpha
from qgsw.scripts.pv import build_pv_funcs
from qgsw.scripts.setups.bases import (
    retrieve_bases_elements,
    setup_basis,
)
from qgsw.scripts.setups.callbacks import (
    pre_cycle_bases_setup,
)
from qgsw.scripts.setups.model import setup_model
from qgsw.scripts.setups.optim import optim_setup
from qgsw.scripts.setups.regularization import (
    inline_reg_setup,
    post_cycle_reg_setup,
)
from qgsw.scripts.setups.scaling import retrieve_scaling_func
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
)
from qgsw.spatial.core.grid import Grid2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils.interpolation import QuadraticInterpolation
from qgsw.utils.reshaping import crop, crop_xr
from qgsw.utils.storage import get_path_from_env

if TYPE_CHECKING:
    import xarray as xr

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)
b = 4

if __name__ == "__main__":
    ## Config

    presets = parse_presets()

    args = ScriptsArgsParser.va_setup(
        prefix_default="eNATL60",
        cycles_default=4,
    )
    args.add_dtdxdy(7200, 10000, 10000)
    args.add_regularization(gamma_dyn_default=1, gamma_bkg_default=1)
    args.add_season(default="summer")
    args.add_reg_exp(default=2)
    args.add_wind_optim(1e-2)
    args.add_alpha(1e-2)
    args.add_sigma_icbc(16, 16)
    args.set_presets(presets)
    args.retrieve()

    with_alpha = not args.no_alpha
    with_reg = not args.no_reg
    with_obs_track = args.obs_track
    with_wind = not args.no_wind

    specs = defaults.get()

    setup_root_logger(args.verbose)
    logger = getLogger(__name__)

    bases_types = retrieve_bases_elements(
        args.model_type,
        args.basis_dyn,
        args.basis_ext,
    )
    bases_lrs = retrieve_bases_elements(
        args.model_type,
        args.basis_dyn_lr,
        args.basis_ext_lr,
    )

    ROOT_PATH = Path(__file__).parent.parent
    config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))

    output_folder = create_output_folder(
        config.io.output.directory, prefix=args.prefix + "_" + args.model_type
    )

    output = build_output_folder(output_folder)

    # Simulation parameters
    dt = args.dt
    data_dt = 7200
    if data_dt % dt != 0:
        msg = f"Timestep must divide data timestep, hence {data_dt}."
        raise ValueError(msg)

    data_n_steps = 240
    dt_ratios = data_dt // dt
    n_file_per_cycle = 20
    n_steps_per_cyle = data_n_steps * dt_ratios - 1

    separation = int(args.separation * dt / 3600 / 24)

    ## Load eNATL60 grid

    ### Data folder

    data_folder = get_path_from_env(key="eNATL60_FOLDER")
    basis_nc_folder = get_path_from_env(key="BASIS_NC_FOLDER")
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
    lons, lats = compute_lonlat_from_regular_xy_grid(
        ds[LONGITUDE],
        ds[LATITUDE],
        dx=args.dx,
        dy=args.dy,
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
        if args.comparison != 1:
            msg = (
                "Using Satellite track, comparison interval "
                "inferred from tracks trajectory."
            )
            logger.warning(box(msg, style="="))
        n_obs = obs_mask.compute_obs_nb(n_steps_per_cyle, dt)
        msg_obs = (
            "Surface observed along satellite tracks,"
            f" {n_obs} pixels observed."
        )
    else:
        obs_mask = FullDomainMask(
            space_interior.psi.xy.x,
            space_interior.psi.xy.y,
            dt=args.comparison * dt,
        )
        msg_obs = (
            f"Full surface observed every {sec2text(args.comparison * dt)}"
        )

    ## Regularization

    gamma_dyn = args.gamma_reg_dyn
    gamma_bkg = args.gamma_reg_bkg

    if with_reg:
        msg_reg = (
            f"[Dynamical regularization] ɣ_dyn = {gamma_dyn:#8.3g}"  # noqa: RUF001
            f"\n[Background regularization] ɣ_reg = {gamma_bkg:#8.3g}"  # noqa: RUF001
        )
    else:
        msg_reg = "No regularization."

    ## Logs

    msg_simu = (
        f"Performing {args.cycles} cycles of {n_steps_per_cyle} "
        f"steps with up to {args.optim} optimization steps."
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
        f"\nInterpolated on a {args.dx} m x{args.dy} m cartesian grid."
    )
    if with_wind:
        msg_wind = "Using wind from ERA interim DFS5."
        if args.wind_optim:
            msg_wind += "\nOptimizing wind surface conversion."
    else:
        msg_wind = "No wind considered."
    msg_output = f"Output will be saved to {output.folder}."
    output.params.save(args.namespace.__dict__)

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
    H1, H2 = H[0], H[1]
    g1, g2 = g_prime[0], g_prime[1]

    # Model
    model = setup_model(
        args.model_type,
        space_interior,
        H,
        g_prime,
        beta_plane,
        dt,
        config.physics.slip_coef,
    )
    # PV computation

    y_w = space_2d.q.xy.y[0, :].unsqueeze(0)
    beta_effect = beta_plane.beta * (y_w - model.y0)
    build_q_rg = build_pv_funcs(
        dx, dy, beta_plane.f0, beta_effect, b, space_2d.nx
    )

    outputs = {"results": [], "optim": [], "basis": []}

    L: float = dx.item()

    for c in range(args.cycles):
        losses = {
            "obs_losses": [],
            "dynamical_reg": [],
            "background_reg": [],
        }

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
                args.sigma_ic,
                args.sigma_bc,
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
                ds_interp["mask"] = (
                    ("time", "i", "j"),
                    create_masks(
                        obs_mask,
                        data_n_steps,
                        data_dt,
                        pad_width=b,
                    ),
                )

        with logger.timeit("Building tensors"):
            psi0 = da_to_tensor(psi0_filt_da, **specs) / f0
            ms = [
                obs_mask.at_time(torch.tensor(i * data_dt)).ravel()
                for i in range(data_n_steps)
            ]
            ts = (
                t
                for t in retrieve_masked(
                    ds_interp[STREAMFUNCTION],
                    ds_interp["mask"],
                    b,
                )
            )
            tm = [
                (t.ravel()[m] / f0, m) if m.sum() != 0 else (None, None)
                for m, t in zip(ms, ts)
            ]
            psis, masks_psi = zip(*tm)
            psis_f = da_to_tensor(ds_interp["psi_filt"], **specs) / f0
            t0 = ds_interp[TIME][0]
            times = (ds_interp[TIME] - t0).dt.total_seconds().to_numpy()
            times = torch.tensor(times, **specs)
        with logger.timeit("Retrieving boundaries"):
            psi_bcs_w = [extract_wide_psi_bc(e, b, clone=True) for e in psis_f]
            psi_bcs = [extract_psi_bc(e, b, clone=True) for e in psis_f]

        del psis_f
        gc.collect()
        var_psi = (
            crop_xr(ds_interp[STREAMFUNCTION], b).var().values.item()
        ) / f0**2

        if with_wind:
            U = ZONAL_WIND_10M
            V = MERIDIONAL_WIND_10M
            uv10 = torch.stack(
                [
                    torch.tensor(crop_xr(ds_interp[U], b).to_numpy(), **specs),
                    torch.tensor(crop_xr(ds_interp[V], b).to_numpy(), **specs),
                ],
            )
        del ds_interp
        gc.collect()
        psi0_mean = psi0.mean()
        U: float = psi0_mean / L
        T = L / U

        s = step(c + 1, args.cycles)
        msg = f"Cycle {s}: eNATL60 data loaded and processed."
        logger.info(box(msg, style="round"))

        psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

        bases, Qs = setup_basis(
            bases_types,
            (
                interpolate(crop(np.rad2deg(lons), b)),
                interpolate(crop(np.rad2deg(lats), b)),
            ),
            space_interior,
            dt,
            n_steps_per_cyle,
            args.reg_exp,
            data_folder / "aux_reduced_basis_MIOST.nc",
            data_folder / "aux_first_baroclinic_speed.nc",
        )

        orders = {k: basis.order for k, basis in bases.items()}

        msg = f"Using bases of order {orders}"
        logger.info(msg)

        coefs = {
            k: DecompositionCoefs.zeros_like(basis.generate_random_coefs())
            for k, basis in bases.items()
        }

        params, coefs, kappa, uv10_to_uvsurf, numel = optim_setup(
            bases_lrs,
            coefs,
            alpha_optim=model.implements_alpha and with_alpha,
            wind_optim=with_wind and args.wind_optim,
            alpha_lr=args.alpha_lr,
            wind_lr=args.wind_lr,
        )

        optimizer = torch.optim.Adam(params)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        lr_callback = LRChangeCallback(optimizer)
        early_stop = EarlyStop()

        scaling_fn = retrieve_scaling_func(
            {k: basis.type for k, basis in bases.items()},
            model.get_type(),
            L,
            U,
            psi0_mean,
            n_steps_per_cyle * dt,
        )

        alpha = kappa_to_alpha(kappa)
        register_params = RegisterParams(
            **scaling_fn(coefs),
            uv10_to_uvsurf=uv10_to_uvsurf,
            **{"alpha": alpha} if model.implements_alpha else {},
        )

        for o in range(args.optim):
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

                coefs_scaled = scaling_fn(coefs)
                alpha = kappa_to_alpha(kappa)
                pre_cycle_bases_setup(model, bases, coefs_scaled, alpha)

                dynamical_reg_fns = inline_reg_setup(
                    bases, H, g_prime, alpha, space_interior, beta_plane, T
                )
                background_reg_fns = post_cycle_reg_setup(bases, Qs)

                if model.A.shape[1] == 1:
                    A11 = model.A[:1, :1]
                    A12 = torch.zeros_like(A11)
                else:
                    A11 = model.A[:1, :1]
                    A12 = model.A[:1, 1:2]

                q_rg, q_rg_bc = build_q_rg(A11, A12)
                q0 = crop(q_rg(psi0), b - 1)
                q_bcs = [q_rg_bc(p1) for p1 in psi_bcs_w]
                q_bc_interp = QuadraticInterpolation(times, q_bcs)

                model.set_psiq(crop(psi0[:, :1], b), q0)
                model.set_boundary_maps(psi_bc_interp, q_bc_interp)

                obs_loss = torch.tensor(0, **specs)
                dynamical_reg = torch.tensor(0, **specs)
                background_reg = torch.tensor(0, **specs)

                loss = torch.tensor(0, **specs)

                if (p := psis[0]) is not None:
                    obs_loss += mse(
                        model.psi[0, 0].ravel()[masks_psi[0]],
                        p,
                        variance=var_psi,
                    )

                for n in range(1, data_n_steps):
                    if with_wind:
                        model.set_wind_forcing(tauxs_i[n - 1], tauys_i[n - 1])
                    for _ in range(dt_ratios):
                        psi1_ = model.psi
                        time = model.time.clone()
                        model.step()
                        psi1 = model.psi

                        if with_reg:
                            dynamical_reg += sum(
                                reg(psi1, (psi1 - psi1_) / dt, time)
                                for reg in dynamical_reg_fns.values()
                            )

                    if (p := psis[n]) is not None:
                        obs_loss += mse(
                            model.psi[0, 0].ravel()[masks_psi[n]],
                            p,
                            variance=var_psi,
                        )
                if with_reg:
                    background_reg += sum(
                        f(coefs[k]) for k, f in background_reg_fns.items()
                    )
                reg_loss = (
                    gamma_dyn * dynamical_reg + gamma_bkg * background_reg
                )
                loss = obs_loss + reg_loss

            if torch.isnan(model.psi).any():
                msg = f"NaN in stream function at {o=}."
                logger.warning(box(msg))
                break
            if torch.isnan(loss.detach()):
                msg = f"Loss has diverged at {o=}."
                logger.warning(box(msg, style="="))
                break

            losses["obs_losses"].append(obs_loss.detach().item())
            losses["dynamical_reg"].append(dynamical_reg.detach().item())
            losses["background_reg"].append(background_reg.detach().item())

            register_params.step(
                loss,
                **{k: c.to_dict() for k, c in coefs_scaled.items()},
                uv10_to_uvsurf=uv10_to_uvsurf,
                **{"alpha": alpha} if model.implements_alpha else {},
            )

            if early_stop.step(loss):
                msg = f"Convergence reached after {o + 1} iterations."
                logger.info(msg)
                break

            loss_ = loss.cpu().item()

            msg = (
                f"Cycle {step(c + 1, args.cycles)} | "
                f"Optimization step {step(o + 1, args.optim)} | "
                f"Loss: {loss_:>#10.5g} | "
                f"Best loss: {register_params.best_loss:>#10.5g}"
            )
            logger.info(msg)

            msg = (
                f"\tObs: {obs_loss.detach().item():>#10.5g}"
                f" | Dyn Reg: {dynamical_reg.detach().item():>#10.5g}"
                f" | Bkg Reg: {background_reg.detach().item():>#10.5g}"
            )
            logger.detail(msg)

            loss.backward()

            for p in params:
                torch.nn.utils.clip_grad_norm_(p["params"], max_norm=1)

            optimizer.step()
            scheduler.step(loss)
            lr_callback.step()

        best_loss = register_params.best_loss
        msg = f"Optimization completed with loss: {best_loss:>#10.5g}"
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
        logger.info(box(msg, msg_mem, style="round"))

        outputs["results"].append(
            {
                "cycle": c,
                **{k: register_params.params[k] for k in bases},
                "uv10_to_uvsurf": register_params.params["uv10_to_uvsurf"],
            }
        )
        outputs["basis"].append(
            {"cycle": c, **{k: c.get_params() for k, c in bases.items()}}
        )
        outputs["optim"].append(
            {
                "cycle": c,
                "numel": numel,
                "max_steps": args.optim,
                "nb_steps": o + 1,
                "loss": best_loss.cpu().item(),
                "max_memory_allocated": max_mem,
            }
        )

        output.results.save(outputs["results"])
        output.optim.save(outputs["optim"])
        output.basis.save(outputs["basis"])

        msg = f"Outputs saved to {output.folder}"
        logger.info(box(msg, style="="))
