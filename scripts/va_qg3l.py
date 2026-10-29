"""Variational analysis."""

# ruff: noqa: B023

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

from qgsw.cli import (
    OptimBasisParser,
    OptimParamParser,
    ScriptsArgsParser,
    parse_presets,
)
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.models.sv.core import SurfaceVorticity
from qgsw.observations import FullDomainMask, SatelliteTrackMask
from qgsw.optim.optimizers import EarlyStopError, build_optimizer
from qgsw.optim.utils import RegisterParams
from qgsw.plots.live_plots import LivePlot
from qgsw.scripts.boundaries import (
    extract_psi_bc,
    extract_wide_psi_bc,
)
from qgsw.scripts.loss import mse
from qgsw.scripts.model_interactions import get_As, get_psi_g, init_model
from qgsw.scripts.output import build_output_folder, create_output_folder
from qgsw.scripts.parameterizations import kappa_to_alpha
from qgsw.scripts.pv import build_pv_funcs
from qgsw.scripts.setups.bases import retrieve_bases_elements, setup_basis
from qgsw.scripts.setups.callbacks import (
    pre_cycle_bases_setup,
)
from qgsw.scripts.setups.model import setup_model
from qgsw.scripts.setups.optim import optim_setup
from qgsw.scripts.setups.regularization import (
    setup_background_regs,
    setup_dynamical_regs,
)
from qgsw.scripts.setups.scaling import retrieve_scaling_func
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
)
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.reshaping import crop

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)
b = 4

if __name__ == "__main__":
    ## Config

    presets = parse_presets()

    args = ScriptsArgsParser.va_setup(
        prefix_default="QG3L",
        cycles_default=4,
    )
    args.add_regs(1, 1)
    args.add_indices()
    args.add_perturbed_params()
    args.set_presets(presets)
    args.retrieve()

    basis_dyn = OptimBasisParser("basis-dyn")
    basis_dyn.set_presets(presets)
    basis_dyn.retrieve()
    basis_ext = OptimBasisParser("basis-ext")
    basis_ext.set_presets(presets)
    basis_ext.retrieve()
    basis_psi = OptimBasisParser("basis-psi")
    basis_psi.set_presets(presets)
    basis_psi.retrieve()
    basis_chi = OptimBasisParser("basis-chi")
    basis_chi.set_presets(presets)
    basis_chi.retrieve()
    alpha_param = OptimParamParser("alpha")
    alpha_param.set_presets(presets)
    alpha_param.retrieve()

    with_alpha_optim = alpha_param.optim
    with_reg = not args.no_reg
    with_obs_track = args.obs_track
    with_wind = not args.no_wind

    specs = defaults.get()

    setup_root_logger(args.verbose)
    logger = getLogger(__name__)

    bases_types = retrieve_bases_elements(
        args.model_type,
        basis_dyn.type,
        basis_ext.type,
        basis_psi.type,
        basis_chi.type,
        check=True,
    )
    bases_scales = retrieve_bases_elements(
        args.model_type,
        basis_dyn.scaling_factor,
        basis_ext.scaling_factor,
        basis_psi.scaling_factor,
        basis_chi.scaling_factor,
    )
    bases_with_bkg = retrieve_bases_elements(
        args.model_type,
        basis_dyn.with_background,
        basis_ext.with_background,
        basis_psi.with_background,
        basis_chi.with_background,
    )
    bases_gammas = retrieve_bases_elements(
        args.model_type,
        basis_dyn.gamma,
        basis_ext.gamma,
        basis_psi.gamma,
        basis_chi.gamma,
    )
    bases_optims = retrieve_bases_elements(
        args.model_type,
        basis_dyn.optim,
        basis_ext.optim,
        basis_psi.optim,
        basis_chi.optim,
    )

    dyn_regs = {}
    dyn_regs["qg"] = args.dyn_reg_qg and (not args.no_reg)
    dyn_regs["div"] = args.dyn_reg_div and (not args.no_reg)
    dyn_gammas = {"qg": args.gamma_dyn_qg, "div": args.gamma_dyn_div}

    ROOT_PATH = Path(__file__).parent.parent
    config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))

    output_folder = create_output_folder(
        config.io.output.directory,
        prefix=args.prefix + "_" + args.model_type,
        mkdir=not args.dry_run,
    )

    output = build_output_folder(output_folder, dry_run=args.dry_run)

    # Simulation parameters
    dt = 7200
    n_steps_per_cyle = 250
    comparison_interval = args.comparison
    n_cycles = args.cycles

    ## Areas

    indices = args.indices
    imin, imax, jmin, jmax = indices
    psi_slices_w = [
        slice(imin - b, imax + b + 1),
        slice(jmin - b, jmax + b + 1),
    ]

    def extract_psi_w(psi: torch.Tensor) -> torch.Tensor:
        """Extract psi."""
        return psi[..., psi_slices_w[0], psi_slices_w[1]]

    space = SpaceDiscretization3D.from_config(
        config.space,
        config.model,
    )

    space_slice = space.remove_h().slice(imin, imax + 1, jmin, jmax + 1)

    space_slice_wide = space.remove_h().slice(
        imin - b, imax + b + 1, jmin - b, jmax + b + 1
    )

    separation = int(args.separation * dt / 3600 / 24)

    ## Observations

    if with_obs_track:
        obs_mask = SatelliteTrackMask(
            space_slice.psi.xy.x,
            space_slice.psi.xy.y,
            track_width=100000,
            track_interval=500000,
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
            space_slice.psi.xy.x,
            space_slice.psi.xy.y,
            dt=args.comparison * dt,
        )
        msg_obs = (
            f"Full surface observed every {sec2text(args.comparison * dt)}"
        )

    ## Regularization

    if with_reg:
        msg_reg = (
            f"[Dynamical regularization] {dyn_regs} with ɣ_dyn = {dyn_gammas}"  # noqa: RUF001
            f"\n[Background regularization] {bases_with_bkg}"
            f" with ɣ_reg = {bases_gammas}"  # noqa: RUF001
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
    msg_area = f"Focusing on i in [{imin}, {imax}] and j in [{jmin}, {jmax}]"
    msg_output = f"Output will be saved to {output.folder}."
    output.params.save(
        args.namespace.__dict__
        | basis_dyn.namespace.__dict__
        | basis_ext.namespace.__dict__
        | basis_psi.namespace.__dict__
        | basis_chi.namespace.__dict__
    )

    logger.info(
        box(msg_simu, msg_area, msg_obs, msg_reg, msg_output, style="=")
    )

    # Parameters

    H = config.model.h
    g_prime = config.model.g_prime
    beta_plane = config.physics.beta_plane
    bottom_drag_coef = config.physics.bottom_drag_coefficient
    slip_coef = config.physics.slip_coef

    P = QGProjector(
        A=compute_A(H=H, g_prime=g_prime),
        H=H.unsqueeze(-1).unsqueeze(-1),
        space=space,
        f0=beta_plane.f0,
        masks=Masks.empty(
            nx=config.space.nx,
            ny=config.space.ny,
        ),
    )
    dx, dy = space.dx, space.dy

    wind = WindForcing.from_config(
        config.windstress,
        config.space,
        config.physics,
    )
    tx, ty = wind.compute()

    uvh0 = UVH.from_file(config.simulation.startup.file)

    U: float = (
        (uvh0.u[0, 0].max() ** 2 + uvh0.v[0, 0].max() ** 2).sqrt().item()
    )
    L: float = dx.item()
    T: float = L / U

    psi_start = P.compute_p(covphys.to_cov(uvh0, dx, dy))[0] / beta_plane.f0

    # Reference model

    model_3l = QGPSIQ(
        space_2d=space.remove_h(),
        H=H,
        beta_plane=config.physics.beta_plane,
        g_prime=g_prime,
    )
    model_3l.set_wind_forcing(tx, ty)
    model_3l.masks = Masks.empty_tensor(
        model_3l.space.nx,
        model_3l.space.ny,
        device=specs["device"],
    )
    model_3l.bottom_drag_coef = bottom_drag_coef
    model_3l.slip_coef = slip_coef
    model_3l.dt = dt
    y0 = model_3l.y0

    model_3l.reset_time()
    model_3l.set_psi(psi_start)

    y_wide = space_slice_wide.q.xy.y[0, :].unsqueeze(0)
    beta_effect_wide = beta_plane.beta * (y_wide - y0)

    if args.perturbed_params:
        msg = "Using perturbed params."
        logger.info(box(msg))
        H1_, H2_ = 150, 1350
        g1_, g2_ = g_prime[0], g_prime[1] * 1
    else:
        H1_, H2_ = H[0], H[1]
        g1_, g2_ = g_prime[0], g_prime[1]

    # Model
    model = setup_model(
        args.model_type,
        space_slice,
        torch.tensor([H1_, H2_], **specs),
        torch.tensor([g1_, g2_], **specs),
        beta_plane,
        dt,
        config.physics.slip_coef,
    )
    model.y0 = y0
    if not args.no_wind:
        model.set_wind_forcing(
            tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
        )
    # PV computation
    build_q_rg = build_pv_funcs(
        dx,
        dy,
        beta_plane.f0,
        beta_effect_wide * (args.model_type != SurfaceVorticity.get_type()),
        b,
        space_slice_wide.nx,
    )

    outputs = {"results": [], "optim": [], "basis": []}

    L: float = dx.item()

    for c in range(args.cycles):
        overflow_encountered = False
        time0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        times = [model_3l.time.item()]

        psi0 = extract_psi_w(model_3l.psi[:, :1])
        psi0_mean = psi0[:, :1].mean()
        var_psi = crop(psi0, b).var()
        psi_bc = extract_psi_bc(psi0, b)

        psi_bcs = [psi_bc]
        psi_bcs_w = [extract_wide_psi_bc(psi0, b, clone=True)]

        mask = obs_mask.at_time(torch.tensor(0 * dt, **specs)).ravel()
        tm = [
            (crop(psi0, b).ravel()[mask], mask)
            if mask.sum() != 0
            else (None, None)
        ]

        for n in range(1, n_steps_per_cyle):
            model_3l.step()

            times.append(model_3l.time.item())

            psi = extract_psi_w(model_3l.psi[:, :1])

            mask = obs_mask.at_time(torch.tensor(n * dt, **specs)).ravel()
            tm.append(
                (crop(psi, b).ravel()[mask], mask)
                if mask.sum() != 0
                else (None, None),
            )
            psi_bcs.append(extract_psi_bc(psi, b))
            psi_bcs_w.append(extract_wide_psi_bc(psi, b, clone=True))
        psis, masks_psi = zip(*tm)
        msg = f"Cycle {step(c + 1, n_cycles)}: Model spin-up completed."
        logger.info(box(msg, style="round"))

        bases, Qs = setup_basis(
            bases_types,
            None,
            space_slice,
            dt,
            n_steps_per_cyle,
            2,
            None,
            None,
        )

        orders = {k: basis.order for k, basis in bases.items()}

        msg = f"Using bases of order {orders}"
        logger.info(msg)

        coefs = {
            k: DecompositionCoefs.zeros_like(basis.generate_random_coefs())
            for k, basis in bases.items()
        }

        params, coefs, kappa, t, numel = optim_setup(
            bases_optims,
            coefs,
            alpha_optim=model.implements_alpha and with_alpha_optim,
            wind_optim=False,
        )

        optimizer = build_optimizer(args.optimizer, params)
        optimizer.max_epoch = args.optim

        scaling_fn = retrieve_scaling_func(
            {k: basis.type for k, basis in bases.items()},
            bases_scales,
            model.get_type(),
            L,
            U,
            psi0_mean,
            n_steps_per_cyle * dt,
        )

        alpha = kappa_to_alpha(kappa * alpha_param.scaling_factor)
        register_params = RegisterParams(
            **scaling_fn(coefs),
            **{"alpha": alpha} if model.implements_alpha else {},
        )
        live_plot = LivePlot(model, active=args.live_plot)

        def closure(*args, **kwargs) -> torch.Tensor:  # noqa: ANN002, ANN003, ARG001
            """Closure for optimizer."""
            model.reset_time()
            coefs_scaled = scaling_fn(coefs)
            alpha = kappa_to_alpha(kappa * alpha_param.scaling_factor)
            pre_cycle_bases_setup(model, bases, coefs_scaled, alpha)

            dyn_reg_fns = setup_dynamical_regs(
                dyn_regs,
                bases,
                get_As(model)[0],
                space_slice,
                beta_plane,
                beta_effect_wide[..., b:-b],
                T,
                wide=True,
            )
            bkg_reg_fns = setup_background_regs(bases_with_bkg, Qs)

            init_model(model, psi0, psi_bcs, psi_bcs_w, build_q_rg, times, b)

            obs = torch.tensor(0, **specs)
            dyn = torch.tensor(0, **specs)
            bkg = torch.tensor(0, **specs)

            if (p := psis[0]) is not None:
                obs += mse(
                    get_psi_g(model)[0, 0].ravel()[masks_psi[0]],
                    p,
                    variance=var_psi,
                )

            for n in range(1, n_steps_per_cyle):
                psi_ = get_psi_g(model)
                t = model.time.clone()
                model.step()
                psi = get_psi_g(model)
                dpsi = (psi - psi_) / model.dt
                if with_reg:
                    dyn += sum(
                        dyn_gammas[k] * dyn_reg_fns[k](psi, dpsi, t)
                        for k in dyn_reg_fns
                    )

                if (p := psis[n]) is not None:
                    obs += mse(
                        psi[0, 0].ravel()[masks_psi[n]], p, variance=var_psi
                    )
            if with_reg:
                bkg += sum(
                    bases_gammas[k] * bkg_reg_fns[k](coefs[k])
                    if bases_with_bkg[k]
                    else 0
                    for k in bkg_reg_fns
                )
            if torch.isnan(psi).any():
                msg = "Overflow"
                raise OverflowError(msg)
            loss = obs + dyn + bkg
            register_params.step(
                loss,
                **{k: c.to_dict() for k, c in coefs_scaled.items()},
                **{"alpha": alpha} if model.implements_alpha else {},
            )
            live_plot.step()
            loss.backward()
            return {
                "tot": loss,
                "obs": obs,
                "dyn": dyn,
                "bkg": bkg,
                "best": register_params.best_loss,
            }

        optimizer.set_closure(closure)

        for _ in range(args.optim):
            try:
                optimizer.step()
            except (EarlyStopError, OverflowError) as e:
                msg = f"{e} at {optimizer.epoch_repr}"
                logger.info(msg)
                overflow_encountered = isinstance(e, OverflowError)
                break

        live_plot.close()

        best_loss = register_params.best_loss
        msg = f"Optimization completed with loss: {best_loss:>#10.5g}"
        max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
        logger.info(box(msg, msg_mem, style="round"))

        outputs["results"].append(
            {
                "cycle": c,
                **register_params.params,
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
                "nb_steps": optimizer.epoch + 1,
                "loss": best_loss.cpu().item(),
                "max_memory_allocated": max_mem,
                "losses": optimizer.losses,
                "overflow": overflow_encountered,
                "elapsed": time.time() - time0,
            }
        )

        output.results.save(outputs["results"])
        output.optim.save(outputs["optim"])
        output.basis.save(outputs["basis"])

        msg = f"Outputs saved to {output.folder}"
        logger.info(box(msg, style="="))
        if c == (n_cycles - 1):
            continue
        for _ in range(args.separation):
            model_3l.step()
        if args.separation > 0:
            msg = f"Performed {args.separation} steps before next cycle."
            logger.info(box(msg, style="="))

        torch.cuda.empty_cache()
        gc.collect()
