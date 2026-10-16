"""Variational analysis."""

from __future__ import annotations

from pathlib import Path

import torch

from qgsw.cli import ScriptsArgsParser, parse_presets
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
from qgsw.observations import FullDomainMask, SatelliteTrackMask
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.scripts.boundaries import (
    extract_psi_bc,
    extract_wide_psi_bc,
)
from qgsw.scripts.loss import mse
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
    inline_reg_setup,
    post_cycle_reg_setup,
)
from qgsw.scripts.setups.scaling import retrieve_scaling_func
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
)
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.interpolation import QuadraticInterpolation
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
    args.add_regularization(gamma_dyn_default=1, gamma_bkg_default=1)
    args.add_indices()
    args.add_reg_exp(default=2)
    args.add_alpha(1e-2)
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
    msg_area = f"Focusing on i in [{imin}, {imax}] and j in [{jmin}, {jmax}]"
    msg_output = f"Output will be saved to {output.folder}."
    output.params.save(args.namespace.__dict__)

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
    nx, ny = space.nx, space.ny

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

    H1_, H2_ = 150, 1350
    g1_, g2_ = g_prime[0], g_prime[1] * 1

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
    if not args.no_wind:
        model.set_wind_forcing(
            tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
        )
    # PV computation
    build_q_rg = build_pv_funcs(
        dx, dy, beta_plane.f0, beta_effect_wide, b, space_slice_wide.nx
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

        psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

        bases, Qs = setup_basis(
            bases_types,
            None,
            space_slice,
            dt,
            n_steps_per_cyle,
            args.reg_exp,
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
            bases_lrs,
            coefs,
            alpha_optim=model.implements_alpha and with_alpha,
            wind_optim=False,
            alpha_lr=args.alpha_lr,
            wind_lr=None,
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
            **{"alpha": alpha} if model.implements_alpha else {},
        )

        for o in range(args.optim):
            optimizer.zero_grad()
            model.reset_time()

            with torch.enable_grad():
                coefs_scaled = scaling_fn(coefs)
                alpha = kappa_to_alpha(kappa)
                pre_cycle_bases_setup(model, bases, coefs_scaled, alpha)

                dynamical_reg_fns = inline_reg_setup(
                    bases, H, g_prime, alpha, space_slice, beta_plane, T
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

                for n in range(1, n_steps_per_cyle):
                    psi1_ = model.psi
                    time = model.time.clone()
                    model.step()
                    psi1 = model.psi

                    if with_reg:
                        dynamical_reg += sum(
                            reg(psi1, (psi1 - psi1_) / dt, time)
                            for reg in dynamical_reg_fns.values()
                        )
                    if torch.isnan(model.psi).any():
                        msg = f"NaN in stream function at {o=} in step {n=}"
                        raise OverflowError(msg)

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

            if torch.isnan(loss.detach()):
                msg = "Loss has diverged."
                logger.warning(box(msg, style="="))
                break

            losses["obs_losses"].append(obs_loss.detach().item())
            losses["dynamical_reg"].append(dynamical_reg.detach().item())
            losses["background_reg"].append(background_reg.detach().item())

            register_params.step(
                loss,
                **{k: c.to_dict() for k, c in coefs_scaled.items()},
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
                f"| Dyn Reg: {dynamical_reg.detach().item():>#10.5g}"
                f"| Bkg Reg: {background_reg.detach().item():>#10.5g}"
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

        for _ in range(args.separation):
            model_3l.step()
        if args.separation > 0:
            msg = f"Performed {args.separation} steps before next cycle."
            logger.info(box(msg, style="="))
