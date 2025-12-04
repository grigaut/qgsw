"""Variational analysis."""

from __future__ import annotations

from pathlib import Path

import torch

from qgsw import logging
from qgsw.cli import ScriptArgsVA
from qgsw.configs.core import Configuration
from qgsw.decomposition.exp_fields.core import ExpField
from qgsw.decomposition.exp_fields.param_generator import subdivisions
from qgsw.decomposition.wavelets.unidimensional.core import WaveletBasis1D
from qgsw.decomposition.wavelets.unidimensional.param_generators import (
    dyadic_decomposition,
)
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.modified.core import (
    QGPSIQCollinearSF,
    QGPSIQMixedReducedOrder,
)
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.pv import (
    compute_q1_interior,
)
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
)
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.interpolation import QuadraticInterpolation
from qgsw.utils.reshaping import crop

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

## Config

args = ScriptArgsVA.from_cli(
    comparison_default=1,
    cycles_default=3,
    prefix_default="results_mixed_ro",
)
specs = defaults.get()

setup_root_logger(args.verbose)
logger = getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent
config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))

output_dir = config.io.output.directory


# Simulation parameters

dt = 7200
optim_max_step = 200
n_steps_per_cyle = 250
comparison_interval = args.comparison
n_cycles = args.cycles

## Areas

indices = args.indices
imin, imax, jmin, jmax = indices

p = 4
psi_slices = [slice(imin, imax + 1), slice(jmin, jmax + 1)]
psi_slices_w = [slice(imin - p, imax + p + 1), slice(jmin - p, jmax + p + 1)]

## Output
prefix = args.prefix
filename = f"{prefix}_{imin}_{imax}_{jmin}_{jmax}.pt"
output_file = output_dir.joinpath(filename)

## Logs

msg_simu = (
    f"Performing {n_cycles} cycles of {n_steps_per_cyle} "
    f"steps with up to {optim_max_step} optimization steps."
)
comp_dt = sec2text(comparison_interval * dt)
msg_loss = f"RMSE will be evaluated every {comp_dt}."
msg_area = f"Focusing on i in [{imin}, {imax}] and j in [{jmin}, {jmax}]"
msg_output = f"Output will be saved to {output_file}."

logger.info(box(msg_simu, msg_loss, msg_area, msg_output, style="="))

# Parameters

H = config.model.h
g_prime = config.model.g_prime
H1, H2 = H[0], H[1]
g1, g2, g3 = g_prime[0], g_prime[1], g_prime[2]
beta_plane = config.physics.beta_plane
bottom_drag_coef = config.physics.bottom_drag_coefficient
slip_coef = config.physics.slip_coef

space = SpaceDiscretization3D.from_config(
    config.space,
    config.model,
)
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

U: float = (uvh0.u[0, 0].max() ** 2 + uvh0.v[0, 0].max() ** 2).sqrt().item()
L: float = dx.item()
T: float = L / U

psi_start = P.compute_p(covphys.to_cov(uvh0, dx, dy))[0] / beta_plane.f0

## Error


def rmse(f: torch.Tensor, f_ref: torch.Tensor) -> float:
    """RMSE."""
    return (f - f_ref).square().mean().sqrt() / f_ref.square().mean().sqrt()


# Models
## Three Layer model

model_3l = QGPSIQ(
    space_2d=space.remove_z_h(),
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


## Inhomogeneous models
def set_inhomogeneous_model(
    model: QGPSIQ | QGPSIQCollinearSF | QGPSIQMixedReducedOrder,
) -> QGPSIQ | QGPSIQCollinearSF | QGPSIQMixedReducedOrder:
    """Set up inhomogeneous model."""
    space = model.space
    model.y0 = y0
    model.masks = Masks.empty_tensor(
        space.nx,
        space.ny,
        device=specs["device"],
    )
    model.bottom_drag_coef = 0
    model.wide = False
    model.slip_coef = slip_coef
    model.dt = dt
    return model


outputs = []
model_3l.reset_time()
model_3l.set_psi(psi_start)

space_slice = P.space.remove_z_h().slice(imin, imax + 1, jmin, jmax + 1)
y = space_slice.q.xy.y[0, :].unsqueeze(0)
beta_effect = beta_plane.beta * (y - y0)

space_slice_w = P.space.remove_z_h().slice(
    imin - p + 1, imax + p, jmin - p + 1, jmax + p
)
space_slice_ww = P.space.remove_z_h().slice(
    imin - p, imax + p + 1, jmin - p, jmax + p + 1
)
space_slice_bc = P.space.remove_z_h().slice(
    imin - 1, imax + 2, jmin - 1, jmax + 2
)
y_w = space_slice_w.q.xy.y[0, :].unsqueeze(0)
beta_effect_w = beta_plane.beta * (y_w - y0)

# PV computation

compute_q_rg = lambda psi1: compute_q1_interior(
    psi1,
    torch.zeros_like(psi1),
    H1,
    g1,
    g2,
    dx,
    dy,
    beta_plane.f0,
    beta_effect_w,
)
compute_q_psi2 = lambda psi1, psi2: compute_q1_interior(
    psi1,
    psi2,
    H1,
    g1,
    g2,
    dx,
    dy,
    beta_plane.f0,
    beta_effect_w,
)

model_mixed = QGPSIQMixedReducedOrder(
    space_2d=space_slice,
    H=H[:2],
    beta_plane=beta_plane,
    g_prime=g_prime[:2],
)
model_mixed: QGPSIQMixedReducedOrder = set_inhomogeneous_model(model_mixed)

if not args.no_wind:
    model_mixed.set_wind_forcing(
        tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
    )


def extract_psi_w(psi: torch.Tensor) -> torch.Tensor:
    """Extract psi."""
    return psi[..., psi_slices_w[0], psi_slices_w[1]]


def extract_psi_bc(psi: torch.Tensor) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, p, -p - 1, p, -p - 1, 2)


for c in range(n_cycles):
    torch.cuda.reset_peak_memory_stats()
    times = [model_3l.time.item()]

    psi0 = extract_psi_w(model_3l.psi[:, :1])
    psi0_mean = psi0[:, :1].mean()
    psi_bc = extract_psi_bc(psi0)

    psis = [psi0]
    psi_bcs = [psi_bc]

    for _ in range(1, n_steps_per_cyle):
        model_3l.step()

        times.append(model_3l.time.item())

        psi = extract_psi_w(model_3l.psi[:, :1])
        psi_bc = extract_psi_bc(psi)

        psis.append(psi)
        psi_bcs.append(psi_bc)

    msg = f"Cycle {step(c + 1, n_cycles)}: Model spin-up completed."
    logger.info(box(msg, style="round"))

    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    params_left = dyadic_decomposition(
        4,
        space_slice_bc.q.xy.y[0, :],
        Lx_max=((H1 + H2) * g1).sqrt() / beta_plane.f0,
        Lt_max=n_steps_per_cyle * dt,
        sigma_x_l_p_ratio=1.13,
    )
    params_right = dyadic_decomposition(
        4,
        space_slice_bc.q.xy.y[-1, :],
        Lx_max=((H1 + H2) * g1).sqrt() / beta_plane.f0,
        Lt_max=n_steps_per_cyle * dt,
        sigma_x_l_p_ratio=1.13,
    )
    params_bottom = dyadic_decomposition(
        4,
        space_slice_bc.q.xy.x[:, 0],
        Lx_max=((H1 + H2) * g1).sqrt() / beta_plane.f0,
        Lt_max=n_steps_per_cyle * dt,
        sigma_x_l_p_ratio=1.13,
    )
    params_top = dyadic_decomposition(
        4,
        space_slice_bc.q.xy.x[:, -1],
        Lx_max=((H1 + H2) * g1).sqrt() / beta_plane.f0,
        Lt_max=n_steps_per_cyle * dt,
        sigma_x_l_p_ratio=1.13,
    )
    bc_left = WaveletBasis1D(*params_left)
    bc_right = WaveletBasis1D(*params_right)
    bc_bottom = WaveletBasis1D(*params_bottom)
    bc_top = WaveletBasis1D(*params_top)
    space_params, time_params = subdivisions(
        space_slice_ww.psi.xy.x,
        space_slice_ww.psi.xy.y,
        subdivision_nb=4,
        Lt_max=n_steps_per_cyle * dt,
    )
    basis = ExpField(space_params, time_params)

    coefs_adim_l = bc_left.generate_random_coefs()
    coefs_adim_l = {
        k: torch.zeros_like(v, requires_grad=True)
        for k, v in coefs_adim_l.items()
    }
    coefs_adim_r = bc_right.generate_random_coefs()
    coefs_adim_r = {
        k: torch.zeros_like(v, requires_grad=True)
        for k, v in coefs_adim_r.items()
    }
    coefs_adim_b = bc_bottom.generate_random_coefs()
    coefs_adim_b = {
        k: torch.zeros_like(v, requires_grad=True)
        for k, v in coefs_adim_b.items()
    }
    coefs_adim_t = bc_top.generate_random_coefs()
    coefs_adim_t = {
        k: torch.zeros_like(v, requires_grad=True)
        for k, v in coefs_adim_t.items()
    }

    coefs_adim = basis.generate_random_coefs()
    coefs_adim = torch.zeros_like(coefs_adim, requires_grad=True)

    alpha = torch.tensor(0, **specs, requires_grad=True)

    numel = (
        alpha.numel()
        + coefs_adim.numel()
        + bc_left.numel()
        + bc_right.numel()
        + bc_bottom.numel()
        + bc_top.numel()
    )
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [
            {"params": [alpha], "lr": 1e-1, "name": "ɑ"},  # noqa: RUF001
            {"params": [coefs_adim], "lr": 1e0, "name": "Coefficients"},
            {
                "params": list(coefs_adim_l.values()),
                "lr": 1e0,
                "name": "Coefs BC left",
            },
            {
                "params": list(coefs_adim_r.values()),
                "lr": 1e0,
                "name": "Coefs BC right",
            },
            {
                "params": list(coefs_adim_b.values()),
                "lr": 1e0,
                "name": "Coefs BC bottom",
            },
            {
                "params": list(coefs_adim_t.values()),
                "lr": 1e0,
                "name": "Coefs BC top",
            },
        ],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    lr_callback = LRChangeCallback(optimizer)
    early_stop = EarlyStop()

    register_params_mixed = RegisterParams(
        alpha=alpha,
        coefs=coefs_adim * psi0_mean,
        **{f"coefs_bc_l_{k}": v * psi0_mean for k, v in coefs_adim_l.items()},
        **{f"coefs_bc_r_{k}": v * psi0_mean for k, v in coefs_adim_r.items()},
        **{f"coefs_bc_b_{k}": v * psi0_mean for k, v in coefs_adim_b.items()},
        **{f"coefs_bc_t_{k}": v * psi0_mean for k, v in coefs_adim_t.items()},
    )

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model_mixed.reset_time()

        with torch.enable_grad():
            coefs_l = {k: v * psi0_mean for k, v in coefs_adim_l.items()}
            coefs_r = {k: v * psi0_mean for k, v in coefs_adim_r.items()}
            coefs_b = {k: v * psi0_mean for k, v in coefs_adim_b.items()}
            coefs_t = {k: v * psi0_mean for k, v in coefs_adim_t.items()}
            bc_left.set_coefs(coefs_l)
            bc_right.set_coefs(coefs_r)
            bc_bottom.set_coefs(coefs_b)
            bc_top.set_coefs(coefs_t)
            coefs = coefs_adim * psi0_mean
            basis.set_coefs(coefs)

            bc_l = bc_left.localize(
                space_slice_bc.q.xy.y[0, :],
            )
            bc_r = bc_right.localize(
                space_slice_bc.q.xy.y[1, :],
            )
            bc_t = bc_top.localize(
                space_slice_bc.q.xy.y[:, 0],
            )
            bc_b = bc_bottom.localize(
                space_slice_bc.q.xy.y[:, -1],
            )
            ef = basis.localize(
                space_slice_ww.psi.xy.x, space_slice_ww.psi.xy.y
            )

            model_mixed.basis = basis
            q0 = crop(
                compute_q_psi2(
                    psi0, ef(model_mixed.time)[None, None, ...] + alpha * psi0
                ),
                p - 1,
            )
            psis_ = (
                (
                    p[:, :1],
                    ef(model_mixed.time + n * dt)[None, None, ...]
                    + alpha * p[:, :1],
                )
                for n, p in enumerate(psis)
            )
            qs = (crop(compute_q_rg(p1), p - 2) for p1 in psis)
            psi_2_bc = [
                Boundaries(
                    top=bc_t(model_mixed.time + i * dt)[:, None],
                    bottom=bc_b(model_mixed.time + i * dt)[:, None],
                    left=bc_l(model_mixed.time + i * dt)[None, :],
                    right=bc_r(model_mixed.time + i * dt)[None, :],
                )
                for i in range(n_steps_per_cyle)
            ]
            q_rg_bcs = [Boundaries.extract(q, 0, -1, 0, -1, 1) for q in qs]

            q_bcs = [
                q_rg + beta_plane.f0**2 / H1.item() / g2.item() * p2
                for q_rg, p2 in zip(q_rg_bcs, psi_2_bc)
            ]

            model_mixed.set_psiq(crop(psi0[:, :1], p), q0)
            q_bc_interp = QuadraticInterpolation(times, q_bcs)
            model_mixed.alpha = torch.ones_like(model_mixed.psi) * alpha
            model_mixed.set_boundary_maps(psi_bc_interp, q_bc_interp)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model_mixed.step()

                psi1 = model_mixed.psi

                if n % comparison_interval == 0:
                    loss += rmse(psi1[0, 0], crop(psis[n][0, 0], p))

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params_mixed.step(
            loss,
            alpha=alpha,
            coefs=coefs,
            **{f"coefs_bc_l_{k}": v for k, v in coefs_l.items()},
            **{f"coefs_bc_r_{k}": v for k, v in coefs_r.items()},
            **{f"coefs_bc_b_{k}": v for k, v in coefs_b.items()},
            **{f"coefs_bc_t_{k}": v for k, v in coefs_t.items()},
        )

        if early_stop.step(loss):
            msg = f"Convergence reached after {o + 1} iterations."
            logger.info(msg)
            break

        loss_ = loss.cpu().item()

        msg = (
            f"Cycle {step(c + 1, n_cycles)} | "
            f"Optimization step {step(o + 1, optim_max_step)} | "
            f"Loss: {loss_:3.5f}"
        )
        logger.info(msg)

        loss.backward()

        grad_alpha = alpha.grad.item()
        torch.nn.utils.clip_grad_value_([alpha], clip_value=1.0)
        grad_alpha_ = alpha.grad.item()

        torch.nn.utils.clip_grad_norm_(
            list(coefs_adim_l.values()),
            max_norm=1e-1,
        )

        torch.nn.utils.clip_grad_norm_(
            list(coefs_adim_r.values()),
            max_norm=1e-1,
        )

        torch.nn.utils.clip_grad_norm_(
            list(coefs_adim_b.values()),
            max_norm=1e-1,
        )

        torch.nn.utils.clip_grad_norm_(
            list(coefs_adim_t.values()),
            max_norm=1e-1,
        )

        norm_grad_cs = coefs_adim.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([coefs_adim], max_norm=1e-1)
        norm_grad_cs_ = coefs_adim.grad.norm().item()

        with logger.section("ɑ parameters:", level=logging.DETAIL):  # noqa: RUF001
            msg = f"Gradient: {grad_alpha:.1e} -> {grad_alpha_:.1e}"
            logger.detail(msg)

        with logger.section("ѱ₂ parameters:", level=logging.DETAIL):
            msg = f"Gradient norm: {norm_grad_cs:.1e} -> {norm_grad_cs_:.1e}"
            logger.detail(msg)

        optimizer.step()
        scheduler.step(loss)
        lr_callback.step()

    best_loss = register_params_mixed.best_loss
    msg = (
        f"ɑ, dɑ, ѱ₂ and dѱ₂ optimization completed with loss: {best_loss:3.5f}"  # noqa: RUF001
    )
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
    logger.info(box(msg, msg_mem, style="round"))
    output = {
        "cycle": c,
        "config": {
            "comparison_interval": comparison_interval,
            "optimization_steps": [optim_max_step],
            "no-wind": args.no_wind,
        },
        "specs": {"max_memory_allocated": max_mem},
        "coords": (imin, imax, jmin, jmax),
        "alpha": register_params_mixed.params["alpha"].detach().cpu(),
        "coefs": register_params_mixed.params["coefs"].detach().cpu(),
    }
    outputs.append(output)

torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
