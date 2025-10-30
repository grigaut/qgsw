"""Variational analysis."""

from __future__ import annotations

from pathlib import Path

import torch

from qgsw import logging
from qgsw.cli import ScriptArgsVA
from qgsw.configs.core import Configuration
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.core.flux import div_flux_5pts_no_pad
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.filtered.core import (
    QGPSIQCollinearSF,
    QGPSIQMixed,
)
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.pv import compute_q1_interior, compute_q2_2l_interior
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import grad_perp
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
    prefix_default="results_mixed_s",
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
g1, g2 = g_prime[0], g_prime[1]
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
    model: QGPSIQ | QGPSIQCollinearSF | QGPSIQMixed,
) -> QGPSIQ | QGPSIQCollinearSF | QGPSIQMixed:
    """Set up inhomogeneous model."""
    space = model.space
    model.y0 = y0
    model.masks = Masks.empty_tensor(
        space.nx,
        space.ny,
        device=specs["device"],
    )
    model.bottom_drag_coef = 0
    model.wide = True
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
y_w = space_slice_w.q.xy.y[0, :].unsqueeze(0)
beta_effect_w = beta_plane.beta * (y_w - y0)


compute_dtq2 = lambda dpsi1, dpsi2: compute_q2_2l_interior(
    dpsi1,
    dpsi2,
    H2,
    g2,
    dx,
    dy,
    beta_plane.f0,
    torch.zeros_like(beta_effect[..., 1:-1]),
)
compute_q2 = lambda psi1, psi2: compute_q2_2l_interior(
    psi1,
    psi2,
    H2,
    g2,
    dx,
    dy,
    beta_plane.f0,
    beta_effect[..., 1:-1],
)


def regularization(
    psi1: torch.Tensor,
    psi2: torch.Tensor,
    dpsi1: torch.Tensor,
    dpsi2: torch.Tensor,
) -> torch.Tensor:
    """Compute regularization.

    Args:
        psi1 (torch.Tensor): Top layer stream function.
        psi2 (torch.Tensor): Bottom layer stream function.
        dpsi1 (torch.Tensor): Top layer stream function derivative.
        dpsi2 (torch.Tensor): Bottom layer stream function derivative

    Returns:
        torch.Tensor: ||∂_t q₂ + J(ѱ₂,q₂)||² (normalized by U / LT)
    """
    dtq2 = compute_dtq2(dpsi1, dpsi2)[..., 1:-1, 1:-1]
    q2 = compute_q2(psi1, psi2)

    u2, v2 = grad_perp(psi2[..., 1:-1, 1:-1])
    u2 /= dy
    v2 /= dx

    dq_2 = div_flux_5pts_no_pad(q2, u2[..., 1:-1, :], v2[..., :, 1:-1], dx, dy)
    return ((dtq2 + dq_2) / U * L * T).square().sum()


gamma = 1 / comparison_interval

# PV computation

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


model_mixed = QGPSIQMixed(
    space_2d=space_slice,
    H=H[:2],
    beta_plane=beta_plane,
    g_prime=g_prime[:2],
)
model_mixed: QGPSIQMixed = set_inhomogeneous_model(model_mixed)

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

    alpha = torch.tensor(0.5, **specs, requires_grad=True)
    psi2_adim = (torch.rand_like(psi0) * 1e-1).requires_grad_()
    dpsi2 = (torch.rand_like(psi2_adim) * 1e-3).requires_grad_()

    numel = alpha.numel() + psi2_adim.numel() + dpsi2.numel()
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [
            {"params": [alpha], "lr": 1e-1},
            {"params": [psi2_adim], "lr": 1e-1},
            {"params": [dpsi2], "lr": 1e-3},
        ],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stop = EarlyStop()
    register_params_mixed = RegisterParams(
        alpha=alpha,
        psi2=psi2_adim * psi0_mean,
        dpsi2=dpsi2,
    )

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model_mixed.reset_time()

        with torch.enable_grad():
            psi2 = psi2_adim * psi0_mean
            q0 = crop(compute_q_psi2(psi0, psi2 + alpha * psi0), p - 1)
            psis_ = (
                (p[:, :1], psi2 + n * dt * dpsi2 + alpha * p[:, :1])
                for n, p in enumerate(psis)
            )
            qs = (compute_q_psi2(p1, p2) for p1, p2 in psis_)
            q_bcs = [
                Boundaries.extract(q, p - 2, -(p - 1), p - 2, -(p - 1), 3)
                for q in qs
            ]

            model_mixed.set_psiq(crop(psi0[:, :1], p), q0)
            q_bc_interp = QuadraticInterpolation(times, q_bcs)
            model_mixed.alpha = torch.ones_like(model_mixed.psi) * alpha
            model_mixed.set_boundary_maps(psi_bc_interp, q_bc_interp)
            model_mixed.dpsi2 = crop(dpsi2, p)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                psi1_ = model_mixed.psi
                psi2_ = crop(psi2 + (n - 1) * dt * dpsi2, p) + alpha * psi1_

                model_mixed.step()

                psi1 = model_mixed.psi
                dpsi1_ = (psi1 - psi1_) / dt
                dpsi2_ = crop(dpsi2, p) + alpha * (psi1 - psi1_) / dt
                reg = gamma * regularization(psi1_, psi2_, dpsi1_, dpsi2_)
                loss += reg

                if n % comparison_interval == 0:
                    loss += rmse(psi1[0, 0], crop(psis[n][0, 0], p))

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params_mixed.step(
            loss,
            alpha=alpha,
            psi2=psi2,
            dpsi2=dpsi2,
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

        lr_alpha = optimizer.param_groups[0]["lr"]
        grad_alpha = alpha.grad.item()
        torch.nn.utils.clip_grad_value_([alpha], clip_value=1.0)
        grad_alpha_ = alpha.grad.item()

        lr_psi2 = optimizer.param_groups[1]["lr"]
        norm_grad_psi2 = psi2_adim.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([psi2_adim], max_norm=1e-1)
        norm_grad_psi2_ = psi2_adim.grad.norm().item()

        lr_dpsi2 = optimizer.param_groups[2]["lr"]
        norm_grad_dpsi2 = dpsi2.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([dpsi2], max_norm=1e-1)
        norm_grad_dpsi2_ = dpsi2.grad.norm().item()

        with logger.section("ɑ parameters:", level=logging.DETAIL):  # noqa: RUF001
            msg = f"Learning rate {lr_alpha:.1e}"
            logger.detail(msg)
            msg = f"Gradient: {grad_alpha:.1e} -> {grad_alpha_:.1e}"
            logger.detail(msg)

        with logger.section("ѱ₂ parameters:", level=logging.DETAIL):
            msg = f"Learning rate {lr_psi2:.1e}"
            logger.detail(msg)
            msg = (
                f"Gradient norm: {norm_grad_psi2:.1e} -> {norm_grad_psi2_:.1e}"
            )
            logger.detail(msg)
        with logger.section("dѱ₂ parameters:", level=logging.DETAIL):
            msg = f"Learning rate {lr_dpsi2:.1e}"
            logger.detail(msg)
            msg = (
                f"Gradient norm: {norm_grad_dpsi2:.1e} ->"
                f" {norm_grad_dpsi2_:.1e}"
            )
            logger.detail(msg)

        optimizer.step()
        scheduler.step(loss)

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
        "psi2": register_params_mixed.params["psi2"].detach().cpu(),
        "dpsi2": register_params_mixed.params["dpsi2"].detach().cpu(),
    }
    outputs.append(output)

torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
