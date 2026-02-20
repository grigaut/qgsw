"""Variational analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import torch

from qgsw.cli import ScriptArgsVAModified
from qgsw.configs.core import Configuration
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.exp_exp.param_generator import gaussian_exp_field
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
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
)
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.reshaping import crop

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

## Config

args = ScriptArgsVAModified.from_cli(
    comparison_default=1,
    cycles_default=3,
    prefix_default="results_psi2",
    gamma_default=1,
)
with_reg = not args.no_reg
with_alpha = not args.no_alpha
with_obs_track = args.obs_track

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
psi_slices_w = [slice(imin - p, imax + p + 1), slice(jmin - p, jmax + p + 1)]

space = SpaceDiscretization3D.from_config(
    config.space,
    config.model,
)

space_slice = space.remove_h().slice(imin, imax + 1, jmin, jmax + 1)

space_slice_w = space.remove_h().slice(
    imin - p + 1, imax + p, jmin - p + 1, jmax + p
)
space_slice_ww = space.remove_h().slice(
    imin - p, imax + p + 1, jmin - p, jmax + p + 1
)
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
    if comparison_interval != 1:
        msg = (
            "Using Satellite track, comparison interval"
            "inferred from tracks trajectory."
        )
        logger.warning(box(msg, style="="))
    msg_obs = "Surface observed along satellite tracks."
else:
    obs_mask = FullDomainMask(
        space_slice.psi.xy.x,
        space_slice.psi.xy.y,
        dt=comparison_interval * dt,
    )
    msg_obs = (
        f"Full surface observed every {sec2text(comparison_interval * dt)}"
    )


def update_loss(
    loss: torch.Tensor,
    f: torch.Tensor,
    f_ref: torch.Tensor,
    time: torch.Tensor,
) -> None:
    """Update loss."""
    mask = obs_mask.at_time(time)
    if not mask.any():
        return loss
    f_sliced = f.flatten()[mask.flatten()]
    f_ref_sliced = f_ref.flatten()[mask.flatten()]
    return loss + mse(f_sliced, f_ref_sliced)


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
filename = f"{prefix}_{imin}_{imax}_{jmin}_{jmax}.pt"
output_file = output_dir.joinpath(filename)

## Logs

msg_simu = (
    f"Performing {n_cycles} cycles of {n_steps_per_cyle} "
    f"steps with up to {optim_max_step} optimization steps."
)
msg_area = f"Focusing on i in [{imin}, {imax}] and j in [{jmin}, {jmax}]"
msg_output = f"Output will be saved to {output_file}."

logger.info(box(msg_simu, msg_area, msg_obs, msg_reg, msg_output, style="="))

# Parameters

H = config.model.h
g_prime = config.model.g_prime
H1, H2 = H[0], H[1]
g1, g2, g3 = g_prime[0], g_prime[1], g_prime[2]
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

U: float = (uvh0.u[0, 0].max() ** 2 + uvh0.v[0, 0].max() ** 2).sqrt().item()
L: float = dx.item()
T: float = L / U

psi_start = P.compute_p(covphys.to_cov(uvh0, dx, dy))[0] / beta_plane.f0

## Error


def mse(f: torch.Tensor, f_ref: torch.Tensor) -> float:
    """RMSE."""
    return (f - f_ref).square().mean() / f_ref.square().mean()


# Models
## Three Layer model

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


## Inhomogeneous models
M = TypeVar("M", bound=QGPSIQ)


def set_inhomogeneous_model(model: M) -> M:
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


def extract_psi_w(psi: torch.Tensor) -> torch.Tensor:
    """Extract psi."""
    return psi[..., psi_slices_w[0], psi_slices_w[1]]


def extract_psi_bc(psi: torch.Tensor) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, p, -p - 1, p, -p - 1, 2)


for c in range(n_cycles):
    torch.cuda.reset_peak_memory_stats()
    times = [model_3l.time.clone()]

    psi0 = extract_psi_w(model_3l.psi[:, :1])
    psi0_mean = psi0[:, :1].mean()
    psis = [psi0]

    for _ in range(1, n_steps_per_cyle):
        model_3l.step()

        times.append(model_3l.time.clone())

        psi = extract_psi_w(model_3l.psi[:, :2])
        psi_bc = extract_psi_bc(psi)

        psis.append(psi)

    msg = f"Cycle {step(c + 1, n_cycles)}: Model spin-up completed."
    logger.info(box(msg, style="round"))

    xx = space_slice_ww.psi.xy.x
    yy = space_slice_ww.psi.xy.y

    space_params, time_params = gaussian_exp_field(
        0, 2, xx, yy, n_steps_per_cyle * dt, n_steps_per_cyle / 4 * 7200
    )
    basis = GaussianExpBasis(space_params, time_params)
    coefs = basis.generate_random_coefs()
    coefs = coefs.requires_grad_()

    if with_alpha:
        numel = coefs.numel()
        params = [
            {
                "params": list(coefs.values()),
                "lr": 1e0,
                "name": "Decomposition coefs",
            },
        ]
    else:
        numel = coefs.numel()
        params = [
            {
                "params": list(coefs.values()),
                "lr": 1e0,
                "name": "Decomposition coefs",
            },
        ]

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
    register_params = RegisterParams(coefs=coefs_scaled.to_dict())

    for o in range(optim_max_step):
        optimizer.zero_grad()

        with torch.enable_grad():
            coefs_scaled = coefs.scale(
                *(
                    1e-1 * psi0_mean / (n_steps_per_cyle * dt) ** k
                    for k in range(basis.order)
                )
            )

            basis.set_coefs(coefs_scaled)

            compute_psi2 = basis.localize(xx, yy)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                psi2 = compute_psi2(times[n] - times[0])

                loss = update_loss(
                    loss,
                    crop(psi2, p),
                    crop(psis[n][0, 1], p),
                    times[n] - times[0],
                )

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params.step(
            loss,
            coefs=coefs_scaled.to_dict(),
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
            "optimization_steps": [optim_max_step],
            "no-wind": args.no_wind,
            "basis": basis.get_params(),
        },
        "specs": {"max_memory_allocated": max_mem},
        "coords": (imin, imax, jmin, jmax),
        "coefs": register_params.params["coefs"],
    }
    outputs.append(output)

torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
