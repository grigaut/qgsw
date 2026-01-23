"""Variational analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import torch

from qgsw.cli import ScriptArgsVARegularized
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.exp_exp.core import GaussianExpBasis
from qgsw.decomposition.exp_exp.param_generator import gaussian_exp_field
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.modified.forced import QGPSIQForced
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import laplacian
from qgsw.spatial.core.discretization import (
    SpaceDiscretization3D,
)
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.interpolation import QuadraticInterpolation
from qgsw.utils.reshaping import crop

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)


## Config

args = ScriptArgsVARegularized.from_cli(
    comparison_default=1,
    cycles_default=3,
    prefix_default="results_forced_rg_dr_ge_reg",
    gamma_default=1e3,
)
with_reg = not args.no_reg
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
psi_slices = [slice(imin, imax + 1), slice(jmin, jmax + 1)]
psi_slices_w = [slice(imin - p, imax + p + 1), slice(jmin - p, jmax + p + 1)]
q_slices = [slice(imin, imax), slice(jmin, jmax)]
q_slices_w = [
    slice(imin - (p - 1), imax + (p - 1)),
    slice(jmin - (p - 1), jmax + (p - 1)),
]

## Observations

if with_obs_track:
    obs_track = torch.zeros(
        (imax - imin + 1, jmax - jmin + 1),
        dtype=torch.bool,
        device=specs["device"],
    )
    for i in range(obs_track.shape[0]):
        for j in range(obs_track.shape[1]):
            if abs(i - j + 20) < 15:
                obs_track[i, j] = True
    obs_track = obs_track.flatten()
    track_ratio = obs_track.sum() / obs_track.numel()
    msg_obs = (
        "Sampling observations along a track "
        f"covering {track_ratio:.2%} of the domain."
    )
else:
    obs_track = torch.ones(
        (imax - imin + 1, jmax - jmin + 1),
        dtype=torch.bool,
        device=specs["device"],
    ).flatten()
    msg_obs = "Sampling observations over the entire domain."


def on_track(f: torch.Tensor) -> torch.Tensor:
    """Project f on the observation track."""
    return f.flatten()[obs_track]


## Regularization

gamma = args.gamma / comparison_interval * obs_track.sum() / obs_track.numel()

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
comp_dt = sec2text(comparison_interval * dt)
msg_loss = f"RMSE will be evaluated every {comp_dt}."
msg_area = f"Focusing on i in [{imin}, {imax}] and j in [{jmin}, {jmax}]"
msg_output = f"Output will be saved to {output_file}."

logger.info(
    box(msg_simu, msg_loss, msg_area, msg_obs, msg_reg, msg_output, style="=")
)

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


def mse(f: torch.Tensor, f_ref: torch.Tensor) -> float:
    """RMSE."""
    return (f - f_ref).square().mean() / f_ref.square().mean()


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
M = TypeVar("M", bound=QGPSIQ)


def set_inhomogeneous_model(
    model: M,
) -> M:
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

space_slice_w = P.space.remove_z_h().slice(
    imin - p + 1, imax + p, jmin - p + 1, jmax + p
)
y_w = space_slice_w.q.xy.y[0, :].unsqueeze(0)
beta_effect_w = beta_plane.beta * (y_w - y0)
space_slice_ww = P.space.remove_z_h().slice(
    imin - p, imax + p + 1, jmin - p, jmax + p + 1
)


model = QGPSIQForced(
    space_2d=space_slice,
    H=H[:1] * H[1:2] / (H[:1] + H[1:2]),
    beta_plane=beta_plane,
    g_prime=g_prime[1:2],
)
model: QGPSIQForced = set_inhomogeneous_model(model)
model.wind_scaling = H[:1].item()
if not args.no_wind:
    model.set_wind_forcing(
        tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
    )

# Compute PV


def extract_psi_w(psi: torch.Tensor) -> torch.Tensor:
    """Extract psi."""
    return psi[..., psi_slices_w[0], psi_slices_w[1]]


def extract_psi_bc(psi: torch.Tensor) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, p, -p - 1, p, -p - 1, 2)


def compute_q_rg(  # noqa: D103
    psi1: torch.Tensor,
) -> torch.Tensor:
    return (
        interpolate(
            laplacian(psi1, dx, dy)
            - beta_plane.f0**2
            * (1 / (H1 * H2 / (H1 + H2)) / g2)
            * psi1[..., 1:-1, 1:-1]
        )
        + beta_effect_w
    )


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

    xx = space_slice_ww.psi.xy.x
    yy = space_slice_ww.psi.xy.y

    space_params, time_params = gaussian_exp_field(
        0, 2, xx, yy, n_steps_per_cyle * dt, n_steps_per_cyle / 4 * 7200
    )
    basis = GaussianExpBasis(space_params, time_params)
    coefs = basis.generate_random_coefs()
    coefs = DecompositionCoefs.zeros_like(coefs).requires_grad_()

    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)
    qs = (compute_q_rg(p1) for p1 in psis)
    q_bcs = [
        Boundaries.extract(q, p - 2, -(p - 1), p - 2, -(p - 1), 3) for q in qs
    ]
    q_bc_interp = QuadraticInterpolation(times, q_bcs)

    numel = coefs.numel()
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [
            {
                "params": list(coefs.values()),
                "lr": 1e-2,
                "name": "Wavelet coefs",
            }
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    lr_callback = LRChangeCallback(optimizer)
    early_stop = EarlyStop()

    coefs_scaled = coefs.scale(
        *(1e-2 * U**2 / L**2 for _ in range(basis.order))
    )

    register_params = RegisterParams(coefs=coefs_scaled.to_dict())

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model.reset_time()
        model.set_boundary_maps(psi_bc_interp, q_bc_interp)

        with torch.enable_grad():
            model.set_psiq(crop(psi0, p), crop(compute_q_rg(psi0), p - 1))

            coefs_scaled = coefs.scale(
                *(1e-2 * U**2 / L**2 for _ in range(basis.order))
            )
            basis.set_coefs(coefs_scaled)

            wv = basis.localize(space_slice.q.xy.x, space_slice.q.xy.y)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model.forcing = wv(model.time)[None, None, ...]
                model.step()

                if n % comparison_interval == 0:
                    loss += mse(
                        on_track(model.psi[0, 0]),
                        on_track(crop(psis[n][0, 0], p)),
                    )

            for coef in coefs.values():
                loss += gamma * coef.square().mean()

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params.step(loss, coefs=coefs_scaled.to_dict())

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

        torch.nn.utils.clip_grad_norm_(list(coefs.values()), max_norm=1)

        optimizer.step()
        scheduler.step(loss)
        lr_callback.step()

    best_loss = register_params.best_loss
    msg = f"Forcing optimization completed with loss: {best_loss:>#10.5g}"
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
