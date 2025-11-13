"""Variational analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from qgsw.cli import ScriptArgsVA
from qgsw.configs.core import Configuration
from qgsw.decomposition.wavelets import WaveletBasis
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.filtered.core import QGPSIQForced
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.optim.utils import EarlyStop, RegisterParams
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

if TYPE_CHECKING:
    from qgsw.models.qg.psiq.filtered.core import (
        QGPSIQCollinearSF,
        QGPSIQMixed,
    )

## Config

args = ScriptArgsVA.from_cli(
    comparison_default=1,
    cycles_default=3,
    prefix_default="results_forced",
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
q_slices = [slice(imin, imax), slice(jmin, jmax)]
q_slices_w = [
    slice(imin - (p - 1), imax + (p - 1)),
    slice(jmin - (p - 1), jmax + (p - 1)),
]

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
    model: QGPSIQ | QGPSIQCollinearSF | QGPSIQMixed | QGPSIQForced,
) -> QGPSIQ | QGPSIQCollinearSF | QGPSIQMixed | QGPSIQForced:
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


model_forced = QGPSIQForced(
    space_2d=space_slice,
    H=H[:1] * H[1:2] / (H[:1] + H[1:2]),
    beta_plane=beta_plane,
    g_prime=g_prime[1:2],
)
model_forced: QGPSIQForced = set_inhomogeneous_model(model_forced)

if not args.no_wind:
    model_forced.set_wind_forcing(
        tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
    )

# Compute PV


def extract_psi_w(psi: torch.Tensor) -> torch.Tensor:
    """Extract psi."""
    return psi[..., psi_slices_w[0], psi_slices_w[1]]


def extract_q_w(q: torch.Tensor) -> torch.Tensor:
    """Extract q."""
    return q[..., q_slices_w[0], q_slices_w[1]]


def extract_psi_bc(psi: torch.Tensor) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, p, -p - 1, p, -p - 1, 2)


def extract_q_bc(q: torch.Tensor) -> Boundaries:
    """Extract q."""
    return Boundaries.extract(q, p - 2, -(p - 1), p - 2, -(p - 1), 3)


for c in range(n_cycles):
    torch.cuda.reset_peak_memory_stats()
    times = [model_3l.time.item()]

    psi0 = extract_psi_w(model_3l.psi[:, :1])
    q0 = extract_q_w(model_3l.q[:, :1])
    psi0_mean = psi0[:, :1].mean()
    psi_bc = extract_psi_bc(psi0)
    q_bc = extract_q_bc(q0)

    psis = [psi0]
    qs = [q0]
    psi_bcs = [psi_bc]
    q_bcs = [q_bc]

    for _ in range(1, n_steps_per_cyle):
        model_3l.step()

        times.append(model_3l.time.item())

        psi = extract_psi_w(model_3l.psi[:, :1])
        psi_bc = extract_psi_bc(psi)

        psis.append(psi)
        psi_bcs.append(psi_bc)

        q = extract_q_w(model_3l.q[:, :1])
        q_bc = extract_q_bc(q)

        qs.append(q)
        q_bcs.append(q_bc)

    msg = f"Cycle {step(c + 1, n_cycles)}: Model spin-up completed."
    logger.info(box(msg, style="round"))

    basis = WaveletBasis(
        space_slice.q.xy.x - space_slice.q.xy.x[:1, :],
        space_slice.q.xy.y - space_slice.q.xy.y[:, :1],
        torch.stack([torch.tensor(t - times[0], **specs) for t in times]),
        order=4,
        Lx_max=((H1 + H2) * g1).sqrt() / beta_plane.f0,
        Ly_max=((H1 + H2) * g1).sqrt() / beta_plane.f0,
    )
    basis.normalize = True
    basis.n_theta = 15

    msg = f"Using basis of order {basis.order}"
    logger.info(msg)

    coefs = basis.generate_random_coefs(**specs)
    coefs = {
        k: torch.zeros_like(v, requires_grad=True) for k, v in coefs.items()
    }

    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    q_bc_interp = QuadraticInterpolation(times, q_bcs)

    numel = basis.numel()
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [{"params": list(coefs.values()), "lr": 1e-10}]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stop = EarlyStop()
    register_params_mixed = RegisterParams(
        **{f"coefs_{k}": v for k, v in coefs.items()}
    )

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model_forced.reset_time()
        model_forced.set_boundary_maps(psi_bc_interp, q_bc_interp)

        with torch.enable_grad():
            model_forced.set_psiq(crop(psi0, p), crop(q0, p - 1))

            loss = torch.tensor(0, **defaults.get())
            basis.set_coefs(coefs)

            for n in range(1, n_steps_per_cyle):
                model_forced.forcing = basis.at_time(model_forced.time)
                model_forced.step()

                if n % comparison_interval == 0:
                    loss += rmse(
                        model_forced.psi[0, 0],
                        crop(psis[n][0, 0], p),
                    )

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params_mixed.step(
            loss, **{f"coefs_{k}": v for k, v in coefs.items()}
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

        lr_forcing = optimizer.param_groups[0]["lr"]
        for v in coefs.values():
            torch.nn.utils.clip_grad_norm_([v], max_norm=1e-1)

        optimizer.step()
        scheduler.step(loss)

    best_loss = register_params_mixed.best_loss
    msg = f"Forcing optimization completed with loss: {best_loss:3.5f}"
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
    logger.info(box(msg, msg_mem, style="round"))
    output = {
        "cycle": c,
        "config": {
            "comparison_interval": comparison_interval,
            "optimization_steps": [optim_max_step],
            "no-wind": args.no_wind,
            "order": basis.order,
            "n_theta": basis.n_theta,
        },
        "specs": {"max_memory_allocated": max_mem},
        "coords": (imin, imax, jmin, jmax),
        **{
            f"coefs_{k}": register_params_mixed.params[f"coefs_{k}"]
            for k in coefs
        },
    }
    outputs.append(output)

torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
