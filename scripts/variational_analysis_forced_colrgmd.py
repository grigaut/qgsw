"""Variational analysis."""

from __future__ import annotations

from pathlib import Path

import torch

from qgsw.cli import ScriptArgsVA
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.wavelets import (
    WaveletBasis,
)
from qgsw.decomposition.wavelets.param_generators import dyadic_decomposition
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForcedRGMDWV,
)
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.pv import compute_q1_interior
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
    prefix_default="results_forced_colrgmd",
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
    model: QGPSIQForcedRGMDWV,
) -> QGPSIQForcedRGMDWV:
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
space_slice_ww = P.space.remove_z_h().slice(
    imin - p, imax + p + 1, jmin - p, jmax + p + 1
)
y_w = space_slice_w.q.xy.y[0, :].unsqueeze(0)
beta_effect_w = beta_plane.beta * (y_w - y0)


model = QGPSIQForcedRGMDWV(
    space_2d=space_slice,
    H=H[:2],
    beta_plane=beta_plane,
    g_prime=g_prime[:2],
)
model: QGPSIQForcedRGMDWV = set_inhomogeneous_model(model)

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

    wv_space, wv_time = dyadic_decomposition(
        order=5,
        xx_ref=space_slice.psi.xy.x,
        yy_ref=space_slice.psi.xy.y,
        Lxy_max=900_000,
        Lt_max=n_steps_per_cyle * dt,
    )
    basis = WaveletBasis(wv_space, wv_time)
    basis.n_theta = 7

    msg = f"Using basis of order {basis.order}"
    logger.info(msg)

    coefs = basis.generate_random_coefs()
    coefs = DecompositionCoefs.zeros_like(coefs)
    coefs = coefs.requires_grad_()
    alpha = torch.tensor(0, **specs, requires_grad=True)

    qs = (compute_q_rg(p1) for p1 in psis)
    q_bcs = [
        Boundaries.extract(q, p - 2, -(p - 1), p - 2, -(p - 1), 3) for q in qs
    ]
    q_bc_interp = QuadraticInterpolation(times, q_bcs)
    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    numel = coefs.numel() + alpha.numel()
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [
            {
                "params": alpha,
                "lr": 1e-2,
                "name": "ɑ",  # noqa: RUF001
            },
            {
                "params": list(coefs.values()),
                "lr": 1e0,
                "name": "Wavelet coefs",
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    lr_callback = LRChangeCallback(optimizer)
    early_stop = EarlyStop()

    coefs_scaled = coefs.scale(*(psi0_mean for _ in range(basis.order)))
    register_params = RegisterParams(alpha=alpha, coefs=coefs_scaled.to_dict())

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model.reset_time()

        with torch.enable_grad():
            model.alpha = torch.ones_like(model.psi) * alpha
            coefs_scaled = coefs.scale(
                *(psi0_mean for _ in range(basis.order))
            )
            basis.set_coefs(coefs_scaled)
            model.basis = basis
            model.set_boundary_maps(psi_bc_interp, q_bc_interp)
            model.set_psiq(
                crop(psi0, p),
                crop(compute_q_rg(psi0), p - 1),
            )

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model.step()

                if n % comparison_interval == 0:
                    loss += rmse(
                        model.psi[0, 0],
                        crop(psis[n][0, 0], p),
                    )

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params.step(loss, alpha=alpha, coefs=coefs_scaled.to_dict())

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

        torch.nn.utils.clip_grad_norm_(list(coefs.values()), max_norm=1)
        torch.nn.utils.clip_grad_value_([alpha], clip_value=1)

        optimizer.step()
        scheduler.step(loss)
        lr_callback.step()

    best_loss = register_params.best_loss
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
            "wv_space": wv_space,
            "wv_time": wv_time,
        },
        "specs": {"max_memory_allocated": max_mem},
        "coords": (imin, imax, jmin, jmax),
        "alpha": register_params.params["alpha"],
        "coefs": register_params.params["coefs"],
    }
    outputs.append(output)

torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
