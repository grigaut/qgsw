"""Variational analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

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
from qgsw.models.qg.psiq.modified.forced import QGPSIQPsi2Transport
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.pv import (
    compute_q1_interior,
    compute_q2_3l_interior,
)
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import grad, laplacian
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
)
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.interpolation import QuadraticInterpolation
from qgsw.utils.reshaping import crop

if TYPE_CHECKING:
    from collections.abc import Callable

    from qgsw.decomposition.base import SpaceTimeDecomposition
    from qgsw.decomposition.supports.space.base import SpaceSupportFunction
    from qgsw.decomposition.supports.time.base import TimeSupportFunction

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

## Config

args = ScriptArgsVAModified.from_cli(
    comparison_default=1,
    cycles_default=3,
    prefix_default="results_mixed_ro_ge",
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
psi_slices = [slice(imin, imax + 1), slice(jmin, jmax + 1)]
psi_slices_w = [slice(imin - p, imax + p + 1), slice(jmin - p, jmax + p + 1)]

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

space_slice = P.space.remove_z_h().slice(imin, imax + 1, jmin, jmax + 1)
y = space_slice.q.xy.y[0, :].unsqueeze(0)
beta_effect = beta_plane.beta * (y - y0)

space_slice_w = P.space.remove_z_h().slice(
    imin - p + 1, imax + p, jmin - p + 1, jmax + p
)
y_w = space_slice_w.q.xy.y[0, :].unsqueeze(0)
beta_effect_w = beta_plane.beta * (y_w - y0)
space_slice_ww = P.space.remove_z_h().slice(
    imin - p, imax + p + 1, jmin - p, jmax + p + 1
)


compute_dtq2 = lambda dpsi1, dpsi2: compute_q2_3l_interior(
    dpsi1,
    dpsi2,
    torch.zeros_like(dpsi2),
    H2,
    g2,
    g3,
    dx,
    dy,
    beta_plane.f0,
    torch.zeros_like(beta_effect[..., 1:-1]),
)
compute_q2 = lambda psi1, psi2: compute_q2_3l_interior(
    psi1,
    psi2,
    torch.zeros_like(psi2),
    H2,
    g2,
    g3,
    dx,
    dy,
    beta_plane.f0,
    beta_effect[..., 1:-1],
)


def compute_regularization_func(
    psi2_basis: SpaceTimeDecomposition[
        SpaceSupportFunction, TimeSupportFunction
    ],
    alpha: torch.Tensor,
    space: SpaceDiscretization2D,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build regularization function.

    Args:
        psi2_basis (SpaceTimeDecomposition): Basis.
        alpha (torch.Tensor): Collinearity coefficient.
        space (SpaceDiscretization2D): Space.

    Returns:
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
            Regularization function.
    """
    q = space.q.xy
    x = crop(q.x, 1)
    y = crop(q.y, 1)

    fpsi2 = psi2_basis.localize(x, y)
    fdx_psi2 = psi2_basis.localize_dx(x, y)
    fdy_psi2 = psi2_basis.localize_dy(x, y)
    flap_psi2 = psi2_basis.localize_laplacian(x, y)
    fdx_lap_psi2 = psi2_basis.localize_dx_laplacian(x, y)
    fdy_lap_psi2 = psi2_basis.localize_dy_laplacian(x, y)

    def compute_reg(
        psi1: torch.Tensor,
        dpsi1: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regularization term.

        Args:
            psi1 (torch.Tensor): Top layer stream function.
            dpsi1 (torch.Tensor): Top layer stream function derivative.
            time (torch.Tensor): Time.

        Returns:
            torch.Tensor: ∂ₜq₂ + J(ѱ₂, q₂)
        """
        dt_lap_psi2 = flap_psi2.dt(time) + alpha * interpolate(
            laplacian(dpsi1, dx, dy)
        )
        dt_psi2 = fpsi2.dt(time) + alpha * interpolate(crop(dpsi1, 1))

        dt_q2 = dt_lap_psi2 - beta_plane.f0**2 * (
            (1 / H2 / g2) * (dt_psi2 - interpolate(crop(dpsi1, 1)))
            + 1 / H2 / g3 * (dt_psi2)
        )

        dx_psi1, dy_psi1 = grad(psi1)
        dx_psi1 /= dx
        dy_psi1 /= dy

        dx_psi1_i = (dx_psi1[..., 1:] + dx_psi1[..., :-1]) / 2
        dy_psi1_i = (dy_psi1[..., 1:, :] + dy_psi1[..., :-1, :]) / 2

        dx_psi2 = fdx_psi2(time) + alpha * crop(dx_psi1_i, 1)
        dy_psi2 = fdy_psi2(time) + alpha * crop(dy_psi1_i, 1)

        lap_dy_psi1 = laplacian(dy_psi1_i, dx, dy)

        dy_q2 = (
            fdy_lap_psi2(time)
            + alpha * lap_dy_psi1
            - beta_plane.f0**2
            * (
                (1 / H2 / g2) * (dy_psi2 - crop(dy_psi1_i, 1))
                + 1 / H2 / g3 * (dy_psi2)
            )
        ) + beta_plane.beta

        lap_dx_psi1 = laplacian(dx_psi1_i, dx, dy)

        dx_q2 = (
            fdx_lap_psi2(time)
            + alpha * lap_dx_psi1
            - beta_plane.f0**2
            * (
                (1 / H2 / g2) * (dx_psi2 - crop(dx_psi1_i, 1))
                + 1 / H2 / g3 * (dx_psi2)
            )
        )

        adv_q2 = -dy_psi2 * dx_q2 + dx_psi2 * dy_q2
        return ((dt_q2 + adv_q2) / U * L * T).square().sum()

    return compute_reg


if with_obs_track:
    obs_track = torch.zeros_like(
        model_3l.psi[0, 0, imin : imax + 1, jmin : jmax + 1], dtype=torch.bool
    )
    for i in range(obs_track.shape[0]):
        for j in range(obs_track.shape[1]):
            if abs(i - j + 20) < 15:
                obs_track[i, j] = True
    obs_track = obs_track.flatten()
    track_ratio = obs_track.sum() / obs_track.numel()
    msg = (
        "Sampling observation along a track "
        f"spanning over {track_ratio:.2%} of the domain."
    )
    logger.info(box(msg, style="round"))
else:
    obs_track = torch.ones_like(
        model_3l.psi[0, 0, imin : imax + 1, jmin : jmax + 1], dtype=torch.bool
    ).flatten()


def on_track(f: torch.Tensor) -> torch.Tensor:
    """Project f on the observation track."""
    return f.flatten()[obs_track]


gamma = 1 / comparison_interval * obs_track.sum() / obs_track.numel() * 0.1

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


model = QGPSIQPsi2Transport(
    space_2d=space_slice,
    H=H[:2],
    beta_plane=beta_plane,
    g_prime=g_prime[:2],
)
model: QGPSIQPsi2Transport = set_inhomogeneous_model(model)

if not args.no_wind:
    model.set_wind_forcing(
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

    xx = space_slice_ww.psi.xy.x
    yy = space_slice_ww.psi.xy.y

    space_params, time_params = gaussian_exp_field(
        0, 2, xx, yy, n_steps_per_cyle * dt, n_steps_per_cyle / 4 * 7200
    )
    basis = GaussianExpBasis(space_params, time_params)
    coefs = basis.generate_random_coefs()
    coefs = coefs.requires_grad_()

    if with_alpha:
        alpha = torch.tensor(0.5, **specs, requires_grad=True)
        numel = alpha.numel() + coefs.numel()
        params = [
            {"params": [alpha], "lr": 1e-2, "name": "ɑ"},  # noqa: RUF001
            {
                "params": list(coefs.values()),
                "lr": 1e0,
                "name": "Decomposition coefs",
            },
        ]
    else:
        alpha = torch.tensor(0, **specs)
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
    register_params = RegisterParams(alpha=alpha, coefs=coefs_scaled.to_dict())

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model.reset_time()

        with torch.enable_grad():
            coefs_scaled = coefs.scale(
                *(
                    1e-1 * psi0_mean / (n_steps_per_cyle * dt) ** k
                    for k in range(basis.order)
                )
            )

            basis.set_coefs(coefs_scaled)

            model.basis = basis

            compute_reg = compute_regularization_func(
                basis, alpha, space_slice
            )

            compute_psi2 = basis.localize(xx, yy)

            q0 = crop(
                compute_q_psi2(psi0, compute_psi2(model.time) + alpha * psi0),
                p - 1,
            )

            psis_ = (
                (
                    p[:, :1],
                    compute_psi2(model.time + n * model.dt) + alpha * p[:, :1],
                )
                for n, p in enumerate(psis)
            )
            qs = (compute_q_psi2(p1, p2) for p1, p2 in psis_)
            q_bcs = [
                Boundaries.extract(q, p - 2, -(p - 1), p - 2, -(p - 1), 3)
                for q in qs
            ]

            model.set_psiq(crop(psi0[:, :1], p), q0)
            q_bc_interp = QuadraticInterpolation(times, q_bcs)
            model.alpha = torch.ones_like(model.psi) * alpha
            model.set_boundary_maps(psi_bc_interp, q_bc_interp)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                psi1_ = model.psi
                time = model.time.clone()

                model.step()

                psi1 = model.psi

                if with_reg:
                    dpsi1_ = (psi1 - psi1_) / dt
                    reg = gamma * compute_reg(psi1_, dpsi1_, time)
                    loss += reg

                if n % comparison_interval == 0:
                    loss += mse(
                        on_track(psi1[0, 0]),
                        on_track(crop(psis[n][0, 0], p)),
                    )

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params.step(
            loss,
            alpha=alpha,
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

        if with_alpha:
            torch.nn.utils.clip_grad_value_([alpha], clip_value=1.0)

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
        "alpha": register_params.params["alpha"],
        "coefs": register_params.params["coefs"],
    }
    outputs.append(output)

torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
