"""Variational analysis."""

from __future__ import annotations

import argparse
import datetime
import pathlib
from dataclasses import dataclass
from pathlib import Path

import torch

from qgsw.configs.core import Configuration
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.logging import getLogger, setup_root_logger
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.filtered.core import (
    QGPSIQCollinearSF,
    QGPSIQFixeddSF2,
)
from qgsw.models.qg.psiq.optim.utils import EarlyStop, RegisterParams
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.models.qg.uvh.projectors.core import QGProjector
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import laplacian
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
)
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.interpolation import QuadraticInterpolation

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

## Config


@dataclass
class ScriptArgs:
    """Script arguments."""

    config: Path
    verbose: int
    indices: list[int]

    @classmethod
    def from_cli(cls) -> Self:
        """Instantiate script arguments from CLI.

        Args:
            default_config (str): Default configuration path.

        Returns:
            Self: ScriptArgs.
        """
        parser = argparse.ArgumentParser(
            description="Retrieve script arguments.",
        )
        parser.add_argument(
            "--config",
            required=True,
            type=pathlib.Path,
            help="Configuration File Path (from qgsw root level)",
        )
        parser.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="Verbose level.",
        )
        parser.add_argument(
            "-i",
            "--indices",
            required=True,
            nargs="+",
            type=int,
            help="Indices (imin, imax, jmin, jmax), "
            "for example (64, 128, 128, 256).",
        )
        return cls(**vars(parser.parse_args()))


args = ScriptArgs.from_cli()
specs = defaults.get()

setup_root_logger(args.verbose)
logger = getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent
config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))

output_dir = config.io.output.directory

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
psi_start = P.compute_p(covphys.to_cov(uvh0, dx, dy))[0] / beta_plane.f0

## Areas

indices = args.indices
imin, imax, jmin, jmax = indices
msg = f"Focusing on i in [{imin}, {imax}] and j in [{jmin}, {jmax}]"
logger.info(msg)

p = 4
psi_slices = [slice(imin, imax + 1), slice(jmin, jmax + 1)]
psi_slices_w = [slice(imin - p, imax + p + 1), slice(jmin - p, jmax + p + 1)]

## Simulation parameters

dt = 7200
optim_max_step = 200
str_optim_len = len(str(optim_max_step))
n_steps_per_cyle = 250
comparison_interval = 1
n_cycles = 3
str_cycles_len = len(str(n_cycles))
msg = (
    f"Performing {n_cycles} cycles of {n_steps_per_cyle} "
    f"steps with up to {optim_max_step} optimization steps."
)
logger.info(msg)

## Error


def rmse(f: torch.Tensor, f_ref: torch.Tensor) -> float:
    """RMSE."""
    return (f - f_ref).square().mean().sqrt() / f_ref.square().mean().sqrt()


# PV computation


def compute_q_alpha(psi: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Compute pv using alpha."""
    return interpolate(
        laplacian(psi, dx, dy)
        - beta_plane.f0**2 * (1 / H1 / g1 + 1 / H1 / g1) * psi[..., 1:-1, 1:-1]
        + beta_plane.f0**2 * (1 / H1 / g2) * alpha * psi[..., 1:-1, 1:-1]
    )


def compute_q_psi2(psi: torch.Tensor, psi2: torch.Tensor) -> torch.Tensor:
    """Compute pv using psi2."""
    return interpolate(
        laplacian(psi, dx, dy)
        - beta_plane.f0**2 * (1 / H1 / g1 + 1 / H1 / g1) * psi[..., 1:-1, 1:-1]
        + beta_plane.f0**2 * (1 / H1 / g2) * psi2[..., 1:-1, 1:-1]
    )


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
    model: QGPSIQ | QGPSIQCollinearSF | QGPSIQFixeddSF2,
) -> QGPSIQ | QGPSIQCollinearSF | QGPSIQFixeddSF2:
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


space_slice = SpaceDiscretization2D.from_tensors(
    x=P.space.remove_z_h().omega.xy.x[imin : imax + 1, 0],
    y=P.space.remove_z_h().omega.xy.y[0, jmin : jmax + 1],
)

model_alpha = QGPSIQCollinearSF(
    space_2d=space_slice,
    H=H[:2],
    beta_plane=beta_plane,
    g_prime=g_prime[:2],
)
model_dpsi = QGPSIQFixeddSF2(
    space_2d=space_slice,
    H=H[:2],
    beta_plane=beta_plane,
    g_prime=g_prime[:2],
)
model_alpha: QGPSIQCollinearSF = set_inhomogeneous_model(model_alpha)
model_dpsi: QGPSIQFixeddSF2 = set_inhomogeneous_model(model_dpsi)

model_alpha.set_wind_forcing(
    tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
)
model_dpsi.set_wind_forcing(
    tx[imin:imax, jmin : jmax + 1], ty[imin : imax + 1, jmin:jmax]
)


def extract_psi_w(psi: torch.Tensor) -> torch.Tensor:
    """Extract psi."""
    return psi[..., psi_slices_w[0], psi_slices_w[1]]


def extract_psi_bc(psi: torch.Tensor) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, p, -p - 1, p, -p - 1, 2)


for c in range(n_cycles):
    c_ = str(c + 1).zfill(str_cycles_len)
    c_max_ = str(n_cycles)
    times = [model_3l.time.item()]

    psi0 = extract_psi_w(model_3l.psi[:, :1])
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

    time = datetime.datetime.now(datetime.timezone.utc)
    time_ = time.strftime("%d/%m/%Y %H:%M:%S")
    msg = f"Cycle {c_}/{c_max_} | Model spin-up completed."
    logger.info(msg)

    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    alpha = torch.tensor(0.5, requires_grad=True)
    dalpha = torch.tensor(0.5, requires_grad=True)

    optimizer = torch.optim.Adam([alpha, dalpha], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stop = EarlyStop()
    register_params_alpha = RegisterParams()

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model_alpha.reset_time()

        with torch.enable_grad():
            q0 = compute_q_alpha(psi0, alpha)[..., 3:-3, 3:-3]
            q_bcs = [
                Boundaries.extract(
                    compute_q_alpha(psi[:, :1], alpha), 2, -3, 2, -3, 3
                )
                for psi in psis
            ]

            model_alpha.set_psiq(psi0[:, :1, p:-p, p:-p], q0)
            q_bc_interp = QuadraticInterpolation(times, q_bcs)
            model_alpha.alpha = torch.ones_like(model_alpha.psi) * dalpha
            model_alpha.set_boundary_maps(psi_bc_interp, q_bc_interp)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model_alpha.step()

                if (n + 1) % comparison_interval == 0:
                    loss += rmse(
                        model_alpha.psi[0, 0], psis[n][0, 0, p:-p, p:-p]
                    )

        register_params_alpha.step(loss, alpha=alpha, dalpha=dalpha)
        if early_stop.step(loss):
            msg = f"Convergence reached after {o + 1} iterations."
            logger.info(msg)
            break

        o_ = str(o + 1).zfill(str_optim_len)
        o_max_ = str(optim_max_step).zfill(str_optim_len)
        loss_ = loss.cpu().item()
        msg = (
            f"Cycle {c_}/{c_max_} | "
            f"ɑ optimization step {o_}/{o_max_} | "  # noqa: RUF001
            f"Loss: {loss_:3.5f}"
        )
        logger.info(msg)

        lr = scheduler.get_last_lr()[0]
        msg = f"\tLearning rate {lr:.1e}"
        logger.detail(msg)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    best_loss = register_params_alpha.best_loss
    time = datetime.datetime.now(datetime.timezone.utc)
    time_ = time.strftime("%d/%m/%Y %H:%M:%S")
    msg_ = (
        f"ɑ and dɑ optimization completed with "  # noqa: RUF001
        f"loss: {best_loss:3.5f}"
    )
    msg = (
        f"##{''.join(['#'] * len(msg_))}##\n"
        f"# {''.join([' '] * len(msg_))} #\n"
        f"# {msg_} #\n# {''.join([' '] * len(msg_))} #\n#"
        f"#{''.join(['#'] * len(msg_))}##"
    )
    logger.info(msg)

    psi2 = (torch.ones_like(psis[0]) * psis[0].mean()).requires_grad_()
    dpsi2 = (torch.ones_like(psi2) * 1e-3).requires_grad_()

    optimizer = torch.optim.Adam([psi2, dpsi2], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stop = EarlyStop()
    register_params_dpsi2 = RegisterParams()

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model_dpsi.reset_time()

        with torch.enable_grad():
            q0 = compute_q_psi2(psi0, psi2)[
                ..., (p - 1) : -(p - 1), (p - 1) : -(p - 1)
            ]
            q_bcs = [
                Boundaries.extract(
                    compute_q_psi2(psi[:, :1], psi2 + n * dt * dpsi2),
                    p - 2,
                    -(p - 1),
                    p - 2,
                    -(p - 1),
                    3,
                )
                for n, psi in enumerate(psis)
            ]

            model_dpsi.set_psiq(psi0[:, :1, p:-p, p:-p], q0)
            q_bc_interp = QuadraticInterpolation(times, q_bcs)
            model_dpsi.set_boundary_maps(psi_bc_interp, q_bc_interp)
            model_dpsi.dpsi2 = dpsi2[..., p:-p, p:-p]

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model_dpsi.step()

                if (n + 1) % comparison_interval == 0:
                    loss += rmse(
                        model_dpsi.psi[0, 0], psis[n][0, 0, p:-p, p:-p]
                    )

        register_params_dpsi2.step(loss, psi2=psi2, dpsi2=dpsi2)
        if early_stop.step(loss):
            msg = f"Convergence reached after {o + 1} iterations."
            logger.info(msg)
            break

        o_ = str(o + 1).zfill(str_optim_len)
        o_max_ = str(optim_max_step)
        loss_ = loss.cpu().item()

        msg = (
            f"Cycle {c_}/{c_max_} | "
            f"dѱ2 optimization step {o_}/{o_max_} | "
            f"Loss: {loss_:3.5f}"
        )
        logger.info(msg)

        lr = scheduler.get_last_lr()[0]
        msg = f"\tLearning rate {lr:.1e}"
        logger.detail(msg)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    best_loss = register_params_dpsi2.best_loss
    time = datetime.datetime.now(datetime.timezone.utc)
    time_ = time.strftime("%d/%m/%Y %H:%M:%S")
    msg_ = f"ѱ2 and dѱ2 optimization completed with loss: {best_loss:3.5f}"
    msg = (
        f"##{''.join(['#'] * len(msg_))}##\n"
        f"# {''.join([' '] * len(msg_))} #\n"
        f"# {msg_} #\n# {''.join([' '] * len(msg_))} #\n#"
        f"#{''.join(['#'] * len(msg_))}##"
    )
    logger.info(msg)
    output = {
        "cycle": c,
        "coords": (imin, imax, jmin, jmax),
        "alpha": register_params_alpha.params["alpha"].detach().cpu(),
        "dalpha": register_params_alpha.params["dalpha"].detach().cpu(),
        "psi2": register_params_dpsi2.params["psi2"].detach().cpu(),
        "dpsi2": register_params_dpsi2.params["dpsi2"].detach().cpu(),
    }
    outputs.append(output)
torch.save(
    outputs, output_dir.joinpath(f"results_{imin}_{imax}_{jmin}_{jmax}.pt")
)
