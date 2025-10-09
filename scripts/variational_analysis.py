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


def compute_slices(
    imin: int, imax: int, jmin: int, jmax: int
) -> tuple[list[slice, slice], list[slice, slice]]:
    """Compute horizontal slices."""
    psi_slices = [slice(imin, imax + 1), slice(jmin, jmax + 1)]
    q_slices = [slice(imin, imax), slice(jmin, jmax)]

    return psi_slices, q_slices


## Simulation parameters

dt = 7200
optim_max_step = 100
str_optim_len = len(str(optim_max_step))
n_steps_per_cyle = 500
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

psi_slices, q_slices = compute_slices(imin, imax, jmin, jmax)

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


def extract_psi(psi: torch.Tensor) -> tuple[torch.Tensor, Boundaries]:
    """Extract psi."""
    return psi[..., psi_slices[0], psi_slices[1]], Boundaries.extract(
        psi, imin, imax + 1, jmin, jmax + 1, 2
    )


def extract_q(q: torch.Tensor) -> tuple[torch.Tensor, Boundaries]:
    """Extract q."""
    return q[..., q_slices[0], q_slices[1]], Boundaries.extract(
        q, imin - 1, imax + 1, jmin - 1, jmax + 1, 3
    )


for c in range(n_cycles):
    c_ = str(c + 1).zfill(str_cycles_len)
    c_max_ = str(n_cycles)
    times = [model_3l.time.item()]

    psi0, psi_bc = extract_psi(model_3l.psi[:, :1])
    q0, q_bc = extract_q(model_3l.q[:, :1])

    psis = [psi0]
    psi_bcs = [psi_bc]
    qs = [q0]
    q_bcs = [q_bc]

    for _ in range(1, n_steps_per_cyle):
        model_3l.step()

        times.append(model_3l.time.item())

        psi, psi_bc = extract_psi(model_3l.psi[:, :1])
        q, q_bc = extract_q(model_3l.q[:, :1])

        psis.append(psi)
        psi_bcs.append(psi_bc)
        qs.append(q)
        q_bcs.append(q_bc)
    time = datetime.datetime.now(datetime.timezone.utc)
    time_ = time.strftime("%d/%m/%Y %H:%M:%S")
    msg = f"Cycle {c_}/{c_max_} | Model spin-up completed."
    logger.info(msg)

    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)
    q_bc_interp = QuadraticInterpolation(times, q_bcs)

    alpha = torch.tensor(0.5, requires_grad=True)

    optimizer = torch.optim.Adam([alpha], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stop = EarlyStop()
    register_params = RegisterParams()

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model_alpha.reset_time()
        model_alpha.set_psiq(psi0, q0)

        with torch.enable_grad():
            model_alpha.alpha = torch.ones_like(psi0) * alpha
            model_alpha.set_boundary_maps(psi_bc_interp, q_bc_interp)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model_alpha.step()

                if (n + 1) % comparison_interval == 0:
                    loss += rmse(model_alpha.psi[0, 0], psis[n][0, 0])

        register_params.step(loss, alpha)
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

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    best_loss = register_params.best_loss
    time = datetime.datetime.now(datetime.timezone.utc)
    time_ = time.strftime("%d/%m/%Y %H:%M:%S")
    msg = (
        f"Cycle {c_}/{c_max_} | "
        f"ɑ optimization completed | "  # noqa: RUF001
        f"Loss: {best_loss:3.5f}"
    )
    logger.info(msg)
    dpsi2 = (torch.ones_like(model_dpsi.psi) * 1e-2).requires_grad_()

    optimizer = torch.optim.Adam([dpsi2], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    early_stop = EarlyStop()
    register_params = RegisterParams()

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model_dpsi.reset_time()
        model_dpsi.set_psiq(psi0, q0)
        model_dpsi.set_boundary_maps(psi_bc_interp, q_bc_interp)

        with torch.enable_grad():
            model_dpsi.dpsi2 = dpsi2

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model_dpsi.step()

                if (n + 1) % comparison_interval == 0:
                    loss += rmse(model_dpsi.psi[0, 0], psis[n][0, 0])

        register_params.step(loss, dpsi2)
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
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

    best_loss = register_params.best_loss
    time = datetime.datetime.now(datetime.timezone.utc)
    time_ = time.strftime("%d/%m/%Y %H:%M:%S")
    msg = (
        f"Cycle {c_}/{c_max_} | "
        f"dѱ2 optimization completed | "
        f"Loss: {best_loss:3.5f}"
    )
    logger.info(msg)
    output = {
        "cycle": c,
        "coords": (imin, imax, jmin, jmax),
        "alpha": alpha.detach().cpu(),
        "dpsi2": dpsi2.detach().cpu(),
    }
    outputs.append(output)
torch.save(
    outputs, output_dir.joinpath(f"results_{imin}_{imax}_{jmin}_{jmax}.pt")
)
