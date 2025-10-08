"""Variational analysis."""

# ruff: noqa:B023

from __future__ import annotations

import datetime
from pathlib import Path

import torch

from qgsw import verbose
from qgsw.cli import ScriptArgs
from qgsw.configs.core import Configuration
from qgsw.fields.variables.tuples import UVH
from qgsw.forcing.wind import WindForcing
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
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
)
from qgsw.specs import defaults
from qgsw.utils import covphys
from qgsw.utils.interpolation import QuadraticInterpolation

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

args = ScriptArgs.from_cli()
specs = defaults.get()

verbose.set_level(args.verbose)

ROOT_PATH = Path(__file__).parent.parent
config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))

output_dir = config.io.output.directory

# Parameters

H = config.model.h
g_prime = config.model.g_prime
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

imins = [32, 32, 112, 112]
i_len = 64
imaxs = [i + i_len for i in imins]

jmins = [64, 256, 64, 256]
j_len = 128
jmaxs = [j + j_len for j in jmins]


def compute_slices(
    imin: int, imax: int, jmin: int, jmax: int
) -> tuple[list[slice, slice], list[slice, slice]]:
    """Compute horizontal slices."""
    psi_slices = [slice(imin, imax + 1), slice(jmin, jmax + 1)]
    q_slices = [slice(imin, imax), slice(jmin, jmax)]

    return psi_slices, q_slices


n_areas = len(imins)
str_area_len = len(str(n_areas))

## Simulation parameters

dt = 3600
optim_max_step = 100
str_optim_len = len(str(optim_max_step))
n_steps_per_cyle = 500
comparison_interval = 100
n_cycles = 5
str_cycles_len = len(str(n_cycles))

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


for i, indices in enumerate(zip(imins, imaxs, jmins, jmaxs)):
    i_ = str(i + 1).zfill(str_area_len)
    i_max_ = str(n_areas)

    imin, imax, jmin, jmax = indices

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
        verbose.display(
            f"[{time_}] "
            f"Area {i_}/{i_max_}: "
            f"Cycle {c_}/{c_max_}: "
            f"Model spin-up completed.",
            trigger_level=1,
        )

        psi_bc_interp = QuadraticInterpolation(times, psi_bcs)
        q_bc_interp = QuadraticInterpolation(times, q_bcs)

        alpha = torch.tensor(0.5, requires_grad=True)

        optimizer = torch.optim.Adam([alpha], lr=1e-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        early_stop = EarlyStop()
        register_params = RegisterParams()

        for o in range(optim_max_step):
            model_alpha.reset_time()
            model_alpha.set_psiq(psi0, q0)

            with torch.enable_grad():
                model_alpha.alpha = torch.ones_like(psi0) * alpha
                model_alpha.set_boundary_maps(psi_bc_interp, q_bc_interp)

                loss = torch.tensor(0, **defaults.get())

                for n in range(1, n_steps_per_cyle):
                    model_alpha.step()

                    if (n + 1) % 1 == 0:
                        loss += rmse(model_alpha.psi[0, 0], psis[n][0, 0])

            register_params.step(loss, alpha)
            if early_stop.step(loss):
                verbose.display(
                    f"Convergence reached after {o + 1} iterations.",
                    trigger_level=1,
                )
                break

            o_ = str(o + 1).zfill(str_optim_len)
            o_max_ = str(optim_max_step).zfill(str_optim_len)
            loss_ = loss.cpu().item()

            verbose.display(
                f"[{time_}] "
                f"Area {i_}/{i_max_}: "
                f"Cycle {c_}/{c_max_}: "
                f"Optimizing ɑ [{o_}/{o_max_}] - Loss: {loss_:3.5f}",  # noqa: RUF001
                trigger_level=1,
            )

            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        best_loss = register_params.best_loss
        time = datetime.datetime.now(datetime.timezone.utc)
        time_ = time.strftime("%d/%m/%Y %H:%M:%S")
        verbose.display(
            f"[{time_}] "
            f"Area {i_}/{i_max_}: "
            f"Cycle {c_}/{c_max_}: "
            f"ɑ optimization completed - Loss: {best_loss:3.5f}",  # noqa: RUF001
            trigger_level=1,
        )
        dpsi2 = torch.ones_like(model_dpsi.psi, requires_grad=True)

        optimizer = torch.optim.Adam([dpsi2], lr=1e-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5
        )
        early_stop = EarlyStop()
        register_params = RegisterParams()

        for o in range(optim_max_step):
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
                verbose.display(
                    f"Convergence reached after {o + 1} iterations.",
                    trigger_level=1,
                )
                break

            o_ = str(o + 1).zfill(str_optim_len)
            o_max_ = str(optim_max_step)
            loss_ = loss.cpu().item()

            verbose.display(
                f"[{time_}] "
                f"Area {i_}/{i_max_}: "
                f"Cycle {c_}/{c_max_}: "
                f"Optimizing dѱ2 [{o_}/{o_max_}] - Loss: {loss_:3.5f}",
                trigger_level=1,
            )
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

        best_loss = register_params.best_loss
        time = datetime.datetime.now(datetime.timezone.utc)
        time_ = time.strftime("%d/%m/%Y %H:%M:%S")
        verbose.display(
            f"[{time_}] "
            f"Area {i_}/{i_max_}: "
            f"Cycle {c_}/{c_max_}: "
            f"dѱ2 optimization completed - Loss: {best_loss:3.5f}",
            trigger_level=1,
        )
        output = {
            "cycle": c,
            "coords": (imin, imax, jmin, jmax),
            "alpha": alpha.detach().cpu(),
            "dpsi2": dpsi2.detach().cpu(),
        }
        outputs.append(output)
    torch.save(outputs, output_dir.joinpath(f"results_area_{i + 1}.pt"))
