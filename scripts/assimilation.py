"""Data assimilation pipeline."""

from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress

from qgsw import verbose
from qgsw.cli import ScriptArgs
from qgsw.configs.core import Configuration
from qgsw.fields.variables.coefficients import (
    create_coefficient,
)
from qgsw.fields.variables.uvh import UVH
from qgsw.forcing.wind import WindForcing
from qgsw.models import matching
from qgsw.models.instantiation import instantiate_model
from qgsw.models.qg.projected.modified.collinear.core import (
    QGCollinearSF,
)
from qgsw.models.qg.projected.modified.collinear.stretching_matrix import (
    compute_g_tilde as g_tilde_qg_col,
)
from qgsw.models.qg.projected.modified.filtered.core import (
    QGCollinearFilteredSF,
)
from qgsw.models.qg.projected.modified.filtered.pv import (
    compute_g_tilde as g_tilde_qg_filt,
)
from qgsw.models.qg.projected.modified.utils import is_modified
from qgsw.perturbations.core import Perturbation
from qgsw.run_summary import RunSummary
from qgsw.simulation.steps import Steps
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
)
from qgsw.specs import DEVICE
from qgsw.utils import time_params

torch.backends.cudnn.deterministic = True

args = ScriptArgs.from_cli()

verbose.set_level(args.verbose)

ROOT_PATH = Path(__file__).parent.parent
config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))
summary = RunSummary.from_configuration(config)

if config.io.output.save:
    summary.to_file(config.io.output.directory)


# Common Set-up
## Wind Forcing
wind = WindForcing.from_config(config.windstress, config.space, config.physics)
taux, tauy = wind.compute()
## Rossby
Ro = 0.1

# Model Set-up
## Vortex
perturbation = Perturbation.from_config(
    perturbation_config=config.perturbation,
)
space_2d = SpaceDiscretization2D.from_config(config.space)

model_ref = instantiate_model(
    config.simulation.reference,
    config.physics.beta_plane,
    space_2d,
    perturbation,
    Ro=0.1,
)

model_ref.slip_coef = config.physics.slip_coef
model_ref.bottom_drag_coef = config.physics.bottom_drag_coefficient
if np.isnan(config.simulation.dt):
    model_ref.dt = time_params.compute_dt(
        model_ref.prognostic.uvh,
        model_ref.space,
        model_ref.g_prime,
        model_ref.H,
    )
else:
    model_ref.dt = config.simulation.dt
model_ref.compute_time_derivatives(model_ref.prognostic.uvh)
model_ref.set_wind_forcing(taux, tauy)

model = instantiate_model(
    config.model,
    config.physics.beta_plane,
    space_2d,
    perturbation,
    Ro=0.1,
)

model.slip_coef = config.physics.slip_coef
model.bottom_drag_coef = config.physics.bottom_drag_coefficient
if np.isnan(config.simulation.dt):
    model.dt = time_params.compute_dt(
        model.prognostic.uvh,
        model.space,
        model.g_prime,
        model.H,
    )
else:
    model.dt = config.simulation.dt
model.compute_time_derivatives(model.prognostic.uvh)
model.set_wind_forcing(taux, tauy)


verbose.display("\n[Reference Model]", trigger_level=1)
verbose.display(msg=model_ref.__repr__(), trigger_level=1)
verbose.display("\n[Model]", trigger_level=1)
verbose.display(msg=model.__repr__(), trigger_level=1)

nl_ref = model_ref.space.nl
nl = model.space.nl

if nl not in [1, nl_ref]:
    msg = (
        "This script only runs for model with 1 layer or as many layers as "
        f"the reference model, hence {nl_ref}"
    )
    raise ValueError(msg)

nx = model.space.nx
ny = model.space.ny

dtype = torch.float64
device = DEVICE.get()

if (startup_file := config.simulation.startup_file) is None:
    uvh0 = UVH.steady(
        n_ens=1,
        nl=nl_ref,
        nx=config.space.nx,
        ny=config.space.ny,
        dtype=torch.float64,
        device=DEVICE.get(),
    )
else:
    uvh0 = UVH.from_file(startup_file, dtype=dtype, device=device)
    horizontal_shape = uvh0.h.shape[-2:]
    if horizontal_shape != (nx, ny):
        msg = (
            f"Horizontal shape {horizontal_shape} from {startup_file}"
            f" should be ({nx},{ny})."
        )
        raise ValueError(msg)

model_ref.set_uvh(
    uvh0.u[:, :nl_ref, ...],
    uvh0.v[:, :nl_ref, ...],
    uvh0.h[:, :nl_ref, ...],
)
model.set_uvh(
    uvh0.u[:, :nl, ...],
    uvh0.v[:, :nl, ...],
    uvh0.h[:, :nl, ...],
)

dt = model.dt
t_end = config.simulation.duration

steps = Steps(t_end=t_end, dt=dt)

ns = steps.simulation_steps()
forks = steps.steps_from_interval(interval=config.simulation.fork_interval)
saves = config.io.output.get_saving_steps(steps)

summary.register_outputs(model.io)
summary.register_steps(t_end=t_end, dt=dt, n_steps=steps.n_tot)

t = 0

summary.register_start()
prefix_ref = config.simulation.reference.prefix
prefix = config.model.prefix
output_dir = config.io.output.directory

# Collinearity Coefficient
if is_modified(config.model.type):
    alpha = create_coefficient(config)
    model.alpha = alpha.compute(model_ref.prognostic)

with Progress() as progress:
    simulation = progress.add_task(
        rf"\[n=00000/{steps.n_tot:05d}]",
        total=steps.n_tot,
    )
    for n, fork, save in zip(ns, forks, saves):
        progress.update(
            simulation,
            description=rf"\[n={n:05d}/{steps.n_tot:05d}]",
        )
        progress.advance(simulation)
        if fork:
            prognostic = model_ref.prognostic
            if config.model.type == QGCollinearSF.get_type():
                model.alpha = alpha.compute_no_slice(prognostic)
                uvh = matching.match_psi(
                    prognostic.uvh,
                    config.simulation.reference.g_prime,
                    g_tilde_qg_col(config.model.g_prime),
                )
            if config.model.type == QGCollinearFilteredSF.get_type():
                model.alpha = alpha.compute_no_slice(prognostic)
                uvh = matching.match_psi(
                    prognostic.uvh,
                    config.simulation.reference.g_prime,
                    g_tilde_qg_filt(config.model.g_prime),
                )
            else:
                uvh = matching.match_psi(
                    prognostic.uvh,
                    config.simulation.reference.g_prime,
                    config.model.g_prime,
                )
            model.set_uvh(
                torch.clone(uvh.u),
                torch.clone(uvh.v),
                torch.clone(uvh.h),
            )
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}] - Forked",
                trigger_level=1,
            )
        if save:
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}]",
                trigger_level=1,
            )
            verbose.display(
                msg="[Reference Model]: ",
                trigger_level=1,
                end="",
            )
            # Save Reference Model
            model_ref.io.save(output_dir.joinpath(f"{prefix_ref}{n}.pt"))
            verbose.display(
                msg="[     Model     ]: ",
                trigger_level=1,
                end="",
            )
            # Save Model
            model.io.save(output_dir.joinpath(f"{prefix}{n}.pt"))
            summary.register_step(n)

        model_ref.step()
        model.step()
        t += dt

    summary.register_end()
