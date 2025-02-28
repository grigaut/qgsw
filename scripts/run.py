"""Run a single model with a Vortex forcing."""

from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress

from qgsw import verbose
from qgsw.cli import ScriptArgs
from qgsw.configs.core import Configuration
from qgsw.fields.variables.coefficients.instantiation import instantiate_coef
from qgsw.forcing.wind import WindForcing
from qgsw.models.instantiation import instantiate_model
from qgsw.models.qg.projected.modified.utils import is_modified
from qgsw.perturbations.core import Perturbation
from qgsw.physics import compute_burger
from qgsw.run_summary import RunSummary
from qgsw.simulation.steps import Steps
from qgsw.spatial.core.discretization import SpaceDiscretization2D
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
## Grid
space_2d = SpaceDiscretization2D.from_config(config.space)
## Set model parameters

model = instantiate_model(
    model_config=config.model,
    beta_plane=config.physics.beta_plane,
    space_2d=space_2d,
    perturbation=perturbation,
    Ro=Ro,
)
# Collinearity Coefficient
modified = is_modified(config.model.type)
if modified:
    coef = instantiate_coef(config.model, config.space)
    model.alpha = coef.update(config.model.collinearity_coef.initial)

model.slip_coef = config.physics.slip_coef
model.bottom_drag_coef = config.physics.bottom_drag_coefficient
if np.isnan(config.simulation.dt):
    model.dt = time_params.compute_dt(
        model.prognostic,
        model.space,
        model.g_prime,
        model.H,
    )
else:
    model.dt = config.simulation.dt
model.set_wind_forcing(taux, tauy)

## Compute Burger Number
Bu = compute_burger(
    g=config.model.g_prime[0],
    h_scale=config.model.h[0],
    f0=config.physics.f0,
    length_scale=perturbation.compute_scale(model.space.omega),
)
verbose.display(
    msg=f"Burger Number: {Bu:.2f}",
    trigger_level=1,
)


## time params
t = 0
dt = model.dt

t_end = config.simulation.duration


steps = Steps(t_end=t_end, dt=dt)
ns = steps.simulation_steps()
saves = config.io.output.get_saving_steps(steps)
logs = steps.steps_from_total(100)

summary.register_outputs(model.io)
summary.register_steps(t_end=t_end, dt=dt, n_steps=steps.n_tot)

verbose.display(msg=model.__repr__(), trigger_level=1)
verbose.display(msg=f"Total Duration: {t_end:.2f} seconds", trigger_level=1)


summary.register_start()
prefix = config.model.prefix
# Start runs
with Progress() as progress:
    simulation = progress.add_task(
        rf"\[n=00000/{steps.n_tot:05d}]",
        total=steps.n_tot,
    )
    for n, save, log in zip(ns, saves, logs):
        progress.update(
            simulation,
            description=rf"\[n={n:05d}/{steps.n_tot:05d}]",
        )
        progress.advance(simulation)
        # Save step
        if save:
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}]",
                trigger_level=1,
            )
            directory = config.io.output.directory
            model.io.save(directory.joinpath(f"{prefix}{n}.pt"))

        model.step()
        t += dt
        # Step log
        if log:
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}]",
                trigger_level=1,
            )
            verbose.display(msg=model.io.print_step(), trigger_level=1)
            summary.register_step(n)

    summary.register_end()
