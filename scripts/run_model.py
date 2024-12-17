"""Run a single model with a Vortex forcing."""

import argparse
from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress

from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.wind import WindForcing
from qgsw.models.instantiation import instantiate_model
from qgsw.perturbations import Perturbation
from qgsw.physics import compute_burger
from qgsw.run_summary import RunSummary
from qgsw.simulation.steps import Steps
from qgsw.spatial.dim_3 import SpaceDiscretization3D
from qgsw.utils import time_params

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

parser = argparse.ArgumentParser(description="Retrieve Configuration file.")
parser.add_argument(
    "--config",
    default="config/run_model.toml",
    help="Configuration File Path (from qgsw root level)",
)
args = parser.parse_args()

ROOT_PATH = Path(__file__).parent.parent
CONFIG_PATH = ROOT_PATH.joinpath(args.config)
config = Configuration.from_file(CONFIG_PATH)
summary = RunSummary.from_configuration(config)

if config.io.results.save:
    summary.to_file(config.io.results.directory)

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
space = SpaceDiscretization3D.from_config(config.space, config.model)
## Compute Burger Number
Bu = compute_burger(
    g=config.model.g_prime[0],
    h_scale=config.model.h[0],
    f0=config.physics.f0,
    length_scale=perturbation.compute_scale(space.omega),
)
verbose.display(
    msg=f"Burger Number: {Bu:.2f}",
    trigger_level=1,
)

## Set model parameters

model = instantiate_model(
    config=config,
    space_3d=space,
    perturbation=perturbation,
    Ro=Ro,
)
model.set_wind_forcing(taux, tauy)

## time params
t = 0
dt = model.dt

if config.simulation.reference == "tau":
    if np.isnan(config.simulation.tau):
        tau = time_params.compute_tau(model.omega, model.space)
    else:
        tau = config.simulation.tau / config.physics.f0
    verbose.display(
        msg=f"tau = {tau *config.physics.f0:.2f} f0⁻¹",
        trigger_level=1,
    )
    t_end = config.simulation.duration * tau
else:
    t_end = config.simulation.duration


steps = Steps(t_end=t_end, dt=dt)
ns = steps.simulation_steps()
saves = steps.steps_from_total(config.io.results.quantity)
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
        if config.io.results.save and save:
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}]",
                trigger_level=1,
            )
            directory = config.io.results.directory
            name = config.model.name_sc
            model.io.save(directory.joinpath(f"{prefix}{n}.npz"))

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
