"""Run a single model with a Vortex forcing."""

import argparse
from pathlib import Path

import numpy as np
import torch

from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.wind import WindForcing
from qgsw.models.qg.instantiation import instantiate_model
from qgsw.perturbations import Perturbation
from qgsw.spatial.dim_3 import SpaceDiscretization3D
from qgsw.utils import time_params

torch.backends.cudnn.deterministic = True
verbose.set_level(0)

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

if np.isnan(config.simulation.tau):
    tau = time_params.compute_tau(model.omega, model.space)
else:
    tau = config.simulation.tau / config.physics.f0

t_end = config.simulation.duration * tau
freq_plot = int(t_end / config.io.plots.quantity / dt) + 1
freq_save = int(t_end / config.io.results.quantity / dt) + 1
freq_log = int(t_end / 100 / dt) + 1
n_steps = int(t_end / dt) + 1

# Start runs
for _ in range(n_steps + 1):
    model.step()
    t += dt
