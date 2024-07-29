"""Run a single model with a Vortex forcing."""

import argparse
from pathlib import Path

import numpy as np
import torch
from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.wind import WindForcing
from qgsw.models import QG, QGCollinearSublayerStreamFunction
from qgsw.models.base import Model
from qgsw.models.qg.alpha import coefficient_from_config
from qgsw.models.qg.collinear_sublayer import (
    QGCollinearSublayerPV,
    QGCollinearSublayerSFModifiedA,
    QGPVMixture,
)
from qgsw.perturbations import Perturbation
from qgsw.physics import compute_burger
from qgsw.run_summary import RunSummary
from qgsw.spatial.core.discretization import keep_top_layer
from qgsw.spatial.dim_3 import SpaceDiscretization3D
from qgsw.specs import DEVICE
from qgsw.utils import time_params

torch.backends.cudnn.deterministic = True
verbose.set_level(0)
DEVICE.set_manually("cpu")

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

if config.io.plots.save:
    save_file = config.io.plots.directory.joinpath("_summary.toml")
    summary.to_file(save_file)
if config.io.results.save:
    save_file = config.io.results.directory.joinpath("_summary.toml")
    summary.to_file(save_file)

supported_models = [
    "QG",
    "QGCollinearSublayerStreamFunction",
    "QGCollinearSublayerPV",
    "QGPVMixture",
    "QGCollinearSublayerSFModifiedA",
]

if config.model.type not in supported_models:
    msg = "Unsupported model type, possible values are: QG."
    raise ValueError(msg)

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

if config.model.type == "QG":
    model = QG(
        space_3d=space,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
    )
    p0 = perturbation.compute_initial_pressure(
        space.omega,
        config.physics.f0,
        Ro,
    )
    uvh0 = model.G(p0)
else:
    p0 = perturbation.compute_initial_pressure(
        keep_top_layer(space).omega,
        config.physics.f0,
        Ro,
    )
    # Initial coefficient for initial prognostic variables set up
    if perturbation.type == "vortex-baroclinic":
        coef0 = 0
    elif perturbation.type == "vortex-half-barotropic":
        coef0 = 0.5
    elif perturbation.type == "vortex-barotropic":
        coef0 = 1
    else:
        msg = f"Unknown perturbation type: {perturbation.type}"
        raise ValueError(msg)
    # Instantiate model
    if config.model.type == "QGCollinearSublayerStreamFunction":
        model = QGCollinearSublayerStreamFunction(
            space_3d=space,
            g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
            beta_plane=config.physics.beta_plane,
            coefficient=coef0,
        )
    elif config.model.type == "QGCollinearSublayerPV":
        model = QGCollinearSublayerPV(
            space_3d=space,
            g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
            beta_plane=config.physics.beta_plane,
            coefficient=coef0,
        )
    elif config.model.type == "QGPVMixture":
        model = QGPVMixture(
            space_3d=space,
            g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
            beta_plane=config.physics.beta_plane,
            coefficient=coef0,
        )
    elif config.model.type == "QGCollinearSublayerSFModifiedA":
        model = QGCollinearSublayerSFModifiedA(
            space_3d=space,
            g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
            beta_plane=config.physics.beta_plane,
            coefficient=coef0,
        )
    # Project Model
    uvh0 = model.G(p0)
    # Set model coefficient
    model.coefficient = coefficient_from_config(config.model.collinearity_coef)

model.slip_coef = config.physics.slip_coef
model.bottom_drag_coef = config.physics.bottom_drag_coef
model.set_wind_forcing(taux, tauy)

if np.isnan(config.simulation.dt):
    dt = time_params.compute_dt(uvh0, model.space, model.g_prime, model.H)
else:
    dt = config.simulation.dt

# Instantiate Model
model.dt = dt
model.set_uvh(
    u=torch.clone(uvh0.u),
    v=torch.clone(uvh0.v),
    h=torch.clone(uvh0.h),
)

## time params
model.compute_diagnostic_variables(model.uvh)
model.compute_time_derivatives(model.uvh)

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


def run_model(model: Model, n_steps: int, t: float) -> None:
    """Run the model.

    Args:
        model (Model): Model to run
        n_steps (int): Number of steps
        t (float): Time
    """
    for _ in range(n_steps + 1):
        model.step()

        t += dt


run_model(model, n_steps, 0)
