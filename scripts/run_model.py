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
from qgsw.physics import compute_burger
from qgsw.plots.vorticity import (
    SecondLayerVorticityAxes,
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)
from qgsw.run_summary import RunSummary
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

if config.io.plots.save:
    save_file = config.io.plots.directory.joinpath("_summary.toml")
    summary.to_file(save_file)
if config.io.results.save:
    save_file = config.io.results.directory.joinpath("_summary.toml")
    summary.to_file(save_file)

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

if np.isnan(config.simulation.tau):
    tau = time_params.compute_tau(model.omega, model.space)
else:
    tau = config.simulation.tau / config.physics.f0
verbose.display(
    msg=f"tau = {tau *config.physics.f0:.2f} f0⁻¹",
    trigger_level=1,
)

t_end = config.simulation.duration * tau
freq_plot = int(t_end / config.io.plots.quantity / dt) + 1
freq_save = int(t_end / config.io.results.quantity / dt) + 1
freq_log = int(t_end / 100 / dt) + 1
n_steps = int(t_end / dt) + 1

summary.register_steps(t_end=t_end, dt=dt, n_steps=n_steps)

plots_required = config.io.plots.save or config.io.plots.show

verbose.display(msg=f"Total Duration: {t_end:.2f}", trigger_level=1)


# Instantiate Figures
qg_top_axes = SurfaceVorticityAxes.from_kwargs()
qg_top_axes.set_title(r"$\omega_{QG-TOP}$")
qg_inf_axes = SecondLayerVorticityAxes.from_kwargs()
qg_inf_axes.set_title(r"$\omega_{QG-INF}$")
plot = VorticityComparisonFigure(qg_top_axes, qg_inf_axes, common_cbar=False)

summary.register_start()
prefix = config.model.prefix
# Start runs
for n in range(n_steps + 1):
    if plots_required and (n % freq_plot == 0 or n == n_steps):
        plot.figure.suptitle(
            f"Ro={Ro:.2f}, Bu={Bu:.2f} t={t/tau:.2f}$\\tau$",
        )
        plot.update_with_models(model, model)
        if config.io.plots.show:
            plot.show()
        if config.io.plots.save:
            output_dir = config.io.plots.directory
            output_name = Path(f"{config.io.name_sc}_{n}.png")
            plot.savefig(output_dir.joinpath(output_name))

    if config.io.results.save and (n % freq_save == 0 or n == n_steps):
        directory = config.io.results.directory
        name = config.model.name_sc
        model.save_uvhwp(directory.joinpath(f"{prefix}{n}.npz"))
    model.step()

    t += dt

    if (freq_log > 0 and n % freq_log == 0) or (n == n_steps):
        verbose.display(
            msg=f"QG-1l: n={n:05d}, {model.get_print_info()}",
            trigger_level=1,
        )
        summary.register_step(n)
summary.register_end()
