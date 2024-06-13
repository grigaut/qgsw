"""Comparison of a single-layer vs a double-layer model under vortex shear."""

from pathlib import Path

import torch
from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.wind import WindForcing
from qgsw.models import QG
from qgsw.models.qg.colinear_sublayer import (
    QGColinearSublayerStreamFunction,
)
from qgsw.perturbations import Perturbation
from qgsw.physics import compute_burger
from qgsw.plots.vorticity import (
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)
from qgsw.run_summary import RunSummary
from qgsw.spatial.dim_3 import SpaceDiscretization3D
from qgsw.specs import DEVICE

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

ROOT_PATH = Path(__file__).parent.parent
CONFIG_PATH = ROOT_PATH.joinpath("config/compare_models.toml")
config = Configuration.from_file(CONFIG_PATH)
summary = RunSummary.from_configuration(config)

if config.io.plots.save:
    save_file = config.io.plots.directory.joinpath("_summary.toml")
    summary.to_file(save_file)
if config.io.results.save:
    save_file = config.io.results.directory.joinpath("_summary.toml")
    summary.to_file(save_file)

possible_models = ["QG", "QGColinearSublayerStreamFunction"]

supported_1 = config.models[0].type in possible_models
supported_2 = config.models[1].type in possible_models
supported_3 = config.models[2].type in possible_models

if not all([supported_1, supported_2, supported_3]):
    msg = f"Unsupported model type, possible values are: {possible_models}."
    raise ValueError(msg)

# Common Set-up
## Wind Forcing
wind = WindForcing.from_config(config.windstress, config.space, config.physics)
taux, tauy = wind.compute()
## Rossby
Ro = 0.1

cfl_adv = 0.5
cfl_gravity = 0.5

## Perturbation
perturbation = Perturbation.from_config(
    perturbation_config=config.perturbation,
)

# Single Layer Set-up
prefix_1 = config.models[0].prefix
## Grid
space_1 = SpaceDiscretization3D.from_config(config.space, config.models[0])
## Compute Burger Number
Bu_1 = compute_burger(
    g=config.models[0].g_prime[0],
    h_scale=config.models[0].h[0],
    f0=config.physics.f0,
    length_scale=perturbation.compute_scale(space_1.omega),
)
verbose.display(
    msg=f"Single-Layer Burger Number: {Bu_1:.2f}",
    trigger_level=1,
)
## Set model parameters
qg_1 = QG(
    space_3d=space_1,
    g_prime=config.models[0].g_prime.unsqueeze(1).unsqueeze(1),
    beta_plane=config.physics.beta_plane,
)
qg_1.slip_coef = config.physics.slip_coef
qg_1.bottom_drag_coef = config.physics.bottom_drag_coef
qg_1.set_wind_forcing(taux, tauy)
p0_1 = perturbation.compute_initial_pressure(
    space_1.omega,
    config.physics.f0,
    Ro,
)
u0_1, v0_1, h0_1 = qg_1.G(p0_1)

## Max speed
u_max_1, v_max_1, c_1 = (
    torch.abs(u0_1).max().item() / config.space.dx,
    torch.abs(v0_1).max().item() / config.space.dy,
    torch.sqrt(config.models[0].g_prime[0] * config.models[0].h.sum()),
)
## Timestep
dt_1 = min(
    cfl_adv * config.space.dx / u_max_1,
    cfl_adv * config.space.dy / v_max_1,
    cfl_gravity * config.space.dx / c_1,
)

# Two Layers Set-up
prefix_2 = config.models[1].prefix
## Grid
space_2 = SpaceDiscretization3D.from_config(config.space, config.models[1])
## Compute Burger Number
h1 = config.models[1].h[0]
h2 = config.models[1].h[1]
h_eq = (h1 * h2) / (h1 + h2)

Bu_2 = compute_burger(
    g=config.models[1].g_prime[0],
    h_scale=h_eq,
    f0=config.physics.f0,
    length_scale=perturbation.compute_scale(space_2.omega),
)
verbose.display(
    msg=f"Colinear-Layers Burger Number: {Bu_2:.2f}",
    trigger_level=1,
)
## Set model parameters
qg_2 = QGColinearSublayerStreamFunction(
    space_3d=space_2,
    g_prime=config.models[1].g_prime.unsqueeze(1).unsqueeze(1),
    beta_plane=config.physics.beta_plane,
)
qg_2.slip_coef = config.physics.slip_coef
qg_2.bottom_drag_coef = config.physics.bottom_drag_coef
qg_2.set_wind_forcing(taux, tauy)
p0_2 = perturbation.compute_initial_pressure(
    space_2.omega,
    config.physics.f0,
    Ro,
)
u0_2, v0_2, h0_2 = qg_2.G(p0_2)
## Max Speed
u_max_2, v_max_2, c_2 = (
    torch.abs(u0_2).max().item() / config.space.dx,
    torch.abs(v0_2).max().item() / config.space.dy,
    torch.sqrt(config.models[1].g_prime[0] * config.models[1].h.sum()),
)
## Timestep
dt_2 = min(
    cfl_adv * config.space.dx / u_max_2,
    cfl_adv * config.space.dy / v_max_2,
    cfl_gravity * config.space.dx / c_2,
)

# Two Layers Set-up
prefix_3 = config.models[2].prefix
## Grid
space_3 = SpaceDiscretization3D.from_config(config.space, config.models[2])
## Compute Burger Number
h1 = config.models[2].h[0]
h2 = config.models[2].h[1]
h_eq = (h1 * h2) / (h1 + h2)

Bu_3 = compute_burger(
    g=config.models[2].g_prime[0],
    h_scale=h_eq,
    f0=config.physics.f0,
    length_scale=perturbation.compute_scale(space_3.omega),
)
verbose.display(
    msg=f"Two-Layers Burger Number: {Bu_3:.2f}",
    trigger_level=1,
)
## Set model parameters
qg_3 = QGColinearSublayerStreamFunction(
    space_3d=space_3,
    g_prime=config.models[2].g_prime.unsqueeze(1).unsqueeze(1),
    beta_plane=config.physics.beta_plane,
)
qg_3.slip_coef = config.physics.slip_coef
qg_3.bottom_drag_coef = config.physics.bottom_drag_coef
qg_3.set_wind_forcing(taux, tauy)
p0_3 = perturbation.compute_initial_pressure(
    space_3.omega,
    config.physics.f0,
    Ro,
)
u0_3, v0_3, h0_3 = qg_3.G(p0_3)
## Max Speed
u_max_3, v_max_3, c_3 = (
    torch.abs(u0_3).max().item() / config.space.dx,
    torch.abs(v0_3).max().item() / config.space.dy,
    torch.sqrt(config.models[2].g_prime[0] * config.models[2].h.sum()),
)
## Timestep
dt_3 = min(
    cfl_adv * config.space.dx / u_max_3,
    cfl_adv * config.space.dy / v_max_3,
    cfl_gravity * config.space.dx / c_3,
)

# Instantiate Models
dt = min(dt_1, dt_2, dt_3)
qg_1.dt = dt
qg_1.set_uvh(
    u=torch.clone(u0_1),
    v=torch.clone(v0_1),
    h=torch.clone(h0_1),
)

qg_2.dt = dt
qg_2.set_uvh(
    u=torch.clone(u0_2),
    v=torch.clone(v0_2),
    h=torch.clone(h0_2),
)

qg_3.dt = dt
qg_3.set_uvh(
    u=torch.clone(u0_3),
    v=torch.clone(v0_3),
    h=torch.clone(h0_3),
)

## time params
t = 0

w_0_1 = qg_1.omega.squeeze() / qg_1.space.dx / qg_1.space.dy


tau_1 = 1.0 / torch.sqrt(w_0_1.pow(2).mean()).to(device=DEVICE).item()
verbose.display(
    msg=f"tau (single layer) = {tau_1 *config.physics.f0:.2f} f0-1",
    trigger_level=1,
)

w_0_2 = qg_2.omega.squeeze() / qg_2.space.dx / qg_2.space.dy


tau_2 = 1.0 / torch.sqrt(w_0_2.pow(2).mean()).to(device=DEVICE).item()
verbose.display(
    msg=f"tau (colinear layer) = {tau_2 *config.physics.f0:.2f} f0-1",
    trigger_level=1,
)

w_0_3 = qg_3.omega.squeeze() / qg_3.space.dx / qg_3.space.dy


tau_3 = 1.0 / torch.sqrt(w_0_3.pow(2).mean()).to(device=DEVICE).item()
verbose.display(
    msg=f"tau (two layers) = {tau_3 *config.physics.f0:.2f} f0-1",
    trigger_level=1,
)

tau = max(tau_1, tau_2, tau_3)
verbose.display(
    msg=f"tau = {tau *config.physics.f0:.2f} f0-1",
    trigger_level=1,
)

t_end = config.simulation.duration * tau
freq_plot = int(t_end / config.io.plots.quantity / dt) + 1
freq_save = int(t_end / config.io.results.quantity / dt) + 1
freq_checknan = 100
freq_log = int(t_end / 100 / dt) + 1
n_steps = int(t_end / dt) + 1

summary.register_steps(t_end=t_end, dt=dt.cpu().item(), n_steps=n_steps)

plots_required = config.io.plots.save or config.io.plots.show

verbose.display(msg=f"Total Duration: {t_end:.2f}", trigger_level=1)


# Instantiate Figures
qg_1_axes = SurfaceVorticityAxes.from_kwargs()
qg_1_axes.set_title(r"$\omega_{QG-1L-TOP}$")
qg_2_top_axes = SurfaceVorticityAxes.from_kwargs()
qg_2_top_axes.set_title(r"$\omega_{QG-CL-TOP}$")
qg_3_top_axes = SurfaceVorticityAxes.from_kwargs()
qg_3_top_axes.set_title(r"$\omega_{QG-2L-TOP}$")
plot = VorticityComparisonFigure(qg_1_axes, qg_2_top_axes, qg_3_top_axes)

summary.register_start()
# Start runs
for n in range(n_steps + 1):
    if plots_required and (n % freq_plot == 0 or n == n_steps):
        plot.figure.suptitle(
            f"Ro={Ro:.2f}, Bu_1={Bu_1:.2f}, Bu_2={Bu_2:.2f},"
            f"Bu_3={Bu_3:.2f}, t={t/tau:.2f}$\\tau$",
        )
        plot.update_with_models(qg_1, qg_2, qg_3)
        if config.io.plots.show:
            plot.show()
        if config.io.plots.save:
            output_dir = config.io.plots.directory
            output_name = Path(f"{config.io.name_sc}_{n}.png")
            plot.savefig(output_dir.joinpath(output_name))

    if config.io.results.save and (n % freq_save == 0 or n == n_steps):
        directory = config.io.results.directory
        qg_1.save_omega(directory.joinpath(f"{prefix_1}{n}.npz"))
        qg_2.save_omega(directory.joinpath(f"{prefix_2}{n}.npz"))
        qg_3.save_omega(directory.joinpath(f"{prefix_3}{n}.npz"))
    qg_1.step()
    qg_2.step()
    qg_3.step()
    t += dt

    if (freq_log > 0 and n % freq_log == 0) or (n == n_steps):
        verbose.display(
            msg=f"QG-1L: n={n:05d}, {qg_1.get_print_info()}",
            trigger_level=1,
        )
        verbose.display(
            msg=f"QG-CL: n={n:05d}, {qg_2.get_print_info()}",
            trigger_level=1,
        )
        verbose.display(
            msg=f"QG-2L: n={n:05d}, {qg_3.get_print_info()}",
            trigger_level=1,
        )
        summary.register_step(n)
summary.register_end()
