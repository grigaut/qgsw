"""Comparison of a single-layer vs a double-layer model under vortex shear."""

from pathlib import Path

import torch
from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.vortex import RankineVortexForcing
from qgsw.forcing.wind import WindForcing
from qgsw.mesh import Meshes3D
from qgsw.models import QG
from qgsw.physics import compute_burger, coriolis
from qgsw.plots.vorticity import (
    SecondLayerVorticityAxes,
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)
from qgsw.run_summary import RunSummary
from qgsw.specs import DEVICE

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

ROOT_PATH = Path(__file__).parent.parent
CONFIG_PATH = ROOT_PATH.joinpath("config/single_vs_double_layers.toml")
config = Configuration.from_file(CONFIG_PATH)
summary = RunSummary.from_configuration(config)

if config.io.plots.save:
    save_file = config.io.plots.directory.joinpath("_summary.toml")
    summary.to_file(save_file)
if config.io.results.save:
    save_file = config.io.results.directory.joinpath("_summary.toml")
    summary.to_file(save_file)

if config.models[0].type != "QG" or config.models[1].type != "QG":
    msg = "Unsupported model type, possible values are: QG."
    raise ValueError(msg)

# Common Set-up
## Wind Forcing
wind = WindForcing.from_config(config.windstress, config.mesh, config.physics)
taux, tauy = wind.compute()
## Rossby
Ro = 0.1

cfl_adv = 0.5
cfl_gravity = 0.5

# Single Layer Set-up
prefix_1l = config.models[0].prefix
## Vortex
vortex_1l = RankineVortexForcing.from_config(
    vortex_config=config.vortex,
    mesh_config=config.mesh,
    model_config=config.models[0],
)
## Mesh
mesh_1l = Meshes3D.from_config(config.mesh, config.models[0])
# "" Coriolis
f = coriolis.compute_beta_plane(
    mesh=mesh_1l.omega.remove_z_h(),
    f0=config.physics.f0,
    beta=config.physics.beta,
)
## Compute Burger Number
Bu_1l = compute_burger(
    g=config.models[0].g_prime[0],
    h_scale=config.models[0].h[0],
    f0=config.physics.f0,
    length_scale=vortex_1l.r0,
)
verbose.display(
    msg=f"Single-Layer Burger Number: {Bu_1l:.2f}",
    trigger_level=1,
)
## Set model parameters
params_1l = {
    "nx": config.mesh.nx,
    "ny": config.mesh.ny,
    "nl": config.models[0].nl,
    "dx": config.mesh.dx,
    "dy": config.mesh.dy,
    "H": mesh_1l.h.xyh[2],
    "g_prime": config.models[0].g_prime.unsqueeze(1).unsqueeze(1),
    "f": f,
    "taux": taux,
    "tauy": tauy,
    "bottom_drag_coef": config.physics.bottom_drag_coef,
    "device": DEVICE,
    "dtype": torch.float64,
    "mask": torch.ones_like(mesh_1l.h.remove_z_h().xy[0]),
    "compile": True,
    "slip_coef": 1.0,
    "dt": 0.0,
}
qg_1l = QG(params_1l)
u0_1l, v0_1l, h0_1l = qg_1l.G(vortex_1l.compute(config.physics.f0, Ro))

## Max speed
u_max_1l, v_max_1l, c_1l = (
    torch.abs(u0_1l).max().item() / config.mesh.dx,
    torch.abs(v0_1l).max().item() / config.mesh.dy,
    torch.sqrt(config.models[0].g_prime[0] * config.models[0].h.sum()),
)
## Timestep
dt_1l = min(
    cfl_adv * config.mesh.dx / u_max_1l,
    cfl_adv * config.mesh.dy / v_max_1l,
    cfl_gravity * config.mesh.dx / c_1l,
)

# Two Layers Set-up
prefix_2l = config.models[1].prefix
## Vortex
vortex_2l = RankineVortexForcing.from_config(
    vortex_config=config.vortex,
    mesh_config=config.mesh,
    model_config=config.models[1],
)
## Mesh
mesh_2l = Meshes3D.from_config(config.mesh, config.models[1])
## Coriolis
f = coriolis.compute_beta_plane(
    mesh=mesh_2l.omega.remove_z_h(),
    f0=config.physics.f0,
    beta=config.physics.beta,
)
## Compute Burger Number
h1 = config.models[1].h[0]
h2 = config.models[1].h[1]
h_eq = (h1 * h2) / (h1 + h2)

Bu_2l = compute_burger(
    g=config.models[1].g_prime[0],
    h_scale=h_eq,
    f0=config.physics.f0,
    length_scale=vortex_2l.r0,
)
verbose.display(
    msg=f"Multi-Layers Burger Number: {Bu_2l:.2f}",
    trigger_level=1,
)
## Set model parameters
params_2l = {
    "nx": config.mesh.nx,
    "ny": config.mesh.ny,
    "nl": config.models[1].nl,
    "dx": config.mesh.dx,
    "dy": config.mesh.dy,
    "H": mesh_2l.h.xyh[2],
    "g_prime": config.models[1].g_prime.unsqueeze(1).unsqueeze(1),
    "f": f,
    "taux": taux,
    "tauy": tauy,
    "bottom_drag_coef": config.physics.bottom_drag_coef,
    "device": DEVICE,
    "dtype": torch.float64,
    "mask": torch.ones_like(mesh_2l.h.remove_z_h().xy[0]),
    "compile": True,
    "slip_coef": 1.0,
    "dt": 0.0,
}
qg_2l = QG(params_2l)
u0_2l, v0_2l, h0_2l = qg_2l.G(vortex_2l.compute(config.physics.f0, Ro))
## Max Speed
u_max_2l, v_max_2l, c_2l = (
    torch.abs(u0_2l).max().item() / config.mesh.dx,
    torch.abs(v0_2l).max().item() / config.mesh.dy,
    torch.sqrt(config.models[1].g_prime[0] * config.models[1].h.sum()),
)
## Timestep
dt_2l = min(
    cfl_adv * config.mesh.dx / u_max_2l,
    cfl_adv * config.mesh.dy / v_max_2l,
    cfl_gravity * config.mesh.dx / c_2l,
)


# Instantiate Models
dt = min(dt_1l, dt_2l)
qg_1l.dt = dt
qg_1l.u = torch.clone(u0_1l)
qg_1l.v = torch.clone(v0_1l)
qg_1l.h = torch.clone(h0_1l)
qg_2l.dt = dt
qg_2l.u = torch.clone(u0_2l)
qg_2l.v = torch.clone(v0_2l)
qg_2l.h = torch.clone(h0_2l)

## time params
t = 0

qg_1l.compute_diagnostic_variables()
qg_1l.compute_time_derivatives()

w_0_1l = qg_1l.omega.squeeze() / qg_1l.dx / qg_1l.dy


tau_1l = 1.0 / torch.sqrt(w_0_1l.pow(2).mean()).to(device=DEVICE).item()
verbose.display(
    msg=f"tau (single layer) = {tau_1l *config.physics.f0:.2f} f0-1",
    trigger_level=1,
)

qg_2l.compute_diagnostic_variables()
qg_2l.compute_time_derivatives()

w_0_2l = qg_2l.omega.squeeze() / qg_2l.dx / qg_2l.dy


tau_2l = 1.0 / torch.sqrt(w_0_2l.pow(2).mean()).to(device=DEVICE).item()
verbose.display(
    msg=f"tau (multi layer) = {tau_2l *config.physics.f0:.2f} f0-1",
    trigger_level=1,
)

tau = max(tau_1l, tau_2l)
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
mask = qg_1l.masks.not_w[0, 0].cpu().numpy()
qg_1l_axes = SurfaceVorticityAxes.from_mask(mask=mask)
qg_1l_axes.set_title(r"$\omega_{QG-1L-TOP}$")
qg_2l_top_axes = SurfaceVorticityAxes.from_mask(mask=mask)
qg_2l_top_axes.set_title(r"$\omega_{QG-ML-TOP}$")
qg_2l_inf_axes = SecondLayerVorticityAxes.from_mask(mask=mask)
qg_2l_inf_axes.set_title(r"$\omega_{QG-ML-INF}$")
plot = VorticityComparisonFigure(qg_1l_axes, qg_2l_top_axes, qg_2l_inf_axes)

summary.register_start()
# Start runs
for n in range(n_steps + 1):
    if plots_required and (n % freq_plot == 0 or n == n_steps):
        plot.figure.suptitle(
            f"Ro={Ro:.2f}, Bu_1l={Bu_1l:.2f}, Bu_2l={Bu_2l:.2f},"
            f" t={t/tau:.2f}$\\tau$",
        )
        plot.update_with_models(qg_1l, qg_2l, qg_2l)
        if config.io.plots.show:
            plot.show()
        if config.io.plots.save:
            output_dir = config.io.plots.directory
            output_name = Path(f"{config.io.name_sc}_{n}.png")
            plot.savefig(output_dir.joinpath(output_name))

    if config.io.results.save and (n % freq_save == 0 or n == n_steps):
        directory = config.io.results.directory
        qg_1l.save_omega(directory.joinpath(f"{prefix_1l}{n}.npz"))
        qg_2l.save_omega(directory.joinpath(f"{prefix_2l}{n}.npz"))
    qg_1l.step()
    qg_2l.step()
    t += dt

    if (freq_log > 0 and n % freq_log == 0) or (n == n_steps):
        verbose.display(
            msg=f"QG-1l: n={n:05d}, {qg_1l.get_print_info()}",
            trigger_level=1,
        )
        verbose.display(
            msg=f"QG_ML: n={n:05d}, {qg_2l.get_print_info()}",
            trigger_level=1,
        )
        summary.register_step(n)
summary.register_end()
