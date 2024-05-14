"""Run a single model with a Vortex forcing."""

from pathlib import Path

import torch
from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.wind import WindForcing
from qgsw.mesh import Meshes3D
from qgsw.models import QG
from qgsw.perturbations import Perturbation
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
CONFIG_PATH = ROOT_PATH.joinpath("config/single_model_vortex.toml")
config = Configuration.from_file(CONFIG_PATH)
summary = RunSummary.from_configuration(config)

if config.io.plots.save:
    save_file = config.io.plots.directory.joinpath("_summary.toml")
    summary.to_file(save_file)
if config.io.results.save:
    save_file = config.io.results.directory.joinpath("_summary.toml")
    summary.to_file(save_file)

if config.model.type != "QG":
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

# Model Set-up
## Vortex
perturbation = Perturbation.from_config(
    perturbation_config=config.perturbation,
)
## Mesh
mesh = Meshes3D.from_config(config.mesh, config.model)
# "" Coriolis
f = coriolis.compute_beta_plane(
    mesh=mesh.omega.remove_z_h(),
    f0=config.physics.f0,
    beta=config.physics.beta,
)
## Compute Burger Number
Bu = compute_burger(
    g=config.model.g_prime[0],
    h_scale=config.model.h[0],
    f0=config.physics.f0,
    length_scale=perturbation.compute_scale(mesh.omega),
)
verbose.display(
    msg=f"Burger Number: {Bu:.2f}",
    trigger_level=1,
)
## Set model parameters
params = {
    "nx": config.mesh.nx,
    "ny": config.mesh.ny,
    "nl": config.model.nl,
    "dx": config.mesh.dx,
    "dy": config.mesh.dy,
    "H": mesh.h.xyh[2],
    "g_prime": config.model.g_prime.unsqueeze(1).unsqueeze(1),
    "f": f,
    "taux": taux,
    "tauy": tauy,
    "bottom_drag_coef": config.physics.bottom_drag_coef,
    "device": DEVICE,
    "dtype": torch.float64,
    "mask": torch.ones_like(mesh.h.remove_z_h().xy[0]),
    "compile": True,
    "slip_coef": config.physics.slip_coef,
    "dt": 0.0,
}
model = QG(params)
p0 = perturbation.compute_initial_pressure(mesh.omega, config.physics.f0, Ro)
u0, v0, h0 = model.G(p0)

## Max speed
u_max, v_max, c = (
    torch.abs(u0).max().item() / config.mesh.dx,
    torch.abs(v0).max().item() / config.mesh.dy,
    torch.sqrt(config.model.g_prime[0] * config.model.h.sum()),
)
## Timestep
dt = min(
    cfl_adv * config.mesh.dx / u_max,
    cfl_adv * config.mesh.dy / v_max,
    cfl_gravity * config.mesh.dx / c,
)

# Instantiate Model
model.dt = dt
model.u = torch.clone(u0)
model.v = torch.clone(v0)
model.h = torch.clone(h0)

## time params
t = 0

model.compute_diagnostic_variables()
model.compute_time_derivatives()

w_0 = model.omega.squeeze() / model.dx / model.dy


tau = 1.0 / torch.sqrt(w_0.pow(2).mean()).to(device=DEVICE).item()
verbose.display(
    msg=f"tau = {tau *config.physics.f0:.2f} f0⁻¹",
    trigger_level=1,
)

t_end = config.simulation.duration * tau
freq_plot = int(t_end / config.io.plots.quantity / dt) + 1
freq_save = int(t_end / config.io.results.quantity / dt) + 1
freq_log = int(t_end / 100 / dt) + 1
n_steps = int(t_end / dt) + 1

summary.register_steps(t_end=t_end, dt=dt.cpu().item(), n_steps=n_steps)

plots_required = config.io.plots.save or config.io.plots.show

verbose.display(msg=f"Total Duration: {t_end:.2f}", trigger_level=1)


# Instantiate Figures
qg_top_axes = SurfaceVorticityAxes.from_mask()
qg_top_axes.set_title(r"$\omega_{QG-TOP}$")
qg_inf_axes = SecondLayerVorticityAxes.from_mask()
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
        model.save_omega(directory.joinpath(f"{prefix}{n}.npz"))
    model.step()

    t += dt

    if (freq_log > 0 and n % freq_log == 0) or (n == n_steps):
        verbose.display(
            msg=f"QG-1l: n={n:05d}, {model.get_print_info()}",
            trigger_level=1,
        )
        summary.register_step(n)
summary.register_end()
