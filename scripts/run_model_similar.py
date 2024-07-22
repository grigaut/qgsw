"""Run a single model with a Vortex forcing."""

import argparse
from pathlib import Path

import numpy as np
import torch
from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.wind import WindForcing
from qgsw.models import QG, QGCollinearSublayerStreamFunction
from qgsw.models.qg.alpha import coefficient_from_config
from qgsw.models.qg.collinear_sublayer import (
    QGCollinearSublayerPV,
    QGCollinearSublayerSFModifiedA,
    QGPVMixture,
)
from qgsw.models.qg.core import G, compute_A
from qgsw.models.variables import UVH
from qgsw.perturbations.base import _Perturbation, grad_perp
from qgsw.perturbations.vortex import (
    BaroclinicVortex,
    BarotropicVortex,
    HalfBarotropicVortex,
)
from qgsw.physics import compute_burger
from qgsw.plots.vorticity import (
    SecondLayerVorticityAxes,
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)
from qgsw.run_summary import RunSummary
from qgsw.spatial.dim_3 import SpaceDiscretization3D
from qgsw.specs import DEVICE
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
perturbations = {
    BaroclinicVortex.get_type(): BaroclinicVortex,
    BarotropicVortex.get_type(): BarotropicVortex,
    HalfBarotropicVortex.get_type(): HalfBarotropicVortex,
}
if config.perturbation.type not in perturbations:
    msg = (
        "Unrecognized perturbation type. "
        f"Possible values are {perturbations.keys()}"
    )
    raise KeyError(msg)
perturbation: _Perturbation = perturbations[config.perturbation.type](
    magnitude=config.perturbation.perturbation_magnitude,
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

psi = torch.ones(
    (1, 2, space.omega.nx, space.omega.ny),
    device=DEVICE,
    dtype=torch.float64,
)
psi_2d = perturbation._compute_streamfunction_2d(space.omega.remove_z_h())  # noqa: SLF001
psi[0, 0, ...] = psi_2d
for i in range(1, 2):
    psi_2d = perturbation._compute_streamfunction_2d(space.omega.remove_z_h())  # noqa: SLF001
    psi[0, i, ...] = perturbation.layer_ratio * psi_2d

u, v = grad_perp(
    psi,
    space.omega.dx,
    space.omega.dy,
)
u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
# set psi amplitude to have a correct Rossby number
psi = psi * (
    Ro
    * config.physics.f0
    * perturbation.compute_scale(space.omega)
    / u_norm_max
)

p0 = psi * config.physics.f0

if space.h.xyh.h.shape[0] > 2:  # noqa: PLR2004
    msg = "Only 1 and 2 layers models are supported."
    raise ValueError(msg)
A = compute_A(
    torch.tensor([200, 800]).to(dtype=torch.float64, device=DEVICE),
    torch.tensor([10.0, 0.05]).to(dtype=torch.float64, device=DEVICE),
    torch.float64,
    DEVICE,
)
uvh0 = G(
    p0,
    space,
    space.h.xyh.h,
    A,
    config.physics.f0,
)

if config.model.type == "QG":
    model = QG(
        space_3d=space,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
    )
    uvh0 = UVH(
        uvh0.u[: space.nl, ...],
        uvh0.v[: space.nl, ...],
        uvh0.h[: space.nl, ...],
    )
elif config.model.type == "QGCollinearSublayerStreamFunction":
    model = QGCollinearSublayerStreamFunction(
        space_3d=space,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
        coefficient=coefficient_from_config(config.model.collinearity_coef),
    )
    uvh0 = UVH(
        uvh0.u[:1, ...],
        uvh0.v[:1, ...],
        uvh0.h[:1, ...],
    )
elif config.model.type == "QGCollinearSublayerPV":
    model = QGCollinearSublayerPV(
        space_3d=space,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
        coefficient=coefficient_from_config(config.model.collinearity_coef),
    )
    uvh0 = UVH(
        uvh0.u[:1, ...],
        uvh0.v[:1, ...],
        uvh0.h[:1, ...],
    )
elif config.model.type == "QGPVMixture":
    model = QGPVMixture(
        space_3d=space,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
        coefficient=coefficient_from_config(config.model.collinearity_coef),
    )
    uvh0 = UVH(
        uvh0.u[:1, ...],
        uvh0.v[:1, ...],
        uvh0.h[:1, ...],
    )
elif config.model.type == "QGCollinearSublayerSFModifiedA":
    model = QGCollinearSublayerSFModifiedA(
        space_3d=space,
        g_prime=config.model.g_prime.unsqueeze(1).unsqueeze(1),
        beta_plane=config.physics.beta_plane,
        coefficient=coefficient_from_config(config.model.collinearity_coef),
    )
    uvh0 = UVH(
        uvh0.u[:1, ...],
        uvh0.v[:1, ...],
        uvh0.h[:1, ...],
    )

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
t = 0

model.compute_diagnostic_variables(model.uvh)
model.compute_time_derivatives(model.uvh)

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
