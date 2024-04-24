"""Comparison of a single-layer vs a double-layer model under vortex shear."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from qgsw import verbose
from qgsw.configs import VortexShearConfig
from qgsw.forcing.vortex import RankineVortexForcing
from qgsw.forcing.wind import WindForcing
from qgsw.mesh import Meshes3D
from qgsw.models import QG
from qgsw.physics import compute_burger, coriolis
from qgsw.specs import DEVICE

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

config_1l = VortexShearConfig.from_file(
    "config/single_vs_double_layers/single_layer.toml"
)
config_2l = VortexShearConfig.from_file(
    "config/single_vs_double_layers/two_layers.toml"
)

# Common Set-up
## Wind Forcing
wind = WindForcing.from_config(config_1l)
taux, tauy = wind.compute()
## Rossby
Ro = 0.1

cfl_adv = 0.5
cfl_gravity = 0.5

# Single Layer Set-up
## Vortex
vortex_1l = RankineVortexForcing.from_config(config_1l)
## Mesh
mesh_1l = Meshes3D.from_config(config_1l)
# "" Coriolis
f = coriolis.compute_beta_plane(
    mesh=mesh_1l.omega.remove_z_h(),
    f0=config_1l.physics.f0,
    beta=config_1l.physics.beta,
)
## Compute Burger Number
Bu_1l = compute_burger(
    g=config_1l.layers.g_prime[0],
    h_scale=config_1l.layers.h[0],
    f0=config_1l.physics.f0,
    length_scale=vortex_1l.r0,
)
verbose.display(
    msg=f"Single-Layer Burger Number: {Bu_1l:.2f}",
    trigger_level=1,
)
## Set model parameters
params_1l = {
    "nx": config_1l.mesh.nx,
    "ny": config_1l.mesh.ny,
    "nl": config_1l.layers.nl,
    "dx": config_1l.mesh.dx,
    "dy": config_1l.mesh.dy,
    "H": mesh_1l.h.xyh[2],
    "g_prime": config_1l.layers.g_prime.unsqueeze(1).unsqueeze(1),
    "f": f,
    "taux": taux,
    "tauy": tauy,
    "bottom_drag_coef": config_1l.physics.bottom_drag_coef,
    "device": DEVICE,
    "dtype": torch.float64,
    "mask": torch.ones_like(mesh_1l.h.remove_z_h().xy[0]),
    "compile": True,
    "slip_coef": 1.0,
    "dt": 0.0,
}
qg_1l = QG(params_1l)
u0_1l, v0_1l, h0_1l = qg_1l.G(vortex_1l.compute(config_1l.physics.f0, Ro))

## Max speed
u_max_1l, v_max_1l, c_1l = (
    torch.abs(u0_1l).max().item() / config_1l.mesh.dx,
    torch.abs(v0_1l).max().item() / config_1l.mesh.dy,
    torch.sqrt(config_1l.layers.g_prime[0] * config_1l.layers.h.sum()),
)
## Timestep
dt_1l = min(
    cfl_adv * config_1l.mesh.dx / u_max_1l,
    cfl_adv * config_1l.mesh.dy / v_max_1l,
    cfl_gravity * config_1l.mesh.dx / c_1l,
)

# Two Layers Set-up
## Vortex
vortex_2l = RankineVortexForcing.from_config(config_2l)
## Mesh
mesh_2l = Meshes3D.from_config(config_2l)
## Coriolis
f = coriolis.compute_beta_plane(
    mesh=mesh_2l.omega.remove_z_h(),
    f0=config_1l.physics.f0,
    beta=config_1l.physics.beta,
)
## Compute Burger Number
h1 = config_2l.layers.h[0]
h2 = config_2l.layers.h[1]
h_eq = (h1 * h2) / (h1 + h2)

Bu_2l = compute_burger(
    g=config_2l.layers.g_prime[0],
    h_scale=h_eq,
    f0=config_2l.physics.f0,
    length_scale=vortex_2l.r0,
)
verbose.display(
    msg=f"Multi-Layers Burger Number: {Bu_2l:.2f}",
    trigger_level=1,
)
## Set model parameters
params_2l = {
    "nx": config_2l.mesh.nx,
    "ny": config_2l.mesh.ny,
    "nl": config_2l.layers.nl,
    "dx": config_2l.mesh.dx,
    "dy": config_2l.mesh.dy,
    "H": mesh_2l.h.xyh[2],
    "g_prime": config_2l.layers.g_prime.unsqueeze(1).unsqueeze(1),
    "f": f,
    "taux": taux,
    "tauy": tauy,
    "bottom_drag_coef": config_2l.physics.bottom_drag_coef,
    "device": DEVICE,
    "dtype": torch.float64,
    "mask": torch.ones_like(mesh_2l.h.remove_z_h().xy[0]),
    "compile": True,
    "slip_coef": 1.0,
    "dt": 0.0,
}
qg_2l = QG(params_2l)
u0_2l, v0_2l, h0_2l = qg_2l.G(vortex_2l.compute(config_2l.physics.f0, Ro))
## Max Speed
u_max_2l, v_max_2l, c_2l = (
    torch.abs(u0_2l).max().item() / config_2l.mesh.dx,
    torch.abs(v0_2l).max().item() / config_2l.mesh.dy,
    torch.sqrt(config_2l.layers.g_prime[0] * config_2l.layers.h.sum()),
)
## Timestep
dt_2l = min(
    cfl_adv * config_2l.mesh.dx / u_max_2l,
    cfl_adv * config_2l.mesh.dy / v_max_2l,
    cfl_gravity * config_2l.mesh.dx / c_2l,
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


tau_1l = 1.0 / torch.sqrt(w_0_1l.pow(2).mean()).cpu().item()
verbose.display(
    msg=f"tau (single layer) = {tau_1l *config_1l.physics.f0:.2f} f0-1",
    trigger_level=1,
)

qg_2l.compute_diagnostic_variables()
qg_2l.compute_time_derivatives()

w_0_2l = qg_2l.omega.squeeze() / qg_2l.dx / qg_2l.dy


tau_2l = 1.0 / torch.sqrt(w_0_2l.pow(2).mean()).cpu().item()
verbose.display(
    msg=f"tau (multi layer) = {tau_2l *config_2l.physics.f0:.2f} f0-1",
    trigger_level=1,
)

tau = max(tau_1l, tau_2l)
verbose.display(
    msg=f"tau = {tau *config_2l.physics.f0:.2f} f0-1", trigger_level=1
)

t_end = 8 * tau
freq_plot = int(t_end / 100 / dt) + 1
freq_checknan = 100
freq_log = int(t_end / 100 / dt) + 1
n_steps = int(t_end / dt) + 1

verbose.display(msg=f"Total Duration: {t_end:.2f}", trigger_level=1)


# Instantiate Plots
mpl.rcParams.update({"font.size": 18})
palette = plt.cm.bwr  # .with_extremes(bad='grey')

plt.ion()

f, a = plt.subplots(1, 3, figsize=(18, 8))
a[0].set_title(r"$\omega_{QG Single Layer}$")
a[1].set_title(r"$\omega_{QG MultiLayers}$")
a[2].set_title(r"$\omega_{QG Single Layer} - \omega_{QG MultiLayers}$")
[(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(3)]
f.tight_layout()
plt.pause(0.1)

# Start runs
for n in range(n_steps + 1):
    if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
        w_1l = (qg_1l.omega / qg_1l.area / qg_1l.f0).cpu().numpy()
        w_2l = (qg_2l.omega / qg_2l.area / qg_2l.f0).cpu().numpy()
        w_m = max(np.abs(w_1l).max(), np.abs(w_2l).max())

        kwargs = {
            "cmap": palette,
            "origin": "lower",
            "vmin": -w_m,
            "vmax": w_m,
            "animated": True,
        }
        a[0].imshow(w_1l[0, 0].T, **kwargs)
        a[1].imshow(w_2l[0, 0].T, **kwargs)
        a[2].imshow((w_1l - w_2l)[0, 0].T, **kwargs)
        f.suptitle(
            f"Ro={Ro:.2f}, Bu_1l={Bu_1l:.2f}, Bu_2l={Bu_2l:.2f},"
            f" t={t/tau:.2f}$\\tau$, f0={config_1l.physics.f0:.2f}"
        )
        plt.pause(0.05)

    qg_1l.step()
    qg_2l.step()
    t += dt

    if freq_log > 0 and n % freq_log == 0:
        verbose.display(
            msg=f"n={n:05d}, {qg_1l.get_print_info()}",
            trigger_level=1,
        )
        verbose.display(
            msg=f"n={n:05d}, {qg_2l.get_print_info()}",
            trigger_level=1,
        )
