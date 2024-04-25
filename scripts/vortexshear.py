# ruff: noqa
"""Comparison between QG and SW solutions in vortex shear instability.

Problem scales:
lx, ly : x,y box dimensions
r0 : vortex core radius -> length scale
ld : deformation length scale -> √(Bu)*r0
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from icecream import ic
from qgsw import verbose
from qgsw.configs import VortexShearConfig
from qgsw.forcing.vortex import (
    RankineVortexForcing,
)
from qgsw.forcing.wind import WindForcing
from qgsw.mesh import Meshes3D
from qgsw.physics import compute_burger
from qgsw.models import SW, QG
from qgsw.specs import DEVICE
from qgsw.plots.vorticity import (
    SurfaceVorticityAxes,
    SurfaceVorticityComparisonFigure,
)

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

config = VortexShearConfig.from_file("config/vortexshear.toml")
mesh = Meshes3D.from_config(config)
wind = WindForcing.from_config(config)
taux, tauy = wind.compute()
vortex = RankineVortexForcing.from_config(config)

h1 = config.layers.h[0]
h2 = config.layers.h[1]

Bu = compute_burger(
    g=config.layers.g_prime[0],
    h_scale=(h1 * h2) / (h1 + h2),
    f0=config.physics.f0,
    length_scale=vortex.r0,
)

mesh_2d = mesh.remove_z_h()
x, y = mesh_2d.omega.xy
xc, yc = mesh_2d.h.xy
rc = torch.sqrt(xc**2 + yc**2)
# circular domain mask
apply_mask = False
mask = (
    (rc < config.mesh.lx / 2).type(torch.float64)
    if apply_mask
    else torch.ones_like(xc)
)
# Burger and Rossby
for Ro in [
    # 0.5,
    0.1,
    # 0.02,
]:
    verbose.display(msg=f"Ro={Ro} Bu={Bu}", trigger_level=1)

    # vortex set up
    r0, r1, r2 = (
        0.1 * config.mesh.lx,
        0.1 * config.mesh.lx,
        0.14 * config.mesh.lx,
    )

    # set coriolis with burger number
    f0 = config.physics.f0
    beta = config.physics.beta
    f = f0 + beta * (y - config.mesh.ly / 2)

    param_qg = {
        "nx": config.mesh.nx,
        "ny": config.mesh.ny,
        "nl": config.layers.nl,
        "dx": config.mesh.dx,
        "dy": config.mesh.dy,
        "H": mesh.h.xyh[2],
        "g_prime": config.layers.g_prime.unsqueeze(1).unsqueeze(1),
        "f": f,
        "taux": taux,
        "tauy": tauy,
        "bottom_drag_coef": config.physics.bottom_drag_coef,
        "device": DEVICE,
        "dtype": torch.float64,
        "mask": mask,
        "compile": True,
        "slip_coef": 1.0,
        "dt": 0.0,
    }

    param_sw = {
        "nx": config.mesh.nx,
        "ny": config.mesh.ny,
        "nl": config.layers.nl,
        "dx": config.mesh.dx,
        "dy": config.mesh.dy,
        "H": mesh.h.xyh[2],
        "rho": config.physics.rho,
        "g_prime": config.layers.g_prime.unsqueeze(1).unsqueeze(1),
        "f": f,
        "taux": taux,
        "tauy": tauy,
        "mask": mask,
        "bottom_drag_coef": config.physics.bottom_drag_coef,
        "device": DEVICE,
        "dtype": torch.float64,
        "slip_coef": 1,
        "compile": True,
        "barotropic_filter": False,
        "barotropic_filter_spectral": False,
        "dt": 0.0,  # time-step (s)
    }

    qg_multilayer = QG(param_qg)

    u_init, v_init, h_init = qg_multilayer.G(vortex.compute(f0, Ro))

    u_max, v_max, c = (
        torch.abs(u_init).max().item() / config.mesh.dx,
        torch.abs(v_init).max().item() / config.mesh.dy,
        torch.sqrt(config.layers.g_prime[0] * config.layers.h.sum()),
    )
    verbose.display(
        msg=f"u_max {u_max:.2e}, v_max {v_max:.2e}, c {c:.2e}",
        trigger_level=1,
    )
    cfl_adv = 0.5
    cfl_gravity = 5 if param_sw["barotropic_filter"] else 0.5
    dt = min(
        cfl_adv * config.mesh.dx / u_max,
        cfl_adv * config.mesh.dy / v_max,
        cfl_gravity * config.mesh.dx / c,
    )

    qg_multilayer.dt = dt
    qg_multilayer.u = torch.clone(u_init)
    qg_multilayer.v = torch.clone(v_init)
    qg_multilayer.h = torch.clone(h_init)
    qg_multilayer.compute_diagnostic_variables()

    w_qg = (qg_multilayer.omega.squeeze() / qg_multilayer.area).cpu().numpy()

    # update time step
    param_sw["dt"] = dt
    sw_multilayer = SW(param_sw)
    sw_multilayer.u = torch.clone(u_init)
    sw_multilayer.v = torch.clone(v_init)
    sw_multilayer.h = torch.clone(h_init)
    sw_multilayer.compute_diagnostic_variables()

    # time params
    t = 0

    qg_multilayer.compute_time_derivatives()
    wa_0 = qg_multilayer.omega_a.squeeze().cpu().numpy()

    w_0 = qg_multilayer.omega.squeeze() / qg_multilayer.dx / qg_multilayer.dy
    tau = 1.0 / torch.sqrt(w_0.pow(2).mean()).cpu().item()
    verbose.display(
        msg=f"tau = {tau *f0:.2f} f0-1",
        trigger_level=1,
    )

    t_end = 8 * tau
    freq_plot = int(t_end / 100 / dt) + 1
    freq_checknan = 100
    freq_log = int(t_end / 10 / dt) + 1
    n_steps = int(t_end / dt) + 1

    if freq_plot > 0:
        sw_axes = SurfaceVorticityAxes.from_mask(
            mask=sw_multilayer.masks.not_w[0, 0].cpu().numpy(),
        )
        sw_axes.set_title(r"$\omega_{SW}$")
        qg_axes = SurfaceVorticityAxes.from_mask(
            mask=sw_multilayer.masks.not_w[0, 0].cpu().numpy(),
        )
        qg_axes.set_title(r"$\omega_{QG}$")
        diff_axes = SurfaceVorticityAxes.from_mask(
            mask=sw_multilayer.masks.not_w[0, 0].cpu().numpy(),
        )
        diff_axes.set_title(r"$\omega_{SW} - \omega_{QG}$")
        plot = SurfaceVorticityComparisonFigure(sw_axes, qg_axes, diff_axes)

    for n in range(n_steps + 1):
        if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            mask_w = sw_multilayer.masks.not_w[0, 0].cpu().numpy()
            w_qg = (
                (qg_multilayer.omega / qg_multilayer.area / qg_multilayer.f0)
                .cpu()
                .numpy()
            )
            w_sw = (
                (sw_multilayer.omega / sw_multilayer.area / sw_multilayer.f0)
                .cpu()
                .numpy()
            )
            w_m = max(np.abs(w_qg).max(), np.abs(w_sw).max())

            kwargs = {
                "vmin": -w_m,
                "vmax": w_m,
            }
            plot.update(w_sw, w_qg, w_sw - w_qg, **kwargs)

        qg_multilayer.step()
        sw_multilayer.step()
        t += dt
        if n % freq_checknan == 0:
            if torch.isnan(qg_multilayer.h).any():
                msg = f"Stopping, NAN number in QG h at iteration {n}."
                raise ValueError(msg)
            if torch.isnan(sw_multilayer.h).any():
                msg = f"Stopping, NAN number in SW h at iteration {n}."
                raise ValueError(msg)

        if freq_log > 0 and n % freq_log == 0:
            verbose.display(
                msg=f"n={n:05d}, {qg_multilayer.get_print_info()}",
                trigger_level=1,
            )
