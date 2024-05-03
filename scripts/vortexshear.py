# ruff: noqa
"""Comparison between QG and SW solutions in vortex shear instability.

Problem scales:
lx, ly : x,y box dimensions
r0 : vortex core radius -> length scale
ld : deformation length scale -> √(Bu)*r0
"""

import torch
from qgsw import verbose
from qgsw.configs import Configuration
from qgsw.forcing.vortex import (
    RankineVortexForcing,
)
from pathlib import Path
from qgsw.forcing.wind import WindForcing
from qgsw.mesh import Meshes3D
from qgsw.physics import compute_burger
from qgsw.models import SW, QG
from qgsw.specs import DEVICE
from qgsw.plots.vorticity import (
    SurfaceVorticityAxes,
    VorticityComparisonFigure,
)

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

ROOT_PATH = Path(__file__).parent.parent
CONFIG_PATH = ROOT_PATH.joinpath("config/vortexshear.toml")
config = Configuration.from_file(CONFIG_PATH)

mesh = Meshes3D.from_config(config.mesh, config.model)
wind = WindForcing.from_config(config.windstress, config.mesh, config.physics)
taux, tauy = wind.compute()
vortex = RankineVortexForcing.from_config(
    config.vortex, config.mesh, config.model
)

h1 = config.model.h[0]
h2 = config.model.h[1]

Bu = compute_burger(
    g=config.model.g_prime[0],
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
        "mask": mask,
        "compile": True,
        "slip_coef": 1.0,
        "dt": 0.0,
    }

    param_sw = {
        "nx": config.mesh.nx,
        "ny": config.mesh.ny,
        "nl": config.model.nl,
        "dx": config.mesh.dx,
        "dy": config.mesh.dy,
        "H": mesh.h.xyh[2],
        "rho": config.physics.rho,
        "g_prime": config.model.g_prime.unsqueeze(1).unsqueeze(1),
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

    qg_ml = QG(param_qg)

    u_init, v_init, h_init = qg_ml.G(vortex.compute(f0, Ro))

    u_max, v_max, c = (
        torch.abs(u_init).max().item() / config.mesh.dx,
        torch.abs(v_init).max().item() / config.mesh.dy,
        torch.sqrt(config.model.g_prime[0] * config.model.h.sum()),
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

    qg_ml.dt = dt
    qg_ml.u = torch.clone(u_init)
    qg_ml.v = torch.clone(v_init)
    qg_ml.h = torch.clone(h_init)
    qg_ml.compute_diagnostic_variables()

    w_qg = (qg_ml.omega.squeeze() / qg_ml.area).cpu().numpy()

    # update time step
    param_sw["dt"] = dt
    sw_ml = SW(param_sw)
    sw_ml.u = torch.clone(u_init)
    sw_ml.v = torch.clone(v_init)
    sw_ml.h = torch.clone(h_init)
    sw_ml.compute_diagnostic_variables()

    # time params
    t = 0

    qg_ml.compute_time_derivatives()
    wa_0 = qg_ml.omega_a.squeeze().cpu().numpy()

    w_0 = qg_ml.omega.squeeze() / qg_ml.dx / qg_ml.dy
    tau = 1.0 / torch.sqrt(w_0.pow(2).mean()).to(device=DEVICE).item()
    verbose.display(
        msg=f"tau = {tau *f0:.2f} f0-1",
        trigger_level=1,
    )

    t_end = 8 * tau
    freq_plot = int(t_end / config.io.plots.quantity / dt) + 1
    freq_save = int(t_end / config.io.results.quantity / dt) + 1
    freq_checknan = 100
    freq_log = int(t_end / 10 / dt) + 1
    n_steps = int(t_end / dt) + 1

    plots_required = config.io.plots.save or config.io.plots.show

    if plots_required:
        mask = sw_ml.masks.not_w[0, 0].cpu().numpy()
        sw_axes = SurfaceVorticityAxes.from_mask(mask=mask)
        sw_axes.set_title(r"$\omega_{SW}$")
        qg_axes = SurfaceVorticityAxes.from_mask(mask=mask)
        qg_axes.set_title(r"$\omega_{QG}$")
        diff_axes = SurfaceVorticityAxes.from_mask(mask=mask)
        diff_axes.set_title(r"$\omega_{SW} - \omega_{QG}$")
        plot = VorticityComparisonFigure(sw_axes, qg_axes, diff_axes)

    for n in range(n_steps + 1):
        if plots_required and (n % freq_plot == 0 or n == n_steps):
            w_qg = (qg_ml.omega / qg_ml.area / qg_ml.f0).cpu().numpy()
            w_sw = (sw_ml.omega / sw_ml.area / sw_ml.f0).cpu().numpy()
            plot.update_with_arrays(w_sw, w_qg, w_sw - w_qg)
            if config.io.plots.show:
                plot.show()
            if config.io.plots.save:
                output_dir = config.io.plots.directory
                output_name = Path(f"{config.io.name}_{n}.png")
                plot.savefig(output_dir.joinpath(output_name))

        qg_ml.step()
        sw_ml.step()
        t += dt
        if n % freq_checknan == 0:
            if torch.isnan(qg_ml.h).any():
                msg = f"Stopping, NAN number in QG h at iteration {n}."
                raise ValueError(msg)
            if torch.isnan(sw_ml.h).any():
                msg = f"Stopping, NAN number in SW h at iteration {n}."
                raise ValueError(msg)

        if freq_log > 0 and n % freq_log == 0:
            verbose.display(
                msg=f"n={n:05d}, {qg_ml.get_print_info()}",
                trigger_level=1,
            )
