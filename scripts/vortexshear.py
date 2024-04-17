# ruff: noqa
"""Comparison between QG and SW solutions in vortex shear instability."""

import sys

import numpy as np
import torch

sys.path.append("../src")

import matplotlib.pyplot as plt
from qgsw.configs import ScriptConfig
from qgsw.forcing.vortex import (
    PassiveLayersVortexForcing,
)
from qgsw.forcing.wind import WindForcing
from qgsw.grid import Grid3D
from qgsw.qg import QG
from qgsw.specs import DEVICE
from qgsw.sw import SW

torch.backends.cudnn.deterministic = True

config = ScriptConfig.from_file("config/vortexshear.toml")
grid = Grid3D.from_runconfig(config)
wind = WindForcing.from_runconfig(config)
taux, tauy = wind.compute()
vortex = PassiveLayersVortexForcing(grid, 1e-3)


x, y = grid.xy.omega_xy
xc, yc = grid.xy.h_xy
rc = torch.sqrt(xc**2 + yc**2)
# circular domain mask
apply_mask = False
mask = (
    (rc < config.grid.lx / 2).type(torch.float64)
    if apply_mask
    else torch.ones_like(xc)
)

flip_sign = False
# Burger and Rossby
for Bu, Ro in [
    # (1., 0.5),
    (1.0, 0.1),
    # (1., 0.02),
]:
    print(f"Ro={Ro} Bu={Bu}")

    # vortex set up
    r0, r1, r2 = (
        0.1 * config.grid.lx,
        0.1 * config.grid.lx,
        0.14 * config.grid.lx,
    )

    # set coriolis with burger number
    f0 = torch.sqrt(
        config.layers.g_prime[0, 0, 0] * config.layers.h[0, 0, 0] / Bu / r0**2
    )
    if flip_sign:
        f0 *= -1
    beta = config.physics.beta
    f = f0 + beta * (y - config.grid.ly / 2)

    param_qg = {
        "nx": config.grid.nx,
        "ny": config.grid.ny,
        "nl": config.layers.nl,
        "dx": config.grid.dx,
        "dy": config.grid.dy,
        "H": config.layers.h,
        "g_prime": config.layers.g_prime,
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
        "nx": config.grid.nx,
        "ny": config.grid.ny,
        "nl": config.layers.nl,
        "dx": config.grid.dx,
        "dy": config.grid.dy,
        "H": config.layers.h,
        "rho": config.physics.rho,
        "g_prime": config.layers.g_prime,
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
        torch.abs(u_init).max().item() / config.grid.dx,
        torch.abs(v_init).max().item() / config.grid.dy,
        torch.sqrt(config.layers.g_prime[0, 0, 0] * config.layers.h.sum()),
    )
    print(f"u_max {u_max:.2e}, v_max {v_max:.2e}, c {c:.2e}")
    cfl_adv = 0.5
    cfl_gravity = 5 if param_sw["barotropic_filter"] else 0.5
    dt = min(
        cfl_adv * config.grid.dx / u_max,
        cfl_adv * config.grid.dy / v_max,
        cfl_gravity * config.grid.dx / c,
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
    print(f"tau = {tau *f0:.2f} f0-1")

    t_end = 8 * tau
    freq_plot = int(t_end / 100 / dt) + 1
    freq_checknan = 100
    freq_log = int(t_end / 10 / dt) + 1
    n_steps = int(t_end / dt) + 1

    if freq_plot > 0:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update({"font.size": 18})
        palette = plt.cm.bwr  # .with_extremes(bad='grey')

        plt.ion()

        f, a = plt.subplots(1, 3, figsize=(18, 8))
        a[0].set_title(r"$\omega_{qg}$")
        a[1].set_title(r"$\omega_{sw}$")
        a[2].set_title(r"$\omega_{qg} - \omega_{sw}$")
        [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(3)]
        f.tight_layout()
        plt.pause(0.1)

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
                "cmap": palette,
                "origin": "lower",
                "vmin": -w_m,
                "vmax": w_m,
                "animated": True,
            }
            a[0].imshow(np.ma.masked_where(mask_w, w_qg[0, 0]).T, **kwargs)
            a[1].imshow(np.ma.masked_where(mask_w, w_sw[0, 0]).T, **kwargs)
            a[2].imshow(
                np.ma.masked_where(mask_w, (w_qg - w_sw)[0, 0]).T, **kwargs
            )
            f.suptitle(
                f'Ro={Ro:.2f}, Bu={Bu:.2f}, t={t/tau:.2f}$\\tau$, '
                f'{"neg." if flip_sign else "pos"} $f_0$'
            )
            plt.pause(0.05)

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
            print(f"n={n:05d}, {qg_multilayer.get_print_info()}")
