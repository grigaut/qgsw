# ruff : noqa
import os
import numpy as np
import sys
import torch
from pathlib import Path

sys.path.append("../src")

from qgsw.sw import SW
from qgsw.qg import QG
from qgsw.configs import RunConfig
from qgsw.grid import Grid
from qgsw.forcing.wind import CosineZonalWindForcing
from qgsw.specs import DEVICE

torch.backends.cudnn.deterministic = True


config = RunConfig.from_file(Path("config/doublegyre.toml"))
grid = Grid.from_runconfig(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

mask = torch.ones(config.grid.nx, config.grid.ny, dtype=dtype, device=device)
# mask[0,0] = 0
# mask[0,-1] = 0

wind_forcing = CosineZonalWindForcing.from_runconfig(run_config=config)
taux, tauy = wind_forcing.compute_over_grid(grid=grid)

param = {
    "nx": config.grid.nx,
    "ny": config.grid.ny,
    "nl": config.layers.nl,
    "dx": config.grid.dx,
    "dy": config.grid.dy,
    "H": config.layers.h,
    "rho": config.physics.rho,
    "g_prime": config.layers.g_prime,
    "bottom_drag_coef": config.physics.bottom_drag_coef,
    "device": DEVICE,
    "dtype": dtype,
    "slip_coef": config.physics.slip_coef,
    "interp_fd": False,
    "dt": config.grid.dt,
    "compile": True,
    "barotropic_filter": True,
    "barotropic_filter_spectral": True,
    "mask": mask,
    "f": grid.generate_coriolis_grid(
        f0=config.physics.f0, beta=config.physics.beta
    ),
    "taux": taux,
    "tauy": tauy,
}


for model, name, dt, start_file in [
    # (QG, 'qg', 4000, ''),
    (SW, "sw", 4000, ""),
    # (SW, 'sw', 4000, 'run_outputs/qg_256x256_dt4000_slip1.0/uvh_100y_010d.npz'),
]:
    # set time step given barotropic mode for SW
    if model == SW:
        c = (
            torch.sqrt(config.layers.h.sum() * config.layers.g_prime[0, 0, 0])
            .cpu()
            .item()
        )
        print(c)
        cfl = 20 if param["barotropic_filter"] else 0.5
        print(cfl)
        dt = float(int(cfl * min(config.grid.dx, config.grid.dy) / c))
        print(f"dt = {dt:.1f} s.")
        param["dt"] = dt
    # exit()
    print(
        f"Double gyre config, {name} model, {config.grid.nx}x{config.grid.ny} grid, dt {dt:.1f}s."
    )

    qgsw_multilayer = model(param)

    if start_file:
        print(f"Starting from file {start_file}...")
        zipf = np.load(start_file)
        qgsw_multilayer.set_physical_uvh(zipf["u"], zipf["v"], zipf["h"])

    t = 0

    freq_checknan = 10
    freq_log = 100

    n_steps = int(10 * 365 * 24 * 3600 / dt) + 1
    n_steps_save = int(2 * 365 * 24 * 3600 / dt)
    freq_save = int(15 * 24 * 3600 / dt)
    freq_plot = int(15 * 24 * 3600 / dt)

    uM, vM, hM = 0, 0, 0

    if freq_save > 0:
        output_dir = (
            f'run_outputs/{name}_{config.grid.nx}x{config.grid.ny}_dt{dt}_'
            f'slip{param["slip_coef"]}/'
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Outputs will be saved to {output_dir}")

    if freq_plot > 0:
        import matplotlib.pyplot as plt

        plt.ion()
        nl_plot = 0
        if model == QG:
            npx, npy = 2, 1
            f, a = plt.subplots(npy, npx, figsize=(12, 12))
            a[0].set_title("$\\omega_g$")
            a[1].set_title("$\\omega_a$")
            [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(npx)]
        else:
            npx, npy = 3, 1
            f, a = plt.subplots(npy, npx, figsize=(16, 6))
            a[0].set_title("$u$")
            a[1].set_title("$v$")
            a[2].set_title("$h$")
            [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(npx)]

        plt.tight_layout()
        plt.pause(0.1)
        plot_kwargs = {"cmap": "bwr", "origin": "lower", "animated": True}

    for n in range(1, n_steps + 1):
        qgsw_multilayer.step()
        t += dt

        n_years = int(t // (365 * 24 * 3600))
        n_days = int(t % (365 * 24 * 3600) // (24 * 3600))

        if n % freq_checknan == 0 and torch.isnan(qgsw_multilayer.p).any():
            raise ValueError(f"Stopping, NAN number in p at iteration {n}.")

        if freq_log > 0 and n % freq_log == 0:
            print(
                f"n={n:05d}, t={n_years:02d}y{n_days:03d}d, {qgsw_multilayer.get_print_info()}"
            )

        if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
            u, v, h = qgsw_multilayer.get_physical_uvh(numpy=True)
            uM, vM = (
                max(uM, 0.8 * np.abs(u).max()),
                max(vM, 0.8 * np.abs(v).max()),
            )
            hM = max(hM, 0.8 * np.abs(h).max())
            if model == QG:
                wM = 0.2
                w = (
                    (
                        qgsw_multilayer.omega
                        / qgsw_multilayer.area
                        / qgsw_multilayer.f0
                    )
                    .cpu()
                    .numpy()
                )
                w_a = (
                    (qgsw_multilayer.omega_a / qgsw_multilayer.f0)
                    .cpu()
                    .numpy()
                )

                a[0].imshow(w[0, nl_plot].T, vmin=-wM, vmax=wM, **plot_kwargs)
                a[1].imshow(
                    w_a[0, nl_plot].T,
                    vmin=-0.2 * wM,
                    vmax=0.2 * wM,
                    **plot_kwargs,
                )
            else:
                a[0].imshow(u[0, nl_plot].T, vmin=-uM, vmax=uM, **plot_kwargs)
                a[1].imshow(v[0, nl_plot].T, vmin=-vM, vmax=vM, **plot_kwargs)
                a[2].imshow(h[0, nl_plot].T, vmin=-hM, vmax=hM, **plot_kwargs)

            f.suptitle(f"{n_years} yrs, {n_days:03d} days")
            plt.pause(0.05)

        if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
            filename = os.path.join(
                output_dir, f"uvh_{n_years:03d}y_{n_days:03d}d.npz"
            )
            u, v, h = qgsw_multilayer.get_physical_uvh(numpy=True)
            if model == QG:
                u_a = qgsw_multilayer.u_a.cpu().numpy()
                v_a = qgsw_multilayer.v_a.cpu().numpy()
                np.savez(
                    filename,
                    u=u.astype("float32"),
                    v=v.astype("float32"),
                    u_a=u_a.astype("float32"),
                    v_a=v_a.astype("float32"),
                    h=h.astype("float32"),
                )
                print(f"saved u,v,h,u_a,v_a to {filename}")
            else:
                np.savez(
                    filename,
                    u=u.astype("float32"),
                    v=v.astype("float32"),
                    h=h.astype("float32"),
                )
                print(f"saved u,v,h to {filename}")
    break
