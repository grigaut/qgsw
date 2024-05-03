# ruff : noqa
import os
import numpy as np
import sys
import torch
from pathlib import Path

sys.path.append("../src")

from qgsw.models import SW, QG
from qgsw.configs import DoubleGyreConfig
from qgsw.physics import coriolis
from qgsw.mesh import Meshes3D
from qgsw.forcing.wind import WindForcing
from qgsw.specs import DEVICE
from qgsw import verbose
from icecream import ic

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

ROOT_PATH = Path(__file__).parent.parent
CONFIG_PATH = ROOT_PATH.joinpath("config/vortexshear.toml")
config = DoubleGyreConfig.from_file(CONFIG_PATH)

mesh = Meshes3D.from_config(config.mesh, config.model)
wind = WindForcing.from_config(config.windstress, config.mesh, config.physics)

mask = torch.ones(
    config.mesh.nx,
    config.mesh.ny,
    dtype=torch.float64,
    device=DEVICE,
)

taux, tauy = wind.compute()

param = {
    "nx": config.mesh.nx,
    "ny": config.mesh.ny,
    "nl": config.model.nl,
    "dx": config.mesh.dx,
    "dy": config.mesh.dy,
    "H": config.model.h.unsqueeze(1).unsqueeze(1),
    "rho": config.physics.rho,
    "g_prime": config.model.g_prime.unsqueeze(1).unsqueeze(1),
    "bottom_drag_coef": config.physics.bottom_drag_coef,
    "device": DEVICE,
    "dtype": torch.float64,
    "slip_coef": config.physics.slip_coef,
    "interp_fd": False,
    "dt": config.mesh.dt,
    "compile": True,
    "barotropic_filter": True,
    "barotropic_filter_spectral": True,
    "mask": mask,
    "f": coriolis.compute_beta_plane(
        mesh=mesh.omega.remove_z_h(),
        f0=config.physics.f0,
        beta=config.physics.beta,
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
            torch.sqrt(config.model.h.sum() * config.model.g_prime[0])
            .to(device=DEVICE)
            .item()
        )
        cfl = 20 if param["barotropic_filter"] else 0.5
        dt = float(int(cfl * min(config.mesh.dx, config.mesh.dy) / c))
        param["dt"] = dt
    # exit()
    verbose.display(
        msg=(
            f"Double gyre config, {name} model, "
            f"{config.mesh.nx}x{config.mesh.ny} mesh, dt {dt:.1f}s."
        ),
        trigger_level=1,
    )

    qgsw_multilayer = model(param)

    if start_file:
        verbose.display(
            msg=f"Starting from file {start_file}...",
            trigger_level=1,
        )
        zipf = np.load(start_file)
        qgsw_multilayer.set_physical_uvh(zipf["u"], zipf["v"], zipf["h"])

    t = 0

    freq_checknan = 10
    freq_log = 100

    n_steps = int(10 * 365 * 24 * 3600 / dt) + 1
    freq_save = int(n_steps / config.io.results.quantity) + 1
    freq_plot = int(n_steps / config.io.plots.quantity) + 1

    plots_required = config.io.plots.save or config.io.plots.show

    uM, vM, hM = 0, 0, 0

    if config.io.plots.save:
        output_dir = (
            f"{config.io.results.directory}/{name}_{config.mesh.nx}x"
            f"{config.mesh.ny}_dt{dt}_slip{param['slip_coef']}/"
        )
        os.makedirs(output_dir, exist_ok=True)
        verbose.display(
            msg=f"Outputs will be saved to {output_dir}", trigger_level=1
        )

    if config.io.plots.show:
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
            verbose.display(
                msg=(
                    f"n={n:05d}, t={n_years:02d}y{n_days:03d}d, "
                    f"{qgsw_multilayer.get_print_info()}"
                ),
                trigger_level=1,
            )

        if plots_required and (n % freq_plot == 0 or n == n_steps):
            u, v, h = qgsw_multilayer.get_physical_uvh_as_ndarray()
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
            if config.io.plots.show:
                plt.pause(0.05)
            if config.io.plots.save:
                plt.savefig(config.io.plots.directory.joinpath(f"{n}.png"))

        if config.io.results.save and n % freq_save == 0:
            filename = os.path.join(
                output_dir, f"uvh_{n_years:03d}y_{n_days:03d}d.npz"
            )
            u, v, h = qgsw_multilayer.get_physical_uvh_as_ndarray()
            if model == QG:
                qgsw_multilayer.save_uvh(Path(filename))
                filename_a = os.path.join(
                    output_dir, f"uv_a_{n_years:03d}y_{n_days:03d}d.npz"
                )
                qgsw_multilayer.save_uv_ageostrophic(Path(filename_a))
            else:
                qgsw_multilayer.save_uvh(Path(filename))
