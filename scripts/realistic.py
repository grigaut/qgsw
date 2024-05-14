# ruff : noqa
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append("../src")
from qgsw.forcing.bathymetry import Bathymetry
from qgsw.forcing.wind import WindForcing
from qgsw.mesh import Meshes3D
from qgsw.models import SW, QG
from qgsw.physics import coriolis
from qgsw.specs import DEVICE
from qgsw import verbose
from qgsw.configs import Configuration

torch.backends.cudnn.deterministic = True
verbose.set_level(2)

ROOT_PATH = Path(__file__).parent.parent
CONFIG_PATH = ROOT_PATH.joinpath("config/realistic.toml")
config = Configuration.from_file(CONFIG_PATH)

mesh = Meshes3D.from_config(config.mesh, config.model)
bathy = Bathymetry.from_config(config.bathymetry)
wind = WindForcing.from_config(config.windstress, config.mesh, config.physics)

verbose.display(
    msg=f"Grid lat: {config.mesh.box.y_min:.1f}, {config.mesh.box.y_max:.1f}, ",
    trigger_level=1,
)
verbose.display(
    msg=f"lon: {config.mesh.box.x_min:.1f}, {config.mesh.box.x_max:.1f}, ",
    trigger_level=1,
)
verbose.display(
    msg=f"dx={config.mesh.dx/1e3:.1f}km, dy={config.mesh.dy/1e3:.1f}km .",
    trigger_level=1,
)
verbose.display(
    msg=f"Topo lat: {bathy.lats.min():.2f} - {bathy.lats.max():.2f}, ",
    trigger_level=1,
)

verbose.display(
    msg=f"lon: {bathy.lons.min():.2f} - {bathy.lons.max():.2f}",
    trigger_level=1,
)

verbose.display(
    msg=(
        "Interpolating bathymetry on mesh with"
        f" {config.bathymetry.interpolation_method} interpolation ..."
    ),
    trigger_level=1,
)
# Land Mask Generation

mask_land = bathy.compute_land_mask(mesh.h.remove_z_h())
mask_land_w = bathy.compute_land_mask_w(mesh.h.remove_z_h())

# coriolis beta plane
f = coriolis.compute_beta_plane(
    mesh=mesh.omega.remove_z_h(),
    f0=config.physics.f0,
    beta=config.physics.beta,
)
verbose.display(
    msg=(
        f"Coriolis param min {f.min().to(device=DEVICE).item():.2e},"
        f" {f.max().to(device=DEVICE).item():.2e}"
    ),
    trigger_level=1,
)
taux, tauy = wind.compute()

param = {
    "nx": config.mesh.nx,
    "ny": config.mesh.ny,
    "nl": config.model.nl,
    "H": config.model.h.unsqueeze(1).unsqueeze(1),
    "dx": config.mesh.dx,
    "dy": config.mesh.dy,
    "rho": config.physics.rho,
    "g_prime": config.model.g_prime.unsqueeze(1).unsqueeze(1),
    "bottom_drag_coef": config.physics.bottom_drag_coef,
    "f": f,
    "device": DEVICE,
    "dtype": torch.float64,
    "slip_coef": config.physics.slip_coef,
    "dt": config.simulation.dt,  # time-step (s)
    "compile": True,
    "mask": bathy.compute_ocean_mask(mesh.h.remove_z_h()),
    "taux": taux[0, 1:-1, :],
    "tauy": tauy[0, :, 1:-1],
}


model = SW
model = QG
qg = model(param)

name = f"qg_"  # {config}"

######### Probably to avoid restarting
start_file = ""
if start_file:
    verbose.display(msg=f"Starting from file {start_file}...", trigger_level=1)
    zipf = np.load(start_file)
    qg.set_physical_uvh(zipf["u"], zipf["v"], zipf["h"])
#########
t = 0

freq_checknan = 100
freq_log = int(24 * 3600 / config.simulation.dt)
n_steps = (
    int(config.simulation.duration * 365 * 24 * 3600 / config.simulation.dt)
    + 1
)
freq_save = int(n_steps / config.io.results.quantity) + 1
freq_plot = int(n_steps / config.io.plots.quantity) + 1

plots_required = config.io.plots.save or config.io.plots.show

uM, vM, hM = 0, 0, 0


if config.io.log_performance:
    from time import time as cputime

    nmeshpoints = param["nl"] * param["nx"] * param["ny"]
    mperf = 0
    [qg.step() for _ in range(5)]  # warm up


if config.io.plots.save:
    output_dir = (
        f'{config.io.results.directory}/{name}_{config.mesh.nx}x{config.mesh.ny}_dt{config.mesh.dt}_'
        f'slip{param["slip_coef"]}/'
    )
    os.makedirs(output_dir, exist_ok=True)
    verbose.display(
        msg=f"Outputs will be saved to {output_dir}",
        trigger_level=1,
    )
    np.save(os.path.join(output_dir, "mask_land_h.npz"), mask_land)
    np.save(
        os.path.join(output_dir, "mask_land_w.npz"),
        mask_land_w.cpu().numpy(),
    )
    torch.save(param, os.path.join(output_dir, "param.pth"))


if plots_required:
    import matplotlib.pyplot as plt

    plt.ion()
    palette = plt.cm.bwr  # .with_extremes(bad='grey')
    nl_plot = 0
    if model == QG:
        npx, npy = 1, 1
        f, a = plt.subplots(npy, npx, figsize=(12, 12))
        a.set_title("$\\omega_g$")
        # a[1].set_title('$\\omega_a$')
        a.set_xticks([]), a.set_yticks([])
        # [(a[i].set_xticks([]), a[i].set_yticks([]))
        # for i in range(npx)]
    else:
        npx, npy = 3, 1
        f, a = plt.subplots(npy, npx, figsize=(16, 6))
        a[0].set_title("$u$")
        a[1].set_title("$v$")
        a[2].set_title("$h$")
        [(a[i].set_xticks([]), a[i].set_yticks([])) for i in range(npx)]

    plt.tight_layout()
    plt.pause(0.1)
    plot_kwargs = {"cmap": palette, "origin": "lower", "animated": True}


for n in range(1, n_steps + 1):
    ## update wind forcing
    month = 12 * (t % (365 * 24 * 3600)) / (365 * 24 * 3600)
    m_i, m_r = int(month), month - int(month)
    taux_t = (1 - m_r) * taux[m_i, 1:-1, :] + m_r * taux[m_i + 1, 1:-1, :]
    tauy_t = (1 - m_r) * tauy[m_i, :, 1:-1] + m_r * tauy[m_i + 1, :, 1:-1]
    qg.set_wind_forcing(taux_t, tauy_t)

    if config.io.log_performance:
        walltime0 = cputime()

    ## one step
    qg.step()
    t += config.simulation.dt

    if config.io.log_performance:
        walltime = cputime()
        perf = (walltime - walltime0) / (nmeshpoints)
        mperf += perf
        verbose.display(
            msg=f"\rkt={n:4} time={t:.2f} perf={perf:.2e} ({mperf/n:.2e}) s",
            trigger_level=1,
        )

    n_years = int(t // (365 * 24 * 3600))
    n_days = int(t % (365 * 24 * 3600) // (24 * 3600))

    if n % freq_checknan == 0 and torch.isnan(qg.p).any():
        raise ValueError(f"Stopping, NAN number in p at iteration {n}.")

    if freq_log > 0 and n % freq_log == 0:
        verbose.display(
            msg=f"n={n:05d}, t={n_years:02d}y{n_days:03d}d, {qg.get_print_info()}",
            trigger_level=1,
        )

    if plots_required and (n % freq_plot == 0 or n == n_steps):
        u, v, h = qg.get_physical_uvh_as_ndarray()
        uM, vM = max(uM, 0.8 * np.abs(u).max()), max(vM, 0.8 * np.abs(v).max())
        hM = max(hM, 0.8 * np.abs(h).max())
        if model == QG:
            wM = 0.05
            w = (qg.omega / qg.area / qg.f0).cpu().numpy()[0, nl_plot]
            w = np.ma.masked_where(mask_land_w.cpu().numpy(), w)
            a.imshow(w.T, vmin=-wM, vmax=wM, **plot_kwargs)
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
        u, v, h = qg.get_physical_uvh_as_ndarray()
        if model == QG:
            qg.save_uvh(Path(filename))
            filename_a = os.path.join(
                output_dir, f"uv_a_{n_years:03d}y_{n_days:03d}d.npz"
            )
            qg.save_uv_ageostrophic(Path(filename_a))
        else:
            qg.save_uvh(Path(filename))
