# ruff : noqa
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import scipy.interpolate
import scipy.io
import scipy.ndimage
import torch
from icecream import ic

sys.path.append("../src")
from qgsw.bathymetry import Bathymetry
from qgsw.forcing.wind import WindForcing
from qgsw.configs import RealisticConfig
from qgsw.mesh import Meshes2D
from qgsw.qg import QG
from qgsw.specs import DEVICE
from qgsw.sw import SW


torch.backends.cudnn.deterministic = True

config = RealisticConfig.from_file(Path("config/realistic.toml"))
mesh = Meshes2D.from_config(config)
bathy = Bathymetry.from_config(config)
wind = WindForcing.from_config(config)

print(
    f"Grid lat: {config.mesh.y_min:.1f}, {config.mesh.y_max:.1f}, "
    f"lon: {config.mesh.x_min:.1f}, {config.mesh.x_max:.1f}, "
    f"dx={config.mesh.dx/1e3:.1f}km, dy={config.mesh.dy/1e3:.1f}km ."
)


print(
    f"Topo lat: {bathy.lats.min():.2f} - {bathy.lats.max():.2f}, "
    f"lon: {bathy.lons.min():.2f} - {bathy.lons.max():.2f}"
)

print(
    "Interpolating bathymetry on mesh with"
    f" {config.bathy.interpolation_method} interpolation ..."
)

# Land Mask Generation
mask_land = bathy.compute_land_mask(mesh.h.xy)
mask_land_w = bathy.compute_land_mask_w(mesh.h.xy)


# coriolis beta plane
f = mesh.generate_coriolis_mesh(f0=config.physics.f0, beta=config.physics.beta)
print(
    f"Coriolis param min {f.min().cpu().item():.2e},"
    f" {f.max().cpu().item():.2e}"
)

taux, tauy = wind.compute()

param = {
    "nx": config.mesh.nx,
    "ny": config.mesh.ny,
    "nl": config.layers.nl,
    "H": config.layers.h.unsqueeze(1).unsqueeze(1),
    "dx": config.mesh.dx,
    "dy": config.mesh.dy,
    "rho": config.physics.rho,
    "g_prime": config.layers.g_prime.unsqueeze(1).unsqueeze(1),
    "bottom_drag_coef": config.physics.bottom_drag_coef,
    "f": f,
    "device": DEVICE,
    "dtype": torch.float64,
    "slip_coef": config.physics.slip_coef,
    "dt": config.mesh.dt,  # time-step (s)
    "compile": True,
    "mask": bathy.compute_ocean_mask(mesh.h.xy),
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
    print(f"Starting from file {start_file}...")
    zipf = np.load(start_file)
    qg.set_physical_uvh(zipf["u"], zipf["v"], zipf["h"])
#########
t = 0

freq_checknan = 100
freq_log = int(24 * 3600 / config.mesh.dt)
n_steps = int(50 * 365 * 24 * 3600 / config.mesh.dt) + 1
n_steps_save = int(0 * 365 * 24 * 3600 / config.mesh.dt)
freq_save = int(5 * 24 * 3600 / config.mesh.dt)
freq_plot = int(config.io.plot_frequency * 24 * 3600 / config.mesh.dt)

uM, vM, hM = 0, 0, 0


if config.io.log_performance:
    from time import time as cputime

    nmeshpoints = param["nl"] * param["nx"] * param["ny"]
    mperf = 0
    [qg.step() for _ in range(5)]  # warm up


if freq_save > 0:
    output_dir = (
        f'{config.io.output_directory}/{name}_{config.mesh.nx}x{config.mesh.ny}_dt{config.mesh.dt}_'
        f'slip{param["slip_coef"]}/'
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to {output_dir}")
    np.save(os.path.join(output_dir, "mask_land_h.npz"), mask_land)
    np.save(
        os.path.join(output_dir, "mask_land_w.npz"),
        mask_land_w.numpy(),
    )
    torch.save(param, os.path.join(output_dir, "param.pth"))


if freq_plot > 0:
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
    t += config.mesh.dt

    if config.io.log_performance:
        walltime = cputime()
        perf = (walltime - walltime0) / (nmeshpoints)
        mperf += perf
        print(
            f"\rkt={n:4} time={t:.2f} perf={perf:.2e} ({mperf/n:.2e}) s",
            end="",
        )

    n_years = int(t // (365 * 24 * 3600))
    n_days = int(t % (365 * 24 * 3600) // (24 * 3600))

    if n % freq_checknan == 0 and torch.isnan(qg.p).any():
        raise ValueError(f"Stopping, NAN number in p at iteration {n}.")

    if freq_log > 0 and n % freq_log == 0:
        print(
            f"n={n:05d}, t={n_years:02d}y{n_days:03d}d, {qg.get_print_info()}"
        )

    if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
        u, v, h = qg.get_physical_uvh(numpy=True)
        uM, vM = max(uM, 0.8 * np.abs(u).max()), max(vM, 0.8 * np.abs(v).max())
        hM = max(hM, 0.8 * np.abs(h).max())
        if model == QG:
            wM = 0.05
            w = (qg.omega / qg.area / qg.f0).cpu().numpy()[0, nl_plot]
            w = np.ma.masked_where(mask_land_w.numpy(), w)
            a.imshow(w.T, vmin=-wM, vmax=wM, **plot_kwargs)
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
        u, v, h = qg.get_physical_uvh(numpy=True)
        if model == QG:
            # u_a = qg.u_a.cpu().numpy()
            # v_a = qg.v_a.cpu().numpy()
            np.savez(
                filename,
                u=u.astype("float32"),
                v=v.astype("float32"),
                # u_a=u_a.astype('float32'), v_a=v_a.astype('float32'),
                h=h.astype("float32"),
            )
            # print(f'saved u,v,h,u_a,v_a to {filename}')
            print(f"saved u,v,h to {filename}")
        else:
            np.savez(
                filename,
                u=u.astype("float32"),
                v=v.astype("float32"),
                h=h.astype("float32"),
            )
            print(f"saved u,v,h to {filename}")
