# ruff : noqa
import argparse
import os
import sys
import numpy as np
import skimage.morphology
import torch
import torch.nn.functional as F
import urllib.request
import netCDF4
import scipy.interpolate
import scipy.io
import scipy.ndimage
import getpass

sys.path.append("../src")
from qgsw.sw import SW
from qgsw.qg import QG

parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help="Configuration: 'med' for Mediterranea " " and 'na' for North-Atlantic.",
)
parser.add_argument("--log_perf", action="store_true", help="Log performance")
parser.add_argument("--freq_plot", type=int, default=0, help="plot frequence in days")
args = parser.parse_args()
config = args.config

# data and output directories
os.makedirs("./data", exist_ok=True)
# os.makedirs('./run_outputs', exist_ok=True)
usr = getpass.getuser()
disk = f"/srv/storage/ithaca@storage2.rennes.grid5000.fr/{usr}"
os.makedirs(disk, exist_ok=True)

## Grids
deg_to_km = 111e3
if config == "na":
    # nx, ny = 512, 256
    nx, ny = 1024, 512
    lat_min, lat_max = 9, 48
    Ly = (lat_max - lat_min) * deg_to_km
    Lx = Ly * nx / ny
    lon_min = -98
    lon_max = lon_min + Lx / (
        0.5
        * (np.cos(lat_min / 180 * np.pi) + np.cos(lat_max / 180 * np.pi))
        * deg_to_km
    )
    htop_ocean = -250
if config == "med":
    # nx, ny = 512, 256
    nx, ny = 1024, 512
    lat_min, lat_max = 31, 45
    Ly = (lat_max - lat_min) * deg_to_km
    Lx = Ly * nx / ny
    lon_min = -5.5
    lon_max = lon_min + Lx / (
        0.5
        * (np.cos(lat_min / 180 * np.pi) + np.cos(lat_max / 180 * np.pi))
        * deg_to_km
    )
    htop_ocean = -100

dx = Lx / nx
dy = Ly / ny
print(
    f"Grid lat: {lat_min:.1f}, {lat_max:.1f}, "
    f"lon: {lon_min:.1f}, {lon_max:.1f}, "
    f"dx={dx/1e3:.1f}km, dy={dy/1e3:.1f}km ."
)

x_cor = np.linspace(lon_min, lon_max, nx + 1)
y_cor = np.linspace(lat_min, lat_max, ny + 1)
# omega grid
lon_w, lat_w = np.meshgrid(x_cor, y_cor, indexing="ij")
x_cen = 0.5 * (x_cor[1:] + x_cor[:-1])
y_cen = 0.5 * (y_cor[1:] + y_cor[:-1])
# h grid
lon_h, lat_h = np.meshgrid(x_cen, y_cen, indexing="ij")
# u and v grid
lon_u, lat_u = np.meshgrid(x_cor, y_cen, indexing="ij")
lon_v, lat_v = np.meshgrid(x_cen, y_cor, indexing="ij")


## Topography
if config == "na":
    topo_file = "./data/northatlantic_topo.mat"
    if not os.path.isfile(topo_file):
        topo_url = "https://www.di.ens.fr/louis.thiry/AN_etopo1.mat"
        print(f"Downloading topo file {topo_file} from {topo_url}...")
        urllib.request.urlretrieve(topo_url, topo_file)
        print("..done")
    data = scipy.io.loadmat(topo_file)
    lon_bath = data["lon_bathy"][:, 0]
    lat_bath = data["lat_bathy"][:, 0]
    bathy = data["bathy"].T
    island_min_area = int(4000 * nx * ny / 1024 / 512)
    lake_min_area = int(40000 * nx * ny / 1024 / 512)
elif config == "med":
    topo_file = "./data/medsea_bathy.nc"
    if not os.path.isfile(topo_file):
        topo_url = "https://www.di.ens.fr/louis.thiry/medsea_bathy.nc"
        print(f"Downloading topo file {topo_file} from {topo_url}...")
        urllib.request.urlretrieve(topo_url, topo_file)
        print("..done")
    ds = netCDF4.Dataset(topo_file, "r")
    lon_bath = ds["lon"][:].data
    lat_bath = ds["lat"][:].data
    bathy = ds["elevation"][:].data.T
    print(
        f"Topo lat: {lat_bath.min():.2f} - {lat_bath.max():.2f}, "
        f"lon: {lon_bath.min():.2f} - {lon_bath.max():.2f}"
    )
    island_min_area = 1000
    lake_min_area = 40000

method = "linear"
print(f"Interpolating bathymetry on grid with {method} interpolation ...")
bathmetry = scipy.interpolate.RegularGridInterpolator(
    (lon_bath, lat_bath), bathy, method=method
)


mask_land = bathmetry((lon_h, lat_h)) > 0
mask_land = skimage.morphology.area_closing(mask_land, area_threshold=lake_min_area)
mask_land = np.logical_not(
    skimage.morphology.area_closing(
        np.logical_not(mask_land), area_threshold=island_min_area
    )
)
mask_land_w = (
    F.avg_pool2d(
        F.pad(
            torch.from_numpy(mask_land).type(torch.float64).unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            value=1.0,
        ),
        (2, 2),
        stride=(1, 1),
    )[0, 0]
    > 0.5
).numpy()

mask_ocean = bathmetry((lon_h, lat_h)) < htop_ocean
mask_ocean = skimage.morphology.area_closing(mask_ocean, area_threshold=island_min_area)
mask_ocean = np.logical_not(
    skimage.morphology.area_closing(
        np.logical_not(mask_ocean), area_threshold=lake_min_area
    )
).astype("float64")

bottom_topography = 4000 + np.clip(bathmetry((lon_h, lat_h)), -4000, 0)
bottom_topography = np.clip(
    scipy.ndimage.gaussian_filter(bottom_topography, 3.0), 0, 150
)

# remove ocean cells surrounded by 3 non-ocean cells
for _ in range(100):
    mask_ocean[1:-1, 1:-1] += (1 - mask_ocean[1:-1, 1:-1]) * (
        (
            mask_ocean[:-2, 1:-1]
            + mask_ocean[2:, 1:-1]
            + mask_ocean[1:-1, 2:]
            + mask_ocean[1:-1, :-2]
        )
        > 2.5
    )
mask_ocean = (mask_ocean > 0.5).astype("float64")
mask = mask_ocean


if config == "na":
    wind_file = "./data/windstress_HellermanRosenstein83.nc"
    if not os.path.isfile(wind_file):
        wind_url = "http://iridl.ldeo.columbia.edu/SOURCES/.HELLERMAN/data.nc"
        print(f"Downloading wind file {wind_file} from {wind_url}...")
        urllib.request.urlretrieve(wind_url, wind_file)

    ds = netCDF4.Dataset(wind_file, "r")
    X = ds.variables["X"][:].data
    Y = ds.variables["Y"][:].data
    T = ds.variables["T"][:].data

    taux = np.zeros((T.shape[0] + 1, nx + 1, ny))
    tauy = np.zeros((T.shape[0] + 1, nx, ny + 1))

    method = "linear"
    print(f"Interpolating wind forcing on grid with {method} interpolation ...")
    for t in range(T.shape[0]):
        ## linear interpolation with scipy
        taux_ref = ds.variables["taux"][:].data[t].T
        tauy_ref = ds.variables["tauy"][:].data[t].T
        taux_interpolator = scipy.interpolate.RegularGridInterpolator(
            (X, Y), taux_ref, method=method
        )
        tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
            (X, Y), tauy_ref, method=method
        )
        taux_i = taux_interpolator((lon_u + 360, lat_u))
        tauy_i = tauy_interpolator((lon_v + 360, lat_v))

        ## bicubic interpolation with torch
        # taux_ref = torch.from_numpy(ds.variables['taux'][:].data[t].T).unsqueeze(0).unsqueeze(0).type(torch.float64)
        # lon_u_norm = torch.from_numpy(((lon_u % 360) - 180) / 180)
        # lat_u_norm = torch.from_numpy(lat_u / 90)
        # grid_u_norm = torch.stack([lat_u_norm, lon_u_norm], dim=-1).unsqueeze(0).type(torch.float64)
        # taux_i = torch.nn.functional.grid_sample(taux_ref, grid_u_norm, mode='bicubic', padding_mode='border', align_corners=False)

        # tauy_ref = torch.from_numpy(ds.variables['tauy'][:].data[t].T).unsqueeze(0).unsqueeze(0).type(torch.float64)
        # lon_v_norm = torch.from_numpy(((lon_v % 360) - 180) / 180)
        # lat_v_norm = torch.from_numpy(lat_v / 90)
        # grid_v_norm = torch.stack([lat_v_norm, lon_v_norm], dim=-1).unsqueeze(0).type(torch.float64)
        # tauy_i = torch.nn.functional.grid_sample(tauy_ref, grid_v_norm, mode='bicubic', padding_mode='border', align_corners=False)

        taux[t, :, :] = taux_i
        tauy[t, :, :] = tauy_i

    taux *= 1e-4
    tauy *= 1e-4
    taux[-1][:] = taux[0][:]
    tauy[-1][:] = tauy[0][:]
elif config == "med":
    wind_file = "./data/wind_medsea_2010.nc"
    if not os.path.isfile(wind_file):
        wind_url = "https://www.di.ens.fr/louis.thiry/wind_medsea_2010.nc"
        print(f"Downloading wind file {wind_file} from {wind_url}...")
        urllib.request.urlretrieve(wind_url, wind_file)

    ds = netCDF4.Dataset(wind_file, "r")
    X = ds.variables["longitude"][:].data.astype("float64")
    Y = ds.variables["latitude"][:].data.astype("float64")[::-1]
    T = ds.variables["time"][:].data.astype("float64")

    print(
        f"Wind lat: {Y.min():.2f} - {Y.max():.2f}, "
        f"lon: {X.min():.2f} - {X.max():.2f}"
    )

    drag_coefficient = 1.3e-3
    rho_ocean = 1e3

    taux = np.zeros((T.shape[0] + 1, nx + 1, ny))
    tauy = np.zeros((T.shape[0] + 1, nx, ny + 1))

    method = "linear"
    print(f"Interpolating wind forcing on grid with {method} interpolation ...")
    for t in range(T.shape[0]):
        u = ds.variables["u10"][:].data[t].T[:, ::-1]
        v = ds.variables["v10"][:].data[t].T[:, ::-1]
        unorm = np.sqrt(u**2 + v**2)
        taux_ref = drag_coefficient / rho_ocean * unorm * u
        tauy_ref = drag_coefficient / rho_ocean * unorm * v
        taux_interpolator = scipy.interpolate.RegularGridInterpolator(
            (X, Y), taux_ref, method=method
        )
        tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
            (X, Y), tauy_ref, method=method
        )
        taux_i = taux_interpolator((lon_u, lat_u))
        tauy_i = tauy_interpolator((lon_v, lat_v))
        taux[t, :, :] = taux_i
        tauy[t, :, :] = tauy_i

    taux[-1][:] = taux[0][:]
    tauy[-1][:] = tauy[0][:]


dtype, device = torch.float64, "cuda" if torch.cuda.is_available() else "cpu"
taux = torch.from_numpy(taux).type(dtype).to(device)
tauy = torch.from_numpy(tauy).type(dtype).to(device)

torch.backends.cudnn.deterministic = True

H = torch.zeros(3, 1, 1, dtype=dtype, device=device)
H[0, 0, 0] = 400.0
H[1, 0, 0] = 1100.0
H[2, 0, 0] = 2600.0

# density/gravity
rho = 1000
g_prime = torch.zeros(3, 1, 1, dtype=dtype, device=device)
g_prime[0, 0, 0] = 9.81
g_prime[1, 0, 0] = 0.025
g_prime[2, 0, 0] = 0.0125


# coriolis beta plane
f0 = 9.375e-5  # mean coriolis (s^-1)
beta = 1.754e-11  # coriolis gradient (m^-1 s^-1)
y = torch.from_numpy(lat_w).type(dtype).to(device) * deg_to_km
f = f0 + beta * (y - y.mean())
print(f"Coriolis param min {f.min().cpu().item():.2e}, {f.max().cpu().item():.2e}")

# bottom drag
bottom_drag_coef = 0.5 * f0 * 2.0 / 2600


dt = 2000
param = {
    "nx": nx,
    "ny": ny,
    "nl": 3,
    "H": H,
    "dx": dx,
    "dy": dy,
    "rho": rho,
    "g_prime": g_prime,
    "bottom_drag_coef": bottom_drag_coef,
    "f": f,
    "device": device,
    "dtype": dtype,
    "slip_coef": 0.6,
    "dt": dt,  # time-step (s)
    "compile": True,
    "mask": torch.from_numpy(mask).type(dtype).to(device),
    "taux": taux[0, 1:-1, :],
    "tauy": tauy[0, :, 1:-1],
}


model = SW
model = QG
qg = model(param)

name = f"qg_{config}"


start_file = ""
if start_file:
    print(f"Starting from file {start_file}...")
    zipf = np.load(start_file)
    qg.set_physical_uvh(zipf["u"], zipf["v"], zipf["h"])

t = 0

freq_checknan = 100
freq_log = int(24 * 3600 / dt)
n_steps = int(50 * 365 * 24 * 3600 / dt) + 1
n_steps_save = int(0 * 365 * 24 * 3600 / dt)
freq_save = int(5 * 24 * 3600 / dt)
freq_plot = int(args.freq_plot * 24 * 3600 / dt)

uM, vM, hM = 0, 0, 0


if args.log_perf:
    from time import time as cputime

    ngridpoints = param["nl"] * param["nx"] * param["ny"]
    mperf = 0
    [qg.step() for _ in range(5)]  # warm up


if freq_save > 0:
    output_dir = f'{disk}/{name}_{nx}x{ny}_dt{dt}_' f'slip{param["slip_coef"]}/'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to {output_dir}")
    np.save(os.path.join(output_dir, "mask_land_h.npz"), mask_land)
    np.save(os.path.join(output_dir, "mask_land_w.npz"), mask_land_w)
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

    if args.log_perf:
        walltime0 = cputime()

    ## one step
    qg.step()
    t += dt

    if args.log_perf:
        walltime = cputime()
        perf = (walltime - walltime0) / (ngridpoints)
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
        print(f"n={n:05d}, t={n_years:02d}y{n_days:03d}d, {qg.get_print_info()}")

    if freq_plot > 0 and (n % freq_plot == 0 or n == n_steps):
        u, v, h = qg.get_physical_uvh(numpy=True)
        uM, vM = max(uM, 0.8 * np.abs(u).max()), max(vM, 0.8 * np.abs(v).max())
        hM = max(hM, 0.8 * np.abs(h).max())
        if model == QG:
            wM = 0.05
            w = (qg.omega / qg.area / qg.f0).cpu().numpy()[0, nl_plot]
            w = np.ma.masked_where(mask_land_w, w)
            a.imshow(w.T, vmin=-wM, vmax=wM, **plot_kwargs)
        else:
            a[0].imshow(u[0, nl_plot].T, vmin=-uM, vmax=uM, **plot_kwargs)
            a[1].imshow(v[0, nl_plot].T, vmin=-vM, vmax=vM, **plot_kwargs)
            a[2].imshow(h[0, nl_plot].T, vmin=-hM, vmax=hM, **plot_kwargs)

        f.suptitle(f"{n_years} yrs, {n_days:03d} days")
        plt.pause(0.05)

    if freq_save > 0 and n > n_steps_save and n % freq_save == 0:
        filename = os.path.join(output_dir, f"uvh_{n_years:03d}y_{n_days:03d}d.npz")
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
