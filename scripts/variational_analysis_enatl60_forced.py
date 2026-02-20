"""Variational analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.ndimage import gaussian_filter

from qgsw.cli import ScriptArgsVAModified
from qgsw.configs.core import Configuration
from qgsw.decomposition.coefficients import DecompositionCoefs
from qgsw.decomposition.wavelets.core import WaveletBasis
from qgsw.decomposition.wavelets.param_generators import dyadic_decomposition
from qgsw.eNATL60.fields_computations import compute_stream_function_ssh_only
from qgsw.eNATL60.interpolation import (
    build_regridder,
    compute_lonlat_from_regular_xy_grid,
    lonlat_to_xy,
)
from qgsw.eNATL60.loading import load_datasets
from qgsw.eNATL60.var_keys import (
    LATITUDE,
    LONGITUDE,
    SSH,
    STREAMFUNCTION,
    TIME,
)
from qgsw.logging import getLogger, setup_root_logger
from qgsw.logging.utils import box, sec2text, step
from qgsw.masks import Masks
from qgsw.models.qg.psiq.core import QGPSIQ
from qgsw.models.qg.psiq.modified.forced import (
    QGPSIQForced,
)
from qgsw.observations import FullDomainMask, SatelliteTrackMask
from qgsw.optim.callbacks import LRChangeCallback
from qgsw.optim.utils import EarlyStop, RegisterParams
from qgsw.physics.constants import EARTH_ANGULAR_ROTATION, EARTH_RADIUS
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.solver.boundary_conditions.base import Boundaries
from qgsw.solver.finite_diff import laplacian
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
)
from qgsw.spatial.core.grid import Grid2D
from qgsw.spatial.core.grid_conversion import interpolate
from qgsw.specs import defaults
from qgsw.utils.interpolation import QuadraticInterpolation
from qgsw.utils.reshaping import crop
from qgsw.utils.storage import get_path_from_env

torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)

## Config

args = ScriptArgsVAModified.from_cli(
    comparison_default=1,
    cycles_default=3,
    prefix_default="results_enatl60_forced",
    gamma_default=0.1,
)
with_reg = not args.no_reg
with_alpha = not args.no_alpha
with_obs_track = args.obs_track

specs = defaults.get()

setup_root_logger(args.verbose)
logger = getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent
config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))

output_dir = config.io.output.directory

# Simulation parameters

dt = 7200
optim_max_step = 200
n_file_per_cycle = 20
n_steps_per_cyle = 240
comparison_interval = args.comparison
n_cycles = args.cycles

## Load eNATL60 grid

### Data folder

data_folder = get_path_from_env(key="DATA_FOLDER") / "MEANDERS"
files = list(
    data_folder.glob("eNATL60-BLB002_1h_2009*_2009*_gridT-2D_2009*-2009*.nc")
)


def sort_files_by_dates(file_paths: list[Path]) -> list[Path]:
    """Sort file names by dates."""
    times = pd.to_datetime([f.name[-20:-12] for f in file_paths])
    args = np.argsort(times)
    return np.array(files)[args].tolist()


def format_ds(ds: xr.Dataset) -> xr.Dataset:
    """Format Dataset."""
    # Drop useless variables
    if "axis_nbounds" in ds.dims:
        ds = ds.drop_dims("axis_nbounds")
    if "time_centered" in ds.coords:
        ds = ds.reset_coords("time_centered", drop=True)
    # Rename
    ds = ds.rename(
        {
            "time_counter": TIME,
            "nav_lon": LONGITUDE,
            "nav_lat": LATITUDE,
            "x": "i",
            "y": "j",
            "sossheig": SSH,
        }
    )
    ds = ds.transpose(TIME, "i", "j")
    return ds.set_coords([LONGITUDE, LATITUDE])


### Load only one file to access grid informations
files = sort_files_by_dates(files)

ds = load_datasets(files[0], format_func=format_ds)

### Compute longitude / latitudes
dx = dy = 10000
lons, lats = compute_lonlat_from_regular_xy_grid(
    ds[LONGITUDE],
    ds[LATITUDE],
    dx=dx,
    dy=dy,
)
xs, ys = lonlat_to_xy(lons, lats)

### Compute β-plane parameters

lat0 = (lats.max() + lats.min()) / 2
beta_plane = BetaPlane(
    f0=2 * EARTH_ANGULAR_ROTATION * np.sin(lat0),
    beta=2 * EARTH_ANGULAR_ROTATION * np.cos(lat0) / EARTH_RADIUS,
)

### Build regridder

psi_regridder = build_regridder(ds, lons, lats)


## Areas
nx, ny = lats.shape
space_2d = SpaceDiscretization2D.from_psi_grid(
    Grid2D(
        x=torch.tile(torch.arange(nx, **specs).reshape(-1, 1) * dx, (1, ny)),
        y=torch.tensor(ys.round(), **specs),
    )
)
### Boundaries offset

b = 4

space_interior = space_2d.slice(
    b,
    space_2d.psi.xy.x.shape[0] - b,
    b,
    space_2d.psi.xy.x.shape[1] - b,
)

nx = space_interior.nx
ny = space_interior.ny
dx = space_interior.dx
dy = space_interior.dy

## Observations

if with_obs_track:
    obs_mask = SatelliteTrackMask(
        space_interior.psi.xy.x,
        space_interior.psi.xy.y,
        track_width=100000,
        track_interval=400000,
        theta=torch.pi / 12,
        full_coverage_time=20 * 3600 * 24,
    )
    if comparison_interval != 1:
        msg = (
            "Using Satellite track, comparison interval"
            "inferred from tracks trajectory."
        )
        logger.warning(box(msg, style="="))
    msg_obs = "Surface observed along satellite tracks."
else:
    obs_mask = FullDomainMask(
        space_interior.psi.xy.x,
        space_interior.psi.xy.y,
        dt=comparison_interval * dt,
    )
    msg_obs = (
        f"Full surface observed every {sec2text(comparison_interval * dt)}"
    )


def update_loss(
    loss: torch.Tensor,
    f: torch.Tensor,
    f_ref: torch.Tensor,
    time: torch.Tensor,
) -> None:
    """Update loss."""
    mask = obs_mask.at_time(time)
    if not mask.any():
        return loss
    f_sliced = f.flatten()[mask.flatten()]
    f_ref_sliced = f_ref.flatten()[mask.flatten()]
    return loss + mse(f_sliced, f_ref_sliced)


## Regularization

gamma = args.gamma / comparison_interval

if with_reg:
    msg_reg = f"Using ɣ = {gamma:#8.3g} to weight regularization"  # noqa: RUF001
    if gamma != args.gamma:
        msg_reg += (
            f" (rescaled from ɣ = {args.gamma:#5.3g} to"  # noqa: RUF001
            " account for observations sparsity)."
        )
    else:
        msg_reg += "."
else:
    msg_reg = "No regularization."


## Output
prefix = args.complete_prefix()
filename = f"{prefix}.pt"
output_file = output_dir.joinpath(filename)

## Logs

msg_simu = (
    f"Performing {n_cycles} cycles of {n_steps_per_cyle} "
    f"steps with up to {optim_max_step} optimization steps."
)
lon_min = np.rad2deg(lons.min())
lon_max = np.rad2deg(lons.max())
lat_min = np.rad2deg(lats.min())
lat_max = np.rad2deg(lats.max())
msg_area = (
    f"Longitudes in [{lon_min:#.3g}°, {lon_max:#.3g}°],"
    f" latitudes in [{lat_min:#.3g}°, {lat_max:#.3g}°]."
)
msg_output = f"Output will be saved to {output_file}."

logger.info(box(msg_simu, msg_area, msg_obs, msg_reg, msg_output, style="="))

# Parameters

H = config.model.h
g_prime = config.model.g_prime
H1, H2 = H[0], H[1]
g1, g2 = g_prime[0], g_prime[1]
beta_plane = config.physics.beta_plane
bottom_drag_coef = config.physics.bottom_drag_coefficient
slip_coef = config.physics.slip_coef


## Error


def mse(f: torch.Tensor, f_ref: torch.Tensor) -> float:
    """RMSE."""
    return (f - f_ref).square().mean() / f_ref.square().mean()


# Models

## Inhomogeneous models
M = TypeVar("M", bound=QGPSIQ)


def set_inhomogeneous_model(model: M) -> M:
    """Set up inhomogeneous model."""
    model.masks = Masks.empty_tensor(nx, ny, device=specs["device"])
    model.bottom_drag_coef = 0
    model.wide = True
    model.slip_coef = slip_coef
    model.dt = dt
    return model


model = QGPSIQForced(
    space_2d=space_interior,
    H=H[:1] * H[1:2] / (H[:1] + H[1:2]),
    beta_plane=beta_plane,
    g_prime=g_prime[1:2],
)
model: QGPSIQForced = set_inhomogeneous_model(model)


# PV computation


y_w = space_2d.q.xy.y[0, :].unsqueeze(0)
beta_effect = beta_plane.beta * (y_w - model.y0)


def compute_q_rg(  # noqa: D103
    psi1: torch.Tensor,
) -> torch.Tensor:
    return (
        interpolate(
            laplacian(psi1, dx, dy)
            - beta_plane.f0**2
            * (1 / (H1 * H2 / (H1 + H2)) / g2)
            * psi1[..., 1:-1, 1:-1]
        )
        + beta_effect[:, 1:-1]
    )


def extract_psi_bc(psi: torch.Tensor) -> Boundaries:
    """Extract psi."""
    return Boundaries.extract(psi, b, -b - 1, b, -b - 1, 2)


def extract_q_bc(q: torch.Tensor) -> Boundaries:
    """Extract q."""
    return Boundaries.extract(q, b - 2, -(b - 1), b - 2, -(b - 1), 3)


outputs = []

L: float = dx.item()

for c in range(n_cycles):
    torch.cuda.reset_peak_memory_stats()

    if (c + 1) * n_file_per_cycle > len(files):
        msg = f"Not enough files to perform cycle {c} and above."
        logger.warning(msg)
        break

    ds = load_datasets(*sort_files_by_dates(files)[:20], format_func=format_ds)

    msg = f"Cycle {step(c + 1, n_cycles)}: eNATL60 data loaded."
    logger.info(box(msg, style="round"))

    ds[STREAMFUNCTION] = compute_stream_function_ssh_only(
        ds,
        g_prime[0].item(),
        remove_avg=True,
    )

    with logger.section("Filtering stream function..."):
        ds["psi_filt"] = xr.apply_ufunc(
            gaussian_filter,
            ds[STREAMFUNCTION].load(),
            kwargs={"sigma": 7},
            input_core_dims=[["i", "j"]],
            output_core_dims=[["i", "j"]],
            vectorize=True,
        )

    with logger.section("Interpolating stream function..."):
        regridded_psi: xr.DataArray = psi_regridder(ds[STREAMFUNCTION])
        regridded_psi_filt: xr.DataArray = psi_regridder(ds["psi_filt"])
        ds_interp = xr.Dataset(
            {
                LONGITUDE: (["i", "j"], lons),
                LATITUDE: (["i", "j"], lats),
                STREAMFUNCTION: ([TIME, "i", "j"], regridded_psi.data),
                "psi_filt": ([TIME, "i", "j"], regridded_psi_filt.data),
            },
            regridded_psi_filt.coords,
        )
        ds_interp = ds_interp.set_coords([LONGITUDE, LATITUDE])
        ds_interp = ds_interp.load()

    psis_ref = [
        torch.tensor(p, **specs).unsqueeze(0).unsqueeze(0) / beta_plane.f0
        for p in ds_interp[STREAMFUNCTION].to_numpy()
    ]

    with logger.section("Computing psi boundaries..."):
        psis_filt = [
            torch.tensor(p, **specs).unsqueeze(0).unsqueeze(0) / beta_plane.f0
            for p in ds_interp["psi_filt"].to_numpy()
        ]
        psi_bcs = [extract_psi_bc(psi) for psi in psis_filt]

    t0 = ds_interp[TIME][0]
    times = (ds_interp[TIME] - t0).dt.total_seconds().to_numpy()
    times = torch.tensor(times, **specs)

    psi0 = psis_filt[0]
    psi0_mean: float = psi0.mean()

    U: float = psi0_mean / L
    T = L / U

    psi_bc = extract_psi_bc(psi0)

    msg = f"Cycle {step(c + 1, n_cycles)}: eNATL60 data loaded and processed."
    logger.info(box(msg, style="round"))

    psi_bc_interp = QuadraticInterpolation(times, psi_bcs)

    xx = space_interior.psi.xy.x
    yy = space_interior.psi.xy.y

    space_params, time_params = dyadic_decomposition(
        order=5,
        xx_ref=space_2d.psi.xy.x,
        yy_ref=space_2d.psi.xy.y,
        Lxy_max=900_000,
        Lt_max=n_steps_per_cyle * dt,
    )

    basis = WaveletBasis(space_params, time_params)
    basis.n_theta = 7

    msg = f"Using basis of order {basis.order}"
    logger.info(msg)

    coefs = basis.generate_random_coefs()
    coefs = DecompositionCoefs.zeros_like(coefs)
    coefs = coefs.requires_grad_()

    numel = coefs.numel()
    msg = f"Control vector contains {numel} elements."
    logger.info(box(msg, style="round"))

    optimizer = torch.optim.Adam(
        [
            {
                "params": list(coefs.values()),
                "lr": 1e-2,
                "name": "Wavelet coefs",
            }
        ]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    lr_callback = LRChangeCallback(optimizer)
    early_stop = EarlyStop()

    coefs_scaled = coefs.scale(*(U**2 / L**2 for _ in range(basis.order)))
    register_params = RegisterParams(
        coefs=coefs_scaled.to_dict(),
    )

    for o in range(optim_max_step):
        optimizer.zero_grad()
        model.reset_time()

        with torch.enable_grad():
            coefs_scaled = coefs.scale(
                *(U**2 / L**2 for _ in range(basis.order))
            )

            basis.set_coefs(coefs_scaled)

            wv = basis.localize(space_interior.q.xy.x, space_interior.q.xy.y)

            q0 = crop(compute_q_rg(psi0), b - 1)

            qs = (compute_q_rg(p1) for p1 in psis_filt)
            q_bcs = [
                Boundaries.extract(q, b - 2, -(b - 1), b - 2, -(b - 1), 3)
                for q in qs
            ]
            q_bc_interp = QuadraticInterpolation(times, q_bcs)

            model.set_psiq(crop(psi0[:, :1], b), q0)
            model.set_boundary_maps(psi_bc_interp, q_bc_interp)

            loss = torch.tensor(0, **defaults.get())

            for n in range(1, n_steps_per_cyle):
                model.forcing = wv(model.time)[None, None, ...]
                model.step()

                loss = update_loss(
                    loss,
                    model.psi[0, 0],
                    crop(psis_ref[n][0, 0], b),
                    model.time,
                )
            if with_reg:
                for coef in coefs.values():
                    loss += gamma * coef.square().mean()

        if torch.isnan(loss.detach()):
            msg = "Loss has diverged."
            logger.warning(box(msg, style="="))
            break

        register_params.step(
            loss,
            coefs=coefs_scaled.to_dict(),
        )

        if early_stop.step(loss):
            msg = f"Convergence reached after {o + 1} iterations."
            logger.info(msg)
            break

        loss_ = loss.cpu().item()

        msg = (
            f"Cycle {step(c + 1, n_cycles)} | "
            f"Optimization step {step(o + 1, optim_max_step)} | "
            f"Loss: {loss_:>#12.7g} | "
            f"Best loss: {register_params.best_loss:>#12.7g}"
        )
        logger.info(msg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(coefs.values()), max_norm=1e0)

        optimizer.step()
        scheduler.step(loss)
        lr_callback.step()

    best_loss = register_params.best_loss
    msg = f"Optimization completed with loss: {best_loss:>#10.5g}"
    max_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    msg_mem = f"Max memory allocated: {max_mem:.1f} MB."
    logger.info(box(msg, msg_mem, style="round"))
    output = {
        "cycle": c,
        "config": {
            "comparison_interval": comparison_interval,
            "optimization_steps": [optim_max_step],
            "no-wind": args.no_wind,
            "basis": basis.get_params(),
        },
        "specs": {"max_memory_allocated": max_mem},
        "coefs": register_params.params["coefs"],
    }
    outputs.append(output)

torch.save(outputs, output_file)
msg = f"Outputs saved to {output_file}"
logger.info(box(msg, style="="))
