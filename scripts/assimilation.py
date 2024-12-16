"""Data assimilation pipeline."""

from pathlib import Path

import numpy as np
import torch
from rich.progress import Progress

from qgsw import verbose
from qgsw.models.qg.core import QG
from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.core.discretization import SpaceDiscretization3D
from qgsw.spatial.units._units import METERS
from qgsw.specs import DEVICE

verbose.set_level(2)

t_end = 31_536_000
dt = 3600

x_min = 0
x_max = 2_560_000

y_min = 0
y_max = 5_120_000

nx = 128
ny = 256

x = torch.linspace(
    x_min,
    x_max,
    nx + 1,
    dtype=torch.float64,
    device=DEVICE.get(),
)
y = torch.linspace(
    y_min,
    y_max,
    ny + 1,
    dtype=torch.float64,
    device=DEVICE.get(),
)
h_3l = torch.tensor(
    [400, 1100, 2600],
    dtype=torch.float64,
    device=DEVICE.get(),
)
h_1l = torch.tensor(
    [4100],
    dtype=torch.float64,
    device=DEVICE.get(),
)

space_3l = SpaceDiscretization3D.from_tensors(
    x_unit=METERS,
    y_unit=METERS,
    zh_unit=METERS,
    x=x,
    y=y,
    h=h_3l,
)

space_1l = SpaceDiscretization3D.from_tensors(
    x_unit=METERS,
    y_unit=METERS,
    zh_unit=METERS,
    x=x,
    y=y,
    h=h_1l,
)

g_prime_3l = torch.tensor(
    [9.81, 0.025, 0.0125],
    dtype=torch.float64,
    device=DEVICE.get(),
)

g_prime_1l = torch.tensor(
    [9.81],
    dtype=torch.float64,
    device=DEVICE.get(),
)

model_3l = QG(
    space_3l,
    g_prime_3l.unsqueeze(-1).unsqueeze(-1),
    BetaPlane(f0=9.375e-5, beta=1.754e-11),
)

model_1l = QG(
    space_1l,
    g_prime_1l.unsqueeze(-1).unsqueeze(-1),
    BetaPlane(f0=9.375e-5, beta=1.754e-11),
)

model_3l.slip_coef = 1.0
model_3l.bottom_drag_coef = 3.60577e-8
model_3l.dt = dt

model_1l.slip_coef = 1.0
model_1l.bottom_drag_coef = 3.60577e-8
model_1l.dt = dt

file0 = np.load("output/g5k/double_gyre_qg_long/results_step_157698.npz")

model_3l.set_uvh(
    torch.tensor(file0["u"], dtype=torch.float64, device=DEVICE.get()),
    torch.tensor(file0["v"], dtype=torch.float64, device=DEVICE.get()),
    torch.tensor(file0["h"], dtype=torch.float64, device=DEVICE.get()),
)
model_1l.set_uvh(
    torch.tensor(file0["u"], dtype=torch.float64, device=DEVICE.get())[
        :,
        :1,
        ...,
    ],
    torch.tensor(file0["v"], dtype=torch.float64, device=DEVICE.get())[
        :,
        :1,
        ...,
    ],
    torch.tensor(file0["h"], dtype=torch.float64, device=DEVICE.get())[
        :,
        :1,
        ...,
    ],
)
verbose.display("\n[Model 3 Layers]", trigger_level=1)
verbose.display(msg=model_3l.__repr__(), trigger_level=1)
verbose.display("\n[Model 1 Layer]", trigger_level=1)
verbose.display(msg=model_1l.__repr__(), trigger_level=1)

n_steps = t_end // dt + 1
t = 0
fork = -1
save_dir = Path("output/local/assimilation")

with Progress() as progress:
    simulation = progress.add_task(
        rf"\[n=00000/{n_steps:05d}]",
        total=n_steps,
    )
    for n in range(n_steps + 1):
        progress.update(
            simulation,
            description=rf"\[n={n:05d}/{n_steps:05d}]",
        )
        progress.advance(simulation)
        if abs(n * dt - (fork + 1) * dt * 24 * 20) < dt:
            verbose.display(
                msg=f"[n={n:05d}/{n_steps:05d}] - Forked",
                trigger_level=1,
            )
            verbose.display(
                msg=f"Model 3 Layers: {model_3l.io.print_step()}",
                trigger_level=1,
            )
            model_3l.io.save(save_dir.joinpath(f"model_3l_{n}.npz"))
            verbose.display(
                msg=f"Model 1 Layer: {model_1l.io.print_step()}",
                trigger_level=1,
            )
            model_1l.io.save(save_dir.joinpath(f"model_1l_{n}.npz"))
            uvh = model_3l.uvh
            model_1l.set_uvh(
                torch.clone(uvh.u)[:, :1, ...],
                torch.clone(uvh.v)[:, :1, ...],
                torch.clone(uvh.h)[:, :1, ...],
            )
            fork += 1

        model_1l.step()
        model_3l.step()
        t += dt
