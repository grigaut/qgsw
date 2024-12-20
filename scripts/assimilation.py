"""Data assimilation pipeline."""

import argparse
from pathlib import Path

import torch
from rich.progress import Progress

from qgsw import verbose
from qgsw.configs.core import Configuration
from qgsw.fields.variables.uvh import UVH
from qgsw.models.qg.core import QG
from qgsw.run_summary import RunSummary
from qgsw.simulation.steps import Steps
from qgsw.spatial.core.coordinates import Coordinates1D
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
)
from qgsw.spatial.units._units import Unit
from qgsw.specs import DEVICE

torch.backends.cudnn.deterministic = True
verbose.set_level(2)


parser = argparse.ArgumentParser(description="Retrieve Configuration file.")
parser.add_argument(
    "--config",
    default="config/assimilation.toml",
    help="Configuration File Path (from qgsw root level)",
)
args = parser.parse_args()


ROOT_PATH = Path(__file__).parent.parent
CONFIG_PATH = ROOT_PATH.joinpath(args.config)
config = Configuration.from_toml(CONFIG_PATH)
summary = RunSummary.from_configuration(config)

if config.io.output.save:
    summary.to_file(config.io.output.directory)

space_2d = SpaceDiscretization2D.from_config(config.space)

h_coords_ref = Coordinates1D(
    points=config.simulation.reference.h,
    unit=Unit.M,
)
h_coords = Coordinates1D(points=config.model.h, unit=Unit.M)

space_3d_ref = space_2d.add_h(h_coords_ref)
space_3d_model = space_2d.add_h(h_coords)

tmp = SpaceDiscretization3D.from_config(
    config.space,
    config.model,
)

model_ref = QG(
    space_3d_ref,
    config.simulation.reference.g_prime.unsqueeze(-1).unsqueeze(-1),
    config.physics.beta_plane,
)

model = QG(
    space_3d_model,
    config.model.g_prime.unsqueeze(-1).unsqueeze(-1),
    config.physics.beta_plane,
)

model_ref.slip_coef = 1.0
model_ref.bottom_drag_coef = 3.60577e-8
model_ref.dt = config.simulation.dt

model.slip_coef = 1.0
model.bottom_drag_coef = 3.60577e-8
model.dt = config.simulation.dt

verbose.display("\n[Reference Model]", trigger_level=1)
verbose.display(msg=model_ref.__repr__(), trigger_level=1)
verbose.display("\n[Model]", trigger_level=1)
verbose.display(msg=model.__repr__(), trigger_level=1)

nl_ref = model_ref.space.nl
nl = model.space.nl
nx = model.space.nx
ny = model.space.ny

dtype = torch.float64
device = DEVICE.get()

if (startup_file := config.simulation.startup_file) is None:
    uvh0 = UVH.steady(
        n_ens=1,
        nl=nl_ref,
        nx=config.space.nx,
        ny=config.space.ny,
        dtype=torch.float64,
        device=DEVICE.get(),
    )
else:
    uvh0 = UVH.from_file(startup_file, dtype=dtype, device=device)
    horizontal_shape = uvh0.h.shape[-2:]
    if horizontal_shape != (nx, ny):
        msg = (
            f"Horizontal shape {horizontal_shape} from {startup_file}"
            f" should be ({nx},{ny})."
        )
        raise ValueError(msg)

model_ref.set_uvh(
    uvh0.u[:, :nl_ref, ...],
    uvh0.v[:, :nl_ref, ...],
    uvh0.h[:, :nl_ref, ...],
)
model.set_uvh(
    uvh0.u[:, :nl, ...],
    uvh0.v[:, :nl, ...],
    uvh0.h[:, :nl, ...],
)

dt = config.simulation.dt
t_end = config.simulation.duration

steps = Steps(t_end=t_end, dt=dt)

ns = steps.simulation_steps()
forks = steps.steps_from_interval(interval=config.simulation.fork_interval)
saves = config.io.output.get_saving_steps(steps)

summary.register_outputs(model.io)
summary.register_steps(t_end=t_end, dt=dt, n_steps=steps.n_tot)

t = 0

summary.register_start()
prefix_ref = config.simulation.reference.prefix
prefix = config.model.prefix
output_dir = config.io.output.directory
with Progress() as progress:
    simulation = progress.add_task(
        rf"\[n=00000/{steps.n_tot:05d}]",
        total=steps.n_tot,
    )
    for n, fork, save in zip(ns, forks, saves):
        progress.update(
            simulation,
            description=rf"\[n={n:05d}/{steps.n_tot:05d}]",
        )
        progress.advance(simulation)
        if save:
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}]",
                trigger_level=1,
            )
            verbose.display(
                msg="[Reference Model]: ",
                trigger_level=1,
                end="",
            )
            # Save Reference Model
            model_ref.io.save(output_dir.joinpath(f"{prefix_ref}{n}.npz"))
            verbose.display(
                msg="[     Model     ]: ",
                trigger_level=1,
                end="",
            )
            # Save Model
            model.io.save(output_dir.joinpath(f"{prefix}{n}.npz"))
            summary.register_step(n)
        if fork:
            uvh = model_ref.uvh
            model.set_uvh(
                torch.clone(uvh.u)[:, :nl, ...],
                torch.clone(uvh.v)[:, :nl, ...],
                torch.clone(uvh.h)[:, :nl, ...],
            )
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}] - Forked",
                trigger_level=1,
            )

        model_ref.step()
        model.step()
        t += dt

    summary.register_end()
