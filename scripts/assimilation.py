"""Data assimilation pipeline."""

from pathlib import Path

import torch
from rich.progress import Progress

from qgsw import verbose
from qgsw.cli import ScriptArgs
from qgsw.configs.core import Configuration
from qgsw.fields.variables.coefficients.instantiation import instantiate_coef
from qgsw.models.instantiation import (
    instantiate_model_from_config,
)
from qgsw.models.names import ModelName
from qgsw.models.qg.uvh.modified.utils import is_modified
from qgsw.models.references.core import load_reference
from qgsw.models.synchronization.initial_conditions import InitialCondition
from qgsw.models.synchronization.sync import Synchronizer
from qgsw.run_summary import RunSummary
from qgsw.simulation.steps import Steps
from qgsw.specs import defaults

torch.backends.cudnn.deterministic = True

args = ScriptArgs.from_cli()

verbose.set_level(args.verbose)
specs = defaults.get()

ROOT_PATH = Path(__file__).parent.parent
config = Configuration.from_toml(ROOT_PATH.joinpath(args.config))
summary = RunSummary.from_configuration(config)

if config.io.output.save:
    summary.to_file(config.io.output.directory)

model = instantiate_model_from_config(
    config.model,
    config.space,
    config.windstress,
    config.physics,
    config.perturbation,
    config.simulation,
    **specs,
)
ic = InitialCondition(model)
ref = load_reference(config)

synchronize = Synchronizer(ref, model)

nl = model.space.nl
if model.get_type() == ModelName.QG_SANITY_CHECK:
    nl += 1

nx = model.space.nx
ny = model.space.ny

dt = model.dt
t_end = config.simulation.duration

steps = Steps(t_start=0, t_end=t_end, dt=dt)
verbose.display(steps.__repr__(), trigger_level=1)

ns = steps.simulation_steps()
syncs = steps.steps_from_interval(interval=config.simulation.fork_interval)
saves = config.io.output.get_saving_steps(steps)

summary.register_outputs(model.io)
summary.register_steps(t_end=t_end, dt=dt, n_steps=steps.n_tot)

t = 0

summary.register_start()
prefix = config.model.prefix
output_dir = config.io.output.directory

# Collinearity Coefficient
modified = is_modified(config.model.type)
if modified:
    coef = instantiate_coef(config.model, config.space)
    model.alpha = coef.get()


with Progress() as progress:
    simulation = progress.add_task(
        rf"\[n=00000/{steps.n_tot:05d}]",
        total=steps.n_tot,
    )
    for n, sync, save in zip(ns, syncs, saves):
        progress.update(
            simulation,
            description=rf"\[n={n:05d}/{steps.n_tot:05d}]",
        )
        progress.advance(simulation)
        if sync:
            synchronize()
        if save:
            verbose.display(
                msg=f"[n={n:05d}/{steps.n_tot:05d}]",
                trigger_level=1,
            )
            # Save Model
            model.io.save(output_dir.joinpath(f"{prefix}{n}.pt"))
            summary.register_step(n)
        model.step()
        t += dt

    summary.register_end()
