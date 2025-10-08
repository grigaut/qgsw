"""Run a single model with a Vortex forcing."""

from pathlib import Path

import torch
from rich.progress import Progress

from qgsw.cli import ScriptArgs
from qgsw.configs.core import Configuration
from qgsw.logging import getLogger, setup_root_logger
from qgsw.models.instantiation import (
    instantiate_model_from_config,
)
from qgsw.run_summary import RunSummary
from qgsw.simulation.steps import Steps
from qgsw.specs import defaults

torch.backends.cudnn.deterministic = True

args = ScriptArgs.from_cli()
specs = defaults.get()

setup_root_logger(args.verbose)
logger = getLogger(__name__)

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

## time params
t = 0
dt = model.dt

t_end = config.simulation.duration


steps = Steps(t_start=0, t_end=t_end, dt=dt)
ns = steps.simulation_steps()
saves = config.io.output.get_saving_steps(steps)
logs = steps.steps_from_total(100)

summary.register_outputs(model.io)
summary.register_steps(t_end=t_end, dt=dt, n_steps=steps.n_tot)

logger.info(model.__repr__())
msg = f"Total Duration: {t_end:.2f} seconds"
logger.info(msg)


summary.register_start()
prefix = config.model.prefix
# Start runs
with Progress() as progress:
    simulation = progress.add_task(
        rf"\[n=00000/{steps.n_tot:05d}]",
        total=steps.n_tot,
    )
    for n, save, log in zip(ns, saves, logs):
        progress.update(
            simulation,
            description=rf"\[n={n:05d}/{steps.n_tot:05d}]",
        )
        progress.advance(simulation)
        # Save step
        if save:
            msg = f"[n={n:05d}/{steps.n_tot:05d}]"
            logger.info(msg)
            directory = config.io.output.directory
            model.io.save(directory.joinpath(f"{prefix}{n}.pt"))

        model.step()
        t += dt
        # Step log
        if log:
            msg = f"[n={n:05d}/{steps.n_tot:05d}]"
            logger.info(msg)
            logger.info(model.io.print_step())
            summary.register_step(n)

    summary.register_end()
