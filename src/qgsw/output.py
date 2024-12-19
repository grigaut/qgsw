"""Output management."""

from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from qgsw.run_summary import RunSummary
from qgsw.utils.sorting import sort_files
from qgsw.variables.base import ParsedVariable

if TYPE_CHECKING:
    from collections.abc import Iterator


class OutputFile(NamedTuple):
    """Output file wrapper."""

    step: int
    second: float
    timestep: timedelta
    path: Path

    def read(self) -> np.ndarray:
        """Read the file data.

        Returns:
            np.ndarray: Data.
        """
        return np.load(file=self.path)


class RunOutput:
    """Run output."""

    def __init__(self, folder: Path, prefix: str | None = None) -> None:
        """Instantiate run output.

        Args:
            folder (Path): Run output folder.
            prefix (str | None): Prefix to use to select files.
        """
        self._folder = Path(folder)
        self._summary = RunSummary.from_folder(self.folder)
        if prefix is None:
            prefix = self.summary.configuration.model.prefix
        files = list(self.folder.glob(f"{prefix}*.npz"))
        steps, files = sort_files(files, prefix, ".npz")
        dt = self._summary.configuration.simulation.dt
        seconds = [step * dt for step in steps]
        timesteps = [timedelta(seconds=sec) for sec in seconds]

        self._outputs = [
            OutputFile(
                step=steps[i],
                timestep=timesteps[i],
                path=files[i],
                second=seconds[i],
            )
            for i in range(len(files))
        ]

        self._vars = {
            var["name"]: ParsedVariable.from_dict(var, outputs=self._outputs)
            for var in self._summary.output_vars
        }

    @cached_property
    def folder(self) -> Path:
        """Run output folder."""
        return Path(self._folder)

    @property
    def summary(self) -> RunSummary:
        """Run summary."""
        return self._summary

    @property
    def output_vars(self) -> list[ParsedVariable]:
        """Output variables."""
        return list(self._vars.values())

    def __repr__(self) -> str:
        """Output Representation."""
        vars_txt = "\n\t".join(str(var) for var in self.output_vars)
        msg_txts = [
            f"Simulation: {self._summary.configuration.io.name}.",
            f"Starting time: {self._summary.started_at}",
            f"Ending time: {self._summary.ended_at}",
            f"Package version: {self._summary.qgsw_version}.",
            f"Folder: {self.folder}.",
            f"Variables:\n\t{vars_txt}",
        ]
        return "\n".join(msg_txts)

    def __getitem__(self, key: str) -> ParsedVariable:
        """Get variable based on its name.

        Args:
            key (str): Variable name.

        Returns:
            ParsedVariable: _description_
        """
        if key not in self._vars:
            msg = f"Variables present in the output are {self._vars.keys()}."
            raise KeyError(msg)
        return self._vars[key]

    def steps(self) -> Iterator[int]:
        """Sorted list of steps.

        Yields:
            Iterator[float]: Steps iterator.
        """
        return (output.step for output in iter(self._outputs))

    def timesteps(self) -> Iterator[timedelta]:
        """Sorted list of timesteps.

        Yields:
            Iterator[datetime.timedelta]: Timesteps iterator.
        """
        return (output.timestep for output in iter(self._outputs))

    def seconds(self) -> Iterator[float]:
        """Sorted list of seconds.

        Yields:
            Iterator[float]: Seconds iterator.
        """
        return (output.second for output in iter(self._outputs))

    def outputs(self) -> Iterator[OutputFile]:
        """Sorted outputs.

        Returns:
            Iterator[OutputFile]: Outputs iterator.
        """
        return iter(self._outputs)


def check_time_compatibility(*runs: RunOutput) -> None:
    """Check time compatibilities between run outputs.

    Args:
        *runs (RunOutput): Run outputs.

    Raises:
        ValueError: If the timestep don't match.
    """
    dt0 = runs[0].summary.configuration.simulation.dt
    dts = [run.summary.configuration.simulation.dt for run in runs]
    if not all(dt == dt0 for dt in dts):
        msg = "Incompatible timesteps."
        raise ValueError(msg)
