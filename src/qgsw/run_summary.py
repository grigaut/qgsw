"""Generate run summaries."""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from qgsw.utils.sorting import sort_files
from qgsw.variables.base import ParsedVariable

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import toml

from qgsw.configs.core import Configuration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qgsw.models.io import IO


class SummaryError(Exception):
    """Summary Related Error."""


class RunSummary:
    """Run summary."""

    _summary_file_name = "_summary.toml"
    _config_file_name = "_config.toml"
    _time_start = "time_start"
    _time_end = "time_end"
    _qgsw_version = "qgsw_version"
    _duration = "total_duration"
    _dt = "dt"
    _total_steps = "n_steps"
    _n = "last_registered_step"
    _finished = "finished_run"
    _variables = "output-vars"

    def __init__(
        self,
        run_params: dict[str, Any],
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Instantiate run summary.

        Args:
            run_params (dict[str, Any]): Run parameters.
            summary (dict[str, Any]): Run summary.
        """
        if summary is not None:
            self._has_summary = True
            self._summary = summary
        else:
            self._has_summary = False
            self._summary = {}
            self._summary[self._variables] = []
            self._summary[self._qgsw_version] = version("qgsw")
        self._files: list[Path] = []
        self._config = Configuration(**run_params)

    @property
    def configuration(self) -> Configuration:
        """Run Configuration."""
        return self._config

    @property
    def output_vars(self) -> list[dict[str, str]]:
        """Output variables."""
        if self._variables in self._summary:
            return self._summary[self._variables]
        msg = "Output varibales not registered."
        raise SummaryError(msg)

    @property
    def qgsw_version(self) -> str:
        """QGSW version."""
        if self._qgsw_version in self._summary:
            return self._summary[self._qgsw_version]
        msg = "QGSW version not registered."
        raise SummaryError(msg)

    @property
    def duration(self) -> float:
        """Duration (in seconds)."""
        if self._duration in self._summary:
            return self._summary[self._duration]
        msg = "Duration not registered."
        raise SummaryError(msg)

    @property
    def dt(self) -> float:
        """Timestep (in seconds)."""
        if self._dt in self._summary:
            return self._summary[self._dt]
        msg = "Timestep not registered."
        raise SummaryError(msg)

    @property
    def total_steps(self) -> int:
        """Total Steps."""
        if self._total_steps in self._summary:
            return self._summary[self._total_steps]
        msg = "Total steps number not registered."
        raise SummaryError(msg)

    @property
    def last_step(self) -> int:
        """Last registered step."""
        if self._n in self._summary:
            return self._summary[self._n]
        msg = "No step registered."
        raise SummaryError(msg)

    @property
    def is_finished(self) -> bool:
        """Whether the run was finished or not."""
        if self._finished in self._summary:
            return self._summary[self._finished]
        msg = "Run status not registered."
        raise SummaryError(msg)

    @property
    def started_at(self) -> str:
        """Starting time."""
        if self._time_start in self._summary:
            return self._summary[self._time_start]
        return "Unknown."

    @property
    def ended_at(self) -> str:
        """Ending time."""
        if self._time_end in self._summary:
            return self._summary[self._time_end]
        if not self.is_finished:
            return "Uncomplete simulation."
        return "Unknown."

    def raise_if_not_writable(self) -> None:
        """Raise an error is the configuration ins not writable.

        Raises:
            SummaryError: If the summary section existed at instanciation.
        """
        if self._has_summary:
            msg = "Summary section already existed, impossible to override."
            raise SummaryError(msg)

    def register_steps(self, *, t_end: float, dt: float, n_steps: int) -> None:
        """Register steps parameters.

        Args:
            t_end (float): Total simulation time in seconds.
            dt (float): Timestep in seconds.
            n_steps (int): Number of steps.
        """
        self.raise_if_not_writable()
        self._summary[self._duration] = t_end
        self._summary[self._dt] = dt
        self._summary[self._total_steps] = n_steps
        if self._files:
            self.update()

    def register_start(self) -> None:
        """Register the start of the simulation."""
        self.raise_if_not_writable()
        self._summary[self._finished] = False
        time_start = datetime.now().astimezone().isoformat(" ", "seconds")
        self._summary[self._time_start] = time_start
        if self._files:
            self.update()

    def register_step(self, step: int) -> None:
        """Register a simulation step.

        Args:
            step (int): Number of the step.
        """
        self.raise_if_not_writable()
        self._summary[self._n] = step
        if self._files:
            self.update()

    def register_end(self) -> None:
        """Register the end of the simulation."""
        self.raise_if_not_writable()
        self._summary[self._finished] = True
        time_end = datetime.now().astimezone().isoformat(" ", "seconds")
        self._summary[self._time_end] = time_end
        if self._files:
            self.update()

    def register_outputs(self, io: IO) -> None:
        """Register the model outputs.

        Args:
            io (IO): Input/Output manager.
        """
        if not self.configuration.io.output.save:
            return
        for var in io.tracked_vars:
            self._summary[self._variables].append(
                var.to_dict(),
            )

    def to_file(self, folder: Path) -> None:
        """Save the summary into a file.

        Multiple calls will add as many files as required.

        Args:
            folder (Path): Folder to save in.
        """
        self.raise_if_not_writable()
        summary_file = folder.joinpath(self._summary_file_name)
        toml.dump(self._summary, summary_file.open("w"))
        if summary_file not in self._files:
            self._files.append(summary_file)
        config_file = folder.joinpath(self._config_file_name)
        toml.dump(
            self._config.model_dump(by_alias=True),
            config_file.open("w"),
        )

    def update(self) -> None:
        """Update the saved files.

        Raises:
            SummaryError: If no files are registered.
        """
        self.raise_if_not_writable()
        if not self._files:
            msg = "Impossible to update the summary, use .to_file first."
            raise SummaryError(msg)
        if not self._files:
            msg = "Impossible to update the summary, use .to_file first."
            raise SummaryError(msg)
        for file in self._files:
            toml.dump(self._summary, file.open("w"))

    @classmethod
    def from_folder(cls, folder: Path) -> Self:
        """Create the summary from a TOML file.

        Args:
            folder (Path): Folder to create from.

        Returns:
            Self: Summary.
        """
        config_file = folder.joinpath(cls._config_file_name)
        if not config_file.is_file():
            msg = f"No configuration file to load from at {config_file}."
            raise SummaryError(msg)

        run_params = toml.load(config_file)

        summary_file = folder.joinpath(cls._summary_file_name)
        if not summary_file.is_file():
            return cls(run_params, None)

        summary = toml.load(summary_file)

        return cls(run_params, summary)

    @classmethod
    def from_configuration(cls, configuration: Configuration) -> Self:
        """Create the summary from a configuration.

        Args:
            configuration (Configuration): Configuration.

        Returns:
            Self: Summary.
        """
        return cls(configuration.model_dump(by_alias=True), None)


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
