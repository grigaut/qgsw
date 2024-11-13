"""Generate run summaries."""

import datetime
from collections.abc import Iterator
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from qgsw.utils.sorting import sort_files

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import toml

from qgsw.configs import Configuration


class SummaryError(Exception):
    """Summary Related Error."""


class RunSummary:
    """Run summary."""

    _summary_section = "run-summary"
    _qgsw_version = "qgsw_version"
    _configuration = "configuration"
    _duration = "total_duration"
    _dt = "dt"
    _total_steps = "n_steps"
    _n = "last_registered_step"
    _finished = "finished_run"

    def __init__(self, run_params: dict[str, Any]) -> None:
        """Instantiate run summary.

        Args:
            run_params (dict[str, Any]): Run parameters.
        """
        if self._summary_section in run_params:
            self._has_summary = True
            self._summary = run_params
        else:
            self._has_summary = False
            self._summary = {self._configuration: run_params}
            self._summary[self._summary_section] = {}
            self._summary[self._qgsw_version] = version("qgsw")
        self._files: list[Path] = []
        self._config = Configuration(self._summary[self._configuration])

    @property
    def configuration(self) -> Configuration:
        """Run Configuration."""
        return self._config

    @property
    def has_summary(self) -> bool:
        """Whether the summary was built or not."""
        return self._summary_section in self._summary

    @property
    def qgsw_version(self) -> str:
        """QGSW version.."""
        if self._qgsw_version in self._summary[self._summary_section]:
            return self._summary[self._summary_section][self._qgsw_version]
        msg = "QGSW version not registered."
        raise SummaryError(msg)

    @property
    def duration(self) -> float:
        """Duration (in seconds)."""
        if self._duration in self._summary[self._summary_section]:
            return self._summary[self._summary_section][self._duration]
        msg = "Duration not registered."
        raise SummaryError(msg)

    @property
    def dt(self) -> float:
        """Timestep (in seconds)."""
        if self._dt in self._summary[self._summary_section]:
            return self._summary[self._summary_section][self._dt]
        msg = "Timestep not registered."
        raise SummaryError(msg)

    @property
    def total_steps(self) -> int:
        """Total Steps."""
        if self._total_steps in self._summary[self._summary_section]:
            return self._summary[self._summary_section][self._total_steps]
        msg = "Total steps number not registered."
        raise SummaryError(msg)

    @property
    def last_step(self) -> int:
        """Last registered step."""
        if self._n in self._summary[self._summary_section]:
            return self._summary[self._summary_section][self._n]
        msg = "No step registered."
        raise SummaryError(msg)

    @property
    def is_finished(self) -> bool:
        """Whether the run was finished or not."""
        if self._finished in self._summary[self._summary_section]:
            return self._summary[self._summary_section][self._finished]
        msg = "Run status not registered."
        raise SummaryError(msg)

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
        self._summary[self._summary_section][self._duration] = t_end
        self._summary[self._summary_section][self._dt] = dt
        self._summary[self._summary_section][self._total_steps] = n_steps
        if self._files:
            self.update()

    def register_start(self) -> None:
        """Register the start of the simulation."""
        self.raise_if_not_writable()
        self._summary[self._summary_section][self._finished] = False
        if self._files:
            self.update()

    def register_step(self, step: int) -> None:
        """Register a simulation step.

        Args:
            step (int): Number of the step.
        """
        self.raise_if_not_writable()
        self._summary[self._summary_section][self._n] = step
        if self._files:
            self.update()

    def register_end(self) -> None:
        """Register the end of the simulation."""
        self.raise_if_not_writable()
        if self._summary_section not in self._summary:
            self._summary[self._summary_section] = {}
        self._summary[self._summary_section][self._finished] = True
        if self._files:
            self.update()

    def to_file(self, file: Path) -> None:
        """Save the summary into a file.

        Multiple calls will add as many files as required.

        Args:
            file (Path): File to save in.

        Raises:
            SummaryError: If the file is not a TOML file.
        """
        self.raise_if_not_writable()
        if file.suffix != ".toml":
            msg = "Summary can only be saved into a .toml file."
            raise SummaryError(msg)
        toml.dump(self._summary, file.open("w"))
        if file not in self._files:
            self._files.append(file)

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
    def from_file(cls, file: Path) -> Self:
        """Create the summary from a TOML file.

        Args:
            file (Path): File to create from.

        Returns:
            Self: Summary.
        """
        return cls(run_params=toml.load(file))

    @classmethod
    def from_configuration(cls, configuration: Configuration) -> Self:
        """CReate the summary from a configuration.

        Args:
            configuration (Configuration): Configuration.

        Returns:
            Self: Summary.
        """
        return cls(run_params=configuration.params)


class OutputFile(NamedTuple):
    """Output file wrapper."""

    step: int
    timestep: datetime.timedelta
    path: Path

    def read(self) -> np.ndarray:
        """Read the file data.

        Returns:
            np.ndarray: Data.
        """
        return np.load(file=self.path)


class RunOutput:
    """Run output."""

    def __init__(self, folder: Path) -> None:
        """Instantiate run output.

        Args:
            folder (Path): Run output folder.
        """
        self._folder = Path(folder)
        self._summary = RunSummary.from_file(
            self.folder.joinpath("_summary.toml"),
        )
        prefix = self.summary.configuration.model.prefix
        files = list(self.folder.glob(f"{prefix}*.npz"))
        steps, files = sort_files(files, prefix, ".npz")
        dt = self._summary.configuration.simulation.dt
        timesteps = [datetime.timedelta(seconds=step * dt) for step in steps]

        self._outputs = [
            OutputFile(step=steps[i], timestep=timesteps[i], path=files[i])
            for i in range(len(files))
        ]

    @cached_property
    def folder(self) -> Path:
        """Run output folder."""
        return Path(self._folder)

    @property
    def summary(self) -> RunSummary:
        """Run summary."""
        return self._summary

    def files(self) -> Iterator[Path]:
        """Sorted list of files.

        Returns:
            Iterator[float]: Files iterator.
        """
        return (output.path for output in self.outputs())

    def steps(self) -> Iterator[int]:
        """Sorted list of steps.

        Returns:
            Iterator[float]: Steps iterator.
        """
        return (output.step for output in self.outputs())

    def timesteps(self) -> Iterator[datetime.timedelta]:
        """Sorted list of timesteps.

        Returns:
            Iterator[float]: Timesteps iterator.
        """
        return (output.timestep for output in self.outputs())

    def outputs(self) -> Iterator[OutputFile]:
        """Sorted outputs.

        Returns:
            Iterator[OutputFile]: Outputs iterator.
        """
        return iter(self._outputs)

    def datas(self) -> Iterator[np.ndarray]:
        """Sorted datas.

        Returns:
            Iterator[np.ndarray]: Datas iterators.
        """
        return (output.read() for output in self.outputs())
