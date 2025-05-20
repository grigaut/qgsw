"""Output management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import torch

from qgsw.exceptions import ConfigurationError
from qgsw.fields.variables.physical import (
    LayerDepthAnomaly,
    MeridionalVelocity,
    ZonalVelocity,
)
from qgsw.fields.variables.prognostic import CollinearityCoefficient, Time
from qgsw.fields.variables.tuples import (
    PSIQT,
    UVHT,
    BaseTuple,
    UVHTAlpha,
)
from qgsw.models.names import ModelName
from qgsw.models.qg.uvh.modified.utils import is_modified
from qgsw.run_summary import RunSummary
from qgsw.simulation.names import SimulationName
from qgsw.specs import DEVICE
from qgsw.utils import tensorio
from qgsw.utils.sorting import sort_files

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qgsw.configs.models import ModelConfig

T = TypeVar("T", bound=BaseTuple)


class _OutputFile(NamedTuple):
    """Output file tuple."""

    step: int
    dt: float
    path: Path
    dtype = torch.float64
    device = DEVICE.get()

    @property
    def second(self) -> float:
        return self.step * self.dt

    @property
    def timestep(self) -> float:
        return timedelta(seconds=self.second)


class _OutputReader(ABC, Generic[T]):
    """Output file wrapper."""

    @abstractmethod
    def read(self) -> T: ...


class OutputFilePSIQ(_OutputReader[PSIQT], _OutputFile):
    """Output file wrapper."""

    def read(self) -> PSIQT:
        """Read the file data.

        Returns:
            PSIQT: Data.
        """
        return PSIQT.from_file(
            self.path,
            dtype=torch.float64,
            device=DEVICE.get(),
        )


class OutputFileUVH(_OutputReader[UVHT], _OutputFile):
    """Output file wrapper."""

    def read(self) -> UVHT:
        """Read the file data.

        Returns:
            UVh: Data.
        """
        data: dict[str, torch.Tensor] = tensorio.load(
            self.path,
            dtype=self.dtype,
            device=self.device,
        )
        t = data[Time.get_name()]
        u = data[ZonalVelocity.get_name()]
        v = data[MeridionalVelocity.get_name()]
        h = data[LayerDepthAnomaly.get_name()]
        return UVHT(u, v, h, t)


class OutputFileAlpha(_OutputReader[UVHTAlpha], _OutputFile):
    """Output file wrapper."""

    def read(self) -> UVHTAlpha:
        """Read the file data.

        Returns:
            UVh: Data.
        """
        data: dict[str, torch.Tensor] = tensorio.load(
            self.path,
            dtype=self.dtype,
            device=self.device,
        )
        t = data[Time.get_name()]
        u = data[ZonalVelocity.get_name()]
        v = data[MeridionalVelocity.get_name()]
        h = data[LayerDepthAnomaly.get_name()]
        alpha = data[CollinearityCoefficient.get_name()]
        return UVHTAlpha(u, v, h, t, alpha)


class RunOutput:
    """Run output."""

    def __init__(
        self,
        folder: Path,
        model_config: ModelConfig | None = None,
    ) -> None:
        """Instantiate run output.

        Args:
            folder (Path): Run output folder.
            model_config (ModelConfig | None): Model configuration.
        """
        self._folder = Path(folder)
        self._summary = RunSummary.from_folder(self.folder)
        if model_config is None:
            prefix = self.summary.configuration.model.prefix
            model_type = self.summary.configuration.model.type
        else:
            prefix = model_config.prefix
            model_type = model_config.type
        files = list(self.folder.glob(f"{prefix}*.pt"))
        steps, files = sort_files(files, prefix, ".pt")
        self._dt = self._summary.configuration.simulation.dt

        if is_modified(model_type):
            self._outputs = [
                OutputFileAlpha(
                    step=steps[i],
                    path=files[i],
                    dt=self._dt,
                )
                for i in range(len(files))
            ]
        elif model_type == ModelName.QUASI_GEOSTROPHIC_USUAL:
            self._outputs = [
                OutputFilePSIQ(
                    step=steps[i],
                    path=files[i],
                    dt=self._dt,
                )
                for i in range(len(files))
            ]
        else:
            self._outputs = [
                OutputFileUVH(
                    step=steps[i],
                    path=files[i],
                    dt=self._dt,
                )
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

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        return [
            f"Simulation: {self._summary.configuration.io.name}.",
            f"Starting time: {self._summary.started_at}",
            f"Ending time: {self._summary.ended_at}",
            f"Package version: {self._summary.qgsw_version}.",
            f"Folder: {self.folder}.",
        ]

    def __repr__(self) -> str:
        """Output Representation."""
        return "\n".join(self.get_repr_parts())

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
        return (output.second for output in self.outputs())

    def outputs(self) -> Iterator[_OutputReader]:
        """Sorted outputs.

        Returns:
            Iterator[_OutputFile]: Outputs iterator.
        """
        return iter(self._outputs)

    def ref_outputs(self) -> Iterator[OutputFileUVH]:
        """Reference outputs.

        Raises:
            ConfigurationError: If the run is not an 'assimilation' run.

        Yields:
            Iterator[OutputFileUVH]: Outputs.
        """
        sim_config = self._summary.configuration.simulation
        if sim_config.type != SimulationName.ASSIMILATION:
            msg = "Reference outputs are for 'assimilation' simulations only."
            raise ConfigurationError(msg)
        prefix = sim_config.reference.prefix
        files = list(self.folder.glob(f"{prefix}*.pt"))
        steps, files = sort_files(files, prefix, ".pt")
        return (
            OutputFileUVH(step=steps[i], path=files[i], dt=self._dt)
            for i in range(len(files))
        )


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
