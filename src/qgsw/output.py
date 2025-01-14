"""Output management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Generic, NamedTuple, TypeVar

import torch

from qgsw.fields.variables.dynamics import (
    LayerDepthAnomalyDiag,
    MeridionalVelocityDiag,
    ZonalVelocityDiag,
)
from qgsw.fields.variables.prognostic import CollinearityCoefficient, Time
from qgsw.fields.variables.uvh import UVHT, PrognosticTuple, UVHTAlpha
from qgsw.models.qg.collinear_sublayer.core import QGCollinearSF
from qgsw.run_summary import RunSummary
from qgsw.specs import DEVICE
from qgsw.utils.sorting import sort_files

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qgsw.configs.models import ModelConfig

T = TypeVar("T", bound=PrognosticTuple)


class _OutputFile(NamedTuple):
    """Output file tuple."""

    step: int
    second: float
    timestep: timedelta
    path: Path
    dtype = torch.float64
    device = DEVICE.get()


class _OutputReader(ABC, Generic[T]):
    """Output file wrapper."""

    step: int
    second: float
    timestep: timedelta
    path: Path
    dtype = torch.float64
    device = DEVICE.get()

    @abstractmethod
    def read(self) -> T: ...


class OutputFile(_OutputReader[UVHT], _OutputFile):
    """Output file wrapper."""

    def read(self) -> UVHT:
        """Read the file data.

        Returns:
            UVh: Data.
        """
        data: dict[str, torch.Tensor] = torch.load(
            self.path,
            weights_only=True,
        )
        t = data[Time.get_name()]
        u = data[ZonalVelocityDiag.get_name()]
        v = data[MeridionalVelocityDiag.get_name()]
        h = data[LayerDepthAnomalyDiag.get_name()]
        return UVHT(
            u.to(dtype=self.dtype, device=self.device),
            v.to(dtype=self.dtype, device=self.device),
            h.to(dtype=self.dtype, device=self.device),
            t.to(dtype=self.dtype, device=self.device),
        )


class OutputFileAlpha(_OutputReader[UVHTAlpha], _OutputFile):
    """Output file wrapper."""

    def read(self) -> UVHTAlpha:
        """Read the file data.

        Returns:
            UVh: Data.
        """
        data: dict[str, torch.Tensor] = torch.load(
            self.path,
            weights_only=True,
        )
        t = data[Time.get_name()]
        u = data[ZonalVelocityDiag.get_name()]
        v = data[MeridionalVelocityDiag.get_name()]
        h = data[LayerDepthAnomalyDiag.get_name()]
        alpha = data[CollinearityCoefficient.get_name()]
        return UVHTAlpha(
            u.to(dtype=self.dtype, device=self.device),
            v.to(dtype=self.dtype, device=self.device),
            h.to(dtype=self.dtype, device=self.device),
            t.to(dtype=self.dtype, device=self.device),
            alpha.to(dtype=self.dtype, device=self.device),
        )


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
        dt = self._summary.configuration.simulation.dt
        seconds = [step * dt for step in steps]
        timesteps = [timedelta(seconds=sec) for sec in seconds]

        if model_type == QGCollinearSF.get_type():
            self._outputs = [
                OutputFileAlpha(
                    step=steps[i],
                    timestep=timesteps[i],
                    path=files[i],
                    second=seconds[i],
                )
                for i in range(len(files))
            ]

        else:
            self._outputs = [
                OutputFile(
                    step=steps[i],
                    timestep=timesteps[i],
                    path=files[i],
                    second=seconds[i],
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
        return (output.second for output in iter(self._outputs))

    def outputs(self) -> Iterator[_OutputReader]:
        """Sorted outputs.

        Returns:
            Iterator[_OutputFile]: Outputs iterator.
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
