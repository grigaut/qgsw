"""Output management."""

from __future__ import annotations

from datetime import timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

from qgsw.masks import Masks
from qgsw.models.qg.stretching_matrix import compute_A
from qgsw.run_summary import RunSummary
from qgsw.spatial.core.discretization import SpaceDiscretization3D
from qgsw.specs import DEVICE
from qgsw.utils.sorting import sort_files
from qgsw.variables.dynamics import (
    Enstrophy,
    MeridionalVelocityFlux,
    ParsedLayerDepthAnomaly,
    ParsedMeridionalVelocity,
    ParsedZonalVelocity,
    PhysicalLayerDepthAnomaly,
    PhysicalMeridionalVelocity,
    PhysicalVorticity,
    PhysicalZonalVelocity,
    PotentialVorticity,
    Pressure,
    StreamFunction,
    SurfaceHeightAnomaly,
    TotalEnstrophy,
    Vorticity,
    ZonalVelocityFlux,
)
from qgsw.variables.energetics import (
    ModalAvailablePotentialEnergy,
    ModalEnergy,
    ModalKineticEnergy,
    TotalAvailablePotentialEnergy,
    TotalEnergy,
    TotalKineticEnergy,
)
from qgsw.variables.state import State
from qgsw.variables.uvh import UVH

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qgsw.configs.models import ModelConfig
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.variables.base import DiagnosticVariable


class OutputFile(NamedTuple):
    """Output file wrapper."""

    step: int
    second: float
    timestep: timedelta
    path: Path
    dtype = torch.float64
    device = DEVICE.get()

    def read(self) -> UVH:
        """Read the file data.

        Returns:
            UVh: Data.
        """
        data = np.load(file=self.path)
        u = torch.tensor(data[ParsedZonalVelocity.get_name()])
        v = torch.tensor(data[ParsedMeridionalVelocity.get_name()])
        h = torch.tensor(data[ParsedLayerDepthAnomaly.get_name()])
        return UVH(
            u.to(dtype=self.dtype, device=self.device),
            v.to(dtype=self.dtype, device=self.device),
            h.to(dtype=self.dtype, device=self.device),
        )


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
        self._state = State(
            self._outputs[0].read(),
        )
        self._vars: dict[str, DiagnosticVariable] = {
            ParsedZonalVelocity.get_name(): ParsedZonalVelocity(),
            ParsedMeridionalVelocity.get_name(): ParsedMeridionalVelocity(),
            ParsedLayerDepthAnomaly.get_name(): ParsedZonalVelocity(),
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
    def state(self) -> State:
        """State."""
        return self._state

    @property
    def vars(self) -> list[DiagnosticVariable]:
        """Variable from state."""
        return list(self._vars.values())

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        vars_txt = "\n\t".join(str(var) for var in self.vars)
        return [
            f"Simulation: {self._summary.configuration.io.name}.",
            f"Starting time: {self._summary.started_at}",
            f"Ending time: {self._summary.ended_at}",
            f"Package version: {self._summary.qgsw_version}.",
            f"Folder: {self.folder}.",
            f"Variables:\n\t{vars_txt}",
        ]

    def __repr__(self) -> str:
        """Output Representation."""
        return "\n".join(self.get_repr_parts())

    def __getitem__(self, key: str) -> Iterator[torch.Tensor]:
        """Get variable based on its name.

        Args:
            key (str): Variable name.

        Yields:
            Iterator[torch.tensor]: Values of a given variable.
        """
        return (self._vars[key].compute(out.read()) for out in self.outputs())

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

    def with_variable(self, variable: DiagnosticVariable) -> None:
        """Add a variable to the state.

        Args:
            variable (DiagnosticVariable): Variable to consider.
        """
        self._vars[variable.name] = variable


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


def add_qg_variables(
    run_output: RunOutput,
    physics_config: PhysicsConfig,
    space_config: SpaceConfig,
    model_config: ModelConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Add QG variables to run output.

    Args:
        run_output (RunOutput): Run output.
        physics_config (PhysicsConfig): Physics configuration
        space_config (SpaceConfig): Space configuration
        model_config (ModelConfig): Model configuration
        dtype (torch.dtype): Data type
        device (torch.device): Device
    """
    space = SpaceDiscretization3D.from_config(space_config, model_config)
    dx = space.dx
    dy = space.dy
    ds = space.area
    H = model_config.h.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # noqa: N806
    g_prime = model_config.g_prime
    A = compute_A(H.squeeze((0, -3, -2)), g_prime, dtype, device)  # noqa: N806
    u_phys = PhysicalZonalVelocity(dx)
    v_phys = PhysicalMeridionalVelocity(dy)
    h_phys = PhysicalLayerDepthAnomaly(ds)
    U = ZonalVelocityFlux(dx)  # noqa: N806
    V = MeridionalVelocityFlux(dy)  # noqa: N806
    vorticity = Vorticity(
        Masks.empty(space.nx, space.ny, device),
        slip_coef=physics_config.slip_coef,
    )
    vorticity_phys = PhysicalVorticity(vorticity, ds)
    enstrophy = Enstrophy(vorticity_phys)
    enstrophy_tot = TotalEnstrophy(vorticity_phys)
    eta = SurfaceHeightAnomaly(h_phys)
    p = Pressure(g_prime, eta)
    psi = StreamFunction(p, physics_config.f0)
    pv = PotentialVorticity(vorticity, H * ds, ds, physics_config.f0)
    ke_hat = ModalKineticEnergy(A, psi, H, dx, dy)
    ape_hat = ModalAvailablePotentialEnergy(A, psi, H, physics_config.f0)
    energy_hat = ModalEnergy(ke_hat, ape_hat)
    ke = TotalKineticEnergy(psi, H, dx, dy)
    ape = TotalAvailablePotentialEnergy(A, psi, H, physics_config.f0)
    energy = TotalEnergy(ke, ape)

    run_output.with_variable(u_phys)
    run_output.with_variable(v_phys)
    run_output.with_variable(h_phys)
    run_output.with_variable(U)
    run_output.with_variable(V)
    run_output.with_variable(vorticity)
    run_output.with_variable(vorticity_phys)
    run_output.with_variable(enstrophy)
    run_output.with_variable(enstrophy_tot)
    run_output.with_variable(eta)
    run_output.with_variable(p)
    run_output.with_variable(psi)
    run_output.with_variable(pv)
    run_output.with_variable(ke_hat)
    run_output.with_variable(ape_hat)
    run_output.with_variable(energy_hat)
    run_output.with_variable(ke)
    run_output.with_variable(ape)
    run_output.with_variable(energy)
