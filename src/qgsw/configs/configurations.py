"""Configurations."""

from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import toml
import torch
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.spatial.units._units import Unit
from qgsw.specs import DEVICE
from qgsw.utils.storage import get_absolute_storage_path

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qgsw.simulation.steps import Steps


class ModelConfig(BaseModel):
    """Model configuration."""

    type: str
    prefix: str
    layers: list[PositiveFloat]
    reduced_gravity: list[PositiveFloat]
    collinearity_coef: (
        ConstantCollinearityCoefConfig | ChangingCollinearityCoefConfig | None
    ) = Field(None, discriminator="type")

    @cached_property
    def h(self) -> torch.Tensor:
        """Vertical layers."""
        return torch.tensor(
            self.layers,
            dtype=torch.float64,
            device=DEVICE.get(),
        )

    @cached_property
    def g_prime(self) -> torch.Tensor:
        """Reduced gravity."""
        return torch.tensor(
            self.reduced_gravity,
            dtype=torch.float64,
            device=DEVICE.get(),
        )


class ConstantCollinearityCoefConfig(BaseModel):
    """Constant collinearity model configuration."""

    type: Literal["constant"]
    value: float


class ChangingCollinearityCoefConfig(BaseModel):
    """Changing collinearity coefficient configuration."""

    type: Literal["changing"]
    source_file: FilePath


class PhysicsConfig(BaseModel):
    """Physics configuration."""

    rho: PositiveFloat
    slip_coef: float = Field(ge=0, le=1)
    f0: float
    beta: float
    bottom_drag_coefficient: float = Field(ge=0, le=1)

    @cached_property
    def beta_plane(self) -> BetaPlane:
        """Beta Plane."""
        return BetaPlane(f0=self.f0, beta=self.beta)


class ModelRunSimulationConfig(BaseModel):
    """Model run simulaton configuration."""

    kind: Literal["simple-run"]
    duration: PositiveFloat
    dt: PositiveFloat


class AssimilationSimulationConfig(BaseModel):
    """Assimilation simulation configuration."""

    kind: Literal["assimilation"]
    duration: PositiveFloat
    dt: PositiveFloat
    reference: ModelConfig


class IOConfig(BaseModel):
    """Input/Output ."""

    name: str
    output: IntervalSaveConfig | QuantitySaveConfig | None = Field(
        None,
        discriminator="type",
    )


class IntervalSaveConfig(BaseModel):
    """Interval save configuration."""

    type: Literal["interval"]
    save: bool
    interval_duration: PositiveFloat
    directory_str: str = Field(
        alias="directory",
    )

    @cached_property
    def directory(self) -> Path:
        """Output directory."""
        directory = get_absolute_storage_path(Path(self.directory_str))
        if not directory.is_dir():
            directory.mkdir()
            gitignore = directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")
        return directory

    def get_saving_steps(self, steps: Steps) -> Iterator[bool]:
        """Get saving steps.

        Args:
            steps (Steps): Steps manager.

        Yields:
            Iterator[bool]: Boolean iterator with the same length as the one
            returned by `simulation_steps`, True when the corresponding step
            matches the interval, always ends with True.
        """
        if not self.save:
            return (False for _ in steps.simulation_steps())
        return steps.steps_from_interval(interval=self.interval_duration)


class QuantitySaveConfig(BaseModel):
    """Quantity save configuration."""

    type: Literal["quantity"]
    save: bool
    quantity: PositiveFloat
    directory_str: str = Field(
        alias="directory",
    )

    @cached_property
    def directory(self) -> Path:
        """Output directory."""
        directory = get_absolute_storage_path(Path(self.directory_str))
        if not directory.is_dir():
            directory.mkdir()
            gitignore = directory.joinpath(".gitignore")
            with gitignore.open("w") as file:
                file.write("*")
        return directory

    def get_saving_steps(self, steps: Steps) -> Iterator[bool]:
        """Get saving steps.

        Args:
            steps (Steps): Steps manager.

        Yields:
            Iterator[bool]: Boolean iterator with the same length as the one
            returned by `simulation_steps`, True when the corresponding step is
            selected, always ends with True. May include one additional step.
        """
        if not self.save:
            return (False for _ in steps.simulation_steps())
        return steps.steps_from_total(total=self.quantity)


class SpaceConfig(BaseModel):
    """Space configuration."""

    nx: PositiveInt
    ny: PositiveInt
    unit_str: str = Field(alias="unit")
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @cached_property
    def unit(self) -> Unit:
        """Space Unit."""
        return Unit(self.unit_str)


class WindStressConfig(BaseModel):
    """Windstress configuration."""

    type: str
    magnitude: NonNegativeFloat | None = None
    drag_coefficient: NonNegativeFloat | None = None


class PerturbationConfig(BaseModel):
    """Perturbation configuration."""

    type: str
    perturbation_magnitude: NonNegativeFloat


class Configuration(BaseModel):
    """Configuration."""

    io: IOConfig
    physics: PhysicsConfig
    simulation: ModelRunSimulationConfig | AssimilationSimulationConfig = (
        Field(discriminator="kind")
    )
    model: ModelConfig
    space: SpaceConfig
    windstress: WindStressConfig
    perturbation: PerturbationConfig

    @classmethod
    def from_toml(cls, file: Path) -> Self:
        """Load from a TOML file.

        Args:
            file (Path): File to load from.

        Returns:
            Self: Configuration.
        """
        return cls(**toml.load(file))