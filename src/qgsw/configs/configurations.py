"""Configurations."""

from __future__ import annotations

from functools import cached_property
from typing import Literal

import toml
import torch

from qgsw.physics.coriolis.beta_plane import BetaPlane
from qgsw.specs import DEVICE

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pathlib import Path  # noqa: TCH003

from pydantic import (
    BaseModel,
    Field,
    FilePath,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
)

from qgsw.spatial.units._units import Unit  # noqa: TCH001


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


class SimulationConfig(BaseModel):
    """Simulation configuration."""

    duration: PositiveFloat
    dt: PositiveFloat


class IOConfig(BaseModel):
    """Input/Output ."""

    name: str
    save: bool
    output_directory: Path | None = None
    save_method: IntervalSaveConfig | QuantitySaveConfig | None = Field(
        None,
        discriminator="type",
    )


class IntervalSaveConfig(BaseModel):
    """Interval save configuration."""

    type: Literal["interval"]
    interval_duration: PositiveFloat


class QuantitySaveConfig(BaseModel):
    """Quantity save configuration."""

    type: Literal["quantity"]
    quantity: PositiveFloat


class SpaceConfig(BaseModel):
    """Space configuration."""

    nx: PositiveInt
    ny: PositiveInt
    unit: Unit
    x_min: float
    x_max: float
    y_min: float
    y_max: float


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
    simulation: SimulationConfig
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
