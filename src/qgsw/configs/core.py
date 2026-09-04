"""Configurations."""

# ruff: noqa: TCH001, TCH003, UP007

from __future__ import annotations

from typing import Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from pathlib import Path

import toml
from pydantic import (
    BaseModel,
    Field,
)

from qgsw.configs.io import IOConfig
from qgsw.configs.models import CoefConfig, ModelConfig
from qgsw.configs.perturbation import PerturbationConfig
from qgsw.configs.physics import PhysicsConfig
from qgsw.configs.simulations import (
    AssimilationSimulationConfig,
    ModelRunSimulationConfig,
)
from qgsw.configs.space import SpaceConfig
from qgsw.configs.windstress import WindStressConfig


class Configuration(BaseModel):
    """Configuration."""

    io: IOConfig
    physics: PhysicsConfig
    simulation: Union[
        ModelRunSimulationConfig,
        AssimilationSimulationConfig,
    ] = Field(discriminator="type")
    model: ModelConfig[Union[CoefConfig, None]]
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


Configuration.model_rebuild()
