"""Main Configuration Tool."""

from functools import cached_property
from pathlib import Path
from typing import Any

import toml
from typing_extensions import Self

from qgsw.configs.bathymetry import BathyConfig
from qgsw.configs.io import IOConfig
from qgsw.configs.mesh import MeshConfig
from qgsw.configs.models import ModelConfig
from qgsw.configs.physics import PhysicsConfig
from qgsw.configs.vortex import VortexConfig
from qgsw.configs.windstress import WindStressConfig


class Configuration:
    """Configuration."""

    def __init__(self, configuration: dict[str, Any]) -> None:
        """Instantiate the Configuration.

        Args:
            configuration (dict[str, Any]): Configuration parameters.
        """
        self._config = configuration

    @cached_property
    def io(self) -> IOConfig:
        """Input/output configuration."""
        return IOConfig.parse(self._config)

    @cached_property
    def physics(self) -> PhysicsConfig:
        """Physics configuration."""
        return PhysicsConfig.parse(self._config)

    @cached_property
    def model(self) -> ModelConfig:
        """Model configuration."""
        return ModelConfig.parse(self._config)

    @cached_property
    def models(self) -> list[ModelConfig]:
        """Models configuration."""
        return ModelConfig.parse_several(self._config)

    @cached_property
    def mesh(self) -> MeshConfig:
        """Mesh configuration."""
        return MeshConfig.parse(self._config)

    @cached_property
    def windstress(self) -> WindStressConfig:
        """WindStress configuration."""
        return WindStressConfig.parse(self._config)

    @cached_property
    def bathymetry(self) -> BathyConfig:
        """Bathymetry Configuration."""
        return BathyConfig.parse(self._config)

    @cached_property
    def vortex(self) -> VortexConfig:
        """Vortex Configuration."""
        return VortexConfig.parse(self._config)

    @classmethod
    def from_file(cls, file: Path) -> Self:
        """Instantiate Configuration from a given TOML file.

        More informations on TOML files: https://toml.io/en/.

        Args:
            file (Path): Configuration file path.

        Returns:
            Self: Configuration.
        """
        return cls(configuration=toml.load(file))
