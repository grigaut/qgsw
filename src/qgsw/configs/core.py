"""Main Configuration Tool."""

from functools import cached_property
from pathlib import Path
from typing import Any, Self

import toml

from qgsw.configs.bathymetry import BathyConfig
from qgsw.configs.io import IOConfig
from qgsw.configs.models import ModelConfig
from qgsw.configs.perturbation import PerturbationConfig
from qgsw.configs.physics import PhysicsConfig
from qgsw.configs.simulations import SimulationConfig
from qgsw.configs.space import SpaceConfig
from qgsw.configs.windstress import WindStressConfig


class Configuration:
    """Configuration."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate the Configuration.

        Args:
            params (dict[str, Any]): Configuration parameters.
        """
        self._params = params

    @property
    def params(self) -> dict[str, Any]:
        """Configuration parameters."""
        return self._params

    @property
    def has_io(self) -> bool:
        """Whether the configuration contains an 'io' section or not."""
        return IOConfig.section in self.params

    @cached_property
    def io(self) -> IOConfig:
        """Input/output configuration."""
        return IOConfig.parse(self.params)

    @property
    def has_physics(self) -> bool:
        """Whether the configuration contains a 'physics' section or not."""
        return PhysicsConfig.section in self.params

    @cached_property
    def physics(self) -> PhysicsConfig:
        """Physics configuration."""
        return PhysicsConfig.parse(self.params)

    @property
    def has_model(self) -> bool:
        """Whether the configuration contains a 'model' section or not."""
        return ModelConfig.section in self.params

    @cached_property
    def model(self) -> ModelConfig:
        """Model configuration."""
        return ModelConfig.parse(self.params)

    @property
    def has_models(self) -> bool:
        """Whether the configuration contains a 'models' section or not."""
        return ModelConfig.section_several in self.params

    @cached_property
    def models(self) -> list[ModelConfig]:
        """Models configuration."""
        return ModelConfig.parse_several(self.params)

    @property
    def has_space(self) -> bool:
        """Whether the configuration contains a 'space' section or not."""
        return SpaceConfig.section in self.params

    @cached_property
    def space(self) -> SpaceConfig:
        """Grid configuration."""
        return SpaceConfig.parse(self.params)

    @cached_property
    def has_windstress(self) -> bool:
        """Whether the configuration contains a 'windstress' section or not."""
        return WindStressConfig.section in self.params

    @property
    def windstress(self) -> WindStressConfig:
        """WindStress configuration."""
        return WindStressConfig.parse(self.params)

    @property
    def has_bathymetry(self) -> bool:
        """Whether the configuration contains a 'bathymetry' section or not."""
        return BathyConfig.section in self.params

    @cached_property
    def bathymetry(self) -> BathyConfig:
        """Bathymetry Configuration."""
        return BathyConfig.parse(self.params)

    @property
    def has_perturbation(self) -> bool:
        """If the configuration contains a 'perturbation' section or not."""
        return PerturbationConfig.section in self.params

    @cached_property
    def perturbation(self) -> PerturbationConfig:
        """Vortex Configuration."""
        return PerturbationConfig.parse(self.params)

    @property
    def has_simulation(self) -> bool:
        """Whether the configuration contains a 'simulation' section or not."""
        return SimulationConfig.section in self.params

    @cached_property
    def simulation(self) -> SimulationConfig:
        """Simulation configuration."""
        return SimulationConfig.parse(self.params)

    @classmethod
    def from_file(cls, file: Path) -> Self:
        """Instantiate Configuration from a given TOML file.

        More informations on TOML files: https://toml.io/en/.

        Args:
            file (Path): Configuration file path.

        Returns:
            Self: Configuration.
        """
        return cls(params=toml.load(file))
