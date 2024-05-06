"""Main Configuration Tool."""

from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import Any

import toml
from typing_extensions import Self

from qgsw.configs.bathymetry import BathyConfig
from qgsw.configs.exceptions import ConfigSaveError
from qgsw.configs.io import IOConfig
from qgsw.configs.mesh import MeshConfig
from qgsw.configs.models import ModelConfig
from qgsw.configs.physics import PhysicsConfig
from qgsw.configs.simulations import SimulationConfig
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

    @property
    def has_io(self) -> bool:
        """Whether the configuration contains an 'io' section or not."""
        return IOConfig.section in self._config

    @cached_property
    def io(self) -> IOConfig:
        """Input/output configuration."""
        return IOConfig.parse(self._config)

    @property
    def has_physics(self) -> bool:
        """Whether the configuration contains a 'physics' section or not."""
        return PhysicsConfig.section in self._config

    @cached_property
    def physics(self) -> PhysicsConfig:
        """Physics configuration."""
        return PhysicsConfig.parse(self._config)

    @property
    def has_model(self) -> bool:
        """Whether the configuration contains a 'model' section or not."""
        return ModelConfig.section in self._config

    @cached_property
    def model(self) -> ModelConfig:
        """Model configuration."""
        return ModelConfig.parse(self._config)

    @property
    def has_models(self) -> bool:
        """Whether the configuration contains a 'models' section or not."""
        return ModelConfig.section_several in self._config

    @cached_property
    def models(self) -> list[ModelConfig]:
        """Models configuration."""
        return ModelConfig.parse_several(self._config)

    @property
    def has_mesh(self) -> bool:
        """Whether the configuration contains a 'mesh' section or not."""
        return MeshConfig.section in self._config

    @cached_property
    def mesh(self) -> MeshConfig:
        """Mesh configuration."""
        return MeshConfig.parse(self._config)

    @cached_property
    def has_windstress(self) -> bool:
        """Whether the configuration contains a 'windstress' section or not."""
        return WindStressConfig.section in self._config

    @property
    def windstress(self) -> WindStressConfig:
        """WindStress configuration."""
        return WindStressConfig.parse(self._config)

    @property
    def has_bathymetry(self) -> bool:
        """Whether the configuration contains a 'bathymetry' section or not."""
        return BathyConfig.section in self._config

    @cached_property
    def bathymetry(self) -> BathyConfig:
        """Bathymetry Configuration."""
        return BathyConfig.parse(self._config)

    @property
    def has_vortex(self) -> bool:
        """Whether the configuration contains a 'vortex' section or not."""
        return VortexConfig.section in self._config

    @cached_property
    def vortex(self) -> VortexConfig:
        """Vortex Configuration."""
        return VortexConfig.parse(self._config)

    @property
    def has_simulation(self) -> bool:
        """Whether the configuration contains a 'simulation' section or not."""
        return SimulationConfig.section in self._config

    @cached_property
    def simulation(self) -> SimulationConfig:
        """Simulation configuration."""
        return SimulationConfig.parse(self._config)

    def to_file(self, file: Path) -> None:
        """Save the configuration to a given TOML file.

        More informations on TOML files: https://toml.io/en/.

        Args:
            file (Path): _description_
        """
        if "qgsw-version" in self._config:
            msg = "Impossible to specify qgsw's version."
            raise ConfigSaveError(msg)
        self._config["qgsw-version"] = version("qgsw")
        if file.suffix != ".toml":
            msg = "Configuration can only be saved into a .toml file."
            raise ConfigSaveError(msg)
        toml.dump(self._config, file.open("w"))

    @classmethod
    def from_file(cls, file: Path) -> Self:
        """Instantiate Configuration from a given TOML file.

        More informations on TOML files: https://toml.io/en/.

        Args:
            file (Path): Configuration file path.

        Returns:
            Self: Configuration.
        """
        config = cls(configuration=toml.load(file))
        if not config.has_io:
            return config
        if config.io.plots.save:
            save_file = config.io.plots.directory.joinpath("_config.toml")
            config.to_file(save_file)
        if config.io.results.save:
            save_file = config.io.results.directory.joinpath("_config.toml")
            config.to_file(save_file)
        return config
