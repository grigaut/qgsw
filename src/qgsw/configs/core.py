"""Main Configuration Tool."""

from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.bathymetry import BathyConfig
from qgsw.configs.exceptions import ConfigError, UngivenFieldError
from qgsw.configs.grid import GridConfig, LayersConfig
from qgsw.configs.io import IOConfig
from qgsw.configs.physics import PhysicsConfig


class RunConfig(_Config):
    """Configuration for a run."""

    _layers_section: str = keys.LAYERS["section"]
    _physics_section: str = keys.PHYSICS["section"]
    _grid_section: str = keys.GRID["section"]
    _bathy_section: str = keys.BATHY["section"]
    _io_section: str = keys.IO["section"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate RunConfig.

        Args:
            params (dict[str, Any]): Run configuration dictionnary.
        """
        super().__init__(params)
        self._layers = LayersConfig(params=self.params[self._layers_section])
        self._physics = PhysicsConfig(
            params=self.params[self._physics_section]
        )
        self._grid = GridConfig(params=self.params[self._grid_section])
        self._bathy = BathyConfig(params=self.params[self._bathy_section])
        self._io = IOConfig(params=self.params[self._io_section])

    @property
    def layers(self) -> LayersConfig:
        """Configuration parameters dictionnary for layers."""
        return self._layers

    @property
    def physics(self) -> PhysicsConfig:
        """Configuration parameters dictionnary for physics."""
        return self._physics

    @property
    def grid(self) -> GridConfig:
        """Configuration parameters dictionnary for the grid."""
        return self._grid

    @property
    def bathy(self) -> BathyConfig:
        """Configuartion parameters frot he bathymetry."""
        if self._bathy_section not in self.params:
            msg = (
                "The configuration does not contain a "
                f"bathymetry section, named {self._bathy_section}."
            )
            raise UngivenFieldError(msg)
        return self._bathy

    @property
    def io(self) -> IOConfig:
        """Input-Output  Configuration."""
        return self._io

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration parameters.

        Args:
            params (dict[str, Any]): Configuration parameters.

        Raises:
            ConfigError: If the configuration doesn't have a layers section.

        Returns:
            dict[str, Any]: Configuration parameters.
        """
        # Verify that the layers section is present.
        if self._layers_section not in params:
            msg = (
                "The configuration must contain a "
                f"layers section, named {self._layers_section}."
            )
            raise ConfigError(msg)
        # Verify that the physics section is present.
        if self._physics_section not in params:
            msg = (
                "The configuration must contain a "
                f"physics section, named {self._physics_section}."
            )
            raise ConfigError(msg)
        # Verify that the grid section is present.
        if self._grid_section not in params:
            msg = (
                "The configuration must contain a "
                f"grid section, named {self._grid_section}."
            )
            raise ConfigError(msg)
        # Verify that the io section is present.
        if self._io_section not in params:
            msg = (
                "The configuration must contain a "
                f"io section, named {self._io_section}."
            )
            raise ConfigError(msg)
        return super()._validate_params(params)
