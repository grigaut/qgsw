"""Custom configuration file parser."""

from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.bathymetry import BathyConfig
from qgsw.configs.exceptions import ConfigError, UnexpectedFieldError
from qgsw.configs.io import IOConfig
from qgsw.configs.mesh import MeshConfig
from qgsw.configs.models import ModelConfig
from qgsw.configs.physics import PhysicsConfig
from qgsw.configs.vortex import VortexConfig
from qgsw.configs.windstress import WindStressConfig


class ModelComparisonConfig(_Config):
    """Configuration for a run."""

    _layers_section: str = keys.LAYERS["section"]
    _physics_section: str = keys.PHYSICS["section"]
    _mesh_section: str = keys.MESH["section"]
    _io_section: str = keys.IO["section"]
    _models_section: str = "models"
    _vortex_section: str = keys.VORTEX["section"]
    _windstress_section: str = keys.WINDSTRESS["section"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate ModelComparisonConfig.

        Args:
            params (dict[str, Any]): Script Configuration dictionnary.
        """
        super().__init__(params)
        self._models = [
            ModelConfig(params=p) for p in self.params[self._models_section]
        ]
        self._physics = PhysicsConfig(
            params=self.params[self._physics_section],
        )
        self._mesh = MeshConfig(params=self.params[self._mesh_section])
        self._io = IOConfig(params=self.params[self._io_section])
        self._windstress = WindStressConfig(
            params=self.params[self._windstress_section],
        )
        self._vortex = VortexConfig(params=self.params[self._vortex_section])

    @property
    def models(self) -> list[ModelConfig]:
        """List of configuration for models."""
        return self._models

    @property
    def physics(self) -> PhysicsConfig:
        """Configuration parameters dictionnary for physics."""
        return self._physics

    @property
    def mesh(self) -> MeshConfig:
        """Configuration parameters dictionnary for the mesh."""
        return self._mesh

    @property
    def io(self) -> IOConfig:
        """Input-Output  Configuration."""
        return self._io

    @property
    def bathy(self) -> BathyConfig:
        """Configuration parameters for the bathymetry."""
        msg = "No Bathymetry configuration for this configuration."
        raise UnexpectedFieldError(msg)

    @property
    def windstress(self) -> WindStressConfig:
        """WindStress Configuration."""
        return self._windstress

    @property
    def vortex(self) -> VortexConfig:
        """Vortex Configuration."""
        return self._vortex

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate configuration parameters.

        Args:
            params (dict[str, Any]): Configuration parameters.

        Raises:
            ConfigError: If the configuration doesn't have a layers section.

        Returns:
            dict[str, Any]: Configuration parameters.
        """
        # Verify that the models section is present.
        if self._models_section not in params:
            msg = (
                "The configuration must contain a "
                f"models section, named {self._models_section}."
            )
            raise ConfigError(msg)
        # Verify that the physics section is present.
        if self._physics_section not in params:
            msg = (
                "The configuration must contain a "
                f"physics section, named {self._physics_section}."
            )
            raise ConfigError(msg)
        # Verify that the mesh section is present.
        if self._mesh_section not in params:
            msg = (
                "The configuration must contain a "
                f"mesh section, named {self._mesh_section}."
            )
            raise ConfigError(msg)
        # Verify that the io section is present.
        if self._io_section not in params:
            msg = (
                "The configuration must contain a "
                f"io section, named {self._io_section}."
            )
            raise ConfigError(msg)
        # Verify that the windstress section is present.
        if self._windstress_section not in params:
            msg = (
                "The configuration must contain a "
                f"windstress section, named {self._windstress_section}."
            )
            raise ConfigError(msg)
        # Verify that the vortex section is present.
        if self._vortex_section not in params:
            msg = (
                "The configuration must contain a "
                f"vortex section, named {self._vortex_section}."
            )
            raise ConfigError(msg)
        return super()._validate_params(params)
