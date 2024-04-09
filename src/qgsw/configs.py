"""Parsing tools for configuration files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import toml
import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from pathlib import Path

from qgsw.specs import DEVICE

PHYSICS_KEYS = {
    "section": "physics",
    "rho": "rho",
    "slip coef": "slip_coef",
    "f0": "f0",
    "beta": "beta",
    "wind stress magnitude": "wind_stress_mag",
}

LAYERS_KEYS = {
    "section": "layers",
    "layer thickness": "h",
    "reduced gravity": "g_prime",
}

GRID_KEYS = {
    "section": "grid",
    "points nb x": "nx",
    "points nb y": "ny",
    "x length": "Lx",
    "y length": "Ly",
    "timestep": "dt",
}


class ConfigError(Exception):
    """Configuration-Related Error."""


class _Config(ABC):
    """Configuration."""

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate configuration from configuration parameters dictionnary.

        Args:
            params (dict[str, Any]): Configuration parameters.
        """
        self._params = self._validate_params(params=params)

    @property
    def params(self) -> dict[str, Any]:
        """Configuration parameters dictionnary."""
        return self._params

    def __repr__(self) -> str:
        """Representation of the configuration.

        Returns:
            str: Configuration params.
        """
        return self.params.__repr__()

    @abstractmethod
    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate prameters values."""

    @classmethod
    def from_file(cls, config_path: Path) -> Self:
        """Instantiate Parser from configuration filepath.

        Args:
            config_path (Path): Configuration file path.

        Returns:
            Self: Parser.
        """
        return cls(params=toml.load(config_path))


class RunConfig(_Config):
    """Configuration for a run."""

    _layers_section: str = LAYERS_KEYS["section"]
    _physics_section: str = PHYSICS_KEYS["section"]
    _grid_section: str = GRID_KEYS["section"]

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
        return params


class PhysicsConfig(_Config):
    """Physics Configuration."""

    _rho: str = PHYSICS_KEYS["rho"]
    _slip_coef: str = PHYSICS_KEYS["slip coef"]
    _coriolis_param: str = PHYSICS_KEYS["f0"]
    _beta: str = PHYSICS_KEYS["beta"]
    _wstress_mag: str = PHYSICS_KEYS["wind stress magnitude"]

    @property
    def slip_coef(self) -> float:
        """Slip coefficient value."""
        return self.params[self._slip_coef]

    @property
    def rho(self) -> float:
        """Density."""
        return self.params[self._rho]

    @property
    def f0(self) -> float:
        """Coriolis Parameter."""
        return self.params[self._coriolis_param]

    @property
    def beta(self) -> float:
        """Beta (from beta-plane approximation) value."""
        return self.params[self._beta]

    @property
    def bottom_drag_coef(self) -> float:
        """Drag Coefficient."""
        return 0.5 * self.f0 * 2.0 / 2600  # Source ?

    @property
    def wind_stress_magnitude(self) -> float:
        """Wind Stress Magnitude (in Pa m-1 kg s-2)."""
        return self.params[self._wstress_mag]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate Physics Configuration.

        Args:
            params (dict[str, Any]): Physics configuration.

        Returns:
            dict[str, Any]: Physics Configuration.
        """
        # Verify slip coefficient value
        slip_coef = params[self._slip_coef]
        if (slip_coef < 0) or (slip_coef > 1):
            msg = f"Slip coefficient must be in [0, 1], got {slip_coef}."
            raise ConfigError(msg)

        return params


class LayersConfig(_Config):
    """Layers Configuration."""

    _h: str = LAYERS_KEYS["layer thickness"]
    _g_prime: str = LAYERS_KEYS["reduced gravity"]

    @property
    def h(self) -> torch.Tensor:
        """Values of layers thickness (h)."""
        h = torch.zeros(
            size=(self.nl, 1, 1),
            dtype=torch.float64,
            device=DEVICE,
        )
        h[:, 0, 0] = torch.Tensor(self.params[self._h])
        return h

    @property
    def g_prime(self) -> torch.Tensor:
        """Values of reduced gravity (g')."""
        g_prime = torch.zeros(
            size=(self.nl, 1, 1),
            dtype=torch.float64,
            device=DEVICE,
        )
        g_prime[:, 0, 0] = torch.Tensor(self.params[self._g_prime])
        return g_prime

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self._nl

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate that H and g' shapes match.

        Args:
            params (dict[str, Any]): Configuration Parameters.

        Raises:
            ConfigError: H and g' shapes don't match.

        Returns:
            dict[str, Any]: Layers Configuration.
        """
        h_shape = len(params[self._h])
        g_prime_shape = len(params[self._g_prime])

        if h_shape != g_prime_shape:
            msg = (
                f"H shape ({h_shape}) and "
                f"g' 's shape ({g_prime_shape}) don't match."
            )
            raise ConfigError(msg)
        self._nl = h_shape
        return params


class GridConfig(_Config):
    """Grid Configuration."""

    _nx: str = GRID_KEYS["points nb x"]
    _ny: str = GRID_KEYS["points nb y"]
    _lx: str = GRID_KEYS["x length"]
    _ly: str = GRID_KEYS["y length"]
    _dt: str = GRID_KEYS["timestep"]

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.params[self._nx]

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.params[self._ny]

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self.params[self._lx]

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self.params[self._ly]

    @property
    def dt(self) -> int:
        """Timestep (in seconds)."""
        return self.params[self._dt]

    @property
    def dx(self) -> float:
        """Elementary length in the x direction."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """Elementary length in the y direction."""
        return self.ly / self.ny

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate grid parameters.

        Args:
            params (dict[str, Any]): Grid parameters.

        Returns:
            dict[str, Any]: Grid parameters.
        """
        return params
