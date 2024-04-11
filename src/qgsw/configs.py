"""Parsing tools for configuration files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
import toml
import torch
from typing_extensions import Self

from qgsw import conversion
from qgsw.bathymetry import BathyLoader
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
    "box unit": "box_unit",
    "points nb x": "nx",
    "points nb y": "ny",
    "x length": "Lx",
    "y length": "Ly",
    "timestep": "dt",
}

BOX_KEYS = {
    "section": "box",
    "x": "x",
    "y": "y",
}

BATHY_KEYS = {
    "section": "bathymetry",
    "url": "URL",
    "folder": "folder",
    "h top ocean": "htop_ocean",
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
    _bathy_section: str = BATHY_KEYS["section"]

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
        return self._bathy

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
        # Verify that the bathymetry section is present.
        if self._bathy_section not in params:
            msg = (
                "The configuration must contain a "
                f"bathymetry section, named {self._bathy_section}."
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
    _dt: str = GRID_KEYS["timestep"]
    _box_section: str = BOX_KEYS["section"]
    _box_unit: str = GRID_KEYS["box unit"]

    _conversion: ClassVar[dict[str, Callable[[float], float]]] = {
        "deg": conversion.deg_to_m_lat,
        "km": conversion.km_to_m,
        "m": conversion.m_to_m,
    }

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate Grid Configuration.

        Args:
            params (dict[str, Any]): Grid configuration dictionnary.
        """
        super().__init__(params)
        self._box = BoxConfig(params=params[self._box_section])

    @property
    def xy_unit(self) -> str:
        """Unit of extremums."""
        return self.params[self._box_unit]

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.params[self._nx]

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.params[self._ny]

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

    @property
    def x_min(self) -> float:
        """X min."""
        return self._box.x_min

    @property
    def x_max(self) -> float:
        """X max."""
        x_max = self._box.x_max
        if np.isnan(x_max):
            x_max = self._infer_deg_xmax()
        return x_max

    @property
    def y_min(self) -> float:
        """Y min."""
        return self._box.y_min

    @property
    def y_max(self) -> float:
        """Y max."""
        return self._box.y_max

    @property
    def lx(self) -> float:
        """Total distance along x (meters)."""
        if np.isnan(self._box.x_max):
            return self._infer_lx()
        return self._conversion[self.xy_unit](self.x_max - self.x_min)

    @property
    def ly(self) -> float:
        """Total distance along y (meters)."""
        return self._conversion[self.xy_unit](self.y_max - self.y_min)

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate grid parameters.

        Args:
            params (dict[str, Any]): Grid parameters.

        Returns:
            dict[str, Any]: Grid parameters.
        """
        # Verify that the layers section is present.
        if self._box_section not in params:
            msg = (
                "The grid configuration must contain a "
                f"box section, named {self._box_section}."
            )
            raise ConfigError(msg)
        return params

    def _infer_deg_xmax(self) -> float:
        if self._box_unit == "deg":
            msg = "X max can only be infered for degree extremums."
            raise ConfigError(msg)
        ymin_cos = np.cos(self.y_min / 180 * np.pi)
        ymax_cos = np.cos(self.y_max / 180 * np.pi)
        mean_cos = 0.5 * (ymin_cos + ymax_cos)
        return self.x_min + self.lx / (conversion.deg_to_m_lat(mean_cos))

    def _infer_lx(self) -> float:
        return self.ly * self.nx / self.ny


class BoxConfig(_Config):
    """Space Configuration."""

    _x: str = BOX_KEYS["x"]
    _y: str = BOX_KEYS["y"]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate Spatial configuration.

        Args:
            params (dict[str, Any]): Spatial configuration dictionnary.

        Raises:
            ConfigError: If latitude has no min entry.
            ConfigError: If latitude has no max entry.
            ConfigError: If longitude has no min entry.
            ConfigError: If longitude has no max entry.
            ConfigError: If some latitude or longitude are not given.

        Returns:
            dict[str, Any]: Configuration dictionnary.
        """
        if "min" not in params[self._x]:
            msg = "X section must contain a `min` entry"
            raise ConfigError(msg)
        if "max" not in params[self._x]:
            msg = "X section must contain a `max` entry"
            raise ConfigError(msg)
        if "min" not in params[self._y]:
            msg = "Y section must contain a `min` entry"
            raise ConfigError(msg)
        if "max" not in params[self._y]:
            msg = "Y section must contain a `max` entry"
            raise ConfigError(msg)

        are_none = [
            params[self._x]["min"] is None,
            params[self._x]["max"] is None,
            params[self._y]["min"] is None,
            params[self._y]["max"] is None,
        ]
        if sum(are_none) > 0:
            msg = "All coordinates extremums must be renseigned."
            raise ConfigError(msg)

        return params

    @property
    def x_min(self) -> float:
        """X min."""
        return self.params[self._x]["min"]

    @property
    def x_max(self) -> float | None:
        """X max."""
        return self.params[self._x]["max"]

    @property
    def y_min(self) -> float:
        """Y min."""
        return self.params[self._y]["min"]

    @property
    def y_max(self) -> float:
        """Y max."""
        return self.params[self._y]["max"]


class BathyConfig(_Config):
    """Bathymetry Configuration."""

    _url: str = BATHY_KEYS["url"]
    _folder: str = BATHY_KEYS["folder"]
    _htop: str = BATHY_KEYS["h top ocean"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate Bathymetry Config."""
        super().__init__(params)
        loader = BathyLoader.from_url(
            url=self.url,
            savefolder=self.folder,
        )
        self._lon, self._lat, self._bathy = loader.retrieve_bathy()

    @property
    def url(self) -> str:
        """Data URL."""
        return self.params[self._url]

    @property
    def folder(self) -> Path:
        """Data saving folder."""
        return Path(self.params[self._folder])

    @property
    def lons(self) -> np.ndarray:
        """Bathymetry longitude array."""
        return self._lon

    @property
    def lats(self) -> np.ndarray:
        """Bathymetry latitude array."""
        return self._lat

    @property
    def elevation(self) -> np.ndarray:
        """Bahymetry."""
        return self._bathy

    @property
    def htop_ocean(self) -> int:
        """Value of htop_ocean."""
        return self.params[self._htop]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate Bathymetry parameters.

        Args:
            params (dict[str, Any]): Bathymetry Configuration dictionnary.

        Returns:
            dict[str, Any]: Bathymetry Configuration dictionnary.
        """
        folder = Path(params[self._folder])
        if not folder.is_dir():
            folder.mkdir()
        return params
