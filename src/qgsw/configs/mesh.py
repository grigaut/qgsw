"""Space-Discretization Related Configurations."""

from __future__ import annotations

from typing import Any, Callable, ClassVar

import numpy as np
import torch

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import ConfigError
from qgsw.spatial import conversion
from qgsw.spatial.units._units import (
    DEGREES,
    KILOMETERS,
    METERS,
    RADIANS,
    Unit,
)
from qgsw.specs import DEVICE


class LayersConfig(_Config):
    """Layers Configuration."""

    _h: str = keys.LAYERS["layer thickness"]
    _g_prime: str = keys.LAYERS["reduced gravity"]

    @property
    def h(self) -> torch.Tensor:
        """Values of layers thickness (h)."""
        h = torch.zeros(
            size=(self.nl, 1, 1),
            dtype=torch.float64,
            device=DEVICE,
        )
        h[:, 0, 0] = torch.Tensor(self.params[self._h])
        return torch.Tensor(self.params[self._h], device=DEVICE).to(
            dtype=torch.float64
        )

    @property
    def g_prime(self) -> torch.Tensor:
        """Values of reduced gravity (g')."""
        return torch.Tensor(self.params[self._g_prime], device=DEVICE).to(
            dtype=torch.float64
        )

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

        return super()._validate_params(params)


class MeshConfig(_Config):
    """Grid Configuration."""

    _nx: str = keys.MESH["points nb x"]
    _ny: str = keys.MESH["points nb y"]
    _dt: str = keys.MESH["timestep"]
    _box_section: str = keys.BOX["section"]

    _conversion: ClassVar[dict[str, Callable[[float], float]]] = {
        DEGREES.name: conversion.deg_to_m_lat,
        KILOMETERS.name: conversion.km_to_m,
        METERS.name: conversion.m_to_m,
        RADIANS.name: conversion.rad_to_m,
    }

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate Grid Configuration.

        Args:
            params (dict[str, Any]): Grid configuration dictionnary.
        """
        super().__init__(params)
        self._box = BoxConfig(params=params[self._box_section])

    @property
    def box(self) -> BoxConfig:
        """Box Configuration."""
        return self._box

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
    def lx(self) -> float:
        """Total distance along x (meters)."""
        if np.isnan(self._box.x_max):
            return self._infer_lx()
        conversion = self._conversion[self._box.unit.name]
        return conversion(self._box.x_max - self._box.x_min)

    @property
    def ly(self) -> float:
        """Total distance along y (meters)."""
        conversion = self._conversion[self._box.unit.name]
        return conversion(self._box.y_max - self._box.y_min)

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate mesh parameters.

        Args:
            params (dict[str, Any]): Grid parameters.

        Returns:
            dict[str, Any]: Grid parameters.
        """
        # Verify that the layers section is present.
        if self._box_section not in params:
            msg = (
                "The mesh configuration must contain a "
                f"box section, named {self._box_section}."
            )
            raise ConfigError(msg)
        return super()._validate_params(params)

    def _infer_lx(self) -> float:
        """Infer lx value from ly, nx and ny.

        Returns:
            float: lx value (in meters)
        """
        return self.ly * self.nx / self.ny


class BoxConfig(_Config):
    """Space Configuration."""

    _x: str = keys.BOX["x"]
    _y: str = keys.BOX["y"]
    _unit: str = keys.BOX["unit"]
    _units_mapping: ClassVar[dict[str, Unit]] = {
        "deg": DEGREES,
        "m": METERS,
        "km": KILOMETERS,
        "rad": RADIANS,
    }

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate Spatial configuration.

        Args:
            params (dict[str, Any]): Spatial configuration dictionnary.

        Raises:
            ConfigError: If latitude has no min entry.
            ConfigError: If latitude has no max entry.
            ConfigError: If longitude has no min entry.
            ConfigError: If longitude has no max entry.

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

        return super()._validate_params(params)

    @property
    def unit(self) -> Unit:
        """Units."""
        return self._units_mapping[self.params[self._unit]]

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
