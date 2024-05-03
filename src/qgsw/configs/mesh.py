"""Space-Discretization Related Configurations."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, ClassVar

import numpy as np

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


class MeshConfig(_Config):
    """Grid Configuration."""

    section: str = keys.MESH["section"]
    _nx: str = keys.MESH["points nb x"]
    _ny: str = keys.MESH["points nb y"]
    _dt: str = keys.MESH["timestep"]

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

    @cached_property
    def box(self) -> BoxConfig:
        """Box Configuration."""
        return BoxConfig.parse(self.params)

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
        if np.isnan(self.box.x_max):
            return self._infer_lx()
        conversion = self._conversion[self.box.unit.name]
        return conversion(self.box.x_max - self.box.x_min)

    @property
    def ly(self) -> float:
        """Total distance along y (meters)."""
        conversion = self._conversion[self.box.unit.name]
        return conversion(self.box.y_max - self.box.y_min)

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate mesh parameters.

        Args:
            params (dict[str, Any]): Grid parameters.

        Returns:
            dict[str, Any]: Grid parameters.
        """
        return super()._validate_params(params)

    def _infer_lx(self) -> float:
        """Infer lx value from ly, nx and ny.

        Returns:
            float: lx value (in meters)
        """
        return self.ly * self.nx / self.ny


class BoxConfig(_Config):
    """Space Configuration."""

    section: str = keys.BOX["section"]
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
