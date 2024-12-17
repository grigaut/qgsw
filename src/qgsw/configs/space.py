"""Space-Discretization Related Configurations."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import ConfigError
from qgsw.spatial import conversion
from qgsw.spatial.units._units import Unit

if TYPE_CHECKING:
    from collections.abc import Callable


class SpaceConfig(_Config):
    """Grid Configuration."""

    section: str = keys.SPACE["section"]
    _nx: str = keys.SPACE["points nb x"]
    _ny: str = keys.SPACE["points nb y"]

    _conversion: ClassVar[dict[str, Callable[[float], float]]] = {
        Unit.DEGREES: conversion.deg_to_m_lat,
        Unit.KILOMETERS: conversion.km_to_m,
        Unit.METERS: conversion.m_to_m,
        Unit.RADIANS: conversion.rad_to_m,
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
        if np.isnan(self.x_max):
            return self._infer_lx()
        conversion = self._conversion[self.unit.name]
        return conversion(self.x_max - self.x_min)

    @property
    def ly(self) -> float:
        """Total distance along y (meters)."""
        conversion = self._conversion[self.unit.name]
        return conversion(self.y_max - self.y_min)

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate space parameters.

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
        "deg": Unit.DEGREES,
        "m": Unit.METERS,
        "km": Unit.KILOMETERS,
        "rad": Unit.RADIANS,
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
