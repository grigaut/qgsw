"""Bathymetry Configuration Tools."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config, _DataConfig


class BathyConfig(_Config):
    """Bathymetry Configuration."""

    section: str = keys.BATHY["section"]
    _htop: str = keys.BATHY["h top ocean"]
    _lake: str = keys.BATHY["lake minimum area"]
    _island: str = keys.BATHY["island minimum area"]
    _interpolation: str = keys.BATHY["interpolation"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate Bathymetry Config."""
        super().__init__(params)

    @cached_property
    def data(self) -> BathyDataConfig:
        """Bathymetry Data Configuration."""
        return BathyDataConfig.parse(self.params)

    @property
    def htop_ocean(self) -> int:
        """Value of htop_ocean."""
        return self.params[self._htop]

    @property
    def lake_min_area(self) -> int:
        """Lake minimum area (in meters)."""
        return self.params[self._lake]

    @property
    def island_min_area(self) -> int:
        """Lake minimum area (in meters)."""
        return self.params[self._island]

    @property
    def interpolation_method(self) -> str:
        """Bathymetry interpolation method.

        For scipy.interpolate.RegularGridInterpolator instantiation.
        """
        return self.params[self._interpolation]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate Bathymetry parameters.

        Args:
            params (dict[str, Any]): Bathymetry Configuration dictionnary.

        Returns:
            dict[str, Any]: Bathymetry Configuration dictionnary.
        """
        return super()._validate_params(params)


class BathyDataConfig(_DataConfig):
    """Bathymetric Data Configuration."""

    section: str = keys.BATHY_DATA["section"]
    _url: str = keys.BATHY_DATA["url"]
    _folder: str = keys.BATHY_DATA["folder"]
    _lon: str = keys.BATHY_DATA["longitude"]
    _lat: str = keys.BATHY_DATA["latitude"]
    _elev: str = keys.BATHY_DATA["elevation"]

    @property
    def longitude(self) -> str:
        """Longitude key."""
        return self.params[self._lon]

    @property
    def latitude(self) -> str:
        """Latitude key."""
        return self.params[self._lat]

    @property
    def elevation(self) -> str:
        """Elevation key."""
        return self.params[self._elev]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        folder = Path(params[self._folder])
        if not folder.is_dir():
            folder.mkdir()
        return params
