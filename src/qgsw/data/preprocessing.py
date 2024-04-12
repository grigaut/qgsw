"""Base Preprocessor."""

from abc import ABC, abstractmethod

import torch

from qgsw.data.readers import Reader


class Preprocessor(ABC):
    """Base Preprocessor."""

    @abstractmethod
    def __call__(self, data: Reader) -> torch.Tensor:
        """Preprocess data."""


class BathyPreprocessor(Preprocessor):
    """Bathymetry preprocessor."""

    def __init__(
        self,
        longitude_key: str,
        latitude_key: str,
        bathymetry_key: str,
    ) -> None:
        """Instanciate Bathymetry Preprocessor."""
        self._lon = longitude_key
        self._lat = latitude_key
        self._bathy = bathymetry_key

    def __call__(self, data: Reader) -> torch.Tensor:
        """Preprocess Bathymetry data."""
        lon_bath = data.get_1d(self._lon)
        lat_bath = data.get_1d(self._lat)
        bathy = data.get_2d(self._bathy).T
        return lon_bath, lat_bath, bathy
