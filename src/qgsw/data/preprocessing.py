"""Base Preprocessor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from qgsw.data.readers import Reader

if TYPE_CHECKING:
    from qgsw.data.readers import Reader

T = TypeVar("T")
WindStressData = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


class Preprocessor(ABC, Generic[T]):
    """Base Preprocessor."""

    @abstractmethod
    def __call__(self, data: Reader) -> T:
        """Preprocess data."""


class BathyPreprocessor(
    Preprocessor[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """Bathymetry preprocessor."""

    def __init__(
        self,
        longitude_key: str,
        latitude_key: str,
        bathymetry_key: str,
    ) -> None:
        """Instanciate Bathymetry Preprocessor.

        Args:
            longitude_key (str): Longitude field name.
            latitude_key (str): Latitude field name.
            bathymetry_key (str): Bathymetry field name.
        """
        self._lon = longitude_key
        self._lat = latitude_key
        self._bathy = bathymetry_key

    def __call__(
        self, data: Reader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess bathymetric data.

        Args:
            data (Reader): Data Reader

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Longitude,
            Latitude, Elevation
        """
        lon_bath = data.get_1d(self._lon)
        lat_bath = data.get_1d(self._lat)
        bathy = data.get(self._bathy).T
        return lon_bath, lat_bath, bathy


class _WindStressPreprocessor(
    Preprocessor[tuple[torch.Tensor, torch.Tensor]], ABC
):
    """Windtress Preprocessor."""

    def __init__(
        self,
        longitude_key: str,
        latitude_key: str,
        time_key: str,
        *,
        taux_key: None | str = None,
        tauy_key: None | str = None,
        u10_key: None | str = None,
        v10_key: None | str = None,
    ) -> None:
        self._lon = longitude_key
        self._lat = latitude_key
        self._time = time_key
        self._taux = taux_key
        self._tauy = tauy_key
        self._u10 = u10_key
        self._v10 = v10_key

    @abstractmethod
    def __call__(self, data: Reader) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(data)


class WindStressPreprocessorSpeed(_WindStressPreprocessor):
    """WindStress Preprocessor based on speed data."""

    def __call__(self, data: Reader) -> WindStressData:
        """Preprocess data.

        Args:
            data (Reader): Data Reader.

        Returns:
            WindStressData: Longitude, Latitude, Time, Speed X, Speed Y
        """
        lon = data.get_1d(self._lon).astype("float64")
        lat = data.get_1d(self._lat).astype("float64")[::-1]
        time = data.get_1d(self._time).astype("float64")
        speed_x = data.get(self._u10)
        speed_y = data.get(self._v10)

        return lon, lat, time, speed_x, speed_y


class WindStressPreprocessorTaux(_WindStressPreprocessor):
    """WindStress Preprocessor based on tau data."""

    def __call__(self, data: Reader) -> WindStressData:
        """Preprocess data.

        Args:
            data (Reader): Data Reader.

        Returns:
            WindStressData: Longitude, Latitude, Time, Tau X, Tau Y
        """
        lon = data.get_1d(self._lon)
        lat = data.get_1d(self._lat)
        time = data.get_1d(self._time)
        full_taux_ref = data.get(self._taux)
        full_tauy_ref = data.get(self._tauy)

        return lon, lat, time, full_taux_ref, full_tauy_ref