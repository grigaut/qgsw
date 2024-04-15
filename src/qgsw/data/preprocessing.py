"""Base Preprocessor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import scipy.interpolate
import torch

from qgsw.data.readers import Reader
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.data.readers import Reader
    from qgsw.grid import Grid

T = TypeVar("T")


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
        physics_config: PhysicsConfig,
        grid: Grid,
        method: None | str = None,
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
        self._physics = physics_config
        self._grid = grid
        self._method = method

    @abstractmethod
    def __call__(self, data: Reader) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(data)


class WindStressPreprocessorSpeed(_WindStressPreprocessor):
    """WindStress Preprocessor based on speed data."""

    def __call__(self, data: Reader) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data.

        Args:
            data (Reader): Data Reader.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tau_x, Tau_y
        """
        lon = data.get_1d(self._lon).astype("float64")
        lat = data.get_1d(self._lat).astype("float64")[::-1]
        time = data.get_1d(self._time).astype("float64")
        speed_x = data.get(self._u10)
        speed_y = data.get(self._v10)

        taux = np.zeros((time.shape[0] + 1, self._grid.nx + 1, self._grid.ny))
        tauy = np.zeros((time.shape[0] + 1, self._grid.nx, self._grid.ny + 1))
        drag_coef = self._physics.drag_coefficient
        rho = self._physics.rho

        for t in range(time.shape[0]):
            u = speed_x[t].T[:, ::-1]
            v = speed_y[t].T[:, ::-1]
            unorm = np.sqrt(u**2 + v**2)
            taux_ref = drag_coef / rho * unorm * u
            tauy_ref = drag_coef / rho * unorm * v
            taux_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat), taux_ref, method=self._method
            )
            tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat), tauy_ref, method=self._method
            )
            taux_i = taux_interpolator(self._grid.u_xy)
            tauy_i = tauy_interpolator(self._grid.v_xy)
            taux[t, :, :] = taux_i
            tauy[t, :, :] = tauy_i

        taux[-1][:] = taux[0][:]
        tauy[-1][:] = tauy[0][:]

        taux_tensor = torch.from_numpy(taux).type(torch.float64).to(DEVICE)
        tauy_tensor = torch.from_numpy(tauy).type(torch.float64).to(DEVICE)

        return taux_tensor[0, 1:-1, :], tauy_tensor[0, :, 1:-1]


class WindStressPreprocessorTaux(_WindStressPreprocessor):
    """WindStress Preprocessor based on tau data."""

    def __call__(self, data: Reader) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data.

        Args:
            data (Reader): Data Reader.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tau_x, Tau_y
        """
        lon = data.get_1d(self._lon)
        lat = data.get_1d(self._lat)
        time = data.get_1d(self._time)
        full_taux_ref = data.get(self._taux)
        full_tauy_ref = data.get(self._tauy)

        taux = np.zeros((time.shape[0] + 1, self._grid.nx + 1, self._grid.ny))
        tauy = np.zeros((time.shape[0] + 1, self._grid.nx, self._grid.ny + 1))
        for t in range(time.shape[0]):
            ## linear interpolation with scipy
            taux_ref = full_taux_ref[t].T
            tauy_ref = full_tauy_ref[t].T
            taux_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat), taux_ref, method=self._method
            )
            tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat), tauy_ref, method=self._method
            )
            taux_i = taux_interpolator(
                (self._grid.u_xy[0] + 360, self._grid.u_xy[1])
            )
            tauy_i = tauy_interpolator(
                (self._grid.v_xy[0] + 360, self._grid.v_xy[1])
            )

            taux[t, :, :] = taux_i
            tauy[t, :, :] = tauy_i

        taux *= 1e-4
        tauy *= 1e-4
        taux[-1][:] = taux[0][:]
        tauy[-1][:] = tauy[0][:]

        taux_tensor = torch.from_numpy(taux).type(torch.float64).to(DEVICE)
        tauy_tensor = torch.from_numpy(tauy).type(torch.float64).to(DEVICE)

        return taux_tensor[0, 1:-1, :], tauy_tensor[0, :, 1:-1]
