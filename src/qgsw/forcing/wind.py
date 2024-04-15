"""Wind Forcings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate
import torch
from typing_extensions import Self

from qgsw.configs import RunConfig
from qgsw.data.loaders import Loader
from qgsw.data.preprocessing import (
    WindStressPreprocessorSpeed,
    WindStressPreprocessorTaux,
    _WindStressPreprocessor,
)
from qgsw.grid import Grid
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.windstress import WindStressDataConfig


class _WindForcing(ABC):
    """Wind Forcing Representation."""

    def __init__(self, config: RunConfig, grid: Grid) -> None:
        """Instantiate _WindForcing.

        Args:
            config (RunConfig): Run configuration.
            grid (Grid): Grid.
        """
        self._config = config
        self._grid = grid

    @abstractmethod
    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Generate Wind Stress constraints over the given grid.

        Args:
            grid (Grid): Grid to generate wind stress over.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: tau_x, tau_y
        """


class CosineZonalWindForcing(_WindForcing):
    """Simple Cosine Zonal Wind."""

    def __init__(self, config: RunConfig, grid: Grid) -> None:
        """Instantiate CosineZonalWindForcing.

        Args:
            config (RunConfig): Run configuration.
            grid (Grid): Grid.
        """
        super().__init__(config, grid)
        magnitude = self._config.windstress.magnitude
        self._tau0 = magnitude / self._config.physics.rho

    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Compute tau x and tau y based on wind data.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: Tau x, Tau y.
        """
        return self._compute_taux(), self._compute_tauy()

    def _compute_taux(self) -> float | torch.Tensor:
        """Zonal Wind.

        Returns:
            float | torch.Tensor: Tau_x
        """
        y_ugrid = 0.5 * (
            self._grid.omega_xy[1][:, 1:] + self._grid.omega_xy[1][:, :-1]
        )
        print(y_ugrid.shape)
        wind_profile = torch.cos(
            2 * torch.pi * (y_ugrid - self._grid.ly / 2) / self._grid.ly
        )
        return self._tau0 * wind_profile[1:-1, :]

    def _compute_tauy(self) -> float | torch.Tensor:
        """No meridional wind.

        Returns:
            float | torch.Tensor: Tau_y
        """
        return 0.0


class DataWindForcing(_WindForcing):
    """Wind Forcing object to handle data-based wind forcing."""

    def __init__(self, config: RunConfig, grid: Grid) -> None:
        """Instantiate DataWindForcing.

        Args:
            config (RunConfig): Run configuration.
            grid (Grid): Grid.
        """
        super().__init__(config, grid)
        self._loader = WindForcingLoader(config=config)
        self._data_type = self._config.windstress.data.data_type

    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Compute tau x and tau y based on wind data.

        Raises:
            KeyError: If the data type is not recognized.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: Tau x, Tau y.
        """
        if self._data_type == "speed":
            return self._transform_speed_data(*self._loader.retrieve())
        if self._data_type == "tau":
            return self._transform_tau_data(*self._loader.retrieve())
        msg = f"Unrecognized windstress data type: {self._data_type}."
        raise KeyError(msg)

    def _transform_speed_data(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        time: torch.Tensor,
        speed_x: torch.Tensor,
        speed_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform speed data into taux, tauy.

        Args:
            lon (torch.Tensor): Longitude.
            lat (torch.Tensor): Latitude.
            time (torch.Tensor): Time.
            speed_x (torch.Tensor): Speed x.
            speed_y (torch.Tensor): Speed y.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tau x, Tau y
        """
        taux = np.zeros((time.shape[0] + 1, self._grid.nx + 1, self._grid.ny))
        tauy = np.zeros((time.shape[0] + 1, self._grid.nx, self._grid.ny + 1))

        drag_coef = self._config.windstress.drag_coefficient
        rho = self._config.physics.rho

        for t in range(time.shape[0]):
            u = speed_x[t].T[:, ::-1]
            v = speed_y[t].T[:, ::-1]
            unorm = np.sqrt(u**2 + v**2)
            taux_ref = drag_coef / rho * unorm * u
            tauy_ref = drag_coef / rho * unorm * v
            taux_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                taux_ref,
                method=self._config.windstress.data.method,
            )
            tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                tauy_ref,
                method=self._config.windstress.data.method,
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

    def _transform_tau_data(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor,
        time: torch.Tensor,
        full_taux_ref: torch.Tensor,
        full_tauy_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform tau data into taux, tauy.

        Args:
            lon (torch.Tensor): Longitude.
            lat (torch.Tensor): Latitude.
            time (torch.Tensor): Time.
            full_taux_ref (torch.Tensor): Tau x ref.
            full_tauy_ref (torch.Tensor): Tau y ref.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tau x, Tau y
        """
        taux = np.zeros((time.shape[0] + 1, self._grid.nx + 1, self._grid.ny))
        tauy = np.zeros((time.shape[0] + 1, self._grid.nx, self._grid.ny + 1))
        for t in range(time.shape[0]):
            ## linear interpolation with scipy
            taux_ref = full_taux_ref[t].T
            tauy_ref = full_tauy_ref[t].T
            taux_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                taux_ref,
                method=self._config.windstress.data.method,
            )
            tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                tauy_ref,
                method=self._config.windstress.data.method,
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


class WindForcing:
    """Wind Forcing Object."""

    def __init__(self, forcing: _WindForcing) -> None:
        """Instantiate Wind Forcing.

        Args:
            forcing (_WindForcing): Core forcing to use.
        """
        self._forcing = forcing

    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Compute Wind Forcing.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: tau x, tau y
        """
        return self._forcing.compute()

    @classmethod
    def from_runconfig(cls, run_config: RunConfig) -> Self:
        """Construct the Wind Forcing given a RunConfig object.

        The method creates the Gird based on the grid configuration.

        Args:
            run_config (RunConfig): Run Configuration Object.

        Returns:
            Self: Corresponding Wind Forcing.
        """
        ws_type = run_config.windstress.type
        if ws_type == "cosine":
            grid = Grid.from_runconfig(run_config=run_config)
            return cls(
                forcing=CosineZonalWindForcing(config=run_config, grid=grid)
            )
        if ws_type == "data":
            grid = Grid.from_runconfig(run_config=run_config)
            return cls(forcing=DataWindForcing(config=run_config, grid=grid))

        msg = "Unrecognized windstress type."
        raise KeyError(msg)


class WindForcingLoader(
    Loader[
        RunConfig, tuple[torch.Tensor, torch.Tensor], _WindStressPreprocessor
    ]
):
    """Wind Forcing Data Loader."""

    def set_preprocessor(self, config: RunConfig) -> _WindStressPreprocessor:
        """Set WindStress preprocessor.

        Args:
            config (RunConfig): configuration.

        Raises:
            KeyError: If the configuration is not valid.

        Returns:
            _WindStressPreprocessor: Preprocessor.
        """
        ws_data = config.windstress.data
        if ws_data.data_type == "speed":
            return WindStressPreprocessorSpeed(
                longitude_key=ws_data.longitude,
                latitude_key=ws_data.latitude,
                time_key=ws_data.time,
                u10_key=ws_data.field_1,
                v10_key=ws_data.field_2,
            )
        if ws_data.data_type == "tau":
            return WindStressPreprocessorTaux(
                longitude_key=ws_data.longitude,
                latitude_key=ws_data.latitude,
                time_key=ws_data.time,
                u10_key=ws_data.field_1,
                v10_key=ws_data.field_2,
            )
        msg = "Unrecognized data type in windstress.data section."
        raise KeyError(msg)

    def _set_config(self, config: RunConfig) -> WindStressDataConfig:
        """Set Data Configuration.

        Args:
            config (RunConfig): Run Configuration.

        Returns:
            WindStressDataConfig: Data configuration.
        """
        return config.windstress.data
