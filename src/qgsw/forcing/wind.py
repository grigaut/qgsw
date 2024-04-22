"""Wind Forcings."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate
import torch
from typing_extensions import Self

from qgsw.data.loaders import WindForcingLoader
from qgsw.mesh import Meshes2D
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.core import ScriptConfig


class _WindForcing(metaclass=ABCMeta):
    """Wind Forcing Representation."""

    def __init__(self, config: ScriptConfig, mesh: Meshes2D) -> None:
        """Instantiate _WindForcing.

        Args:
            config (ScriptConfig): Script Configuration.
            mesh (Meshes2D): Grid.
        """
        self._config = config
        self._mesh = mesh

    @abstractmethod
    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Generate Wind Stress constraints over the given mesh.

        Args:
            mesh (Meshes2D): Grid to generate wind stress over.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: tau_x, tau_y
        """


class CosineZonalWindForcing(_WindForcing):
    """Simple Cosine Zonal Wind."""

    def __init__(self, config: ScriptConfig, mesh: Meshes2D) -> None:
        """Instantiate CosineZonalWindForcing.

        Args:
            config (ScriptConfig): Script Configuration.
            mesh (Meshes2D): Grid.
        """
        super().__init__(config, mesh)
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
            self._mesh.omega.xy[1][:, 1:] + self._mesh.omega.xy[1][:, :-1]
        )
        wind_profile = torch.cos(
            2 * torch.pi * (y_ugrid - self._mesh.ly / 2) / self._mesh.ly
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

    def __init__(self, config: ScriptConfig, mesh: Meshes2D) -> None:
        """Instantiate DataWindForcing.

        Args:
            config (ScriptConfig): Script Configuration.
            mesh (Meshes2D): Grid.
        """
        super().__init__(config, mesh)
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
        taux = np.zeros((time.shape[0] + 1, self._mesh.nx + 1, self._mesh.ny))
        tauy = np.zeros((time.shape[0] + 1, self._mesh.nx, self._mesh.ny + 1))

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
            taux_i = taux_interpolator(self._mesh.u.xy)
            tauy_i = tauy_interpolator(self._mesh.v.xy)
            taux[t, :, :] = taux_i
            tauy[t, :, :] = tauy_i

        taux[-1][:] = taux[0][:]
        tauy[-1][:] = tauy[0][:]

        taux_tensor = torch.from_numpy(taux).type(torch.float64).to(DEVICE)
        tauy_tensor = torch.from_numpy(tauy).type(torch.float64).to(DEVICE)

        return taux_tensor, tauy_tensor

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
        taux = np.zeros((time.shape[0] + 1, self._mesh.nx + 1, self._mesh.ny))
        tauy = np.zeros((time.shape[0] + 1, self._mesh.nx, self._mesh.ny + 1))
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
                (self._mesh.u.xy[0] + 360, self._mesh.u.xy[1])
            )
            tauy_i = tauy_interpolator(
                (self._mesh.v.xy[0] + 360, self._mesh.v.xy[1])
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


class NoWindForcing(_WindForcing):
    """No wind forcing."""

    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Compute no wind forcing -> return 0.0s.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: Tau x, Tau y.
        """
        return 0.0, 0.0


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
    def from_config(cls, script_config: ScriptConfig) -> Self:
        """Construct the Wind Forcing given a ScriptConfig object.

        The method creates the Gird based on the mesh configuration.

        Args:
            script_config (ScriptConfig): Script Configuration Object.

        Returns:
            Self: Corresponding Wind Forcing.
        """
        ws_type = script_config.windstress.type
        if ws_type == "cosine":
            mesh = Meshes2D.from_config(script_config=script_config)
            return cls(
                forcing=CosineZonalWindForcing(config=script_config, mesh=mesh)
            )
        if ws_type == "data":
            mesh = Meshes2D.from_config(script_config=script_config)
            return cls(
                forcing=DataWindForcing(config=script_config, mesh=mesh)
            )
        if ws_type == "none":
            mesh = Meshes2D.from_config(script_config=script_config)
            return cls(forcing=NoWindForcing(config=script_config, mesh=mesh))
        msg = "Unrecognized windstress type."
        raise KeyError(msg)
