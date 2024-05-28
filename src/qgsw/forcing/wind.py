"""Wind Forcings."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import scipy.interpolate
import torch
from typing_extensions import Self

from qgsw.data.loaders import WindForcingLoader
from qgsw.spatial.core.discretization import SpaceDiscretization2D
from qgsw.specs import DEVICE
from qgsw.utils.type_switch import TypeSwitch

if TYPE_CHECKING:
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.configs.windstress import WindStressConfig


class _WindForcing(TypeSwitch, metaclass=ABCMeta):
    """Wind Forcing Representation."""

    _type: str

    def __init__(
        self,
        windstress_config: WindStressConfig,
        physics_config: PhysicsConfig,
        space_2d: SpaceDiscretization2D,
    ) -> None:
        """Instantiate _WindForcing.

        Args:
            windstress_config (WindStressConfig): Windstress Forcing.
            physics_config (PhysicsConfig): Physics configuration
            space_2d (SpaceDiscretization2D): 2D Space Discretization
        """
        super(TypeSwitch).__init__()
        self._physics = physics_config
        self._windstress = windstress_config
        self._space = space_2d

    @abstractmethod
    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Generate Wind Stress constraints over the space.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: tau_x, tau_y
        """


class CosineZonalWindForcing(_WindForcing):
    """Simple Cosine Zonal Wind."""

    _type = "cosine"

    def __init__(
        self,
        windstress_config: WindStressConfig,
        physics_config: PhysicsConfig,
        spave_2d: SpaceDiscretization2D,
    ) -> None:
        """Instantiate CosineZonalWindForcing.

        Args:
            windstress_config (WindStressConfig): Windstress configuration
            physics_config (PhysicsConfig): Physics configuration
            spave_2d (SpaceDiscretization2D): 2D Space Discretization
        """
        super().__init__(windstress_config, physics_config, spave_2d)
        magnitude = self._windstress.magnitude
        self._tau0 = magnitude / self._physics.rho

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
            self._space.omega.xy.y[:, 1:] + self._space.omega.xy.y[:, :-1]
        )
        wind_profile = torch.cos(
            2 * torch.pi * (y_ugrid - self._space.ly / 2) / self._space.ly,
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

    _type = "data"

    def __init__(
        self,
        windstress_config: WindStressConfig,
        physics_config: PhysicsConfig,
        space: SpaceDiscretization2D,
    ) -> None:
        """Instantiate DataWindForcing.

        Args:
            windstress_config (WindStressConfig): Windstress configuration
            physics_config (PhysicsConfig): Physics configuration
            space (SpaceDiscretization2D): 2D Space Discretization
        """
        super().__init__(windstress_config, physics_config, space)
        self._loader = WindForcingLoader(config=windstress_config)
        self._data_type = windstress_config.data.data_type

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
        time_shape = time.shape[0] + 1
        taux = np.zeros((time_shape, self._space.u.nx, self._space.u.ny))
        tauy = np.zeros((time_shape, self._space.v.nx, self._space.v.ny))

        drag_coef = self._windstress.drag_coefficient
        rho = self._physics.rho

        for t in range(time.shape[0]):
            u = speed_x[t].T[:, ::-1]
            v = speed_y[t].T[:, ::-1]
            unorm = np.sqrt(u**2 + v**2)
            taux_ref = drag_coef / rho * unorm * u
            tauy_ref = drag_coef / rho * unorm * v
            taux_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                taux_ref,
                method=self._windstress.data.method,
            )
            tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                tauy_ref,
                method=self._windstress.data.method,
            )
            taux_i = taux_interpolator(self._space.u.xy)
            tauy_i = tauy_interpolator(self._space.v.xy)
            taux[t, :, :] = taux_i
            tauy[t, :, :] = tauy_i

        taux[-1][:] = taux[0][:]
        tauy[-1][:] = tauy[0][:]

        taux_tensor = (
            torch.from_numpy(taux).type(torch.float64).to(device=DEVICE)
        )
        tauy_tensor = (
            torch.from_numpy(tauy).type(torch.float64).to(device=DEVICE)
        )

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
        time_shape = time.shape[0] + 1
        taux = np.zeros((time_shape, self._space.nx + 1, self._space.ny))
        tauy = np.zeros((time_shape, self._space.nx, self._space.ny + 1))
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
                (self._space.u.xy.x + 360, self._space.u.xy.y),
            )
            tauy_i = tauy_interpolator(
                (self._space.v.xy.x + 360, self._space.v.xy.y),
            )

            taux[t, :, :] = taux_i
            tauy[t, :, :] = tauy_i

        taux *= 1e-4
        tauy *= 1e-4
        taux[-1][:] = taux[0][:]
        tauy[-1][:] = tauy[0][:]

        taux_tensor = (
            torch.from_numpy(taux).type(torch.float64).to(device=DEVICE)
        )
        tauy_tensor = (
            torch.from_numpy(tauy).type(torch.float64).to(device=DEVICE)
        )

        return taux_tensor[0, 1:-1, :], tauy_tensor[0, :, 1:-1]


class NoWindForcing(_WindForcing):
    """No wind forcing."""

    _type = "none"

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
    def from_config(
        cls,
        windstress_config: WindStressConfig,
        grid_config: SpaceConfig,
        physics_config: PhysicsConfig,
    ) -> Self:
        """Construct the Wind Forcing given a SpaceConfig object.

        The method creates the Grid based on the space configuration.

        Args:
            windstress_config (WindStressConfig): Windstress Configuration
            grid_config (SpaceConfig): Space configuration
            physics_config (PhysicsConfig): Physics configuration

        Raises:
            KeyError: If the windstress is not recognized

        Returns:
            Self: Wind Forcing
        """
        wind_forcings = {
            CosineZonalWindForcing.get_type(): CosineZonalWindForcing,
            DataWindForcing.get_type(): DataWindForcing,
            NoWindForcing.get_type(): NoWindForcing,
        }
        if windstress_config.type not in wind_forcings:
            msg = (
                "Unrecognized perturbation type. "
                f"Possible values are {wind_forcings.keys()}"
            )
            raise KeyError(msg)

        space_2d = SpaceDiscretization2D.from_config(grid_config=grid_config)

        forcing = wind_forcings[windstress_config.type](
            windstress_config=windstress_config,
            physics_config=physics_config,
            space_2d=space_2d,
        )
        return cls(forcing=forcing)
