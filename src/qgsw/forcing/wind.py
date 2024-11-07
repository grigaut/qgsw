"""Wind Forcings."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import scipy.interpolate
import torch

from qgsw.configs.physics import PhysicsConfig
from qgsw.configs.windstress import WindStressConfig
from qgsw.data.loaders import WindForcingLoader
from qgsw.spatial.core.discretization import (
    SpaceDiscretization2D,
    SpaceDiscretization3D,
)
from qgsw.specs import DEVICE
from qgsw.utils.type_switch import TypeSwitch

if TYPE_CHECKING:
    from qgsw.configs.physics import PhysicsConfig
    from qgsw.configs.space import SpaceConfig
    from qgsw.configs.windstress import WindStressConfig


class _WindForcing(TypeSwitch, metaclass=ABCMeta):
    """Wind Forcing Representation."""

    _type: str

    @abstractmethod
    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Generate Wind Stress constraints over the space.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: tau_x, tau_y
        """

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        windstress_config: WindStressConfig,
        physics_config: PhysicsConfig,
        space_2d: SpaceDiscretization3D,
    ) -> Self: ...


class CosineZonalWindForcing(_WindForcing):
    """Simple Cosine Zonal Wind."""

    _type = "cosine"

    def __init__(
        self,
        space_2d: SpaceDiscretization2D,
        magnitude: float,
        rho: float,
    ) -> None:
        """Instantiate CosineZonalWindForcing."""
        super().__init__()
        self._space = space_2d
        self._tau0 = magnitude / rho

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
        ly = self._space.omega.ly
        y_ugrid = 0.5 * (
            self._space.omega.xy.y[:, 1:] + self._space.omega.xy.y[:, :-1]
        )
        wind_profile = torch.cos(2 * torch.pi * (y_ugrid - ly / 2) / ly)
        return self._tau0 * wind_profile[1:-1, :]

    def _compute_tauy(self) -> float | torch.Tensor:
        """No meridional wind.

        Returns:
            float | torch.Tensor: Tau_y
        """
        return 0.0

    @classmethod
    def from_config(
        cls,
        windstress_config: WindStressConfig,
        physics_config: PhysicsConfig,
        space_2d: SpaceDiscretization3D,
    ) -> Self:
        """Instantiate the CosineZonalWindForcing.

        Args:
            windstress_config (WindStressConfig): Windstress configuration.
            physics_config (PhysicsConfig): Physics configuration.
            space_2d (SpaceDiscretization3D): 2D space discreztization.

        Returns:
            Self: CosineZonalWindForcing.
        """
        return cls(
            space_2d=space_2d,
            magnitude=windstress_config.magnitude,
            rho=physics_config.rho,
        )


class DataWindForcing(_WindForcing):
    """Wind Forcing object to handle data-based wind forcing."""

    _type = "data"

    def __init__(
        self,
        loader: WindForcingLoader,
        space: SpaceDiscretization2D,
        interpolation_method: str,
        rho: float,
        drag_coef: float,
        data_type: str,
    ) -> None:
        """Instantiate DataWindForcing.

        Args:
            loader (WindForcingLoader): Data Loader.
            space (SpaceDiscretization2D): 2D Space discretization.
            interpolation_method (str): Interpolation method.
            rho (float): Density of air.
            drag_coef (float): Drag coefficient.
            data_type (str): Type of data.
        """
        TypeSwitch.__init__(self)
        self._space = space
        self._rho = rho
        self._drag = drag_coef
        self._interp_method = interpolation_method
        self._loader = loader
        self._data_type = data_type

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

        drag_coef = self._drag
        rho = self._rho

        for t in range(time.shape[0]):
            u = speed_x[t].T[:, ::-1]
            v = speed_y[t].T[:, ::-1]
            unorm = np.sqrt(u**2 + v**2)
            taux_ref = drag_coef / rho * unorm * u
            tauy_ref = drag_coef / rho * unorm * v
            taux_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                taux_ref,
                method=self._interp_method,
            )
            tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                tauy_ref,
                method=self._interp_method,
            )
            taux_i = taux_interpolator(self._space.u.xy)
            tauy_i = tauy_interpolator(self._space.v.xy)
            taux[t, :, :] = taux_i
            tauy[t, :, :] = tauy_i

        taux[-1][:] = taux[0][:]
        tauy[-1][:] = tauy[0][:]

        taux_tensor = (
            torch.from_numpy(taux).type(torch.float64).to(device=DEVICE.get())
        )
        tauy_tensor = (
            torch.from_numpy(tauy).type(torch.float64).to(device=DEVICE.get())
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
                method=self._interp_method,
            )
            tauy_interpolator = scipy.interpolate.RegularGridInterpolator(
                (lon, lat),
                tauy_ref,
                method=self._interp_method,
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
            torch.from_numpy(taux).type(torch.float64).to(device=DEVICE.get())
        )
        tauy_tensor = (
            torch.from_numpy(tauy).type(torch.float64).to(device=DEVICE.get())
        )

        return taux_tensor[0, 1:-1, :], tauy_tensor[0, :, 1:-1]

    @classmethod
    def from_config(
        cls,
        windstress_config: WindStressConfig,
        physics_config: PhysicsConfig,
        space_2d: SpaceDiscretization3D,
    ) -> Self:
        """Instantiate the DataWindForcing from configuration.

        Args:
            windstress_config (WindStressConfig): Windstress configuration.
            physics_config (PhysicsConfig): Physics configuration.
            space_2d (SpaceDiscretization3D): 2D Space discretization.

        Returns:
            Self: DataWindForcing.
        """
        return cls(
            space=space_2d,
            loader=WindForcingLoader.from_config(
                config=windstress_config.data,
            ),
            interpolation_method=windstress_config.data.method,
            rho=physics_config.rho,
            drag_coef=windstress_config.drag_coefficient,
            data_type=windstress_config.data.data_type,
        )


class NoWindForcing(_WindForcing):
    """No wind forcing."""

    _type = "none"

    def __init__(self) -> None:
        """Instantiate the NoWindForcing."""

    def compute(self) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Compute no wind forcing -> return 0.0s.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: Tau x, Tau y.
        """
        return 0.0, 0.0

    @classmethod
    def from_config(
        cls,
        windstress_config: WindStressConfig,  # noqa: ARG003
        physics_config: PhysicsConfig,  # noqa: ARG003
        space_2d: SpaceDiscretization3D,  # noqa: ARG003
    ) -> Self:
        """Instantiate DataWindForcing from configuration.

        Args:
            windstress_config (WindStressConfig): Windstress configuration.
            physics_config (PhysicsConfig): Physics configuration.
            space_2d (SpaceDiscretization3D): 2D Space discretization.

        Returns:
            Self: NoWindForcing.
        """
        return cls()


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

        forcing = wind_forcings[windstress_config.type].from_config(
            windstress_config=windstress_config,
            physics_config=physics_config,
            space_2d=space_2d,
        )
        return cls(forcing=forcing)
