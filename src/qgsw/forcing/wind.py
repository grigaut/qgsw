"""Wind Forcings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from qgsw.configs.physics import PhysicsConfig


class _WindForcing(ABC):
    """Wind Forcing Representation."""

    def __init__(self, physics: PhysicsConfig) -> None:
        """Instantiate the Wind Forcing Object.

        Args:
            physics (PhysicsConfig): _description_
        """
        self._physics = physics

    @property
    def magnitude(self) -> float:
        """Wind Stress Magnitude."""
        return self._physics.wind_stress_magnitude

    @property
    def tau0(self) -> float:
        """Tau0."""
        return self.magnitude / self._physics.rho

    def compute_over_grid(
        self, grid: Grid
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Generate Wind Stress constraints over the given grid.

        Args:
            grid (Grid): Grid to generate wind stress over.

        Returns:
            tuple[float | torch.Tensor, float | torch.Tensor]: tau_x, tau_y
        """
        return self._compute_taux(grid=grid), self._compute_tauy(grid=grid)

    @abstractmethod
    def _compute_taux(self, grid: Grid) -> float | torch.Tensor:
        """Compute tau_x value over the grid.

        Args:
            grid (Grid): Grid over which to compute tau_x.

        Returns:
            float | torch.Tensor: Tau_x
        """

    @abstractmethod
    def _compute_tauy(self, grid: Grid) -> float | torch.Tensor:
        """Compute tau_y value over the grid.

        Args:
            grid (Grid): Grid over which to compute tau_y.

        Returns:
            float | torch.Tensor: Tau_y
        """

    @classmethod
    def from_runconfig(cls, run_config: RunConfig) -> Self:
        """Construct the Wind Forcing given a RunConfig object.

        Args:
            run_config (RunConfig): Run Configuration Object.

        Returns:
            Self: Corresponding Wind Forcing.
        """
        return cls(physics=run_config.physics)


class CosineZonalWindForcing(_WindForcing):
    """Simple Cosine Zonal Wind."""

    def _compute_taux(self, grid: Grid) -> float | torch.Tensor:
        """Zonal Wind.

        Args:
            grid (Grid): Grid to compute wind over.

        Returns:
            float | torch.Tensor: Tau_x
        """
        y_ugrid = 0.5 * (grid.y[:, 1:] + grid.y[:, :-1])
        wind_profile = torch.cos(
            2 * torch.pi * (y_ugrid - grid.ly / 2) / grid.ly
        )
        return self.tau0 * wind_profile[1:-1, :]

    def _compute_tauy(self, grid: Grid) -> float | torch.Tensor:  # noqa: ARG002
        """No meridional wind.

        Args:
            grid (Grid): Grid, for compatibility reason only.

        Returns:
            float | torch.Tensor: Tau_y
        """
        return 0.0


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
                physics_config=config.physics,
                grid=Grid.from_runconfig(config),
                u10_key=ws_data.field_1,
                v10_key=ws_data.field_2,
                method=ws_data.method,
            )
        if ws_data.data_type == "tau":
            return WindStressPreprocessorTaux(
                longitude_key=ws_data.longitude,
                latitude_key=ws_data.latitude,
                time_key=ws_data.time,
                physics_config=config.physics,
                grid=Grid.from_runconfig(config),
                u10_key=ws_data.field_1,
                v10_key=ws_data.field_2,
                method=ws_data.method,
            )
        msg = "Unrecognized data type in windstress.data section."
        raise KeyError(msg)

    def _set_filepath(self, config: RunConfig) -> Path:
        filename = Path(config.windstress.data.url).name
        return config.windstress.data.folder.joinpath(filename)

    def _set_config(self, config: RunConfig) -> None:
        self._config = config.windstress.data
