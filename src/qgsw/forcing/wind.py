"""Wind Forcings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from qgsw.grid import Grid

if TYPE_CHECKING:
    from qgsw.configs import PhysicsConfig, RunConfig
    from qgsw.grid import Grid


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
