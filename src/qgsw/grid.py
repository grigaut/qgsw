"""Grid defining utilities."""

import torch
from typing_extensions import Self

from qgsw.configs import GridConfig, RunConfig
from qgsw.specs import DEVICE


class Grid:
    """Grid Object."""

    def __init__(self, config: GridConfig) -> None:
        """Instantiate the Grid."""
        self._config = config
        self._generate_grid()

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self._config.nx

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self._config.ny

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self._config.lx

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self._config.ly

    @property
    def x(self) -> torch.Tensor:
        """X."""
        return self._x

    @property
    def y(self) -> torch.Tensor:
        """Y."""
        return self._y

    def _generate_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate X,Y grids.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: X,Y
        """
        self._x, self._y = torch.meshgrid(
            torch.linspace(
                0,
                self.lx,
                self.nx + 1,
                dtype=torch.float64,
                device=DEVICE,
            ),
            torch.linspace(
                0,
                self.ly,
                self.ny + 1,
                dtype=torch.float64,
                device=DEVICE,
            ),
            indexing="ij",
        )

    def generate_coriolis_grid(self, f0: float, beta: float) -> torch.Tensor:
        """Generate Coriolis Parameter Grid.

        Args:
            f0 (float): f0 (from beta-plane approximation).
            beta (float): Beta (from beta plane approximation)

        Returns:
            torch.Tensor: _description_
        """
        return f0 + beta * (self.y - self.ly / 2)

    @classmethod
    def from_runconfig(cls, run_config: RunConfig) -> Self:
        """Construct the Grid given a RunConfig object.

        Args:
            run_config (RunConfig): Run Configuration Object.

        Returns:
            Self: Corresponding Grid.
        """
        return cls(config=run_config.grid)
