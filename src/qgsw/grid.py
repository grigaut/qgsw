"""Grid defining utilities."""

import torch
from typing_extensions import Self

from qgsw.configs import RunConfig
from qgsw.configs.grid import GridConfig
from qgsw.specs import DEVICE


class Grid:
    """Grid Object."""

    def __init__(self, config: GridConfig) -> None:
        """Instantiate the Grid."""
        self._config = config
        self._generate_grids()

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
    def omega_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y cordinates of the Omega grid ('classical' grid corners).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._omega_x, self._omega_y

    @property
    def h_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y coordinates of the H grid (center of 'classical' grid cells).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._h_x, self._h_y

    @property
    def u_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y coordinates of the u grid .

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._u_x, self._u_y

    @property
    def v_xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y coordinates of the H grid (center of 'classical' grid cells).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._v_x, self._v_y

    def _generate_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate X,Y grids.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: X,Y
        """
        xs = torch.linspace(
            self._config.x_min,
            self._config.x_max,
            self.nx + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        ys = torch.linspace(
            self._config.y_min,
            self._config.y_max,
            self.ny + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        x_centers = 0.5 * (xs[1:] + xs[:-1])
        y_centers = 0.5 * (ys[1:] + ys[:-1])

        self._omega_x, self._omega_y = torch.meshgrid(
            xs,
            ys,
            indexing="ij",
        )

        self._h_x, self._h_y = torch.meshgrid(
            x_centers,
            y_centers,
            indexing="ij",
        )
        self._u_x, self._u_y = torch.meshgrid(
            xs,
            y_centers,
            indexing="ij",
        )
        self._v_x, self._v_y = torch.meshgrid(
            x_centers,
            ys,
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
        return f0 + beta * (self.omega_xy[1] - self.ly / 2)

    @classmethod
    def from_runconfig(cls, run_config: RunConfig) -> Self:
        """Construct the Grid given a RunConfig object.

        Args:
            run_config (RunConfig): Run Configuration Object.

        Returns:
            Self: Corresponding Grid.
        """
        return cls(config=run_config.grid)
