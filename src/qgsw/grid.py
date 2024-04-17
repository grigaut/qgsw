"""Grid defining utilities."""

import torch
from typing_extensions import Self

from qgsw.configs.core import ScriptConfig
from qgsw.configs.grid import GridConfig, LayersConfig
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
    def dx(self) -> float:
        """dx."""
        return self._config.dx

    @property
    def dy(self) -> float:
        """dy."""
        return self._config.dy

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
    def from_runconfig(cls, script_config: ScriptConfig) -> Self:
        """Construct the Grid given a ScriptConfig object.

        Args:
            script_config (ScriptConfig): Run Configuration Object.

        Returns:
            Self: Corresponding Grid.
        """
        return cls(config=script_config.grid)


class Grid3D:
    """3D Grid."""

    def __init__(
        self,
        grid_config: GridConfig,
        layers_config: LayersConfig,
    ) -> None:
        """Instantiate 3D Grid.

        Args:
            grid_config (GridConfig): Surfacic Grid Configuration.
            layers_config (LayersConfig): Layers Configuration.
        """
        self._2D = Grid(config=grid_config)
        self._layers = layers_config

    @property
    def xy(self) -> Grid:
        """Surfacic grid."""
        return self._2D

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self._2D.nx

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self._2D.ny

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self._layers.nl

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self._2D.lx

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self._2D.ly

    @property
    def lz(self) -> int:
        """Total length in the z direction (in meters)."""
        return self._layers.h.sum()

    @property
    def dx(self) -> float:
        """dx."""
        return self._2D.dx

    @property
    def dy(self) -> float:
        """dy."""
        return self._2D.dy

    @property
    def h(self) -> torch.Tensor:
        """Layers thickness."""
        return self._layers.h.expand((self.nl, self.nx, self.ny))

    @property
    def omega_xyz(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y cordinates of the Omega grid ('classical' grid corners).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return *self._2d_to_3d(self._2D.omega_xy), self.h

    @property
    def h_xyz(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y coordinates of the H grid (center of 'classical' grid cells).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._2d_to_3d(self._2D.h_xy), self.h

    @property
    def u_xyz(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y coordinates of the u grid .

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._2d_to_3d(self._2D.u_xy), self.h

    @property
    def v_xyz(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y coordinates of the H grid (center of 'classical' grid cells).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._2d_to_3d(self._2D.v_xy), self.h

    def _2d_to_3d(
        self, grid_2d: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform 2D grid into 3D grid.

        Args:
            grid_2d (tuple[torch.Tensor,torch.Tensor]): 2D grid to transform.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: 3D grid.
        """
        x, y = grid_2d
        return x.expand((self.nl, *x.shape)), y.expand((self.nl, *y.shape))

    @classmethod
    def from_runconfig(cls, script_config: ScriptConfig) -> Self:
        """Construct the 3D Grid given a ScriptConfig object.

        Args:
            script_config (ScriptConfig): Run Configuration Object.

        Returns:
            Self: Corresponding 3D Grid.
        """
        return cls(
            grid_config=script_config.grid,
            layers_config=script_config.layers,
        )
