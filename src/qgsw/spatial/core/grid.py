"""Grid ensembles."""

from __future__ import annotations

from typing import NamedTuple

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch


class XY(NamedTuple):
    """X,Y grid wrapper."""

    x: torch.Tensor
    y: torch.Tensor


class XYH(NamedTuple):
    """X, Y and H grid wrapper."""

    x: torch.Tensor
    y: torch.Tensor
    h: torch.Tensor


class Grid2D:
    """2D Grid.

    Warning: Since the first coordinate of the Tensor represents
    the x coordinates, the actual Tensor is a 90° clockwise rotation
    of the intuitive X,Y Grid.

    Intuitive Rsepresentation for x and y values:

    y                            y
    ^                            ^

    :     :     :                :     :     :
    x1----x2----x3..             y3----y3----y3..
    |     |     |                |     |     |
    |     |     |                |     |     |
    x1----x2----x3..             y2----y2----y2..
    |     |     |                |     |     |
    |     |     |                |     |     |
    x1----x2----x3..  >x         y1----y1----y1..  >x

    Actual Implementation for x and y values:

    x1----x1----x1..  >y         y1----y2----y3..  >y
    |     |     |                |     |     |
    |     |     |                |     |     |
    x2----x2----x2..             y1----y2----y3..
    |     |     |                |     |     |
    |     |     |                |     |     |
    x3----x3----x3..             y1----y2----y3..
    :     :     :                :     :     :

    v                            v
    x                            x

    """

    def __init__(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        """Instantiate 2D Grid.

        Args:
            x (torch.Tensor): X coordinates tensor.
                └── (nx, ny )-shaped
            y (torch.Tensor): Y coordinates tensor.
                └── (nx, ny )-shaped
        """
        self._x = x
        self._y = y
        self._dx, self._dy = self._ensure_regular(x, y)

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self._x.shape[0]

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self._y.shape[1]

    @property
    def lx(self) -> torch.Tensor:
        """Total length in the x direction (in meters)."""
        return self.dx * (self.nx - 1)

    @property
    def ly(self) -> torch.Tensor:
        """Total length in the y direction (in meters)."""
        return self.dy * (self.ny - 1)

    @property
    def dx(self) -> torch.Tensor:
        """Dx."""
        return self._dx

    @property
    def dy(self) -> torch.Tensor:
        """Dy."""
        return self._dy

    @property
    def xy(self) -> XY:
        """X and Y grids.

        Both tensors shapes are (nx, ny).

        Warning: Since the first coordinate of the Tensor represents
        the x coordinates, the actual Tensor is a 90° clockwise rotation
        of the intuitive X,Y Grid.

        Intuitive Rsepresentation for x and y values:

        y                            y
        ^                            ^

        :     :     :                :     :     :
        x1----x2----x3..             y3----y3----y3..
        |     |     |                |     |     |
        |     |     |                |     |     |
        x1----x2----x3..             y2----y2----y2..
        |     |     |                |     |     |
        |     |     |                |     |     |
        x1----x2----x3..  >x         y1----y1----y1..  >x

        Actual Implementation for x and y values:

        x1----x1----x1..  >y         y1----y2----y3..  >y
        |     |     |                |     |     |
        |     |     |                |     |     |
        x2----x2----x2..             y1----y2----y3..
        |     |     |                |     |     |
        |     |     |                |     |     |
        x3----x3----x3..             y1----y2----y3..
        :     :     :                :     :     :

        v                            v
        x                            x
        """
        return XY(self._x, self._y)

    def _ensure_regular(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify that the given grid is regular.

        Args:
            x (torch.Tensor): X coordinates tensor.
                └── (nx, ny )-shaped
            y (torch.Tensor): Y coordinates tensor.
                └── (nx, ny )-shaped

        Raises:
            ValueError: If not regular along x.
            ValueError: If not regular along y.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dx, dy
        """
        dxs = x[1:] - x[:-1]
        if (dx := dxs.unique().squeeze()).numel() != 1:
            dx_max = dx.max()
            dx_min = dx.min()
            msg = (
                "Grid spacing is not regular along x."
                f"It varies from {dx_min} to {dx_max}"
            )
            raise ValueError(msg)
        dys = y[:, 1:] - y[:, :-1]
        if (dy := dys.unique().squeeze()).numel() != 1:
            dy_max = dy.max()
            dy_min = dy.min()
            msg = (
                "Grid spacing is not regular along y."
                f"It varies from {dy_min} to {dy_max}"
            )
            raise ValueError(msg)
        return dx, dy

    def add_h_coords(self, h_1d: torch.Tensor) -> Grid3D:
        """Switch to 3D Grid adding layers thickness.

        Args:
            h_1d (torch.Tensor): Layers thickness.
                └── (nx, ny )-shaped

        Returns:
            Grid3D: 3D Grid.
        """
        nl = h_1d.shape[0]
        x = self._x.tile((nl, 1, 1))
        y = self._y.tile((nl, 1, 1))
        return Grid3D(x=x, y=y, h=h_1d.unsqueeze(1).unsqueeze(1))

    @classmethod
    def from_coords(
        cls,
        *,
        x_1d: torch.Tensor,
        y_1d: torch.Tensor,
    ) -> Self:
        """Create 2D Grid from X and Y tensors.

        Args:
            x_1d (torch.Tensor): X coordinates Vector.
                └── (nx, )-shaped
            y_1d (torch.Tensor): Y coordinates Vector.
                └── (ny, )-shaped

        Returns:
            Self: 2D Grid.
        """
        x, y = torch.meshgrid(x_1d, y_1d, indexing="ij")
        return cls(x=x, y=y)


class Grid3D:
    """3D Grid.

    Warning: the h (layer thickness) coordinates has smaller
    dimension (nl, 1, 1) than z (nl, nx, ny) to account for constant thickness
    layers and speed up calculations.

    Warning: Since the first coordinate of the Tensor represents
    the x coordinates, the actual Tensor is a 90° clockwise rotation
    of the intuitive X,Y Grid.

    Intuitive Representation for x and y values:

    y                            y
    ^                            ^

    :     :     :                :     :     :
    x1----x2----x3..             y3----y3----y3..
    |     |     |                |     |     |
    |     |     |                |     |     |
    x1----x2----x3..             y2----y2----y2..
    |     |     |                |     |     |
    |     |     |                |     |     |
    x1----x2----x3..  >x         y1----y1----y1..  >x

    Actual Implementation for x and y values:

    x1----x1----x1..  >y         y1----y2----y3..  >y
    |     |     |                |     |     |
    |     |     |                |     |     |
    x2----x2----x2..             y1----y2----y3..
    |     |     |                |     |     |
    |     |     |                |     |     |
    x3----x3----x3..             y1----y2----y3..
    :     :     :                :     :     :

    v                            v
    x                            x
    """

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
    ) -> None:
        """Instantiate 3D Grid.

        Args:
            x (torch.Tensor): X coordinates tensor.
                └── (nl, nx, ny )-shaped
            y (torch.Tensor): Y coordinates tensor.
                └── (nl, nx, ny )-shaped
            h (torch.Tensor): Y coordinates tensor.
                └── (nl, 1, 1 )-shaped
        """
        self._x = x
        self._y = y
        self._h = h
        self._dx, self._dy = self._ensure_regular(x, y)

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self._x.shape[1]

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self._y.shape[2]

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self._h.shape[0]

    @property
    def lx(self) -> torch.Tensor:
        """Total length in the x direction (in meters)."""
        return self.dx * (self.nx - 1)

    @property
    def ly(self) -> torch.Tensor:
        """Total length in the y direction (in meters)."""
        return self.dy * (self.ny - 1)

    @property
    def dx(self) -> torch.Tensor:
        """Dx."""
        return self._dx

    @property
    def dy(self) -> torch.Tensor:
        """Dy."""
        return self._dy

    @property
    def xyh(self) -> XYH:
        """X,Y,H grids.

        X and Y have (nl, nx, ny) shapes and H has (nl,1,1) shape
        (constant thickness layers).

        Warning: Since the first coordinate of the Tensor represents
        the x coordinates, the actual Tensor is a 90° clockwise rotation
        of the intuitive X,Y Grid.

        Intuitive Representation for x and y values:

        y                            y
        ^                            ^

        :     :     :                :     :     :
        x1----x2----x3..             y3----y3----y3..
        |     |     |                |     |     |
        |     |     |                |     |     |
        x1----x2----x3..             y2----y2----y2..
        |     |     |                |     |     |
        |     |     |                |     |     |
        x1----x2----x3..  >x         y1----y1----y1..  >x

        Actual Implementation for x and y values:

        x1----x1----x1..  >y         y1----y2----y3..  >y
        |     |     |                |     |     |
        |     |     |                |     |     |
        x2----x2----x2..             y1----y2----y3..
        |     |     |                |     |     |
        |     |     |                |     |     |
        x3----x3----x3..             y1----y2----y3..
        :     :     :                :     :     :

        v                            v
        x                            x
        """
        return XYH(self._x, self._y, self._h)

    def _ensure_regular(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[float, float]:
        """Verify that the given grid is regular.

        Args:
            x (torch.Tensor): X coordinates tensor.
                └── (nl, nx, ny )-shaped
            y (torch.Tensor): Y coordinates tensor.
                └── (nl, nx, ny )-shaped

        Raises:
            ValueError: If not regular along x.
            ValueError: If not regular along y.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: dx, dy
        """
        dxs = x[:, 1:] - x[:, :-1]
        if (dx := dxs.unique().squeeze()).numel() != 1:
            dx_max = dx.max()
            dx_min = dx.min()
            msg = (
                "Grid spacing is not regular along x."
                f"It varies from {dx_min} to {dx_max}"
            )
            raise ValueError(msg)
        dys = y[:, :, 1:] - y[:, :, :-1]
        if (dy := dys.unique().squeeze()).numel() != 1:
            dy_max = dy.max()
            dy_min = dy.min()
            msg = (
                "Grid spacing is not regular along y."
                f"It varies from {dy_min} to {dy_max}"
            )
            raise ValueError(msg)
        return dx, dy

    def remove_h(self) -> Grid2D:
        """Remove h coordinates.

        Returns:
            Grid2D: 2D Grid for only X and Y.
        """
        return Grid2D(x=self._x[0], y=self._y[0])

    @classmethod
    def from_coords(
        cls,
        *,
        x_1d: torch.Tensor,
        y_1d: torch.Tensor,
        h_1d: torch.Tensor,
    ) -> Self:
        """Create 3D Grid from coordinates Vectors.

        Args:
            x_1d (torch.Tensor): X points.
                └── (nx, )-shaped
            y_1d (torch.Tensor): Y points.
                └── (ny, )-shaped
            h_1d (torch.Tensor | None, optional): H points.
                └── (nl, )-shaped
        Returns:
            Self: Grid3D.
        """
        _, x, y = torch.meshgrid(
            h_1d,
            x_1d,
            y_1d,
            indexing="ij",
        )
        h = h_1d.unsqueeze(1).unsqueeze(1)
        return cls(x=x, y=y, h=h)


def loc_to_indexes_2d(
    grid_2d: Grid2D,
    *,
    x: float,
    y: float,
) -> tuple[int, int]:
    """Match locations to closest indexes.

    i = argmin((X - x).abs())
    j = argmin((Y - y).abs())

    Args:
        grid_2d (Grid2D): Grid.
        x (float): X value.
        y (float): Y value.

    Returns:
        tuple[int, int]: i, j
    """
    i: int = torch.argmin((grid_2d.xy.x[:, 0] - x).abs()).item()
    j: int = torch.argmin((grid_2d.xy.y[0, :] - y).abs()).item()
    return (i, j)


def loc_to_indexes_3d(
    grid_3d: Grid3D,
    *,
    x: float,
    y: float,
    h: float,
) -> tuple[int, int, int]:
    """Match locations to closest indexes.

    k = argmin((H - h).abs())
    i = argmin((X - x).abs())
    j = argmin((Y - y).abs())

    Args:
        grid_3d (Grid3D): Grid.
        x (float): X value.
        y (float): Y value.
        h (float): H value.

    Returns:
        tuple[int, int, int]: k, i, j
    """
    k: int = torch.argmin((grid_3d.xyh.h[:, 0, 0] - h).abs()).item()
    i: int = torch.argmin((grid_3d.xyh.x[0, :, 0] - x).abs()).item()
    j: int = torch.argmin((grid_3d.xyh.y[0, 0, :] - y).abs()).item()
    return (k, i, j)
