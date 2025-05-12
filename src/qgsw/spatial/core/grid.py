"""Grid ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch

from qgsw.spatial.core.coordinates import (
    Coordinates1D,
    Coordinates2D,
    Coordinates3D,
)

if TYPE_CHECKING:
    from qgsw.utils.units._units import Unit


class XY(NamedTuple):
    """X,Y grid wrapper."""

    x: torch.Tensor
    y: torch.Tensor


class XYZ(NamedTuple):
    """X,Y and Z grid wrapper."""

    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor


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

    def __init__(self, coordinates: Coordinates2D) -> None:
        """Instantiate 2D Grid.

        Args:
            coordinates (Coordinates2D): 2D Coordinates.
        """
        self._coords = coordinates
        self._x, self._y = torch.meshgrid(
            self._coords.x.points,
            self._coords.y.points,
            indexing="ij",
        )

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self._coords.x.n

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self._coords.y.n

    @property
    def lx(self) -> torch.Tensor:
        """Total length in the x direction (in meters)."""
        return self._coords.x.l

    @property
    def ly(self) -> torch.Tensor:
        """Total length in the y direction (in meters)."""
        return self._coords.y.l

    @property
    def dx(self) -> torch.Tensor:
        """Dx."""
        return self.lx / (self.nx - 1)

    @property
    def dy(self) -> torch.Tensor:
        """Dy."""
        return self.ly / (self.ny - 1)

    @property
    def xy_unit(self) -> Unit:
        """Grid unit."""
        return self._coords.xy_unit

    @property
    def coordinates(self) -> Coordinates2D:
        """Grid X,Y coordinates."""
        return self._coords

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

    def add_z(self, z: Coordinates1D) -> Grid3D:
        """Switch to 3D Grid adding z coordinates.

        Args:
            z (Coordinates1D): Z coordinates.

        Returns:
            Grid3D: 3D Grid.
        """
        return Grid3D(self.coordinates.add_z(z=z))

    def add_h(self, h: Coordinates1D) -> Grid3D:
        """Switch to 3D Grid adding layers thickness.

        Args:
            h (Coordinates1D): Layers thickness.

        Returns:
            Grid3D: 3D Grid.
        """
        return Grid3D(self.coordinates.add_h(h=h))

    def to_shape(self, nx: int, ny: int) -> Grid2D:
        """Recreate a new 2D Grid.

        Args:
            nx (int): New nx.
            ny (int): New ny.

        Returns:
            Grid2D: New 2D Grid.
        """
        return Grid2D(coordinates=self._coords.to_shape(nx, ny))

    @classmethod
    def from_tensors(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unit: Unit,
        y_unit: Unit,
    ) -> Self:
        """Create 2D Grid from X and Y tensors.

        Args:
            x (torch.Tensor): X coordinates Vector.
                └── (nx, )-shaped
            y (torch.Tensor): Y coordinates Vector.
                └── (ny, )-shaped
            x_unit (Unit): X unit.
            y_unit (Unit): Y unit.

        Returns:
            Self: 2D Grid.
        """
        coords = Coordinates2D.from_tensors(
            x=x,
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        return cls(coords)


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

    def __init__(self, coordinates: Coordinates3D) -> None:
        """Instantiate 3D Grid.

        Args:
            coordinates (Coordinates3D): X,Y,Z (,H) Coordinates.
        """
        self._coords = coordinates
        self._z, self._zx, self._zy = torch.meshgrid(
            self._coords.z.points,
            self._coords.x.points,
            self._coords.y.points,
            indexing="ij",
        )
        _, self._hx, self._hy = torch.meshgrid(
            self._coords.h.points,
            self._coords.x.points,
            self._coords.y.points,
            indexing="ij",
        )
        self._h = self._coords.h.points.unsqueeze(1).unsqueeze(1)

    @property
    def coordinates(self) -> Coordinates3D:
        """X,Y,Z (,H) Coordinates."""
        return self._coords

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self._coords.x.n

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self._coords.y.n

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self._coords.h.n

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self._coords.x.l

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self._coords.y.l

    @property
    def lz(self) -> int:
        """Total length in the z direction (in meters)."""
        return self._coords.z.l

    @property
    def dx(self) -> float:
        """dx."""  # noqa: D403
        return self.lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        """dy."""  # noqa: D403
        return self.ly / (self.ny - 1)

    @property
    def xyz(self) -> XYZ:
        """X,Y,Z grids.

        X,Y and Z tensors all have (nz, nx, ny) shapes.

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
        return XYZ(self._zx, self._zy, self._z)

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
        return XYH(self._hx, self._hy, self._h)

    @property
    def xy_unit(self) -> Unit:
        """X and Y unit."""
        return self._coords.xy_unit

    @property
    def zh_unit(self) -> Unit:
        """Z and H unit."""
        return self._coords.zh_unit

    def remove_z_h(self) -> Grid2D:
        """Remove z coordinates.

        Returns:
            Grid2D: 2D Grid for only X and Y.
        """
        return Grid2D(coordinates=self._coords.remove_z_h())

    def to_shape(self, nx: int, ny: int, nl: int) -> Grid3D:
        """Recreate a new 3D Grid.

        Args:
            nx (int): New nx.
            ny (int): New ny.
            nl (int): New nl

        Returns:
            Grid3D: New 3D Grid.
        """
        return Grid3D(coordinates=self._coords.to_shape(nx, ny, nl))

    @classmethod
    def from_tensors(
        cls,
        *,
        x_unit: Unit,
        y_unit: Unit,
        zh_unit: Unit,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> Self:
        """Create 3D Grid from coordinates Vectors.

        Args:
            x_unit (Unit): X unit.
            y_unit (Unit): Y unit.
            zh_unit (Unit): Z and H unit.
            x (torch.Tensor): X points.
                └── (nx, )-shaped
            y (torch.Tensor): Y points.
                └── (ny, )-shaped
            z (torch.Tensor | None, optional): Z points, set to None if h
            is given. Defaults to None.
                └── (nl+1, )-shaped
            h (torch.Tensor | None, optional): H points, set to None if z
            is given. Defaults to None.
                └── (nl, )-shaped
        Returns:
            Self: Grid3D.
        """
        coords = Coordinates3D.from_tensors(
            x=x,
            y=y,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )
        return cls(coords)


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
