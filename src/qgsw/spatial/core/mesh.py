"""Mesh ensembles."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from typing_extensions import Self

from qgsw.spatial.core.coordinates import (
    Coordinates1D,
    Coordinates2D,
    Coordinates3D,
)

if TYPE_CHECKING:
    from qgsw.spatial.units._units import Unit


class XY(NamedTuple):
    """X,Y mesh wrapper."""

    x: torch.Tensor
    y: torch.Tensor


class XYZ(NamedTuple):
    """X,Y and Z mesh wrapper."""

    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor


class XYH(NamedTuple):
    """X, Y and H mesh wrapper."""

    x: torch.Tensor
    y: torch.Tensor
    h: torch.Tensor


class Mesh2D:
    """2D Mesh."""

    def __init__(self, coordinates: Coordinates2D) -> None:
        """Instantiate 2D Mesh.

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
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self._coords.x.l

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self._coords.y.l

    @property
    def dx(self) -> float:
        """Dx."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """Dy."""
        return self.ly / self.ny

    @property
    def xy_unit(self) -> Unit:
        """Mesh unit."""
        return self._coords.xy_unit

    @property
    def coordinates(self) -> Coordinates2D:
        """Mesh X,Y coordinates."""
        return self._coords

    @property
    def xy(self) -> XY:
        """X and Y meshes.

        Both tensors shapes are (nx, ny).
        """
        return XY(self._x, self._y)

    def add_z(self, z: Coordinates1D) -> Mesh3D:
        """Switch to 3D Mesh adding z coordinates.

        Args:
            z (Coordinates1D): Z coordinates.

        Returns:
            Mesh3D: 3D Mesh.
        """
        return Mesh3D(self.coordinates.add_z(z=z))

    def add_h(self, h: Coordinates1D) -> Mesh3D:
        """Switch to 3D Mesh adding layers thickness.

        Args:
            h (Coordinates1D): Layers thickness.

        Returns:
            Mesh3D: 3D Mesh.
        """
        return Mesh3D(self.coordinates.add_h(h=h))

    @classmethod
    def from_tensors(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unit: Unit,
        y_unit: Unit,
    ) -> Self:
        """Create 2D Mesh from X and Y tensors.

        Args:
            x (torch.Tensor): X coordinates Vector.
            y (torch.Tensor): Y coordinates Vector.
            x_unit (Unit): X unit.
            y_unit (Unit): Y unit.

        Returns:
            Self: 2D Mesh.
        """
        coords = Coordinates2D.from_tensors(
            x=x,
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        return cls(coords)


class Mesh3D:
    """3D Mesh.

    Warning: the h (layer thickness) coordinates has smaller
    dimension (nl, 1, 1) than z (nl, nx, ny) to account for constant thickness
    layers and speed up calculations.
    """

    def __init__(self, coordinates: Coordinates3D) -> None:
        """Instantiate 3D Mesh.

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
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """dy."""  # noqa: D403
        return self.ly / self.ny

    @property
    def xyz(self) -> XYZ:
        """X,Y,Z meshes.

        X,Y and Z tensors all have (nz, nx, ny) shapes.
        """
        return XYZ(self._zx, self._zy, self._z)

    @property
    def xyh(self) -> XYH:
        """X,Y,H meshes.

        X and Y have (nl, nx, ny) shapes and H has (nl,1,1) shape
        (constant thickness layers).
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

    def remove_z_h(self) -> Mesh2D:
        """Remove z coordinates.

        Returns:
            Mesh2D: 2D Mesh for only X and Y.
        """
        return Mesh2D(coordinates=self._coords.remove_z_h())

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
        """Create 3D Mesh from coordinates Vectors.

        Args:
            x_unit (Unit): X unit.
            y_unit (Unit): Y unit.
            zh_unit (Unit): Z and H unit.
            x (torch.Tensor): X points.
            y (torch.Tensor): Y points.
            z (torch.Tensor | None, optional): Z points, set to None if h
            is given. Defaults to None.
            h (torch.Tensor | None, optional): H points, set to None if z
            is given. Defaults to None.

        Returns:
            Self: Mesh3D.
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