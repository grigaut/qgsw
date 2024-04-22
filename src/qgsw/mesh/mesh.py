"""Mesh ensembles."""

from __future__ import annotations

import torch
from typing_extensions import Self

from qgsw.mesh.coordinates import Coordinates2D, Coordinates3D


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
    def coordinates(self) -> Coordinates2D:
        """Mesh X,Y coordinates."""
        return self._coords

    @property
    def xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X and Y meshes.

        Both tensors shapes are (nx, ny).
        """
        return self._x, self._y

    def add_z(self, z: torch.Tensor) -> Mesh3D:
        """Switch to 3D Mesh adding z coordinates.

        Args:
            z (torch.Tensor): Z coordinates.

        Returns:
            Mesh3D: 3D Mesh.
        """
        return Mesh3D(self.coordinates.add_z(z=z))

    def add_h(self, h: torch.Tensor) -> Mesh3D:
        """Switch to 3D Mesh adding layers thickness.

        Args:
            h (torch.Tensor): Layers thickness.

        Returns:
            Mesh3D: 3D Mesh.
        """
        return Mesh3D(self.coordinates.add_h(h=h))

    @classmethod
    def from_tensors(cls, x: torch.Tensor, y: torch.Tensor) -> Self:
        """Create 2D Mesh from X and Y tensors.

        Args:
            x (torch.Tensor): X coordinates Vector.
            y (torch.Tensor): Y coordinates Vector.

        Returns:
            Self: 2D Mesh.
        """
        return cls(Coordinates2D(x=x, y=y))


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
    def xyz(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """X,Y,Z meshes.

        X,Y and Z tensors all have (nz, nx, ny) shapes.
        """
        return self._zx, self._zy, self._z

    @property
    def xyh(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """X,Y,H meshes.

        X and Y have (nl, nx, ny) shapes and H has (nl,1,1) shape
        (constant thickness layers).
        """
        return self._hx, self._hy, self._h

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
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> Self:
        """Create 3D Mesh from coordinates Vectors.

        Args:
            x (torch.Tensor): X Coordinates.
            y (torch.Tensor): Y Coordinates.
            z (torch.Tensor | None, optional): Z Coordinates, must be set to
            None is h is given. Defaults to None.
            h (torch.Tensor | None, optional): H thickness, must be set to
            None is z is given. Defaults to None.

        Returns:
            Self: 3D Mesh.
        """
        return cls(Coordinates3D(x=x, y=y, z=z, h=h))
