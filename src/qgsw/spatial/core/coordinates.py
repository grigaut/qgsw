"""Coordinates storers."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.exceptions import CoordinatesInstanciationError

if TYPE_CHECKING:
    from qgsw.utils.units._units import Unit


class Coordinates1D:
    """1D Coordinates."""

    def __init__(self, *, points: torch.Tensor, unit: Unit) -> None:
        """Instantiate Coordinates.

        Args:
            points (torch.Tensor): Coordinates values.
            unit (Unit): Coordinates unit.
        """
        self._raise_if_multidim(points=points)
        self._points = points
        self._unit = unit

    @property
    def unit(self) -> Unit:
        """Coordinates unit."""
        return self._unit

    @property
    def points(self) -> torch.Tensor:
        """Points values.

        └── (n, )-shaped
        """
        return self._points

    @property
    def n(self) -> int:
        """Number of points."""
        return self._points.shape[0]

    @property
    def l(self) -> torch.Tensor:  # noqa: E743
        """Total length."""
        return self._points[-1] - self._points[0]

    def _raise_if_multidim(self, points: torch.Tensor) -> None:
        """Raise an error is the points is not an unidimensional tensor.

        Args:
            points (torch.Tensor): Coordinates points.
        """
        if len(points.shape) != 1:
            msg = "Only unidimensional tensors are accepted as coordinates."
            raise CoordinatesInstanciationError(msg)

    def to_shape(self, n: int) -> Coordinates1D:
        """Reacreate a new 1D Coordinates.

        Args:
            n (int): New n.

        Returns:
            Coordinates1D: 1D Coordinates.
        """
        return Coordinates1D(
            points=torch.linspace(
                start=self.points[0],
                end=self.points[-1],
                steps=n,
                dtype=self.points.dtype,
                device=self.points.device,
            ),
            unit=self.unit,
        )


class Coordinates2D:
    """2D Coordinates."""

    def __init__(
        self,
        *,
        x: Coordinates1D,
        y: Coordinates1D,
    ) -> None:
        """Instantiate Coordinates 2D.

        Args:
            x (Coordinates1D): x coordinates.
            y (Coordinates1D): y coordinates.

        Raises:
            CoordinatesInstanciationError: If coordinates units don't match.
        """
        if x.unit != y.unit:
            msg = "Both coordinates must have the same unit."
            raise CoordinatesInstanciationError(msg)
        self._x = x
        self._y = y

    @property
    def x(self) -> Coordinates1D:
        """X coordinates."""
        return self._x

    @property
    def y(self) -> Coordinates1D:
        """Y coordinates."""
        return self._y

    @property
    def xy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """X,Y coordinates values.

        X and Y are vectors.
            ├── X: (nx, )-shaped
            └── Y: (ny, )-shaped
        """
        return self.x.points, self.y.points

    @property
    def xy_unit(self) -> Unit:
        """X and Y unit."""
        return self._x.unit

    def add_z(self, z: Coordinates1D) -> Coordinates3D:
        """Switch to 3D coordinates adding z coordinates.

        Args:
            z (Coordinates1D): Z coordinates.

        Returns:
            Coordinates3D: 3D Coordinates.
        """
        return Coordinates3D(x=self.x, y=self.y, z=z)

    def add_h(self, h: Coordinates1D) -> Coordinates3D:
        """Switch to 3D coordinates adding layers thickness.

        Args:
            h (Coordinates1D): Layers thickness.

        Returns:
            Coordinates3D: 3D Coordinates.
        """
        return Coordinates3D(x=self.x, y=self.y, h=h)

    def to_shape(self, nx: int, ny: int) -> Coordinates2D:
        """Recreate 2D Coordinates with new shape.

        Args:
            nx (int): New nx.
            ny (int): New ny.
            nl (int): New nl.

        Returns:
            Coordinates2D: New 2D Coordinates.
        """
        return Coordinates2D(
            x=self.x.to_shape(nx),
            y=self.y.to_shape(ny),
        )

    @classmethod
    def from_tensors(
        cls,
        *,
        x: torch.Tensor,
        x_unit: Unit,
        y: torch.Tensor,
        y_unit: Unit,
    ) -> Self:
        """Creates 2d Coordinates form tensors.

        Args:
            x (torch.Tensor): X points.
                └── (nx, )-shaped
            x_unit (Unit): X units.
            y (torch.Tensor): Y points.
                └── (ny, )-shaped
            y_unit (Unit): Y units.

        Returns:
            Self: 2D coordinates.
        """
        x = Coordinates1D(points=x, unit=x_unit)
        y = Coordinates1D(points=y, unit=y_unit)
        return cls(x=x, y=y)


class Coordinates3D:
    """3D coordinates.

    Z coordinates is considered as increasing with depth. Therefore,
    1000 meters below the surface corresponds to z=1000.
    """

    def __init__(
        self,
        *,
        x: Coordinates1D,
        y: Coordinates1D,
        z: Coordinates1D | None = None,
        h: Coordinates1D | None = None,
    ) -> None:
        """Instantiate the 3D Coordinates.

        Args:
            x (Coordinates1D): X Coordinates.
            y (Coordinates1D): Y Coordinates.
            z (Coordinates1D | None, optional): Z Coordinates, must be set to
            None is h is given. Defaults to None.
            h (Coordinates1D | None, optional): H thickness, must be set to
            None is z is given. Defaults to None.

        Raises:
            CoordinatesInstanciationError: If both z and h are None.
            CoordinatesInstanciationError: If both z and h are not None.
        """
        self._2d = Coordinates2D(x=x, y=y)
        if (z is None) and (h is None):
            msg = "Exactly one of z and h must be given, none were given."
            raise CoordinatesInstanciationError(msg)
        if (z is not None) and (h is not None):
            msg = "Exactly one of z and h must be given, both were given."
            raise CoordinatesInstanciationError(msg)
        if z is None:
            z_from_h = self._convert_h_to_z(h)
            self._h = h
            self._z = z_from_h
        if h is None:
            h_from_z = self._convert_z_to_h(z)
            self._h = h_from_z
            self._z = z

    @property
    def x(self) -> Coordinates1D:
        """X coordinates."""
        return self._2d.x

    @property
    def y(self) -> Coordinates1D:
        """Y coordinates."""
        return self._2d.y

    @property
    def z(self) -> Coordinates1D:
        """Z coordinates."""
        return self._z

    @property
    def h(self) -> Coordinates1D:
        """Layers thickness (H)."""
        return self._h

    @property
    def xyz(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """X,Y,Z coordinates.

        ├── X: (nx, )-shaped
        ├── y: (ny, )-shaped
        └── Z: (nl+1, )-shaped
        """
        return self.x.points, self.y.points, self.z.points

    @property
    def xyh(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """X,Y coordinates and layer thickness (H).

        ├── X: (nx, )-shaped
        ├── y: (ny, )-shaped
        └── H: (nl, )-shaped
        """
        return self.x.points, self.y.points, self.h.points

    @property
    def xy_unit(self) -> Unit:
        """X and Y unit."""
        return self._2d.xy_unit

    @property
    def zh_unit(self) -> Unit:
        """Z and H unit."""
        return self._z.unit

    def _convert_z_to_h(self, z: Coordinates1D) -> Coordinates1D:
        """Convert Z coordinates to layers thickness.

        Args:
            z (Coordinates1D): Z coordinates.

        Returns:
            Coordinates1D: Layers thickness.
        """
        return Coordinates1D(
            points=(z.points[1:] - z.points[:-1]),
            unit=z.unit,
        )

    def _convert_h_to_z(self, h: Coordinates1D) -> Coordinates1D:
        """Convert layers thickness to Z coordinates.

        Args:
            h (Coordinates1D): Layers thickness.

        Returns:
            Coordinates1D: Z coordinates.
        """
        return Coordinates1D(
            points=F.pad(h.points.cumsum(0), (1, 0)),
            unit=h.unit,
        )

    def remove_z_h(self) -> Coordinates2D:
        """Remove Vertical coordinates.

        Returns:
            Coordinates2D: X,Y Coordinates.
        """
        return self._2d

    def to_shape(self, nx: int, ny: int, nl: int) -> Coordinates3D:
        """Recreate 3D Coordinates with new shape.

        Args:
            nx (int): New nx.
            ny (int): New ny.
            nl (int): New nl.

        Returns:
            Coordinates3D: New 3D Coordinates.
        """
        return Coordinates3D(
            x=self.x.to_shape(nx),
            y=self.y.to_shape(ny),
            h=self.h.to_shape(nl),
        )

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
        """Create 3D coordinates form tensors.

        Args:
            x_unit (Unit): X unit.
            y_unit (Unit): Y unit.
            zh_unit (Unit): Z or H unit.
            x (torch.Tensor): X points.
                └── (nx, )-shaped
            y (torch.Tensor): Y points.
                └── (ny, )-shaped
            z (torch.Tensor | None, optional): Z points, set to None if h is
            given. Defaults to None.
                └── (nl+1, )-shaped
            h (torch.Tensor | None, optional): H points, set to None if z is
            given. Defaults to None.
                └── (nl, )-shaped

        Raises:
            CoordinatesInstanciationError: If both z and h are None.
            CoordinatesInstanciationError: If both z and h are given.

        Returns:
            Self: 3D Coordinates.
        """
        coords_2d = Coordinates2D.from_tensors(
            x=x,
            x_unit=x_unit,
            y=y,
            y_unit=y_unit,
        )
        if (z is None) and (h is None):
            msg = "Exactly one of z and h must be given, none were given."
            raise CoordinatesInstanciationError(msg)
        if (z is not None) and (h is not None):
            msg = "Exactly one of z and h must be given, both were given."
            raise CoordinatesInstanciationError(msg)
        if z is None:
            h_coords = Coordinates1D(points=h, unit=zh_unit)
            return coords_2d.add_h(h_coords)
        z_coords = Coordinates1D(points=z, unit=zh_unit)
        return coords_2d.add_z(z_coords)
