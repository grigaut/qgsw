"""Coordinates storers."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


class CoordinatesInstanciationError(Exception):
    """Exception raised when instantiating coordinates."""


class Coordinates1D:
    """1D Coordinates."""

    def __init__(self, *, points: torch.Tensor) -> None:
        """Instantiate Coordinates.

        Args:
            points (torch.Tensor): Coordinates values.
        """
        self._raise_if_multidim(points=points)
        self._points = points

    @property
    def points(self) -> torch.Tensor:
        """Points values."""
        return self._points

    @property
    def n(self) -> int:
        """Number of points."""
        return self._points.shape[0]

    @property
    def l(self) -> int:  # noqa: E743
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


class Coordinates2D:
    """2D Coordinates."""

    def __init__(self, *, x: torch.Tensor, y: torch.Tensor) -> None:
        """Instantiate the Coordinates2D."""
        self._x = Coordinates1D(points=x)
        self._y = Coordinates1D(points=y)

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

        X and Y are vectors with (nx,) and (ny,) shapes.
        """
        return self.x.points, self.y.points

    def add_z(self, z: torch.Tensor) -> Coordinates3D:
        """Switch to 3D coordinates adding z coordinates.

        Args:
            z (torch.Tensor): Z coordinates.

        Returns:
            Coordinates3D: 3D Coordinates.
        """
        return Coordinates3D(x=self.x, y=self.y, z=z)

    def add_h(self, h: torch.Tensor) -> Coordinates3D:
        """Switch to 3D coordinates adding layers thickness.

        Args:
            h (torch.Tensor): Layers thickness.

        Returns:
            Coordinates3D: 3D Coordinates.
        """
        return Coordinates3D(x=self.x, y=self.y, h=h)


class Coordinates3D:
    """3D coordinates.

    Z coordinates is considered as increasing with depth. Therefore,
    1000 meters below the surface corresponds to z=1000.
    """

    def __init__(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> None:
        """Instantiate the 3D Coordinates.

        Args:
            x (torch.Tensor): X Coordinates.
            y (torch.Tensor): Y Coordinates.
            z (torch.Tensor | None, optional): Z Coordinates, must be set to
            None is h is given. Defaults to None.
            h (torch.Tensor | None, optional): H thickness, must be set to
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
            self._h = Coordinates1D(points=h)
            z_from_h = self._convert_h_to_z(h)
            self._z = Coordinates1D(points=z_from_h)
        if h is None:
            h_from_z = self._convert_z_to_h(z)
            self._ = Coordinates1D(points=h_from_z)
            self._z = Coordinates1D(points=z)

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
        """X,Y,Z coordinates."""
        return self.x.points, self.y.points, self.z.points

    @property
    def xyh(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """X,Y coordinates and layer thickness (H)."""
        return self.x.points, self.y.points, self.h.points

    def _convert_z_to_h(self, z: torch.Tensor) -> torch.Tensor:
        """Convert Z coordinates to layers thickness.

        Args:
            z (torch.Tensor): Z coordinates.

        Returns:
            torch.Tensor: Layers thickness.
        """
        return z[1:] - z[:-1]

    def _convert_h_to_z(self, h: torch.Tensor) -> torch.Tensor:
        """Convert layers thickness to Z coordinates.

        Args:
            h (torch.Tensor): Layers thickness.

        Returns:
            torch.Tensor: Z coordinates.
        """
        return F.pad(h.cumsum(0), (1, 0))

    def remove_z_h(self) -> Coordinates2D:
        """Remove Vertical coordinates.

        Returns:
            Coordinates2D: X,Y Coordinates.
        """
        return self._2d
