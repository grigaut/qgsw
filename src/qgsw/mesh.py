"""Meshes management."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from typing_extensions import Self

from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.core import ScriptConfig


class CoordinateInstanciationError(Exception):
    """Exception raised when instantiating coordinates."""


class Coordinates1D:
    """1D Coordinates."""

    def __init__(self, *, points: torch.Tensor) -> None:
        """Instantiate Coordinates.

        Args:
            points (torch.Tensor): Coordinates values.
        """
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
            CoordinateInstanciationError: If both z and h are None.
            CoordinateInstanciationError: If both z and h are not None.
        """
        self._2d = Coordinates2D(x=x, y=y)
        if (z is None) and (h is None):
            msg = "Exactly one of z and h must be given, none were given."
            raise CoordinateInstanciationError(msg)
        if (z is not None) and (h is not None):
            msg = "Exactly one of z and h must be given, both were given."
            raise CoordinateInstanciationError(msg)
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

    def remove_z(self) -> Coordinates2D:
        """Remove Vertical coordinates.

        Returns:
            Coordinates2D: X,Y Coordinates.
        """
        return self._2d


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
    """3D Mesh."""

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

        X,Y and H tensors all have (nh, nx, ny) shapes.
        """
        return self._hx, self._hy, self._h

    def remove_z(self) -> Mesh2D:
        """Remove z coordinates.

        Returns:
            Mesh2D: 2D Mesh for only X and Y.
        """
        return Mesh2D(coordinates=self._coords.remove_z())

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


class Meshes2D:
    """Meshes2D Object."""

    def __init__(
        self,
        *,
        omega_mesh: Mesh2D,
        h_mesh: Mesh2D,
        u_mesh: Mesh2D,
        v_mesh: Mesh2D,
    ) -> None:
        """Instantiate the Meshes2D."""
        self._omega = omega_mesh
        self._h = h_mesh
        self._u = u_mesh
        self._v = v_mesh

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.h.coordinates.x.n

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.h.coordinates.y.n

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self.omega.coordinates.x.l

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self.omega.coordinates.y.l

    @property
    def dx(self) -> float:
        """dx."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """dy."""
        return self.ly / self.ny

    @property
    def omega(self) -> Mesh2D:
        """Omega Mesh.

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._omega

    @property
    def h(self) -> Mesh2D:
        """H Mesh.

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._h

    @property
    def u(self) -> Mesh2D:
        """U Mesh.

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._u

    @property
    def v(self) -> Mesh2D:
        """V Mesh.

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._v

    def generate_coriolis_mesh(self, f0: float, beta: float) -> torch.Tensor:
        """Generate Coriolis Parameter Mesh.

        Args:
            f0 (float): f0 (from beta-plane approximation).
            beta (float): Beta (from beta plane approximation)

        Returns:
            torch.Tensor: Coriolis Mesh.
        """
        return f0 + beta * (self.omega.xy[1] - self.ly / 2)

    @classmethod
    def from_config(cls, script_config: ScriptConfig) -> Self:
        """Construct the Meshes2D given a ScriptConfig object.

        Args:
            script_config (ScriptConfig): Script Configuration Object.

        Returns:
            Self: Corresponding Meshes2D.
        """
        x = torch.linspace(
            script_config.mesh.x_min,
            script_config.mesh.x_max,
            script_config.mesh.nx + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        y = torch.linspace(
            script_config.mesh.y_min,
            script_config.mesh.y_max,
            script_config.mesh.ny + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        return cls.from_tensors(x=x, y=y)

    @classmethod
    def from_tensors(cls, *, x: torch.Tensor, y: torch.Tensor) -> Self:
        """Generate ω, h, u, v meshes from coordinates tensors.

        Args:
            x (torch.Tensor): X Coordinates.
            y (torch.Tensor): Y Coordinates.

        Returns:
            Self: 2D Meshes.
        """
        x_centers = 0.5 * (x[1:] + x[:-1])
        y_centers = 0.5 * (y[1:] + y[:-1])

        omega_mesh = Mesh2D.from_tensors(x=x, y=y)
        h_mesh = Mesh2D.from_tensors(x=x_centers, y=y_centers)
        u_mesh = Mesh2D.from_tensors(x=x, y=y_centers)
        v_mesh = Mesh2D.from_tensors(x=x_centers, y=y)

        return cls(
            omega_mesh=omega_mesh,
            h_mesh=h_mesh,
            u_mesh=u_mesh,
            v_mesh=v_mesh,
        )


class Meshes3D:
    """3D Grid."""

    def __init__(
        self,
        *,
        omega_mesh: Mesh3D,
        h_mesh: Mesh3D,
        u_mesh: Mesh3D,
        v_mesh: Mesh3D,
    ) -> None:
        """Instantiate the Meshes3D.

        Args:
            omega_mesh (Mesh3D): Omega mesh.
            h_mesh (Mesh3D): h mesh.
            u_mesh (Mesh3D): u mesh.
            v_mesh (Mesh3D): v mesh.
        """
        self._omega = omega_mesh
        self._h = h_mesh
        self._u = u_mesh
        self._v = v_mesh

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.h.coordinates.x.n

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.h.coordinates.y.n

    @property
    def nh(self) -> int:
        """Number of layers."""
        return self.h.coordinates.h.n

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self.omega.coordinates.x.l

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self.omega.coordinates.y.l

    @property
    def lz(self) -> int:
        """Total length in the z direction (in meters)."""
        return self.omega.coordinates.z.l

    @property
    def dx(self) -> float:
        """dx."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """dy."""
        return self.ly / self.ny

    @property
    def omega(self) -> Mesh3D:
        """X,Y cordinates of the Omega mesh ('classical' mesh corners).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._omega

    @property
    def h(self) -> Mesh3D:
        """X,Y coordinates of the H mesh (center of 'classical' mesh cells).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._h

    @property
    def u(self) -> Mesh3D:
        """X,Y coordinates of the u mesh .

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._u

    @property
    def v(self) -> Mesh3D:
        """X,Y coordinates of the H mesh (center of 'classical' mesh cells).

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._v

    def remove_z(self) -> Meshes2D:
        """Remove z coordinates.

        Returns:
            Meshes2D: 2D Mesh for only X and Y.
        """
        return Meshes2D(
            omega_mesh=self._omega.remove_z(),
            h_mesh=self._h.remove_z(),
            u_mesh=self._u.remove_z(),
            v_mesh=self._v.remove_z(),
        )

    @classmethod
    def from_config(cls, script_config: ScriptConfig) -> Self:
        """Construct the 3D Grid given a ScriptConfig object.

        Args:
            script_config (ScriptConfig): Script Configuration Object.

        Returns:
            Self: Corresponding 3D Grid.
        """
        x = torch.linspace(
            script_config.mesh.x_min,
            script_config.mesh.x_max,
            script_config.mesh.nx + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        y = torch.linspace(
            script_config.mesh.y_min,
            script_config.mesh.y_max,
            script_config.mesh.ny + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        return cls.from_tensors(x=x, y=y, h=script_config.layers.h)

    @classmethod
    def from_tensors(
        cls,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor | None = None,
        h: torch.Tensor | None = None,
    ) -> Self:
        """Generate ω, h, u, v meshes from coordinates tensors.

        Args:
            x (torch.Tensor): X Coordinates.
            y (torch.Tensor): Y Coordinates.
            z (torch.Tensor | None, optional): Z Coordinates, must be set to
            None is h is given. Defaults to None.
            h (torch.Tensor | None, optional): H thickness, must be set to
            None is z is given. Defaults to None.

        Returns:
            Self: 3D Meshes.
        """
        x_centers = 0.5 * (x[1:] + x[:-1])
        y_centers = 0.5 * (y[1:] + y[:-1])

        omega_mesh = Mesh3D.from_tensors(x=x, y=y, z=z, h=h)
        h_mesh = Mesh3D.from_tensors(x=x_centers, y=y_centers, z=z, h=h)
        u_mesh = Mesh3D.from_tensors(x=x, y=y_centers, z=z, h=h)
        v_mesh = Mesh3D.from_tensors(x=x_centers, y=y, z=z, h=h)

        return cls(
            omega_mesh=omega_mesh,
            h_mesh=h_mesh,
            u_mesh=u_mesh,
            v_mesh=v_mesh,
        )
