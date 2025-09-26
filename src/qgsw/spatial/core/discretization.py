"""Space Discretizations.

Since the first coordinate of the Tensor represents
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

The Space Discretization uses a staggered grid for:
- ω : Vorticity
- h : Layer Thickness Anomaly
- u : Zonal Velocity
- v : Meridional Velocity

The Intuitive Representation of this grid is the following:

y
^

:       :       :       :
ω---v---ω---v---ω---v---ω..
|       |       |       |
u   h   u   h   u   h   u
|       |       |       |
ω---v---ω---v---ω---v---ω..
|       |       |       |
u   h   u   h   u   h   u
|       |       |       |
ω---v---ω---v---ω---v---ω..
|       |       |       |
u   h   u   h   u   h   u
|       |       |       |
ω---v---ω---v---ω---v---ω..   > x

While its actual implementation is:

ω---u---ω---u---ω---u---ω..   > y
|       |       |       |
v   h   v   h   v   h   v
|       |       |       |
ω---u---ω---u---ω---u---ω..
|       |       |       |
v   h   v   h   v   h   v
|       |       |       |
ω---u---ω---u---ω---u---ω..
|       |       |       |
v   h   v   h   v   h   v
|       |       |       |
ω---u---ω---u---ω---u---ω..
:       :       :       :


v
x
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch

from qgsw.spatial.core.grid import Grid2D, Grid3D
from qgsw.specs import DEVICE
from qgsw.utils.units._units import Unit

if TYPE_CHECKING:
    from qgsw.configs.core import SpaceConfig
    from qgsw.configs.models import ModelConfig
    from qgsw.spatial.core.coordinates import Coordinates1D


class MeshesInstanciationError(Exception):
    """Error raised when instantiating grids."""


class SpaceDiscretization2D:
    """SpaceDiscretization2D Object.

    Horizontal Grids Sizes:
        ├── ω: (nx+1, ny+1)-shaped
        ├── u: (nx+1, ny)-shaped
        ├── v: (nx, ny+1)-shaped
        └── h: (nx, ny)-shaped


    Grid Patterns:

    y
    ^

    :       :       :       :
    ω---v---ω---v---ω---v---ω..
    |       |       |       |
    u   h   u   h   u   h   u
    |       |       |       |
    ω---v---ω---v---ω---v---ω..
    |       |       |       |
    u   h   u   h   u   h   u
    |       |       |       |
    ω---v---ω---v---ω---v---ω..
    |       |       |       |
    u   h   u   h   u   h   u
    |       |       |       |
    ω---v---ω---v---ω---v---ω..   > x

    Warning: 2DMesh have x coordinate as first coordinate.
    Therefore, the actual pattern implementation corresponds to a
    90° clockwise rotation of the preceding pattern:

    ω---u---ω---u---ω---u---ω..   > y
    |       |       |       |
    v   h   v   h   v   h   v
    |       |       |       |
    ω---u---ω---u---ω---u---ω..
    |       |       |       |
    v   h   v   h   v   h   v
    |       |       |       |
    ω---u---ω---u---ω---u---ω..
    |       |       |       |
    v   h   v   h   v   h   v
    |       |       |       |
    ω---u---ω---u---ω---u---ω..
    :       :       :       :


    v
    x

    """

    def __init__(
        self,
        *,
        omega_grid: Grid2D,
        h_grid: Grid2D,
        u_grid: Grid2D,
        v_grid: Grid2D,
    ) -> None:
        """Instantiate the SpaceDiscretization2D."""
        self._verify_xy_units(
            omega_xy_unit=omega_grid.xy_unit,
            h_xy_unit=h_grid.xy_unit,
            u_xy_unit=u_grid.xy_unit,
            v_xy_unit=v_grid.xy_unit,
        )
        self._omega = omega_grid
        self._h = h_grid
        self._u = u_grid
        self._v = v_grid

    def __repr__(self) -> str:
        """String representation of the Space."""
        msg_parts = [
            "2D Space.",
            "└── Dimensions:",
            (
                f"     ├── X: {self.nx} points "
                f"- dx = {self.dx} {self.omega.xy_unit.value}"
            ),
            (
                f"     └── Y: {self.ny} points "
                f"- dy = {self.dy} {self.omega.xy_unit.value}"
            ),
        ]
        return "\n".join(msg_parts)

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.h.nx

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.h.ny

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self.h.lx

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self.h.ly

    @property
    def dx(self) -> float:
        """Dx."""
        return self.h.dx

    @property
    def dy(self) -> float:
        """Dy."""
        return self.h.dy

    @property
    def ds(self) -> float:
        """Elementary area surface."""
        return self.dx * self.dy

    @property
    def xy_unit(self) -> Unit:
        """X and Y unit."""
        return self._omega.xy_unit

    @property
    def omega(self) -> Grid2D:
        """Omega Grid.

        └── (nx+1, ny+1)-shaped

        Pattern:

        y
        ^

        :       :       :       :
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

        ω-------ω-------ω-------ω..   > y
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        :       :       :       :


        v
        x

        See https://agupubs.oFinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._omega

    @property
    def psi(self) -> Grid2D:
        """Psi Grid, same as Omega grid.

        └── (nx+1, ny+1)-shaped

        Pattern:

        y
        ^

        :       :       :       :
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

        ѱ-------ѱ-------ѱ-------ѱ..   > y
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        :       :       :       :


        v
        x
        """
        return self.omega

    @property
    def h(self) -> Grid2D:
        """H Grid.

        └── (nx, ny)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ------- ------- ------- ..   > y
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        :       :       :       :


        v
        x

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._h

    @property
    def q(self) -> Grid2D:
        """PV Grid. Same as the h grid.

        └── (nx, ny)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ------- ------- ------- ..   > y
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        :       :       :       :


        v
        x
        """
        return self.h

    @property
    def u(self) -> Grid2D:
        """U Grid.

        └── (nx+1, ny)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ------- ------- ------- ..
        |       |       |       |
        u       u       u       u
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        u       u       u       u
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        u       u       u       u
        |       |       |       |
         ------- ------- ------- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ---u--- ---u--- ---u--- ..   > y
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---u--- ---u--- ---u--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---u--- ---u--- ---u--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---u--- ---u--- ---u--- ..
        :       :       :       :


        v
        x

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._u

    @property
    def v(self) -> Grid2D:
        """V Grid.

        └── (nx, ny+1)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ---v--- ---v--- ---v--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---v--- ---v--- ---v--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---v--- ---v--- ---v--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---v--- ---v--- ---v--- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ------- ------- ------- ..   > y
        |       |       |       |
        v       v       v       v
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        v       v       v       v
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        v       v       v       v
        |       |       |       |
         ------- ------- ------- ..
        :       :       :       :


        v
        x

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._v

    def _verify_xy_units(
        self,
        omega_xy_unit: Grid2D,
        h_xy_unit: Grid2D,
        u_xy_unit: Grid2D,
        v_xy_unit: Grid2D,
    ) -> None:
        """Verify grids xy units equality.

        Args:
            omega_xy_unit (Grid2D): Omega xy unit.
            h_xy_unit (Grid2D): h xy unit.
            u_xy_unit (Grid2D): u xy unit.
            v_xy_unit (Grid2D): v xy unit.

        Raises:
            MeshesInstanciationError: If the unit don't match.
        """
        omega_h = omega_xy_unit == h_xy_unit
        h_u = h_xy_unit == u_xy_unit
        u_v = u_xy_unit == v_xy_unit

        if omega_h and h_u and u_v:
            return
        msg = "All grids xy units must correspond."
        raise MeshesInstanciationError(msg)

    def add_h(self, h: Coordinates1D) -> SpaceDiscretization3D:
        """Switch to 3D Grids adding layers thickness.

        Args:
            h (Coordinates1D): Layers thickness coordinates.

        Returns:
            SpaceDiscretization3D: 3D Grids.
        """
        omega_3d = self._omega.add_h(h=h)
        h_3d = self._h.add_h(h=h)
        u_3d = self._u.add_h(h=h)
        v_3d = self._v.add_h(h=h)
        return SpaceDiscretization3D(
            omega_grid=omega_3d,
            h_grid=h_3d,
            u_grid=u_3d,
            v_grid=v_3d,
        )

    def add_z(self, z: torch.Tensor) -> SpaceDiscretization3D:
        """Switch to 3D Grid adding z coordinates.

        Args:
            z (torch.Tensor): Z coordinates.
                └── (nl+1, )-shaped

        Returns:
            SpaceDiscretization3D: 3D Grid.
        """
        omega_3d = self.h.add_z(z=z)
        h_3d = self._h.add_z(z=z)
        u_3d = self._u.add_z(z=z)
        v_3d = self._v.add_z(z=z)
        return SpaceDiscretization3D(
            omega_grid=omega_3d,
            h_grid=h_3d,
            u_grid=u_3d,
            v_grid=v_3d,
        )

    def to_shape(self, nx: int, ny: int) -> SpaceDiscretization2D:
        """Recreate a new Space discretization 2D.

        Args:
            nx (int): New nx.
            ny (int): New ny.

        Returns:
            SpaceDiscretization2D: 2D space discretization with new shape.
        """
        return SpaceDiscretization2D(
            omega_grid=self.omega.to_shape(nx + 1, ny + 1),
            h_grid=self.h.to_shape(nx, ny),
            u_grid=self.u.to_shape(nx + 1, ny),
            v_grid=self.v.to_shape(nx, ny + 1),
        )

    @classmethod
    def from_config(cls, grid_config: SpaceConfig) -> Self:
        """Construct the SpaceDiscretization2D given a SpaceConfig object.

        Args:
            grid_config (SpaceConfig): Grid Configuration Object.

        Returns:
            Self: Corresponding SpaceDiscretization2D.
        """
        x = torch.linspace(
            grid_config.x_min,
            grid_config.x_max,
            grid_config.nx + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        y = torch.linspace(
            grid_config.y_min,
            grid_config.y_max,
            grid_config.ny + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        return cls.from_tensors(
            x=x,
            y=y,
            x_unit=grid_config.unit,
            y_unit=grid_config.unit,
        )

    @classmethod
    def from_tensors(
        cls,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        x_unit: Unit,
        y_unit: Unit,
    ) -> Self:
        """Generate ω, h, u, v grids from coordinates tensors.

        Args:
            x (torch.Tensor): X Coordinates.
                └── (nx, )-shaped
            y (torch.Tensor): Y Coordinates.
                └── (ny, )-shaped
            x_unit (Unit): X unit.
            y_unit (Unit): Y unit.

        Returns:
            Self: 2D Grids.
        """
        x_centers = 0.5 * (x[1:] + x[:-1])
        y_centers = 0.5 * (y[1:] + y[:-1])

        omega_grid = Grid2D.from_tensors(
            x=x,
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        h_grid = Grid2D.from_tensors(
            x=x_centers,
            y=y_centers,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        u_grid = Grid2D.from_tensors(
            x=x,
            y=y_centers,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        v_grid = Grid2D.from_tensors(
            x=x_centers,
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
        )

        return cls(
            omega_grid=omega_grid,
            h_grid=h_grid,
            u_grid=u_grid,
            v_grid=v_grid,
        )


class SpaceDiscretization3D:
    """3D Space Discretization.

    Horizontal Grids sizes
        ├── ω: (nx+1, ny+1)-shaped
        ├── u: (nx+1, ny)-shaped
        ├── v: (nx, ny+1)-shaped
        └── h: (nx, ny)-shaped


    Grids Pattern:

    y
    ^

    :       :       :       :
    ω---v---ω---v---ω---v---ω..
    |       |       |       |
    u   h   u   h   u   h   u
    |       |       |       |
    ω---v---ω---v---ω---v---ω..
    |       |       |       |
    u   h   u   h   u   h   u
    |       |       |       |
    ω---v---ω---v---ω---v---ω..
    |       |       |       |
    u   h   u   h   u   h   u
    |       |       |       |
    ω---v---ω---v---ω---v---ω..   > x

    Warning: 2DMesh have x coordinate as first coordinate.
    Therefore, the actual pattern implementation corresponds to a
    90° clockwise rotation of the preceding pattern:

    ω---u---ω---u---ω---u---ω..   > y
    |       |       |       |
    v   h   v   h   v   h   v
    |       |       |       |
    ω---u---ω---u---ω---u---ω..
    |       |       |       |
    v   h   v   h   v   h   v
    |       |       |       |
    ω---u---ω---u---ω---u---ω..
    |       |       |       |
    v   h   v   h   v   h   v
    |       |       |       |
    ω---u---ω---u---ω---u---ω..
    :       :       :       :


    v
    x

    """

    def __init__(
        self,
        *,
        omega_grid: Grid3D,
        h_grid: Grid3D,
        u_grid: Grid3D,
        v_grid: Grid3D,
    ) -> None:
        """Instantiate the SpaceDiscretization3D.

        Args:
            omega_grid (Grid3D): Omega grid.
            h_grid (Grid3D): h grid.
            u_grid (Grid3D): u grid.
            v_grid (Grid3D): v grid.
        """
        self._verify_xy_units(
            omega_xy_unit=omega_grid.xy_unit,
            h_xy_unit=h_grid.xy_unit,
            u_xy_unit=u_grid.xy_unit,
            v_xy_unit=v_grid.xy_unit,
        )
        self._verify_zh_units(
            omega_zh_unit=omega_grid.zh_unit,
            h_zh_unit=h_grid.zh_unit,
            u_zh_unit=u_grid.zh_unit,
            v_zh_unit=v_grid.zh_unit,
        )
        self._omega = omega_grid
        self._h = h_grid
        self._u = u_grid
        self._v = v_grid

    def get_repr_parts(self) -> list[str]:
        """String representations parts.

        Returns:
            list[str]: String representation parts.
        """
        return [
            "3D Space",
            "└── Dimensions:",
            (
                f"\t├── X: {self.nx} points "
                f"- dx = {self.dx} {self.omega.xy_unit.value}"
            ),
            (
                f"\t├── Y: {self.ny} points "
                f"- dy = {self.dy} {self.omega.xy_unit.value}"
            ),
            f"\t└── H: {self.nl} layer{'s' if self.nl > 1 else ''}",
        ]

    def __repr__(self) -> str:
        """String representation of the Space."""
        return "\n".join(self.get_repr_parts())

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.h.nx

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.h.ny

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self.h.nl

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self.h.lx

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self.h.ly

    @property
    def lz(self) -> int:
        """Total length in the z direction (in meters)."""
        return self.h.lz

    @property
    def dx(self) -> float:
        """dx."""  # noqa: D403
        return self.h.dx

    @property
    def dy(self) -> float:
        """dy."""  # noqa: D403
        return self.h.dy

    @property
    def ds(self) -> float:
        """Elementary area surface."""
        return self.dx * self.dy

    @property
    def omega(self) -> Grid3D:
        """X,Y cordinates of the Omega grid ('classical' grid corners).

        └── (nx+1, ny+1)-shaped

        Pattern:

        y
        ^

        :       :       :       :
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

        ω-------ω-------ω-------ω..   > y
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ω-------ω-------ω-------ω..
        :       :       :       :


        v
        x

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._omega

    @property
    def psi(self) -> Grid3D:
        """X,Y cordinates of the Psi grid ('classical' grid corners).

        Same as Omega grid.

        └── (nx+1, ny+1)-shaped

        Pattern:

        y
        ^

        :       :       :       :
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

        ѱ-------ѱ-------ѱ-------ѱ..   > y
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
        ѱ-------ѱ-------ѱ-------ѱ..
        :       :       :       :


        v
        x
        """
        return self.omega

    @property
    def h(self) -> Grid3D:
        """X,Y coordinates of the H grid (center of 'classical' grid cells).

        └── (nx, ny)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ------- ------- ------- ..   > y
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   h   |   h   |   h   |
        |       |       |       |
         ------- ------- ------- ..
        :       :       :       :


        v
        x
        """
        return self._h

    @property
    def q(self) -> Grid3D:
        """X,Y coordinates of the q grid (center of 'classical' grid cells).

        Same as H grid.

        └── (nx, ny)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ------- ------- ------- ..   > y
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        |   q   |   q   |   q   |
        |       |       |       |
         ------- ------- ------- ..
        :       :       :       :


        v
        x
        """
        return self.h

    @property
    def u(self) -> Grid3D:
        """X,Y coordinates of the u grid.

        └── (nx+1, ny)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ------- ------- ------- ..
        |       |       |       |
        u       u       u       u
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        u       u       u       u
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        u       u       u       u
        |       |       |       |
         ------- ------- ------- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ---u--- ---u--- ---u--- ..   > y
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---u--- ---u--- ---u--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---u--- ---u--- ---u--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---u--- ---u--- ---u--- ..
        :       :       :       :


        v
        x

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._u

    @property
    def v(self) -> Grid3D:
        """X,Y coordinates of the H grid (center of 'classical' grid cells).

        └── (nx, ny+1)-shaped

        Pattern:

        y
        ^

        :       :       :       :
         ---v--- ---v--- ---v--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---v--- ---v--- ---v--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---v--- ---v--- ---v--- ..
        |       |       |       |
        |       |       |       |
        |       |       |       |
         ---v--- ---v--- ---v--- ..   > x

        Warning: 2DMesh have x coordinate as first coordinate.
        Therefore, the actual pattern implementation corresponds to a
        90° clockwise rotation of the preceding pattern:

         ------- ------- ------- ..   > y
        |       |       |       |
        v       v       v       v
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        v       v       v       v
        |       |       |       |
         ------- ------- ------- ..
        |       |       |       |
        v       v       v       v
        |       |       |       |
         ------- ------- ------- ..
        :       :       :       :


        v
        x

        See https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
        for more details.
        """
        return self._v

    def _verify_xy_units(
        self,
        omega_xy_unit: Grid2D,
        h_xy_unit: Grid2D,
        u_xy_unit: Grid2D,
        v_xy_unit: Grid2D,
    ) -> None:
        """Verify grids xy units equality.

        Args:
            omega_xy_unit (Grid2D): Omega xy unit.
            h_xy_unit (Grid2D): h xy unit.
            u_xy_unit (Grid2D): u xy unit.
            v_xy_unit (Grid2D): v xy unit.

        Raises:
            MeshesInstanciationError: If the unit don't match.
        """
        omega_h = omega_xy_unit == h_xy_unit
        h_u = h_xy_unit == u_xy_unit
        u_v = u_xy_unit == v_xy_unit

        if omega_h and h_u and u_v:
            return
        msg = "All grids xy units must correspond."
        raise MeshesInstanciationError(msg)

    def _verify_zh_units(
        self,
        omega_zh_unit: Grid2D,
        h_zh_unit: Grid2D,
        u_zh_unit: Grid2D,
        v_zh_unit: Grid2D,
    ) -> None:
        """Verify grids zh units equality.

        Args:
            omega_zh_unit (Grid2D): Omega zh unit.
            h_zh_unit (Grid2D): h zh unit.
            u_zh_unit (Grid2D): u zh unit.
            v_zh_unit (Grid2D): v zh unit.

        Raises:
            MeshesInstanciationError: If the unit don't match.
        """
        omega_h = omega_zh_unit == h_zh_unit
        h_u = h_zh_unit == u_zh_unit
        u_v = u_zh_unit == v_zh_unit

        if omega_h and h_u and u_v:
            return
        msg = "All grids zh units must correspond."
        raise MeshesInstanciationError(msg)

    def remove_z_h(self) -> SpaceDiscretization2D:
        """Remove z coordinates.

        Returns:
            SpaceDiscretization2D: 2D Grid for only X and Y.
        """
        return SpaceDiscretization2D(
            omega_grid=self._omega.remove_z_h(),
            h_grid=self._h.remove_z_h(),
            u_grid=self._u.remove_z_h(),
            v_grid=self._v.remove_z_h(),
        )

    def to_shape(self, nx: int, ny: int, nl: int) -> SpaceDiscretization3D:
        """Recreate a new Space discretization 3D.

        Args:
            nx (int): New nx.
            ny (int): New ny.
            nl (int): New nl.

        Returns:
            SpaceDiscretization3D: 3D space discretization with new shape.
        """
        return SpaceDiscretization3D(
            omega_grid=self.omega.to_shape(nx + 1, ny + 1, nl),
            h_grid=self.h.to_shape(nx, ny, nl),
            u_grid=self.u.to_shape(nx + 1, ny, nl),
            v_grid=self.v.to_shape(nx, ny + 1, nl),
        )

    @classmethod
    def from_config(
        cls,
        grid_config: SpaceConfig,
        model_config: ModelConfig,
    ) -> Self:
        """Construct the 3D Grid given a SpaceConfig object.

        Args:
            grid_config (SpaceConfig): Grid Configuration Object.
            model_config (ModelConfig): Model Configuration Object.

        Returns:
            Self: Corresponding 3D Grid.
        """
        x = torch.linspace(
            grid_config.x_min,
            grid_config.x_max,
            grid_config.nx + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        y = torch.linspace(
            grid_config.y_min,
            grid_config.y_max,
            grid_config.ny + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        return cls.from_tensors(
            x=x,
            y=y,
            h=model_config.h,
            x_unit=grid_config.unit,
            y_unit=grid_config.unit,
            zh_unit=Unit.M,
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
        """Generate ω, h, u, v grids from coordinates tensors.

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
            Self: 3D Grids.
        """
        x_centers = 0.5 * (x[1:] + x[:-1])
        y_centers = 0.5 * (y[1:] + y[:-1])

        omega_grid = Grid3D.from_tensors(
            x=x,
            y=y,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )
        h_grid = Grid3D.from_tensors(
            x=x_centers,
            y=y_centers,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )
        u_grid = Grid3D.from_tensors(
            x=x,
            y=y_centers,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )
        v_grid = Grid3D.from_tensors(
            x=x_centers,
            y=y,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )

        return cls(
            omega_grid=omega_grid,
            h_grid=h_grid,
            u_grid=u_grid,
            v_grid=v_grid,
        )


def keep_top_layer(space: SpaceDiscretization3D) -> SpaceDiscretization3D:
    """Keep Only Top Layer.

    Args:
        space (SpaceDiscretization3D): Original Space.

    Returns:
        SpaceDiscretization3D: Top Layer.
    """
    return SpaceDiscretization3D.from_tensors(
        x_unit=space.omega.xy_unit,
        y_unit=space.omega.xy_unit,
        zh_unit=space.omega.zh_unit,
        x=space.omega.xyh.x[0, :, 0],
        y=space.omega.xyh.y[0, 0, :],
        h=space.omega.xyh.h[0, 0, 0].unsqueeze(0),
    )
