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

from qgsw.spatial.core.grid_conversion import interpolate

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch

from qgsw.spatial.core.grid import Grid2D, Grid3D
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.core import SpaceConfig
    from qgsw.configs.models import ModelConfig


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
        self._omega = omega_grid
        self._h = h_grid
        self._u = u_grid
        self._v = v_grid

    def __repr__(self) -> str:
        """String representation of the Space."""
        msg_parts = [
            "2D Space.",
            "└── Dimensions:",
            (f"     ├── X: {self.nx} points - dx = {self.dx}"),
            (f"     └── Y: {self.ny} points - dy = {self.dy}"),
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

    def add_h_coords(self, h_1d: torch.Tensor) -> SpaceDiscretization3D:
        """Switch to 3D Grids adding layers thickness.

        Args:
            h_1d (torch.Tensor): Layers thickness coordinates.

        Returns:
            SpaceDiscretization3D: 3D Grids.
        """
        omega_3d = self._omega.add_h_coords(h_1d=h_1d)
        h_3d = self._h.add_h_coords(h_1d=h_1d)
        u_3d = self._u.add_h_coords(h_1d=h_1d)
        v_3d = self._v.add_h_coords(h_1d=h_1d)
        return SpaceDiscretization3D(
            omega_grid=omega_3d,
            h_grid=h_3d,
            u_grid=u_3d,
            v_grid=v_3d,
        )

    def slice(
        self, imin: int, imax: int, jmin: int, jmax: int
    ) -> SpaceDiscretization2D:
        """Slice the space.

        Args:
            imin (int): Lower bound for x indexes.
            imax (int): Upper bound for x indexes.
            jmin (int): Lower bound for y indexes..
            jmax (int): Upper bound for y indexes..

        Returns:
            SpaceDiscretization2D: Slice space, such that
                x -> X[imin:imax, jmin:jmax] and y -> Y[imin:imax, jmin:jmax].
        """
        x_sliced = self.omega.xy.x[imin:imax, jmin:jmax]
        y_sliced = self.omega.xy.y[imin:imax, jmin:jmax]
        return self.from_omega_grid(Grid2D(x=x_sliced, y=y_sliced))

    @classmethod
    def from_config(cls, grid_config: SpaceConfig) -> Self:
        """Construct the SpaceDiscretization2D given a SpaceConfig object.

        Args:
            grid_config (SpaceConfig): Grid Configuration Object.

        Returns:
            Self: Corresponding SpaceDiscretization2D.
        """
        x_1d = torch.linspace(
            grid_config.x_min,
            grid_config.x_max,
            grid_config.nx + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        y_1d = torch.linspace(
            grid_config.y_min,
            grid_config.y_max,
            grid_config.ny + 1,
            dtype=torch.float64,
            device=DEVICE.get(),
        )
        return cls.from_coords(
            x_1d=x_1d,
            y_1d=y_1d,
        )

    @classmethod
    def from_coords(
        cls,
        *,
        x_1d: torch.Tensor,
        y_1d: torch.Tensor,
    ) -> Self:
        """Generate ω, h, u, v grids from coordinates tensors.

        Args:
            x_1d (torch.Tensor): X Coordinates.
                └── (nx, )-shaped
            y_1d (torch.Tensor): Y Coordinates.
                └── (ny, )-shaped

        Returns:
            Self: 2D Grids.
        """
        omega_grid = Grid2D.from_coords(
            x_1d=x_1d,
            y_1d=y_1d,
        )

        return cls.from_omega_grid(omega_grid)

    @classmethod
    def from_omega_grid(cls, omega_grid: Grid2D) -> Self:
        """Instantiate space from omega grid.

        Args:
            omega_grid (Grid3D): Omega grid.

        Returns:
            Self: SpaceDiscretization3D.
        """
        x, y = omega_grid.xy

        h_grid = Grid2D(x=interpolate(x), y=interpolate(y))
        u_grid = Grid2D(
            x=(x[:, 1:] + x[:, :-1]) / 2,
            y=(y[:, 1:] + y[:, :-1]) / 2,
        )
        v_grid = Grid2D(
            x=(x[1:] + x[:-1]) / 2,
            y=(y[1:] + y[:-1]) / 2,
        )
        return cls(
            omega_grid=omega_grid,
            h_grid=h_grid,
            u_grid=u_grid,
            v_grid=v_grid,
        )

    @classmethod
    def from_psi_grid(cls, psi_grid: Grid2D) -> Self:
        """Instantiate space from psi grid.

        Args:
            psi_grid (Grid2D): Psi grid.

        Returns:
            Self: SpaceDiscretization2D.
        """
        return cls.from_omega_grid(psi_grid)


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
            (f"\t├── X: {self.nx} points - dx = {self.dx} m"),
            (f"\t├── Y: {self.ny} points - dy = {self.dy} m"),
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

    def remove_h(self) -> SpaceDiscretization2D:
        """Remove z coordinates.

        Returns:
            SpaceDiscretization2D: 2D Grid for only X and Y.
        """
        return SpaceDiscretization2D(
            omega_grid=self._omega.remove_h(),
            h_grid=self._h.remove_h(),
            u_grid=self._u.remove_h(),
            v_grid=self._v.remove_h(),
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
        dx = (grid_config.x_max - grid_config.x_min) / grid_config.nx
        x_1d = (
            torch.arange(
                0,
                grid_config.nx + 1,
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            * dx
            + grid_config.x_min
        )

        dy = (grid_config.y_max - grid_config.y_min) / grid_config.ny
        y_1d = (
            torch.arange(
                0,
                grid_config.ny + 1,
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            * dy
            + grid_config.y_min
        )
        return cls.from_coords(
            x_1d=x_1d,
            y_1d=y_1d,
            h_1d=model_config.h,
        )

    @classmethod
    def from_coords(
        cls,
        *,
        x_1d: torch.Tensor,
        y_1d: torch.Tensor,
        h_1d: torch.Tensor,
    ) -> Self:
        """Generate ω, h, u, v grids from coordinates tensors.

        Args:
            x_1d (torch.Tensor): X points.
                └── (nx, )-shaped
            y_1d (torch.Tensor): Y points.
                └── (ny, )-shaped
            h_1d (torch.Tensor | None, optional): H points.
                └── (nl, )-shaped

        Returns:
            Self: 3D Grids.
        """
        omega_grid = Grid3D.from_coords(x_1d=x_1d, y_1d=y_1d, h_1d=h_1d)

        return cls.from_omega_grid(omega_grid)

    @classmethod
    def from_omega_grid(cls, omega_grid: Grid3D) -> Self:
        """Instantiate space from omega grid.

        Args:
            omega_grid (Grid3D): Omega grid.

        Returns:
            Self: SpaceDiscretization3D.
        """
        x, y, h = omega_grid.xyh

        h_grid = Grid3D(x=interpolate(x), y=interpolate(y), h=h)
        u_grid = Grid3D(
            x=(x[:, :, 1:] + x[:, :, :-1]) / 2,
            y=(y[:, :, 1:] + y[:, :, :-1]) / 2,
            h=h,
        )
        v_grid = Grid3D(
            x=(x[:, 1:] + x[:, :-1]) / 2,
            y=(y[:, 1:] + y[:, :-1]) / 2,
            h=h,
        )
        return cls(
            omega_grid=omega_grid,
            h_grid=h_grid,
            u_grid=u_grid,
            v_grid=v_grid,
        )

    @classmethod
    def from_psi_grid(cls, psi_grid: Grid3D) -> Self:
        """Instantiate space from psi grid.

        Args:
            psi_grid (Grid3D): Psi grid.

        Returns:
            Self: SpaceDiscretization3D.
        """
        return cls.from_omega_grid(psi_grid)


def keep_top_layer(space: SpaceDiscretization3D) -> SpaceDiscretization3D:
    """Keep only top layer.

    Args:
        space (SpaceDiscretization3D): Original Space.

    Returns:
        SpaceDiscretization3D: Top Layer.
    """
    return SpaceDiscretization3D.from_coords(
        x_1d=space.omega.xyh.x[0, :, 0],
        y_1d=space.omega.xyh.y[0, 0, :],
        h_1d=space.omega.xyh.h[0, 0, 0].unsqueeze(0),
    )
