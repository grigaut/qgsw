"""Mesh defining objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from qgsw.mesh.mesh import Mesh2D, Mesh3D
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.core import ScriptConfig


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

    def add_h(self, h: torch.Tensor) -> Meshes3D:
        """Switch to 3D Meshes adding layers thickness.

        Args:
            h (torch.Tensor): Layers thickness.

        Returns:
            Meshes3D: 3D Meshes.
        """
        omega_3d = self.omega.add_h(h=h)
        h_3d = self._h.add_h(h=h)
        u_3d = self._u.add_h(h=h)
        v_3d = self._v.add_h(h=h)
        return Meshes3D(
            omega_mesh=omega_3d,
            h_mesh=h_3d,
            u_mesh=u_3d,
            v_mesh=v_3d,
        )

    def add_z(self, z: torch.Tensor) -> Meshes3D:
        """Switch to 3D Mesh adding z coordinates.

        Args:
            z (torch.Tensor): Z coordinates.

        Returns:
            Meshes3D: 3D Mesh.
        """
        omega_3d = self.omega.add_z(z=z)
        h_3d = self._h.add_z(z=z)
        u_3d = self._u.add_z(z=z)
        v_3d = self._v.add_z(z=z)
        return Meshes3D(
            omega_mesh=omega_3d,
            h_mesh=h_3d,
            u_mesh=u_3d,
            v_mesh=v_3d,
        )

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

    def remove_z_h(self) -> Meshes2D:
        """Remove z coordinates.

        Returns:
            Meshes2D: 2D Mesh for only X and Y.
        """
        return Meshes2D(
            omega_mesh=self._omega.remove_z_h(),
            h_mesh=self._h.remove_z_h(),
            u_mesh=self._u.remove_z_h(),
            v_mesh=self._v.remove_z_h(),
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
