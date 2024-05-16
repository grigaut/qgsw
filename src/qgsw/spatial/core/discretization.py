"""Mesh defining objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import Self

from qgsw.spatial.core.mesh import Mesh2D, Mesh3D
from qgsw.spatial.units._units import METERS
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.mesh import MeshConfig
    from qgsw.configs.models import ModelConfig
    from qgsw.spatial.units._units import Unit


class MeshesInstanciationError(Exception):
    """Error raised when instantiating meshes."""


class SpaceDiscretization2D:
    """SpaceDiscretization2D Object."""

    def __init__(
        self,
        *,
        omega_mesh: Mesh2D,
        h_mesh: Mesh2D,
        u_mesh: Mesh2D,
        v_mesh: Mesh2D,
    ) -> None:
        """Instantiate the SpaceDiscretization2D."""
        self._verify_xy_units(
            omega_xy_unit=omega_mesh.xy_unit,
            h_xy_unit=h_mesh.xy_unit,
            u_xy_unit=u_mesh.xy_unit,
            v_xy_unit=v_mesh.xy_unit,
        )
        self._omega = omega_mesh
        self._h = h_mesh
        self._u = u_mesh
        self._v = v_mesh

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.omega.nx

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.omega.ny

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self.omega.lx

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self.omega.ly

    @property
    def dx(self) -> float:
        """Dx."""
        return self.omega.dx

    @property
    def dy(self) -> float:
        """Dy."""
        return self.omega.dy

    @property
    def xy_unit(self) -> Unit:
        """X and Y unit."""
        return self._omega.xy_unit

    @property
    def omega(self) -> Mesh2D:
        """Omega Mesh.

        See https://agupubs.oFinelibrary.wiley.com/doi/epdf/10.1029/2021MS002663#JAME21507.indd%3Ahl_jame21507-fig-0001%3A73
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

    def _verify_xy_units(
        self,
        omega_xy_unit: Mesh2D,
        h_xy_unit: Mesh2D,
        u_xy_unit: Mesh2D,
        v_xy_unit: Mesh2D,
    ) -> None:
        """Verify meshes xy units equality.

        Args:
            omega_xy_unit (Mesh2D): Omega xy unit.
            h_xy_unit (Mesh2D): h xy unit.
            u_xy_unit (Mesh2D): u xy unit.
            v_xy_unit (Mesh2D): v xy unit.

        Raises:
            MeshesInstanciationError: If the unit don't match.
        """
        omega_h = omega_xy_unit == h_xy_unit
        h_u = h_xy_unit == u_xy_unit
        u_v = u_xy_unit == v_xy_unit

        if omega_h and h_u and u_v:
            return
        msg = "All meshes xy units must correspond."
        raise MeshesInstanciationError(msg)

    def add_h(self, h: torch.Tensor) -> SpaceDiscretization3D:
        """Switch to 3D Meshes adding layers thickness.

        Args:
            h (torch.Tensor): Layers thickness.

        Returns:
            SpaceDiscretization3D: 3D Meshes.
        """
        omega_3d = self.omega.add_h(h=h)
        h_3d = self._h.add_h(h=h)
        u_3d = self._u.add_h(h=h)
        v_3d = self._v.add_h(h=h)
        return SpaceDiscretization3D(
            omega_mesh=omega_3d,
            h_mesh=h_3d,
            u_mesh=u_3d,
            v_mesh=v_3d,
        )

    def add_z(self, z: torch.Tensor) -> SpaceDiscretization3D:
        """Switch to 3D Mesh adding z coordinates.

        Args:
            z (torch.Tensor): Z coordinates.

        Returns:
            SpaceDiscretization3D: 3D Mesh.
        """
        omega_3d = self.omega.add_z(z=z)
        h_3d = self._h.add_z(z=z)
        u_3d = self._u.add_z(z=z)
        v_3d = self._v.add_z(z=z)
        return SpaceDiscretization3D(
            omega_mesh=omega_3d,
            h_mesh=h_3d,
            u_mesh=u_3d,
            v_mesh=v_3d,
        )

    @classmethod
    def from_config(cls, mesh_config: MeshConfig) -> Self:
        """Construct the SpaceDiscretization2D given a MeshConfig object.

        Args:
            mesh_config (MeshConfig): Mesh Configuration Object.

        Returns:
            Self: Corresponding SpaceDiscretization2D.
        """
        x = torch.linspace(
            mesh_config.box.x_min,
            mesh_config.box.x_max,
            mesh_config.nx + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        y = torch.linspace(
            mesh_config.box.y_min,
            mesh_config.box.y_max,
            mesh_config.ny + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        return cls.from_tensors(
            x=x,
            y=y,
            x_unit=mesh_config.box.unit,
            y_unit=mesh_config.box.unit,
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
        """Generate ω, h, u, v meshes from coordinates tensors.

        Args:
            x (torch.Tensor): X Coordinates.
            y (torch.Tensor): Y Coordinates.
            x_unit (Unit): X unit.
            y_unit (Unit): Y unit.

        Returns:
            Self: 2D Meshes.
        """
        x_centers = 0.5 * (x[1:] + x[:-1])
        y_centers = 0.5 * (y[1:] + y[:-1])

        omega_mesh = Mesh2D.from_tensors(
            x=x,
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        h_mesh = Mesh2D.from_tensors(
            x=x_centers,
            y=y_centers,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        u_mesh = Mesh2D.from_tensors(
            x=x,
            y=y_centers,
            x_unit=x_unit,
            y_unit=y_unit,
        )
        v_mesh = Mesh2D.from_tensors(
            x=x_centers,
            y=y,
            x_unit=x_unit,
            y_unit=y_unit,
        )

        return cls(
            omega_mesh=omega_mesh,
            h_mesh=h_mesh,
            u_mesh=u_mesh,
            v_mesh=v_mesh,
        )


class SpaceDiscretization3D:
    """3D Grid."""

    def __init__(
        self,
        *,
        omega_mesh: Mesh3D,
        h_mesh: Mesh3D,
        u_mesh: Mesh3D,
        v_mesh: Mesh3D,
    ) -> None:
        """Instantiate the SpaceDiscretization3D.

        Args:
            omega_mesh (Mesh3D): Omega mesh.
            h_mesh (Mesh3D): h mesh.
            u_mesh (Mesh3D): u mesh.
            v_mesh (Mesh3D): v mesh.
        """
        self._verify_xy_units(
            omega_xy_unit=omega_mesh.xy_unit,
            h_xy_unit=h_mesh.xy_unit,
            u_xy_unit=u_mesh.xy_unit,
            v_xy_unit=v_mesh.xy_unit,
        )
        self._verify_zh_units(
            omega_zh_unit=omega_mesh.zh_unit,
            h_zh_unit=h_mesh.zh_unit,
            u_zh_unit=u_mesh.zh_unit,
            v_zh_unit=v_mesh.zh_unit,
        )
        self._omega = omega_mesh
        self._h = h_mesh
        self._u = u_mesh
        self._v = v_mesh

    @property
    def nx(self) -> int:
        """Number of points on the x direction."""
        return self.omega.nx

    @property
    def ny(self) -> int:
        """Number of points on the y direction."""
        return self.omega.ny

    @property
    def nl(self) -> int:
        """Number of layers."""
        return self.omega.nl

    @property
    def lx(self) -> int:
        """Total length in the x direction (in meters)."""
        return self.omega.lx

    @property
    def ly(self) -> int:
        """Total length in the y direction (in meters)."""
        return self.omega.ly

    @property
    def lz(self) -> int:
        """Total length in the z direction (in meters)."""
        return self.omega.lz

    @property
    def dx(self) -> float:
        """dx."""  # noqa: D403
        return self.omega.dx

    @property
    def dy(self) -> float:
        """dy."""  # noqa: D403
        return self.omega.dy

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

    def _verify_xy_units(
        self,
        omega_xy_unit: Mesh2D,
        h_xy_unit: Mesh2D,
        u_xy_unit: Mesh2D,
        v_xy_unit: Mesh2D,
    ) -> None:
        """Verify meshes xy units equality.

        Args:
            omega_xy_unit (Mesh2D): Omega xy unit.
            h_xy_unit (Mesh2D): h xy unit.
            u_xy_unit (Mesh2D): u xy unit.
            v_xy_unit (Mesh2D): v xy unit.

        Raises:
            MeshesInstanciationError: If the unit don't match.
        """
        omega_h = omega_xy_unit == h_xy_unit
        h_u = h_xy_unit == u_xy_unit
        u_v = u_xy_unit == v_xy_unit

        if omega_h and h_u and u_v:
            return
        msg = "All meshes xy units must correspond."
        raise MeshesInstanciationError(msg)

    def _verify_zh_units(
        self,
        omega_zh_unit: Mesh2D,
        h_zh_unit: Mesh2D,
        u_zh_unit: Mesh2D,
        v_zh_unit: Mesh2D,
    ) -> None:
        """Verify meshes zh units equality.

        Args:
            omega_zh_unit (Mesh2D): Omega zh unit.
            h_zh_unit (Mesh2D): h zh unit.
            u_zh_unit (Mesh2D): u zh unit.
            v_zh_unit (Mesh2D): v zh unit.

        Raises:
            MeshesInstanciationError: If the unit don't match.
        """
        omega_h = omega_zh_unit == h_zh_unit
        h_u = h_zh_unit == u_zh_unit
        u_v = u_zh_unit == v_zh_unit

        if omega_h and h_u and u_v:
            return
        msg = "All meshes zh units must correspond."
        raise MeshesInstanciationError(msg)

    def remove_z_h(self) -> SpaceDiscretization2D:
        """Remove z coordinates.

        Returns:
            SpaceDiscretization2D: 2D Mesh for only X and Y.
        """
        return SpaceDiscretization2D(
            omega_mesh=self._omega.remove_z_h(),
            h_mesh=self._h.remove_z_h(),
            u_mesh=self._u.remove_z_h(),
            v_mesh=self._v.remove_z_h(),
        )

    @classmethod
    def from_config(
        cls,
        mesh_config: MeshConfig,
        model_config: ModelConfig,
    ) -> Self:
        """Construct the 3D Grid given a MeshConfig object.

        Args:
            mesh_config (MeshConfig): Mesh Configuration Object.
            model_config (ModelConfig): Model Configuration Object.

        Returns:
            Self: Corresponding 3D Grid.
        """
        x = torch.linspace(
            mesh_config.box.x_min,
            mesh_config.box.x_max,
            mesh_config.nx + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        y = torch.linspace(
            mesh_config.box.y_min,
            mesh_config.box.y_max,
            mesh_config.ny + 1,
            dtype=torch.float64,
            device=DEVICE,
        )
        return cls.from_tensors(
            x=x,
            y=y,
            h=model_config.h,
            x_unit=mesh_config.box.unit,
            y_unit=mesh_config.box.unit,
            zh_unit=METERS,
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
        """Generate ω, h, u, v meshes from coordinates tensors.

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
            Self: 3D Meshes.
        """
        x_centers = 0.5 * (x[1:] + x[:-1])
        y_centers = 0.5 * (y[1:] + y[:-1])

        omega_mesh = Mesh3D.from_tensors(
            x=x,
            y=y,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )
        h_mesh = Mesh3D.from_tensors(
            x=x_centers,
            y=y_centers,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )
        u_mesh = Mesh3D.from_tensors(
            x=x,
            y=y_centers,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )
        v_mesh = Mesh3D.from_tensors(
            x=x_centers,
            y=y,
            z=z,
            h=h,
            x_unit=x_unit,
            y_unit=y_unit,
            zh_unit=zh_unit,
        )

        return cls(
            omega_mesh=omega_mesh,
            h_mesh=h_mesh,
            u_mesh=u_mesh,
            v_mesh=v_mesh,
        )
