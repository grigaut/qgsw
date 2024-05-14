"""Vortex Forcing."""

from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.mesh.mesh import Mesh2D, Mesh3D
from qgsw.models.core import helmholtz
from qgsw.perturbations.base import _Perturbation
from qgsw.spatial.units._units import METERS, Unit
from qgsw.spatial.units.exceptions import UnitError
from qgsw.specs import DEVICE


class RankineVortex2D:
    """2D Rankine Vortex."""

    _norm_factor: int = 100
    _required_xy_unit: Unit = METERS

    def __init__(
        self,
        perturbation_magnitude: float = 1e-3,
    ) -> None:
        """Instantiate Vortex.

        Args:
            mesh (Meshes2D): Spatial Meshes2D.
            perturbation_magnitude (float, optional): Tripolar perturbation
            magnitude. Defaults to 1e-3.
        """
        self._perturbation = perturbation_magnitude

    @property
    def perturbation_magnitude(self) -> float:
        """Tripolar perturbation magnitude."""
        return self._perturbation

    def _raise_if_invalid_unit(self, mesh: Mesh2D) -> None:
        """Check the mesh unit.

        Args:
            mesh (Mesh2D): 2D Mesh.

        Raises:
            UnitError: If the mesh has an invalid unit
        """
        if mesh.xy_unit != self._required_xy_unit:
            msg = (
                "The mesh must have the following xy unit:"
                f" {self._required_xy_unit.name}"
            )
            raise UnitError(msg)

    def compute_vortex_scales(
        self,
        mesh: Mesh2D,
    ) -> tuple[float, float, float]:
        """Compute the vortex radiuses.

        R0 is the radius of the innermost gyre,
        R1 is the radius of the inner part of the surrounding ring,
        R2 is the radius of the outermost part of the ring.

        Args:
            mesh (Mesh3D): 3D Mesh.

        Returns:
            tuple[float, float, float]: R0, R1, R2.
        """
        r0 = 0.1 * mesh.lx
        r1 = r0
        r2 = 0.14 * mesh.lx
        return r0, r1, r2

    def compute_stream_function(self, mesh: Mesh2D) -> torch.Tensor:
        """Compute the value of the streamfunction ψ.

        Returns:
            torch.Tensor: Streamfunction values over the domain,
            (1, 1, nx, ny)-shaped..
        """
        vor = self._compute_vorticity(mesh)
        # Compute Laplacian operator in Fourier Space
        laplacian = helmholtz.compute_laplace_dstI(
            mesh.nx - 1,
            mesh.ny - 1,
            mesh.dx,
            mesh.dy,
            {"device": DEVICE, "dtype": torch.float64},
        )
        # Solve problem in Fourier space : "ψ = ω/∆"
        psi_hat = helmholtz.dstI2D(vor[1:-1, 1:-1]) / laplacian
        # Come back to original space
        psi = F.pad(helmholtz.dstI2D(psi_hat), (1, 1, 1, 1))
        return psi.unsqueeze(0).unsqueeze(0)

    def _compute_vorticity(
        self,
        mesh: Mesh2D,
    ) -> torch.Tensor:
        """Compute the vorticity ω of the vortex.

        Args:
            mesh (Mesh2D): Grid.

        Returns:
            torch.Tensor: Vorticity Value.
        """
        self._raise_if_invalid_unit(mesh)
        x, y = mesh.xy
        r0, r1, r2 = self.compute_vortex_scales(mesh=mesh)
        # Compute cylindrical components
        theta = torch.angle(x + 1j * y)
        r = torch.sqrt(x**2 + y**2)
        r = r * (1 + self._perturbation * torch.cos(theta * 3))
        # Mask vortex's core
        mask_core = torch.sigmoid((r0 - r) / self._norm_factor)
        # Mask vortex's ring
        inner_ring = torch.sigmoid((r - r1) / self._norm_factor)
        outer_ring = torch.sigmoid((r2 - r) / self._norm_factor)
        mask_ring = inner_ring * outer_ring
        # compute vorticity
        return mask_ring / mask_ring.mean() - mask_core / mask_core.mean()


class RankineVortex3D(_Perturbation, metaclass=ABCMeta):
    """3D Rankine Vortex."""

    def __init__(self, magnitude: float = 0.001) -> None:
        """Instantiate 3D Vortex.

        Args:
            magnitude (float, optional): Tripolar perturbation magnitude.
              Defaults to 0.001.
        """
        super().__init__(magnitude)
        self._2d_vortex = RankineVortex2D(perturbation_magnitude=magnitude)

    @property
    def perturbation_magnitude(self) -> float:
        """Tripolar perturbation magnitude."""
        return self._2d_vortex.perturbation_magnitude

    def compute_scale(self, mesh: Mesh3D) -> float:
        """Compute Vortex Scale.

        Args:
            mesh (Mesh3D): 3D Mesh

        Returns:
            float: Inner vortex gyre radius.
        """
        return self._2d_vortex.compute_vortex_scales(mesh.remove_z_h())[0]

    @abstractmethod
    def compute_stream_function(self, mesh: Mesh3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            mesh (Mesh3D): 3D Mesh to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """


class ActiveLayersRankineVortex3D(RankineVortex3D):
    """3D Rankine Vortex with similar vortex behavior accross all layers."""

    def compute_stream_function(self, mesh: Mesh3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            mesh (Mesh3D): 3D Mesh to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        psi_2d = self._2d_vortex.compute_stream_function(mesh.remove_z_h())
        nx, ny = psi_2d.shape[-2:]
        return psi_2d.expand((1, mesh.nl, nx, ny))


class PassiveLayersRankineVortex3D(RankineVortex3D):
    """3D Rankine Vortex with only superior layer active."""

    def compute_stream_function(self, mesh: Mesh3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            mesh (Mesh3D): 3D Mesh to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped.
        """
        psi_2d = self._2d_vortex.compute_stream_function(mesh.remove_z_h())
        nx, ny = psi_2d.shape[-2:]
        psi = torch.ones(
            (1, mesh.nl, nx, ny),
            device=DEVICE,
            dtype=torch.float64,
        )

        psi[0, 0, ...] = psi_2d
        return psi
