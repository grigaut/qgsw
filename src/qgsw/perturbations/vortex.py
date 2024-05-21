"""Vortex Forcing."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import verbose
from qgsw.models.core import helmholtz
from qgsw.perturbations.base import (
    BaroclinicPerturbation,
    BarotropicPerturbation,
    _Perturbation,
)
from qgsw.spatial.core.mesh import Mesh2D, Mesh3D
from qgsw.spatial.units._units import METERS, Unit
from qgsw.spatial.units.exceptions import UnitError
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.spatial.core.mesh import Mesh2D, Mesh3D


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
            perturbation_magnitude (float, optional): Tripolar perturbation
            magnitude. Defaults to 1e-3.
        """
        self._perturbation = perturbation_magnitude

    @property
    def perturbation_magnitude(self) -> float:
        """Tripolar perturbation magnitude."""
        return self._perturbation

    def _raise_if_invalid_unit(self, mesh_2d: Mesh2D) -> None:
        """Check the 2D mesh unit.

        Args:
            mesh_2d (Mesh2D): 2D Mesh.

        Raises:
            UnitError: If the mesh has an invalid unit
        """
        if mesh_2d.xy_unit != self._required_xy_unit:
            msg = (
                "The mesh_2d must have the following xy unit:"
                f" {self._required_xy_unit.name}"
            )
            raise UnitError(msg)

    def compute_vortex_scales(
        self,
        mesh_2d: Mesh2D,
    ) -> tuple[float, float, float]:
        """Compute the vortex radiuses.

        R0 is the radius of the innermost gyre,
        R1 is the radius of the inner part of the surrounding ring,
        R2 is the radius of the outermost part of the ring.

        Args:
            mesh_2d (Mesh3D): 3D Mesh.

        Returns:
            tuple[float, float, float]: R0, R1, R2.
        """
        r0 = 0.1 * mesh_2d.lx
        r1 = r0
        r2 = 0.14 * mesh_2d.lx
        return r0, r1, r2

    def compute_stream_function(self, mesh_2d: Mesh2D) -> torch.Tensor:
        """Compute the value of the streamfunction ψ.

        Returns:
            torch.Tensor: Streamfunction values over the domain,
            (1, 1, nx, ny)-shaped..
        """
        vor = self._compute_vorticity(mesh_2d)
        # Compute Laplacian operator in Fourier Space
        laplacian = helmholtz.compute_laplace_dstI(
            mesh_2d.nx - 1,
            mesh_2d.ny - 1,
            mesh_2d.dx,
            mesh_2d.dy,
            {"device": DEVICE, "dtype": torch.float64},
        )
        # Solve problem in Fourier space : "ψ = ω/∆"
        psi_hat = helmholtz.dstI2D(vor[1:-1, 1:-1]) / laplacian
        # Come back to original space
        psi = F.pad(helmholtz.dstI2D(psi_hat), (1, 1, 1, 1))
        return psi.unsqueeze(0).unsqueeze(0)

    def _compute_vorticity(
        self,
        mesh_2d: Mesh2D,
    ) -> torch.Tensor:
        """Compute the vorticity ω of the vortex.

        Args:
            mesh_2d (Mesh2D): Grid.

        Returns:
            torch.Tensor: Vorticity Value.
        """
        self._raise_if_invalid_unit(mesh_2d)
        x, y = mesh_2d.xy
        r0, r1, r2 = self.compute_vortex_scales(mesh_2d=mesh_2d)
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
        vortex = mask_ring / mask_ring.mean() - mask_core / mask_core.mean()
        return vortex / vortex.abs().max()


class PerturbedVortex2D(RankineVortex2D):
    """2D vortex with random perturbations."""

    _kernel_size_ratio: int = 10
    _sigma: float = 3
    _threshold: float = 1e-6

    def set_random_seed(self, seed: int) -> None:
        """Set the pytorch random seed.

        Args:
            seed (int): Seed to set.
        """
        torch.random.manual_seed(seed=seed)

    def _generate_random_field(
        self,
        nx: int,
        ny: int,
    ) -> torch.Tensor:
        """Generate a random field in the center of the area.

        Args:
            nx (int): Number of cells in the x direction
            ny (int): Number of cells in the y direction

        Returns:
            torch.Tensor: Random field, (nx, ny) shaped.
        """
        # Select a sub area of the entire grid
        random_field = torch.rand(
            (nx, ny),
            dtype=torch.float64,
            device=DEVICE,
        )  # shape (nx_rand_area, ny_rand_area)
        return 2 * random_field - 1

    def _generate_gaussian_kernel(
        self,
        size: int,
    ) -> torch.Tensor:
        """Generate the gaussian kernel.

        Args:
            size (int): 'Radius' of the kernel.

        Returns:
            torch.Tensor: (2*size, 2*size) gaussian kernel.
        """
        mean = (size - 1) / 2.0
        variance = self._sigma**2.0

        x_cord = torch.arange(size, dtype=torch.float64, device=DEVICE)
        x_grid = x_cord.repeat(size).view(size, size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1.0 / (2.0 * torch.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance),
        )
        # Make sure sum of values in gaussian kernel equals 1.
        return gaussian_kernel / torch.sum(gaussian_kernel)

    def _compute_perturbation(self, mesh_2d: Mesh2D) -> torch.Tensor:
        """Compute the perturbation over the entire domain.

        Args:
            mesh_2d (Mesh2D): Mesh to generate the perturbation on.

        Returns:
            torch.Tensor: Perturbation values (nx, ny) shaped.
        """
        kernel_size = int(mesh_2d.nx / self._kernel_size_ratio)
        size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        random_field = self._generate_random_field(
            nx=mesh_2d.nx,  # + (size - 1),
            ny=mesh_2d.ny,  # + (size - 1),
        )
        verbose.display(
            f"Random perturbation: Gaussian kernel size: {size}",
            trigger_level=2,
        )
        kernel = self._generate_gaussian_kernel(size=size)
        filtered_field = F.conv2d(
            random_field.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding="same",
        ).squeeze()  # shape: (nx,ny)
        return filtered_field / filtered_field.abs().max()

    def _generate_vortex(self, mesh_2d: Mesh2D) -> torch.Tensor:
        """Generate the vortex.

        Args:
            mesh_2d (Mesh2D): Mesh to generate vortex on.

        Returns:
            torch.Tensor: Vorticity values, between -1 and 1,
            (nx, ny) shaped.
        """
        self._raise_if_invalid_unit(mesh_2d)
        x, y = mesh_2d.xy
        r0, r1, r2 = self.compute_vortex_scales(mesh_2d=mesh_2d)
        # Compute cylindrical components
        r = torch.sqrt(x**2 + y**2)
        # Mask vortex's core
        mask_core = torch.sigmoid((r0 - r) / self._norm_factor)
        # Mask vortex's ring
        inner_ring = torch.sigmoid((r - r1) / self._norm_factor)
        outer_ring = torch.sigmoid((r2 - r) / self._norm_factor)

        mask_ring = inner_ring * outer_ring
        vortex = mask_ring / mask_ring.mean() - mask_core / mask_core.mean()
        return vortex / vortex.abs().max()

    def _compute_vorticity(self, mesh_2d: Mesh2D) -> torch.Tensor:
        """Compute the vorticity ω of the vortex.

        Args:
            mesh_2d (Mesh2D): Grid.

        Returns:
            torch.Tensor: Vorticity Value, (nx,ny) shaped.
        """
        vortex = self._generate_vortex(mesh_2d=mesh_2d)
        vortex_norm = vortex / vortex.abs().max()
        perturbation = self._compute_perturbation(mesh_2d)
        is_vortex = vortex_norm.abs() > self._threshold
        masked_perturbation = perturbation.where(
            is_vortex,
            torch.zeros_like(perturbation),
        )
        return vortex + self.perturbation_magnitude * masked_perturbation


class RankineVortex3D(_Perturbation, metaclass=ABCMeta):
    """3D Rankine Vortex."""

    _2d_vortex: RankineVortex2D | PerturbedVortex2D

    def __init__(self, magnitude: float = 0.001) -> None:
        """Instantiate 3D Vortex.

        Args:
            magnitude (float, optional): Tripolar perturbation magnitude.
              Defaults to 0.001.
        """
        super().__init__(magnitude)
        self._set_2d_vortex(magnitude)

    @abstractmethod
    def _set_2d_vortex(self, magnitude: float) -> None:
        """Set the 2D  vortex.

        Args:
            magnitude (float): Vortex perturbation magnitude
        """

    def _compute_streamfunction_2d(self, mesh_2d: Mesh2D) -> torch.Tensor:
        """Compute the streamfunction for a single layer.

        Args:
            mesh_2d (Mesh2D): Mesh to use for stream function computation.

        Returns:
            torch.Tensor: Stream function values.
        """
        return self._2d_vortex.compute_stream_function(mesh_2d)

    @property
    def perturbation_magnitude(self) -> float:
        """Perturbation magnitude."""
        return self._2d_vortex.perturbation_magnitude

    def compute_scale(self, mesh_3d: Mesh3D) -> float:
        """Compute Vortex Scale.

        Args:
            mesh_3d (Mesh3D): 3D Mesh

        Returns:
            float: Inner vortex gyre radius.
        """
        return self._2d_vortex.compute_vortex_scales(mesh_3d.remove_z_h())[0]


class BarotropicVortex(RankineVortex3D, BarotropicPerturbation):
    """3D Rankine Vortex with similar vortex behavior accross all layers."""

    _type = "vortex-barotropic"

    def _set_2d_vortex(self, magnitude: float) -> None:
        """Set the 2D  vortex.

        Args:
            magnitude (float): Vortex perturbation magnitude
        """
        self._2d_vortex = RankineVortex2D(perturbation_magnitude=magnitude)


class BaroclinicVortex(RankineVortex3D, BaroclinicPerturbation):
    """3D Rankine Vortex with only superior layer active."""

    _type = "vortex-baroclinic"

    def _set_2d_vortex(self, magnitude: float) -> None:
        """Set the 2D  vortex.

        Args:
            magnitude (float): Vortex perturbation magnitude
        """
        self._2d_vortex = RankineVortex2D(perturbation_magnitude=magnitude)


class PerturbedBarotropicVortex(RankineVortex3D, BarotropicPerturbation):
    """Perturbed Barotropic vortex."""

    _type = "vortex-barotropic-perturbed"

    def _set_2d_vortex(self, magnitude: float) -> None:
        """Set the 2D  vortex.

        Args:
            magnitude (float): Vortex perturbation magnitude
        """
        self._2d_vortex = PerturbedVortex2D(perturbation_magnitude=magnitude)

    def compute_stream_function(self, mesh_3d: Mesh3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            mesh_3d (Mesh3D): 3D Mesh to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        self._2d_vortex.set_random_seed(0)
        return super().compute_stream_function(mesh_3d)


class PerturbedBaroclinicVortex(RankineVortex3D, BaroclinicPerturbation):
    """Perturbed Baroclinic vortex."""

    _type = "vortex-baroclinic-perturbed"

    def _set_2d_vortex(self, magnitude: float) -> None:
        """Set the 2D  vortex.

        Args:
            magnitude (float): Vortex perturbation magnitude
        """
        self._2d_vortex = PerturbedVortex2D(perturbation_magnitude=magnitude)

    def compute_stream_function(self, mesh_3d: Mesh3D) -> torch.Tensor:
        """Value of the stream function ψ.

        Args:
            mesh_3d (Mesh3D): 3D Mesh to generate Stream Function on.

        Returns:
            torch.Tensor: Stream function values, (1, nl, nx, ny)-shaped..
        """
        self._2d_vortex.set_random_seed(0)
        return super().compute_stream_function(mesh_3d)
