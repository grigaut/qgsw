"""Vortex Forcing."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Self

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.models.core import helmholtz
from qgsw.perturbations.base import (
    BaroclinicPerturbation,
    BarotropicPerturbation,
    HalfBarotropicPerturbation,
    _Perturbation,
)
from qgsw.spatial.core.grid import Grid2D, Grid3D
from qgsw.spatial.units._units import METERS, Unit
from qgsw.spatial.units.exceptions import UnitError
from qgsw.specs import DEVICE

if TYPE_CHECKING:
    from qgsw.configs.perturbation import PerturbationConfig
    from qgsw.spatial.core.grid import Grid2D, Grid3D


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

    def _raise_if_invalid_unit(self, grid_2d: Grid2D) -> None:
        """Check the 2D grid unit.

        Args:
            grid_2d (Grid2D): 2D Grid.

        Raises:
            UnitError: If the grid has an invalid unit
        """
        if grid_2d.xy_unit != self._required_xy_unit:
            msg = (
                "The grid_2d must have the following xy unit:"
                f" {self._required_xy_unit.name}"
            )
            raise UnitError(msg)

    def compute_vortex_scales(
        self,
        grid_2d: Grid2D,
    ) -> tuple[float, float, float]:
        """Compute the vortex radiuses.

        R0 is the radius of the innermost gyre,
        R1 is the radius of the inner part of the surrounding ring,
        R2 is the radius of the outermost part of the ring.

        Args:
            grid_2d (Grid3D): 3D Grid.

        Returns:
            tuple[float, float, float]: R0, R1, R2.
        """
        r0 = 0.1 * grid_2d.lx
        r1 = r0
        r2 = 0.14 * grid_2d.lx
        return r0, r1, r2

    def compute_stream_function(self, grid_2d: Grid2D) -> torch.Tensor:
        """Compute the value of the streamfunction ψ.

        Returns:
            torch.Tensor: Streamfunction values over the domain,
            (1, 1, nx, ny)-shaped..
        """
        vor = self._compute_vorticity(grid_2d)
        # Compute Laplacian operator in Fourier Space
        laplacian = helmholtz.compute_laplace_dstI(
            grid_2d.nx - 1,
            grid_2d.ny - 1,
            grid_2d.dx,
            grid_2d.dy,
            device=DEVICE.get(),
            dtype=torch.float64,
        )
        # Solve problem in Fourier space : "ψ = ω/∆"
        psi_hat = helmholtz.dstI2D(vor[1:-1, 1:-1]) / laplacian
        # Come back to original space
        psi = F.pad(helmholtz.dstI2D(psi_hat), (1, 1, 1, 1))
        return psi.unsqueeze(0).unsqueeze(0)

    def _compute_vorticity(
        self,
        grid_2d: Grid2D,
    ) -> torch.Tensor:
        """Compute the vorticity ω of the vortex.

        Args:
            grid_2d (Grid2D): Grid.

        Returns:
            torch.Tensor: Vorticity Value.
        """
        self._raise_if_invalid_unit(grid_2d)
        x, y = grid_2d.xy
        r0, r1, r2 = self.compute_vortex_scales(grid_2d=grid_2d)
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


class RankineVortex3D(_Perturbation, metaclass=ABCMeta):
    """3D Rankine Vortex."""

    _2d_vortex: RankineVortex2D

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

    def _compute_streamfunction_2d(self, grid_2d: Grid2D) -> torch.Tensor:
        """Compute the streamfunction for a single layer.

        Args:
            grid_2d (Grid2D): Grid to use for stream function computation.

        Returns:
            torch.Tensor: Stream function values.
        """
        return self._2d_vortex.compute_stream_function(grid_2d)

    @property
    def perturbation_magnitude(self) -> float:
        """Perturbation magnitude."""
        return self._2d_vortex.perturbation_magnitude

    def compute_scale(self, grid_3d: Grid3D) -> float:
        """Compute Vortex Scale.

        Args:
            grid_3d (Grid3D): 3D Grid

        Returns:
            float: Inner vortex gyre radius.
        """
        return self._2d_vortex.compute_vortex_scales(grid_3d.remove_z_h())[0]

    @classmethod
    def from_config(cls, perturbation_config: PerturbationConfig) -> Self:
        """Instantiate Perturbation from config.

        Args:
            perturbation_config (PerturbationConfig): Perturbation Config.

        Returns:
            Self: Perturbation.
        """
        return cls(
            magnitude=perturbation_config.perturbation_magnitude,
        )


class BarotropicVortex(RankineVortex3D, BarotropicPerturbation):
    """3D Rankine Vortex with similar vortex behavior accross all layers."""

    _type = "vortex-barotropic"

    def _set_2d_vortex(self, magnitude: float) -> None:
        """Set the 2D  vortex.

        Args:
            magnitude (float): Vortex perturbation magnitude
        """
        self._2d_vortex = RankineVortex2D(perturbation_magnitude=magnitude)


class HalfBarotropicVortex(RankineVortex3D, HalfBarotropicPerturbation):
    """3D Rankine Vortex with similar vortex behavior accross all layers."""

    _type = "vortex-half-barotropic"

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
