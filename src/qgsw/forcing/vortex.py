"""Vortex Forcing."""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw import helmholtz
from qgsw.grid import Grid, Grid3D
from qgsw.specs import DEVICE


def grad_perp(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Orthogonal gradient."""
    return (f[..., :-1] - f[..., 1:]) / dy, (
        f[..., 1:, :] - f[..., :-1, :]
    ) / dx


class RankineVortex2D:
    """2D Rankine Vortex."""

    _norm_factor: int = 100

    def __init__(
        self,
        grid: Grid,
        perturbation_magnitude: float = 1e-3,
    ) -> None:
        """Instantiate Vortex.

        Args:
            grid (Grid): Spatial Grid.
            perturbation_magnitude (float, optional): Tripolar perturbation
            magnitude. Defaults to 1e-3.
        """
        self._grid = grid
        self._perturbation = perturbation_magnitude
        self._compute_psi()

    @property
    def perturbation_magnitude(self) -> float:
        """Tripolar perturbation magnitude."""
        return self._perturbation

    @property
    def r0(self) -> float:
        """Core cylinder radius: 0.1*lx."""
        return 0.1 * self._grid.lx

    @property
    def r1(self) -> float:
        """Inner radius of the surrounding ring: r0."""
        return self.r0

    @property
    def r2(self) -> float:
        """Outer radius of the surrounding ring: 0.14*lx."""
        return 0.14 * self._grid.lx

    @property
    def psi(self) -> torch.Tensor:
        """Value of the stream function ψ.

        The Tensor has a shape of (1, 1, nx + 1, ny + 1).
        """
        return self._psi

    def _compute_psi(self) -> torch.Tensor:
        """Compute the value of the streamfunction ψ.

        Returns:
            torch.Tensor: Streamfunction values over the
        (nx + 1, ny + 1) domain.
        """
        vor = self._compute_vorticity(self._grid.omega_xy)
        # Compute Laplacian operator in Fourier Space
        laplacian = helmholtz.compute_laplace_dstI(
            self._grid.nx,
            self._grid.ny,
            self._grid.dx,
            self._grid.dy,
            {"device": DEVICE, "dtype": torch.float64},
        )
        # Solve problem in Fourier space : "ψ = ω/∆"
        psi_hat = helmholtz.dstI2D(vor[1:-1, 1:-1]) / laplacian
        # Come back to original space
        psi = F.pad(helmholtz.dstI2D(psi_hat), (1, 1, 1, 1))
        self._psi = psi.unsqueeze(0).unsqueeze(0)

    def _compute_vorticity(
        self, grid_xy: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the vorticity ω of the vortex.

        Args:
            grid_xy (tuple[torch.Tensor, torch.Tensor]): Grid.

        Returns:
            torch.Tensor: Vorticity Value.
        """
        x, y = grid_xy
        # Compute cylindrical components
        theta = torch.angle(x + 1j * y)
        r = torch.sqrt(x**2 + y**2)
        r = r * (1 + self._perturbation * torch.cos(theta * 3))
        # Mask vortex's core
        mask_core = torch.sigmoid((self.r0 - r) / self._norm_factor)
        # Mask vortex's ring
        inner_ring = torch.sigmoid((r - self.r1) / self._norm_factor)
        outer_ring = torch.sigmoid((self.r2 - r) / self._norm_factor)
        mask_ring = inner_ring * outer_ring
        # compute vorticity
        return mask_ring / mask_ring.mean() - mask_core / mask_core.mean()


class RankineVortex3D(ABC):
    """3D Rankine Vortex."""

    def __init__(
        self,
        grid: Grid3D,
        perturbation_magnitude: float = 1e-3,
    ) -> None:
        """Instantiate 3D Rankine Vortex.

        Args:
            grid (Grid3D): 3D Grid.
            perturbation_magnitude (float, optional): Tripolar perturbation
            magnitude. Defaults to 1e-3.
        """
        self._grid = grid
        self._2d = RankineVortex2D(
            grid=grid.xy,
            perturbation_magnitude=perturbation_magnitude,
        )

    @property
    def perturbation_magnitude(self) -> float:
        """Tripolar perturbation magnitude."""
        return self._2d.perturbation_magnitude

    @property
    def r0(self) -> float:
        """Core cylinder radius: 0.1*lx."""
        return self._2d.r0

    @property
    def r1(self) -> float:
        """Inner radius of the surrounding ring: r0."""
        return self._2d.r1

    @property
    def r2(self) -> float:
        """Outer radius of the surrounding ring: 0.14*lx."""
        return self._2d.r2

    @property
    @abstractmethod
    def psi(self) -> torch.Tensor:
        """Value of the stream function ψ.

        The Tensor has a shape of (1, nl, nx + 1, ny + 1).
        """
        xy_shape = self._2d.psi.shape[-2:]
        return self._2d.psi.expand((1, self._grid.nl, *xy_shape))


class ActiveLayersRankineVortex3D(RankineVortex3D):
    """3D Rankine Vortex with similar vortex behavior accross all layers."""

    @property
    def psi(self) -> torch.Tensor:
        """Value of the stream function ψ.

        The Tensor has a shape of (1, nl, nx + 1, ny + 1).
        """
        xy_shape = self._2d.psi.shape[-2:]
        return self._2d.psi.expand((1, self._grid.nl, *xy_shape))


class PassiveLayersRankineVortex3D(RankineVortex3D):
    """3D Rankine Vortex with only superior layer active."""

    @property
    def psi(self) -> torch.Tensor:
        """Value of the stream function ψ.

        The Tensor has a shape of (1, nl, nx + 1, ny + 1).
        """
        xy_shape = self._2d.psi.shape[-2:]
        psi = torch.ones(
            (1, self._grid.nl, *xy_shape),
            device=DEVICE,
            dtype=torch.float64,
        )
        psi[0, 0, ...] = self._2d.psi
        return psi


class VortexForcing(ABC):
    """Vortex Forcing's abtract class."""

    _vortex: RankineVortex3D

    def __init__(
        self,
        grid_3d: Grid3D,
        perturbation_magnitude: float = 1e-3,
    ) -> None:
        """Instantiate VortexForcing.

        Args:
            grid_3d (Grid3D): 3D Grid.
            perturbation_magnitude (float, optional): Tripolar perturbation
            magnitude. Defaults to 1e-3.
        """
        self._grid = grid_3d
        self._set_vortex(perturbation_magnitude=perturbation_magnitude)

    @abstractmethod
    def _set_vortex(self, perturbation_magnitude: float) -> None:
        """Set Vortex.

        Args:
            perturbation_magnitude (float): Vortex's tripolar
            perturbation magnitude.
        """

    def compute(self, f0: float, Ro: float) -> torch.Tensor:  # noqa: N803
        """Compute Initial pressure based on the vortex's vorticity.

        Args:
            f0 (float): Coriolis Parameter from β-plane approximation.
            Ro (float): Rossby NUmber.

        Returns:
            torch.Tensor: Initial Pressure with the presence of the vortex.
        """
        u, v = grad_perp(self._vortex.psi, self._grid.dx, self._grid.dy)
        u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
        # set psi amplitude to have a correct Rossby number
        psi_norm = self._vortex.psi * (Ro * f0 * self._vortex.r0 / u_norm_max)
        # Return pressure value
        return psi_norm * f0


class ActiveLayersVortexForcing(VortexForcing):
    """Vortex Forcing with similar vortex behavior accross all layers."""

    def _set_vortex(self, perturbation_magnitude: float) -> None:
        self._vortex = ActiveLayersRankineVortex3D(
            grid=self._grid, perturbation_magnitude=perturbation_magnitude
        )


class PassiveLayersVortexForcing(VortexForcing):
    """Vortex Forcing with only superior layer active."""

    def _set_vortex(self, perturbation_magnitude: float) -> None:
        self._vortex = PassiveLayersRankineVortex3D(
            grid=self._grid, perturbation_magnitude=perturbation_magnitude
        )
