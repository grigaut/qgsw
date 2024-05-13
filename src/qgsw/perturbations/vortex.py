"""Vortex Forcing."""

from abc import ABCMeta, abstractmethod

import torch
import torch.nn.functional as F  # noqa: N812
from typing_extensions import Self

from qgsw.configs.mesh import MeshConfig
from qgsw.configs.models import ModelConfig
from qgsw.configs.perturbation import PerturbationConfig
from qgsw.mesh.meshes import Meshes2D, Meshes3D
from qgsw.models.core import helmholtz
from qgsw.spatial.units._units import METERS, Unit
from qgsw.spatial.units.exceptions import UnitError
from qgsw.specs import DEVICE


def grad_perp(f: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Orthogonal gradient."""
    return (f[..., :-1] - f[..., 1:]) / dy, (
        f[..., 1:, :] - f[..., :-1, :]
    ) / dx


class RankineVortex2D:
    """2D Rankine Vortex."""

    _norm_factor: int = 100
    _required_xy_unit: Unit = METERS

    def __init__(
        self,
        mesh: Meshes2D,
        perturbation_magnitude: float = 1e-3,
    ) -> None:
        """Instantiate Vortex.

        Args:
            mesh (Meshes2D): Spatial Meshes2D.
            perturbation_magnitude (float, optional): Tripolar perturbation
            magnitude. Defaults to 1e-3.
        """
        if mesh.xy_unit != self._required_xy_unit:
            msg = f"XY units should be {self._required_xy_unit}."
            raise UnitError(msg)
        self._mesh = mesh
        self._perturbation = perturbation_magnitude
        self._compute_psi()

    @property
    def mesh(self) -> Meshes2D:
        """Underlying mesh."""
        return self._mesh

    @property
    def perturbation_magnitude(self) -> float:
        """Tripolar perturbation magnitude."""
        return self._perturbation

    @property
    def r0(self) -> float:
        """Core cylinder radius: 0.1*lx."""
        return 0.1 * self._mesh.lx

    @property
    def r1(self) -> float:
        """Inner radius of the surrounding ring: r0."""
        return self.r0

    @property
    def r2(self) -> float:
        """Outer radius of the surrounding ring: 0.14*lx."""
        return 0.14 * self._mesh.lx

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
        vor = self._compute_vorticity(self._mesh.omega.xy)
        # Compute Laplacian operator in Fourier Space
        laplacian = helmholtz.compute_laplace_dstI(
            self._mesh.nx,
            self._mesh.ny,
            self._mesh.dx,
            self._mesh.dy,
            {"device": DEVICE, "dtype": torch.float64},
        )
        # Solve problem in Fourier space : "ψ = ω/∆"
        psi_hat = helmholtz.dstI2D(vor[1:-1, 1:-1]) / laplacian
        # Come back to original space
        psi = F.pad(helmholtz.dstI2D(psi_hat), (1, 1, 1, 1))
        self._psi = psi.unsqueeze(0).unsqueeze(0)

    def _compute_vorticity(
        self,
        mesh_xy: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the vorticity ω of the vortex.

        Args:
            mesh_xy (tuple[torch.Tensor, torch.Tensor]): Grid.

        Returns:
            torch.Tensor: Vorticity Value.
        """
        x, y = mesh_xy
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


class RankineVortex3D(metaclass=ABCMeta):
    """3D Rankine Vortex."""

    def __init__(
        self,
        mesh: Meshes3D,
        perturbation_magnitude: float = 1e-3,
    ) -> None:
        """Instantiate 3D Rankine Vortex.

        Args:
            mesh (Meshes3D): 3D Grid.
            perturbation_magnitude (float, optional): Tripolar perturbation
            magnitude. Defaults to 1e-3.
        """
        self._mesh = mesh
        self._2d = RankineVortex2D(
            mesh=mesh.remove_z_h(),
            perturbation_magnitude=perturbation_magnitude,
        )

    @property
    def mesh(self) -> Meshes3D:
        """Underlying mesh."""
        return self._mesh

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
        return self._2d.psi.expand((1, self._mesh.nl, *xy_shape))


class ActiveLayersRankineVortex3D(RankineVortex3D):
    """3D Rankine Vortex with similar vortex behavior accross all layers."""

    @property
    def psi(self) -> torch.Tensor:
        """Value of the stream function ψ.

        The Tensor has a shape of (1, nl, nx + 1, ny + 1).
        """
        xy_shape = self._2d.psi.shape[-2:]
        return self._2d.psi.expand((1, self._mesh.nl, *xy_shape))


class PassiveLayersRankineVortex3D(RankineVortex3D):
    """3D Rankine Vortex with only superior layer active."""

    @property
    def psi(self) -> torch.Tensor:
        """Value of the stream function ψ.

        The Tensor has a shape of (1, nl, nx + 1, ny + 1).
        """
        xy_shape = self._2d.psi.shape[-2:]
        psi = torch.ones(
            (1, self._mesh.nl, *xy_shape),
            device=DEVICE,
            dtype=torch.float64,
        )
        psi[0, 0, ...] = self._2d.psi
        return psi


class RankineVortexForcing:
    """Vortex Forcing's abtract class."""

    def __init__(self, vortex: RankineVortex3D) -> None:
        """Instantiate Vortex.

        Args:
            vortex (RankineVortex3D): Corresponding Vortex
        """
        self._vortex = vortex

    @property
    def perturbation_magnitude(self) -> float:
        """Tripolar perturbation magnitude."""
        return self._vortex.perturbation_magnitude

    @property
    def r0(self) -> float:
        """Core cylinder radius: 0.1*lx."""
        return self._vortex.r0

    @property
    def r1(self) -> float:
        """Inner radius of the surrounding ring: r0."""
        return self._vortex.r1

    @property
    def r2(self) -> float:
        """Outer radius of the surrounding ring: 0.14*lx."""
        return self._vortex.r2

    def compute(self, f0: float, Ro: float) -> torch.Tensor:  # noqa: N803
        """Compute Initial pressure based on the vortex's vorticity.

        Args:
            f0 (float): Coriolis Parameter from β-plane approximation.
            Ro (float): Rossby Number.

        Returns:
            torch.Tensor: Initial Pressure with the presence of the vortex.
        """
        u, v = grad_perp(
            self._vortex.psi,
            self._vortex.mesh.dx,
            self._vortex.mesh.dy,
        )
        u_norm_max = max(torch.abs(u).max(), torch.abs(v).max())
        # set psi amplitude to have a correct Rossby number
        psi_norm = self._vortex.psi * (Ro * f0 * self._vortex.r0 / u_norm_max)
        # Return pressure value
        return psi_norm * f0

    @classmethod
    def from_config(
        cls,
        vortex_config: PerturbationConfig,
        mesh_config: MeshConfig,
        model_config: ModelConfig,
    ) -> Self:
        """Instantiate VortexForcing from ScriptConfig.

        Args:
            vortex_config (PerturbationConfig): Vortex configuration.
            mesh_config (MeshConfig): Mesh configuration.
            model_config (ModelConfig): Model configuration.

        Raises:
            KeyError: If the vortex is not recognized.

        Returns:
            Self: Vortex forcing.
        """
        if vortex_config.type == "active":
            mesh = Meshes3D.from_config(
                mesh_config=mesh_config,
                model_config=model_config,
            )
            vortex = ActiveLayersRankineVortex3D(
                mesh=mesh,
                perturbation_magnitude=vortex_config.perturbation_magnitude,
            )
            return cls(vortex=vortex)
        if vortex_config.type == "passive":
            mesh = Meshes3D.from_config(
                mesh_config=mesh_config,
                model_config=model_config,
            )
            vortex = PassiveLayersRankineVortex3D(
                mesh=mesh,
                perturbation_magnitude=vortex_config.perturbation_magnitude,
            )
            return cls(vortex=vortex)
        msg = "Unrecognized vortex type."
        raise KeyError(msg)
