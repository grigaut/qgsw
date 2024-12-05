"""Energy-related variables."""

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.masks import Masks
from qgsw.models.core import finite_diff
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.models.variables.core import UVH, DiagnosticVariable
from qgsw.models.variables.dynamics import (
    PhysicalMeridionalVelocity,
    PhysicalZonalVelocity,
    Pressure,
)


class KineticEnergy(DiagnosticVariable):
    """Kinetic Energy Variable."""

    def __init__(
        self,
        masks: Masks,
        U: PhysicalZonalVelocity,  # noqa: N803
        V: PhysicalMeridionalVelocity,  # noqa: N803
    ) -> None:
        """Instantiate Kinetic Energy variable.

        Args:
            masks (Masks): Masks.
            U (PhysicalZonalVelocity): Physical Zonal Velocity.
            V (PhysicalMeridionalVelocity): Physical Meridional Velocity.
        """
        self._h_mask = masks.h
        self._U = U
        self._V = V
        self._comp_ke = OptimizableFunction(finite_diff.comp_ke)

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the kinetic energy.

        Args:
            uvh (UVH): u,v,h

        Returns:
            torch.Tensor: Kinetic energy.
        """
        u, v, _ = uvh
        U, V = self._U.compute(uvh), self._V.compute(uvh)  # noqa: N806
        return self._comp_ke(u, U, v, V) * self._h_mask


class TotalKineticEnergy(DiagnosticVariable):
    """Total Kinetic Energy."""

    def __init__(self, kinetic_energy: KineticEnergy) -> None:
        """Instantiate the kinetic energy measure.

        Args:
            kinetic_energy (KineticEnergy): Kinetic energy
        """
        self._ke = kinetic_energy

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute total kinetic energy.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            float: Total kinetic energy, shape: (n_ens).
        """
        return torch.sum(self._ke.compute(uvh), dim=[-1, -2, -3]).cpu().item()


class TotalModalKineticEnergy(DiagnosticVariable):
    """Compute total modal kinetic energy."""

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        pressure: Pressure,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            pressure (Pressure): Pressure diagnostic variable.
        """
        self._p = pressure
        _, _, self._Cl2m = compute_layers_to_mode_decomposition(A)

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal kinetic energy, shape: (n_ens).
        """
        p = self._p.compute(uvh)
        p_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, p)
        # Pad x with 0 on top
        p_hat_pad_x = F.pad(p_hat, (0, 0, 1, 0))
        # Pad y with 0 on the left
        p_hat_pad_y = F.pad(p_hat, (1, 0, 0, 0))
        # Differentiate
        p_hat_dx = torch.diff(p_hat_pad_x, dim=-2)
        p_hat_dy = torch.diff(p_hat_pad_y, dim=-1)
        return torch.sum(p_hat_dx**2 + p_hat_dy**2, dim=(-1, -2, -3))


class TotalModalAvailablePotentialEnergy(DiagnosticVariable):
    """Total modal available potential energy."""

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        pressure: Pressure,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            pressure (Pressure): Pressure diagnostic variable.
        """
        self._p = pressure
        _, self._lambd, self._Cl2m = compute_layers_to_mode_decomposition(A)

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal avalaible potential energy,
            shape: (n_ens).
        """
        p = self._p.compute(uvh)
        p_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, p)
        return torch.sum((self._lambd * p_hat) ** 2, dim=(-1, -2, -3))


class TotalModalEnergy(DiagnosticVariable):
    """Total modal energy."""

    def __init__(
        self,
        ke_hat: TotalModalKineticEnergy,
        ape_hat: TotalModalAvailablePotentialEnergy,
    ) -> None:
        """Instantiate variable.

        Args:
            ke_hat (TotalModalKineticEnergy): Total modal kinetic energy
            ape_hat (TotalModalAvailablePotentialEnergy): Total modal
            available potential energy
        """
        self._ke = ke_hat
        self._ape = ape_hat

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute total modal energy.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal energy, shape: (n_ens)
        """
        return self._ke.compute(uvh) + self._ape.compute(uvh)
