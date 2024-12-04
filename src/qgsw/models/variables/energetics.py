"""Energy-related variables."""

import torch

from qgsw.masks import Masks
from qgsw.models.core import finite_diff
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.variables.core import UVH, DiagnosticVariable
from qgsw.models.variables.dynamics import (
    PhysicalMeridionalVelocity,
    PhysicalZonalVelocity,
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

    def compute(self, uvh: UVH) -> float:
        """Compute total kinetic energy.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            float: Total kinetic energy.
        """
        return torch.sum(self._ke.compute(uvh)).cpu().item()
