"""Energy-related variables."""

from typing import Self

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.masks import Masks
from qgsw.models.core import finite_diff
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.models.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.models.variables.dynamics import (
    MeridionalVelocityFlux,
    Pressure,
    ZonalVelocityFlux,
)
from qgsw.models.variables.state import State
from qgsw.models.variables.uvh import UVH


class KineticEnergy(DiagnosticVariable[torch.Tensor]):
    """Kinetic Energy Variable."""

    _unit = "m²s⁻²"
    _name = "kinetic_energy"
    _description = "Kinetic energy."

    def __init__(
        self,
        masks: Masks,
        U: ZonalVelocityFlux,  # noqa: N803
        V: MeridionalVelocityFlux,  # noqa:N803
    ) -> None:
        """Instantiate Kinetic Energy variable.

        Args:
            masks (Masks): Masks.
            U (ZonalVelocityFlux): Zonal Velocity Flux
            (Contravariant velocity vector).
            V (MeriodionalVelocityFlux): Meriodional Velocity Flux
            (Contravariant velocity vector).
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
        U = self._U.compute(uvh)  # noqa: N806
        V = self._V.compute(uvh)  # noqa: N806
        return self._comp_ke(u, U, v, V) * self._h_mask

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the UV variables
        self._U = self._U.bind(state)
        self._V = self._V.bind(state)
        return super().bind(state)


class TotalKineticEnergy(DiagnosticVariable[torch.Tensor]):
    """Total Kinetic Energy."""

    _unit = "m²s⁻²"
    _name = "total_ke"
    _description = "Total kinetic energy."

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
            torch.Tensor: Total kinetic energy, shape: (n_ens).
        """
        return torch.sum(self._ke.compute(uvh), dim=[-1, -2, -3]).cpu().item()

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the ke variable
        self._ke = self._ke.bind(state)
        return super().bind(state)


class TotalModalKineticEnergy(DiagnosticVariable[torch.Tensor]):
    """Compute total modal kinetic energy."""

    _unit = ""
    _name = "ke_hat"
    _description = "Modal kinetic energy."

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
        p_hat_pad_x = F.pad(p_hat, (0, 0, 0, 1))
        # Pad y with 0 on the left
        p_hat_pad_y = F.pad(p_hat, (0, 1, 0, 0))
        # Differentiate
        p_hat_dx = torch.diff(p_hat_pad_x, dim=-2)
        p_hat_dy = torch.diff(p_hat_pad_y, dim=-1)
        return torch.sum(p_hat_dx**2 + p_hat_dy**2, dim=(-1, -2, -3))

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the p variable
        self._p = self._p.bind(state)
        return super().bind(state)


class TotalModalAvailablePotentialEnergy(DiagnosticVariable[torch.Tensor]):
    """Total modal available potential energy."""

    _unit = ""
    _name = "ape_hat"
    _description = "Modal available potential energy."

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

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the p variable
        self._p = self._p.bind(state)
        return super().bind(state)


class TotalModalEnergy(DiagnosticVariable[torch.Tensor]):
    """Total modal energy."""

    _unit = ""
    _name = "e_tot_hat"
    _description = "Total modal energy."

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

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the ke variable
        self._ke = self._ke.bind(state)
        # Bind the _ape variable
        self.__ape = self._ape.bind(state)
        return super().bind(state)
