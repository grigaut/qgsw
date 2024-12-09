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
from qgsw.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.variables.dynamics import (
    MeridionalVelocityFlux,
    StreamFunction,
    ZonalVelocityFlux,
)
from qgsw.variables.state import State
from qgsw.variables.uvh import UVH


class KineticEnergy(DiagnosticVariable):
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
    ) -> BoundDiagnosticVariable[Self]:
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


class TotalKineticEnergy(DiagnosticVariable):
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
        return torch.sum(self._ke.compute(uvh), dim=[-1, -2, -3])

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the ke variable
        self._ke = self._ke.bind(state)
        return super().bind(state)


class TotalModalKineticEnergy(DiagnosticVariable):
    """Compute total modal kinetic energy."""

    _unit = "m².s⁻²"
    _name = "ke_hat"
    _description = "Modal kinetic energy."

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            dx (float): Elementary distance in the X direction.
            dy (float): Elementary distance in the Y direction.
        """
        self._psi = stream_function
        _, _, self._Cl2m = compute_layers_to_mode_decomposition(A)
        self._dx = dx
        self._dy = dy

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal kinetic energy, shape: (n_ens).
        """
        psi = self._psi.compute(uvh)
        psi_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, psi)
        # Pad x with 0 on top
        psi_hat_pad_x = F.pad(psi_hat, (0, 0, 0, 1))
        # Pad y with 0 on the left
        psi_hat_pad_y = F.pad(psi_hat, (0, 1, 0, 0))
        # Differentiate
        psi_hat_dx = torch.diff(psi_hat_pad_x, dim=-2) / self._dx
        psi_hat_dy = torch.diff(psi_hat_pad_y, dim=-1) / self._dy
        return torch.sum(psi_hat_dx**2 + psi_hat_dy**2, dim=(-1, -2, -3))

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)


class TotalModalAvailablePotentialEnergy(DiagnosticVariable):
    """Total modal available potential energy."""

    _unit = "m².s⁻²"
    _name = "ape_hat"
    _description = "Modal available potential energy."

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        f0: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            f0 (float): Coriolis parameter.
        """
        self._psi = stream_function
        self._f0 = f0
        _, self._lambd, self._Cl2m = compute_layers_to_mode_decomposition(A)

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal avalaible potential energy,
            shape: (n_ens).
        """
        psi = self._psi.compute(uvh)
        p_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, psi)
        return torch.sum(
            self._f0**2 * self._lambd * p_hat**2,
            dim=(-1, -2, -3),
        )

    def bind(
        self,
        state: State,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)


class TotalModalEnergy(DiagnosticVariable):
    """Total modal energy."""

    _unit = "m².s-2"
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
    ) -> BoundDiagnosticVariable[Self]:
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
