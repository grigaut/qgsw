"""Energy-related variables."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.scope import Scope
from qgsw.fields.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.fields.variables.dynamics import (
    MeridionalVelocityFlux,
    StreamFunction,
    ZonalVelocityFlux,
)
from qgsw.fields.variables.state import State
from qgsw.fields.variables.uvh import UVH
from qgsw.masks import Masks
from qgsw.models.core import finite_diff
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.qg.stretching_matrix import (
    compute_layers_to_mode_decomposition,
)
from qgsw.spatial.units._units import Unit


def compute_W(H: torch.Tensor) -> torch.Tensor:  # noqa: N802, N803
    """Compute the weight matrix.

    Args:
        H (torch.Tensor): Layers reference depths.

    Returns:
        torch.Tensor: Weight Matrix
    """
    return torch.diag(H) / torch.sum(H)


class KineticEnergy(DiagnosticVariable):
    """Kinetic Energy Variable."""

    _unit = Unit.M2S_2
    _name = "kinetic_energy"
    _description = "Kinetic energy"
    _scope = Scope.POINT_WISE

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

    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the kinetic energy.

        Args:
            uvh (UVH): u,v,h

        Returns:
            torch.Tensor: Kinetic energy.
        """
        u, v, _ = uvh
        U = self._U.compute_no_slice(uvh)  # noqa: N806
        V = self._V.compute_no_slice(uvh)  # noqa: N806
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


class ModalKineticEnergy(DiagnosticVariable):
    """Compute modal kinetic energy."""

    _unit = Unit.M2S_2
    _name = "ke_hat"
    _description = "Modal kinetic energy"
    _scope = Scope.LEVEL_WISE

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
            dx (float): Elementary distance in the X direction.
            dy (float): Elementary distance in the Y direction.
        """
        self._psi = stream_function
        self._dx = dx
        self._dy = dy
        # Decomposition of A
        Cm2l, _, self._Cl2m = compute_layers_to_mode_decomposition(A)  # noqa: N806
        # Compute W = Diag(H) / h_{tot}
        W = compute_W(H)  # noqa: N806
        # Compute Cl2m^{-T} @ W @ Cl2m⁻¹
        Cm2l_T = Cm2l.transpose(dim0=0, dim1=1)  # noqa: N806
        Cm2lT_W_Cm2l = Cm2l_T @ W @ Cm2l  # noqa: N806
        # Since Cm2lT_W_Cm2l is diagonal
        self._Cm2lT_W_Cm2l = torch.diag(Cm2lT_W_Cm2l)  # Vector

    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Modal kinetic energy, shape: (n_ens, nl).
        """
        psi = self._psi.compute_no_slice(uvh)
        psi_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, psi)
        # Pad x with 0 on top
        psi_hat_pad_x = F.pad(psi_hat, (0, 0, 0, 1))
        # Pad y with 0 on the left
        psi_hat_pad_y = F.pad(psi_hat, (0, 1, 0, 0))
        # Differentiate
        psi_hat_dx = torch.diff(psi_hat_pad_x, dim=-2) / self._dx
        psi_hat_dy = torch.diff(psi_hat_pad_y, dim=-1) / self._dy
        psiT_CT_W_C_psi = torch.einsum(  # noqa: N806
            "l,...lxy->...lxy",
            self._Cm2lT_W_Cm2l,
            (psi_hat_dx**2 + psi_hat_dy**2),
        )
        return 0.5 * torch.sum(psiT_CT_W_C_psi, dim=(-1, -2))

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


class ModalAvailablePotentialEnergy(DiagnosticVariable):
    """Modal available potential energy."""

    _unit = Unit.M2S_2
    _name = "ape_hat"
    _description = "Modal available potential energy"
    _scope = Scope.LEVEL_WISE

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        f0: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
            f0 (float): Coriolis parameter.
        """
        self._psi = stream_function
        self._f0 = f0
        # Decomposition of A
        Cm2l, lambd, self._Cl2m = compute_layers_to_mode_decomposition(A)  # noqa: N806
        # Compute weight matrix
        W = compute_W(H)  # noqa: N806
        # Compute Cl2m^{-T} @ W @ Cl2m⁻¹ @ Λ
        Cm2l_T = Cm2l.transpose(dim0=0, dim1=1)  # noqa: N806
        self._Cm2lT_W_Cm2l_lambda = Cm2l_T @ W @ Cm2l @ lambd  # Vector

    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Modal avalaible potential energy,
            shape: (n_ens,nl).
        """
        psi = self._psi.compute_no_slice(uvh)
        psi_hat = torch.einsum("lm,...mxy->...lxy", self._Cl2m, psi)
        psiT_CT_W_C_lambda_psi = torch.einsum(  # noqa: N806
            "l,...lxy->...lxy",
            self._Cm2lT_W_Cm2l_lambda,
            psi_hat**2,
        )
        ape = torch.sum(psiT_CT_W_C_lambda_psi, dim=(-1, -2))
        return 0.5 * self._f0**2 * ape

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


class ModalEnergy(DiagnosticVariable):
    """Modal energy."""

    _unit = Unit.M2S_2
    _name = "e_tot_hat"
    _description = "Modal energy"
    _scope = Scope.LEVEL_WISE

    def __init__(
        self,
        ke_hat: ModalKineticEnergy,
        ape_hat: ModalAvailablePotentialEnergy,
    ) -> None:
        """Instantiate variable.

        Args:
            ke_hat (TotalModalKineticEnergy): Modal kinetic energy
            ape_hat (TotalModalAvailablePotentialEnergy): Modal
            available potential energy
        """
        self._ke = ke_hat
        self._ape = ape_hat

    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute total modal energy.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Modal total energy, shape: (n_ens)
        """
        return self._ke.compute_no_slice(uvh) + self._ape.compute_no_slice(uvh)

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
        self._ape = self._ape.bind(state)
        return super().bind(state)


class TotalKineticEnergy(DiagnosticVariable):
    """Compute total kinetic energy."""

    _unit = Unit.M2S_2
    _name = "ke"
    _description = "Kinetic energy"
    _scope = Scope.ENSEMBLE_WISE

    def __init__(
        self,
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
            dx (float): Elementary distance in the X direction.
            dy (float): Elementary distance in the Y direction.
        """
        self._psi = stream_function
        self._dx = dx
        self._dy = dy
        # Compute W = Diag(H) / h_{tot}
        self._W = torch.diag(compute_W(H))  # Vector

    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal kinetic energy, shape: (n_ens).
        """
        psi = self._psi.compute_no_slice(uvh)
        # Pad x with 0 on top
        psi_pad_x = F.pad(psi, (0, 0, 0, 1))
        # Pad y with 0 on the left
        psi_pad_y = F.pad(psi, (0, 1, 0, 0))
        # Differentiate
        psi_dx = torch.diff(psi_pad_x, dim=-2) / self._dx
        psi_dy = torch.diff(psi_pad_y, dim=-1) / self._dy
        psiT_W_psi = torch.einsum(  # noqa: N806
            "l,...lxy->...lxy",
            self._W,
            (psi_dx**2 + psi_dy**2),
        )
        return 0.5 * torch.sum(psiT_W_psi, dim=(-1, -2, -3))

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


class TotalAvailablePotentialEnergy(DiagnosticVariable):
    """Total modal available potential energy."""

    _unit = Unit.M2S_2
    _name = "ape"
    _description = "Available potential energy"
    _scope = Scope.ENSEMBLE_WISE

    def __init__(
        self,
        A: torch.Tensor,  # noqa: N803
        stream_function: StreamFunction,
        H: torch.Tensor,  # noqa: N803
        f0: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            A (torch.Tensor): Stetching matrix.
            stream_function (StreamFunction): Stream function diagnostic
            variable.
            H (torch.Tensor): Layers reference depth.
            f0 (float): Coriolis parameter.
        """
        self._psi = stream_function
        self._f0 = f0
        self._A = A
        # Compute weight matrix
        self._W = compute_W(H)  # Matrix

    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal avalaible potential energy,
            shape: (n_ens).
        """
        psi = self._psi.compute_no_slice(uvh)
        W_A_psi = torch.einsum(  # noqa: N806
            "lm,...mxy->...lxy",
            self._W @ self._A,
            psi,
        )
        psiT_W_A_psi = torch.einsum(  # noqa: N806
            "...lxy,...lxy->...lxy",
            psi,
            W_A_psi,
        )
        ape = torch.sum(psiT_W_A_psi, dim=(-1, -2, -3))
        return 0.5 * self._f0**2 * ape

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


class TotalEnergy(DiagnosticVariable):
    """Total modal energy."""

    _unit = Unit.M2S_2
    _name = "e_tot"
    _description = "Total energy"
    _scope = Scope.ENSEMBLE_WISE

    def __init__(
        self,
        ke: TotalKineticEnergy,
        ape: TotalAvailablePotentialEnergy,
    ) -> None:
        """Instantiate variable.

        Args:
            ke (TotalKineticEnergy): Total modal kinetic energy
            ape (TotalAvailablePotentialEnergy): Total modal
            available potential energy
        """
        self._ke = ke
        self._ape = ape

    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute total modal energy.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Total modal energy, shape: (n_ens)
        """
        return self._ke.compute_no_slice(uvh) + self._ape.compute_no_slice(uvh)

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
        self._ape = self._ape.bind(state)
        return super().bind(state)
