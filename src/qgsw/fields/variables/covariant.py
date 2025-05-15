"""Covariant varibales."""

from __future__ import annotations

from qgsw.fields.scope import Scope
from qgsw.models.core import finite_diff
from qgsw.utils.units._units import Unit

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import contextlib
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction

if TYPE_CHECKING:
    from qgsw.fields.variables.prognostic_tuples import (
        BaseUVH,
        UVHTAlpha,
    )
    from qgsw.fields.variables.state import StateUVH
    from qgsw.masks import Masks
    from qgsw.models.qg.uvh.projectors.core import QGProjector


class PhysicalZonalVelocity(DiagnosticVariable):
    """Physical zonal velocity.

    └── (n_ens, nl, nx+1, ny)-shaped
    """

    _unit = Unit.M1S_1
    _name = "u_phys"
    _description = "Physical zonal velocity"
    _scope = Scope.POINT_WISE

    def __init__(self, dx: float) -> None:
        """Instantiate the variable.

        Args:
            dx (float): Elementary distance in the X direction.
        """
        self._dx = dx

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical zonal velocity component.
                └── (n_ens, nl, nx+1, ny)-shaped
        """
        return vars_tuple.u / self._dx


class PhysicalMeridionalVelocity(DiagnosticVariable):
    """Physical meridional velocity.

    └── (n_ens, nl, nx, ny+1)-shaped
    """

    _unit = Unit.M1S_1
    _name = "v_phys"
    _description = "Physical meridional velocity"
    _scope = Scope.POINT_WISE

    def __init__(self, dy: float) -> None:
        """Instantiate the variable.

        Args:
            dy (float): Elementary distance in the Y direction.
        """
        self._dy = dy

    def _compute(self, vars_tuple: BaseUVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            vars_tuple (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical meridional velocity component.
                └── (n_ens, nl, nx, ny+1)-shaped
        """
        return vars_tuple.v / self._dy


class PhysicalLayerDepthAnomaly(DiagnosticVariable):
    """Physical layer depth anomaly.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M
    _name = "h_phys"
    _description = "Physical layer depth anomaly"
    _scope = Scope.POINT_WISE

    def __init__(self, ds: float) -> None:
        """Instantiate the variable.

        Args:
            ds (float): Elementary surface element.
        """
        self._ds = ds

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the variable.

        Args:
            prognostic (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Physical layer depth anomaly.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return prognostic.h / self._ds


class ZonalVelocityFlux(DiagnosticVariable):
    """Velocity flux.

    └── (n_ens, nl, nx+1, ny)-shaped
    """

    _unit = Unit.S_1
    _name = "U"
    _description = "Zonal velocity flux"
    _scope = Scope.POINT_WISE

    def __init__(self, dx: float) -> None:
        """Instantiate the variable.

        Args:
            dx (float): Elementary distance in the x direction.
        """
        self._dx = dx

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            BaseUVH: Value
                └── (n_ens, nl, nx+1, ny)-shaped
        """
        return prognostic.u / self._dx**2


class MeridionalVelocityFlux(DiagnosticVariable):
    """Velocity flux.

    └── (n_ens, nl, nx, ny+1)-shaped
    """

    _unit = Unit.S_1
    _name = "V"
    _description = "Meriodional velocity flux"
    _scope = Scope.POINT_WISE

    def __init__(self, dy: float) -> None:
        """Instantiate the variable.

        Args:
            dy (float): Elementary distance in the y direction.
        """
        self._dy = dy

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            BaseUVH: Value
                └── (n_ens, nl, nx, ny+1)-shaped
        """
        return prognostic.v / self._dy**2


class PhysicalSurfaceHeightAnomaly(DiagnosticVariable):
    """Physical Surface height anomaly.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M
    _name = "eta_phys"
    _description = "Physical Surface height anomaly"
    _scope = Scope.POINT_WISE

    def __init__(self, h_phys: PhysicalLayerDepthAnomaly) -> None:
        """Instantiate variable.

        Args:
            h_phys (PhysicalLayerDepthAnomaly): Physical surface anomaly.
        """
        self._h_phys = h_phys

        self._require_alpha |= h_phys.require_alpha
        self._require_time |= h_phys.require_time

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Surface height anomaly
                └── (n_ens, nl, nx, ny)-shaped
        """
        return reverse_cumsum(
            self._h_phys.compute_no_slice(prognostic),
            dim=-3,
        )

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the h_phys variable
        self._h_phys = self._h_phys.bind(state)
        return super().bind(state)


class MaskedVorticity(DiagnosticVariable):
    """Masked Vorticity Diagnostic Variable.

    └── (n_ens, nl, nx+1, ny+1)-shaped
    """

    _unit = Unit.M2S_1
    _name = "omega_masked"
    _description = "Vorticity matching mask"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        masks: Masks,
        slip_coef: float,
    ) -> None:
        """Instantiate the vorticity variable.

        Args:
            masks (Masks): Masks
            slip_coef (float): Slip coefficient
        """
        self._slip_coef = slip_coef
        self._w_valid = masks.w_valid
        self._w_cornerout_bound = masks.w_cornerout_bound
        self._w_vertical_bound = masks.w_vertical_bound
        self._w_horizontal_bound = masks.w_horizontal_bound

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BaseUVH): Covariant variables

        Returns:
            torch.Tensor: Vorticity
                └── (n_ens, nl, nx+1, ny+1)-shaped
        """
        u = prognostic.u
        v = prognostic.v
        u_ = F.pad(u, (1, 1, 0, 0))
        v_ = F.pad(v, (0, 0, 1, 1))
        dx_v = torch.diff(v_, dim=-2)
        dy_u = torch.diff(u_, dim=-1)
        curl_uv = dx_v - dy_u
        alpha = 2 * (1 - self._slip_coef)
        return (
            self._w_valid * curl_uv
            + self._w_cornerout_bound * (1 - self._slip_coef) * curl_uv
            + self._w_vertical_bound * alpha * dx_v
            - self._w_horizontal_bound * alpha * dy_u
        )


class QGPressure(DiagnosticVariable):
    """QGPressure.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_2
    _name = "p"
    _description = "Pressure per unit of mass"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        P: QGProjector,  # noqa: N803
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            P (QGprojector): Projector to use to retrieve pressure.

        """
        self._P = P

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Pressure.
                └── (n_ens, nl, nx, ny)-shaped
        """
        raise NotImplementedError
        with contextlib.suppress(AttributeError):
            self._P.alpha = prognostic.alpha
        return self._P.compute_p(prognostic)[1]


class Pressure(DiagnosticVariable):
    """Pressure.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_2
    _name = "p"
    _description = "Pressure per unit of mass"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        g_prime: torch.Tensor,
        eta_phys: PhysicalSurfaceHeightAnomaly,
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            g_prime (torch.Tensor): Reduced gravity.
                └── (1, nl, 1, 1)-shaped
            eta_phys (PhysicalSurfaceHeightAnomaly): Surface height anomaly
            variable.
        """
        self._g_prime = g_prime
        self._eta = eta_phys

        self._require_alpha |= eta_phys.require_alpha
        self._require_time |= eta_phys.require_time

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BaseUVH): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Pressure.
                └── (n_ens, nl, nx, ny)-shaped
        """
        return torch.cumsum(
            self._g_prime * self._eta.compute_no_slice(prognostic),
            dim=-3,
        )

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the eta_phys variable
        self._eta = self._eta.bind(state)
        return super().bind(state)


class PressureTilde(Pressure):
    """Pressure tilde.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _name = "p_tilde"
    _description = "Pressure per unit of mass for collinear model"
    _require_alpha = True

    def __init__(
        self,
        g_prime: torch.Tensor,
        eta_phys: PhysicalSurfaceHeightAnomaly,
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            g_prime (torch.Tensor): Reduced gravity.
                └── (1, 2, 1, 1)-shaped
            eta_phys (PhysicalSurfaceHeightAnomaly): Surface height anomaly
            variable.
        """
        if not g_prime.squeeze().shape[0] == 2:  # noqa: PLR2004
            raise ValueError
        self._g1 = g_prime.squeeze()[0]
        self._g2 = g_prime.squeeze()[1]
        self._eta = eta_phys

        self._require_alpha |= eta_phys.require_alpha
        self._require_time |= eta_phys.require_time

    def _compute_g_tilde(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute g_tilde.

        Args:
            alpha (torch.Tensor): Collinearity coefficient.

        Returns:
            torch.Tensor: g_tilde
        """
        return self._g1 * self._g2 / (self._g2 + (1 - alpha) * self._g1)

    def _compute(self, prognostic: UVHTAlpha) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (UVHTAlpha): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Pressure
                └── (n_ens, nl, nx, ny)-shaped
        """
        g_tilde = self._compute_g_tilde(prognostic.alpha)
        return g_tilde * self._eta.compute_no_slice(prognostic)


class KineticEnergy(DiagnosticVariable):
    """Kinetic Energy Variable.

    └── (n_ens, nl, nx, ny)-shaped
    """

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
        self._require_alpha |= U.require_alpha | V.require_alpha
        self._require_time |= U.require_time | V.require_time

    def _compute(self, prognostic: BaseUVH) -> torch.Tensor:
        """Compute the kinetic energy.

        Args:
            prognostic (BasePrognosticTuple): Covariant variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Kinetic energy.
                └── (n_ens, nl, nx, ny)-shaped
        """
        u = prognostic.u
        v = prognostic.v
        U = self._U.compute_no_slice(prognostic)  # noqa: N806
        V = self._V.compute_no_slice(prognostic)  # noqa: N806
        return self._comp_ke(u, U, v, V) * self._h_mask

    def bind(
        self,
        state: StateUVH,
    ) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the UV variables
        self._U = self._U.bind(state)
        self._V = self._V.bind(state)
        return super().bind(state)
