"""Dynamics variables."""

from __future__ import annotations

from qgsw.fields.scope import Scope
from qgsw.fields.variables.prognostic import (
    CollinearityCoefficient,
    LayerDepthAnomaly,
    MeridionalVelocity,
    Time,
    ZonalVelocity,
)
from qgsw.fields.variables.state import State
from qgsw.fields.variables.uvh import BasePrognosticTuple
from qgsw.models.core.helmholtz import (
    compute_laplace_dstI,
    solve_helmholtz_dstI,
)
from qgsw.specs import DEVICE
from qgsw.utils.units._units import Unit

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.fields.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.spatial.core.grid_conversion import points_to_surfaces

if TYPE_CHECKING:
    from qgsw.fields.variables.state import State
    from qgsw.fields.variables.uvh import BasePrognosticTuple, UVHTAlpha
    from qgsw.masks import Masks


class PhysicalZonalVelocity(DiagnosticVariable):
    """Physical zonal velocity."""

    _unit = Unit.M1S_1
    _name = "u_phys"
    _description = "Physical zonal velocity"
    _scope = Scope.POINT_WISE

    def __init__(self, dx: float) -> None:
        """Instantiate the variable.

        Args:
            dx (float): Elementary distance in the x direction.
        """
        self._dx = dx

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Physical zonal velocity component.
        """
        return prognostic.u / self._dx


class PhysicalMeridionalVelocity(DiagnosticVariable):
    """Physical zonal velocity."""

    _unit = Unit.M1S_1
    _name = "v_phys"
    _description = "Physical meridional velocity"
    _scope = Scope.POINT_WISE

    def __init__(self, dy: float) -> None:
        """Instantiate the variable.

        Args:
            dy (float): Elementary distance in the x direction.
        """
        self._dy = dy

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Physical zonal velocity component.
        """
        return prognostic.v / self._dy


class PhysicalLayerDepthAnomaly(DiagnosticVariable):
    """Physical layer depth anomaly."""

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

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Physical layer depth anomaly.
        """
        return prognostic.h / self._ds


class ZonalVelocityFlux(DiagnosticVariable):
    """Velocity flux."""

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

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostioc variables

        Returns:
            BasePrognosticTuple: Value
        """
        return prognostic.u / self._dx**2


class MeridionalVelocityFlux(DiagnosticVariable):
    """Velocity flux."""

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

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostioc variables

        Returns:
            BasePrognosticTuple: Value
        """
        return prognostic.v / self._dy**2


class SurfaceHeightAnomaly(DiagnosticVariable):
    """Surface height anomaly."""

    _unit = Unit.M
    _name = "eta"
    _description = "Surface height anomaly"
    _scope = Scope.POINT_WISE

    def __init__(self) -> None:
        """Instantiate variable."""

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostioc variables

        Returns:
            torch.Tensor: Surface height anomaly
        """
        return reverse_cumsum(prognostic.h, dim=-3)


class PhysicalSurfaceHeightAnomaly(DiagnosticVariable):
    """Physical Surface height anomaly."""

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

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostioc variables

        Returns:
            torch.Tensor: Surface height anomaly
        """
        return reverse_cumsum(
            self._h_phys.compute_no_slice(prognostic),
            dim=-3,
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
        # Bind the h_phys variable
        self._h_phys = self._h_phys.bind(state)
        return super().bind(state)


class MaskedVorticity(DiagnosticVariable):
    """Masked Vorticity Diagnostic Variable."""

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

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables

        Returns:
            torch.Tensor: Vorticity
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


class Vorticity(DiagnosticVariable):
    """Vorticity."""

    _unit = Unit.M2S_1
    _name = "omega"
    _description = "Vorticity"
    _scope = Scope.POINT_WISE

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic tuple.

        Returns:
            torch.Tensor: Vorticity, (n_ens,nl,nx-1,ny-1)-shaped
        """
        return torch.diff(prognostic.v[..., 1:-1], dim=-2) - torch.diff(
            prognostic.u[..., 1:-1, :],
            dim=-1,
        )


class PhysicalVorticity(DiagnosticVariable):
    """Physical vorticity."""

    _unit = Unit.S_1
    _name = "omega_phys"
    _description = "Physical vorticity"
    _scope = Scope.POINT_WISE

    def __init__(self, vorticity: Vorticity, ds: float) -> None:
        """Instantiate the variable.

        Args:
            vorticity (Vorticity): Vorticity.
            ds (float): Elementary surface element.
        """
        self._vorticity = vorticity
        self._ds = ds

        self._require_alpha |= vorticity.require_alpha
        self._require_time |= vorticity.require_time

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Physical vorticity.
        """
        return self._vorticity.compute_no_slice(prognostic) / self._ds

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
        # Bind the vorticity variable
        self._vorticity = self._vorticity.bind(state)
        return super().bind(state)


class Pressure(DiagnosticVariable):
    """Pressure."""

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
            g_prime (torch.Tensor): Reduced gravity
            eta_phys (PhysicalSurfaceHeightAnomaly): Surface height anomaly
            variable.
        """
        self._g_prime = g_prime
        self._eta = eta_phys

        self._require_alpha |= eta_phys.require_alpha
        self._require_time |= eta_phys.require_time

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostioc variables

        Returns:
            torch.Tensor: Pressure
        """
        return torch.cumsum(
            self._g_prime * self._eta.compute_no_slice(prognostic),
            dim=-3,
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
        # Bind the eta_phys variable
        self._eta = self._eta.bind(state)
        return super().bind(state)


class PressureTilde(Pressure):
    """Pressure tilde."""

    _description = "Pressure per unit of mass for collinear model"

    def __init__(
        self,
        g_prime: torch.Tensor,
        eta_phys: PhysicalSurfaceHeightAnomaly,
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            g_prime (torch.Tensor): Reduced gravity
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
            prognostic (UVHTAlpha): Prognostioc variables

        Returns:
            torch.Tensor: Pressure
        """
        g_tilde = self._compute_g_tilde(prognostic.alpha)
        return g_tilde * self._eta.compute_no_slice(prognostic)


class PotentialVorticity(DiagnosticVariable):
    """Potential Vorticity."""

    _unit = Unit.S_1
    _name = "pv"
    _description = "Potential vorticity"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        vorticity_phys: PhysicalVorticity,
        h_ref: torch.Tensor,
        ds: float,
        f0: float,
    ) -> None:
        """Instantiate variable.

        Args:
            vorticity_phys (PhysicalVorticity): Physical vorticity.
            h_ref (torch.Tensor): Reference heights.
            ds (float): Elementary area.
            f0 (float): Coriolis parameter.
        """
        self._h_ref = h_ref
        self._ds = ds
        self._f0 = f0
        self._vorticity_phys = vorticity_phys
        self._interp = OptimizableFunction(points_to_surfaces)

        self._require_alpha |= vorticity_phys.require_alpha
        self._require_time |= vorticity_phys.require_time

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            prognostic (BasePrognosticTuple): Prognostioc variables

        Returns:
            torch.Tensor: Value
        """
        vorticity_phys = self._interp(
            self._vorticity_phys.compute_no_slice(prognostic),
        )
        return vorticity_phys - self._f0 * prognostic.h / self._h_ref

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
        # Bind the vorticity_phys variable
        self._vorticity_phys = self._vorticity_phys.bind(state)
        return super().bind(state)


class StreamFunction(DiagnosticVariable):
    """Stream function variable."""

    _unit = Unit.M2S_1
    _name = "psi"
    _description = "Stream function"
    _scope = Scope.POINT_WISE

    def __init__(self, pressure: Pressure, f0: float) -> None:
        """Instantiate the variable.

        Args:
            pressure (Pressure): Pressure variable.
            f0 (float): Coriolis parameter.
        """
        self._p = pressure
        self._f0 = f0

        self._require_alpha |= pressure.require_alpha
        self._require_time |= pressure.require_time

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Stream function.
        """
        return self._p.compute_no_slice(prognostic) / self._f0

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the pressure variable
        self._p = self._p.bind(state)
        return super().bind(state)


class StreamFunctionFromVorticity(DiagnosticVariable):
    """Stream function variable from vorticity inversion."""

    _unit = Unit.M2S_1
    _name = "psi_from_omega"
    _description = "Stream function from vorticity inversion"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        vorticity: PhysicalVorticity,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
    ) -> None:
        """Instantiate the variable.

        Args:
            vorticity (PhysicalVorticity): Physical vorticity.
            nx (int): Number of poitn in the x direction.
            ny (int): Number of points in the y direction.
            dx (float): Infinitesimal x step.
            dy (float): Infinitesimal y step.
        """
        self._vorticity = vorticity

        self._require_alpha |= vorticity.require_alpha
        self._require_time |= vorticity.require_time

        self._laplacian = (
            compute_laplace_dstI(
                nx,
                ny,
                dx,
                dy,
                dtype=torch.float64,
                device=DEVICE.get(),
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Stream function.
        """
        vorticity = self._vorticity.compute_no_slice(prognostic)
        return solve_helmholtz_dstI(vorticity, self._laplacian)[..., 1:, 1:]

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the vorticity variable
        self._vorticity = self._vorticity.bind(state)
        return super().bind(state)


class Enstrophy(DiagnosticVariable):
    """Layer-wise enstrophy."""

    _unit = Unit.S_2
    _name = "enstrophy"
    _description = "Layer-wise enstrophy"
    _scope = Scope.LEVEL_WISE

    def __init__(self, pv: PotentialVorticity) -> None:
        """Instantiate the variable.

        Args:
            pv (PotentialVorticity): Physical vorticity.
        """
        self._pv = pv
        self._require_alpha |= pv.require_alpha
        self._require_time |= pv.require_time

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Enstrophy.
        """
        return 0.5 * torch.sum(
            self._pv.compute_no_slice(prognostic) ** 2,
            dim=(-1, -2),
        )

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a state.

        Args:
            state (State): State.

        Returns:
            BoundDiagnosticVariable[Self]: Bound variable.
        """
        self._pv.bind(state)
        return super().bind(state)


class TotalEnstrophy(Enstrophy):
    """Total enstrophy."""

    _unit = Unit.S_2
    _name = "enstrophy_tot"
    _description = "Total enstrophy"
    _scope = Scope.ENSEMBLE_WISE

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables.

        Returns:
            torch.Tensor: Enstrophy.
        """
        return 0.5 * torch.sum(
            self._pv.compute_no_slice(prognostic) ** 2,
            dim=(-1, -2, -3),
        )


class ZonalVelocityDiag(DiagnosticVariable):
    """Diagnostic zonal velocity."""

    _unit = ZonalVelocity.get_unit()
    _name = ZonalVelocity.get_name()
    _description = ZonalVelocity.get_description()
    _scope = ZonalVelocity.get_scope()

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return prognostic.u


class MeridionalVelocityDiag(DiagnosticVariable):
    """Diagnostic Meridional Velocity."""

    _unit = MeridionalVelocity.get_unit()
    _name = MeridionalVelocity.get_name()
    _description = MeridionalVelocity.get_description()
    _scope = MeridionalVelocity.get_scope()

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return prognostic.v


class LayerDepthAnomalyDiag(DiagnosticVariable):
    """Diagnostic Layer Depth Anomaly."""

    _unit = LayerDepthAnomaly.get_unit()
    _name = LayerDepthAnomaly.get_name()
    _description = LayerDepthAnomaly.get_description()
    _scope = LayerDepthAnomaly.get_scope()

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return prognostic.h


class TimeDiag(DiagnosticVariable):
    """Diagnostic Layer Depth Anomaly."""

    _unit = Time.get_unit()
    _name = Time.get_name()
    _description = Time.get_description()
    _scope = Time.get_scope()

    def _compute(self, prognostic: BasePrognosticTuple) -> torch.Tensor:
        """Compute variable value.

        Args:
            prognostic (BasePrognosticTuple): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return prognostic.t


class CollinearityCoefficientDiag(DiagnosticVariable):
    """Diagnostic Layer Depth Anomaly."""

    _unit = CollinearityCoefficient.get_unit()
    _name = CollinearityCoefficient.get_name()
    _description = CollinearityCoefficient.get_description()
    _scope = CollinearityCoefficient.get_scope()

    def _compute(self, prognostic: UVHTAlpha) -> torch.Tensor:
        """Compute variable value.

        Args:
            prognostic (UVHTAlpha): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return prognostic.alpha
