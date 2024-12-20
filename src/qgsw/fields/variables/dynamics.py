"""Dynamics variables."""

from __future__ import annotations

from qgsw.fields.scope import Scope
from qgsw.fields.variables.prognostic import (
    LayerDepthAnomaly,
    MeridionalVelocity,
    ZonalVelocity,
)
from qgsw.spatial.units._units import Unit

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
    from qgsw.fields.variables.uvh import UVH
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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Physical zonal velocity component.
        """
        return uvh.u / self._dx


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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Physical zonal velocity component.
        """
        return uvh.v / self._dy


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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Physical layer depth anomaly.
        """
        return uvh.h / self._ds


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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            UVH: Value
        """
        return uvh.u / self._dx**2


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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            UVH: Value
        """
        return uvh.v / self._dy**2


class SurfaceHeightAnomaly(DiagnosticVariable):
    """Surface height anomaly."""

    _unit = Unit.M
    _name = "eta"
    _description = "Surface height anomaly"
    _scope = Scope.POINT_WISE

    def __init__(self, h_phys: PhysicalLayerDepthAnomaly) -> None:
        """Instantiate variable.

        Args:
            h_phys (PhysicalLayerDepthAnomaly): Physical surface anomaly.
        """
        self._h_phys = h_phys

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            torch.Tensor: Surface height anomaly
        """
        return reverse_cumsum(self._h_phys.compute(uvh), dim=-3)

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


class Vorticity(DiagnosticVariable):
    """Vorticity Diagnostic Variable."""

    _unit = Unit.M2S_1
    _name = "omega"
    _description = "Vorticity"
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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostic variables

        Returns:
            torch.Tensor: Vorticity
        """
        u, v, _ = uvh
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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Physical vorticity.
        """
        return self._vorticity.compute(uvh) / self._ds

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
        eta: SurfaceHeightAnomaly,
    ) -> None:
        """Instantiate the pressure variable.

        Args:
            g_prime (torch.Tensor): Reduced gravity
            eta (SurfaceHeightAnomaly): Surface height anomaly variable.
        """
        self._g_prime = g_prime
        self._eta = eta

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            torch.Tensor: Pressure
        """
        return torch.cumsum(self._g_prime * self._eta.compute(uvh), dim=-3)

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
        # Bind the eta variable
        self._eta = self._eta.bind(state)
        return super().bind(state)


class PotentialVorticity(DiagnosticVariable):
    """Potential Vorticity."""

    _unit = Unit.S_1
    _name = "pv"
    _description = "Potential vorticity"
    _scope = Scope.POINT_WISE

    def __init__(
        self,
        vorticity: PhysicalVorticity,
        h_ref: torch.Tensor,
        area: float,
        f0: float,
    ) -> None:
        """Instantiate variable.

        Args:
            vorticity (PhysicalVorticity): Physical vorticity.
            h_ref (torch.Tensor): Reference heights.
            area (float): Elementary area.
            f0 (float): Coriolis parameter.
        """
        self._h_ref = h_ref
        self._area = area
        self._f0 = f0
        self._vorticity = vorticity
        self._interp = OptimizableFunction(points_to_surfaces)

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            torch.Tensor: Value
        """
        vorticity = self._interp(self._vorticity.compute(uvh))
        return vorticity - self._f0 * uvh.h / self._h_ref

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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Stream function.
        """
        return self._p.compute(uvh) / self._f0

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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Enstrophy.
        """
        return 0.5 * torch.sum(self._pv.compute(uvh) ** 2, dim=(-1, -2))

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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Enstrophy.
        """
        return 0.5 * torch.sum(self._pv.compute(uvh) ** 2, dim=(-1, -2, -3))


class ZonalVelocityDiag(DiagnosticVariable):
    """Diagnostic zonal velocity."""

    _unit = ZonalVelocity.get_unit()
    _name = ZonalVelocity.get_name()
    _description = ZonalVelocity.get_description()
    _scope = ZonalVelocity.get_scope()

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return uvh.u


class MeridionalVelocityDiag(DiagnosticVariable):
    """Diagnostic Meridional Velocity."""

    _unit = MeridionalVelocity.get_unit()
    _name = MeridionalVelocity.get_name()
    _description = MeridionalVelocity.get_description()
    _scope = MeridionalVelocity.get_scope()

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return uvh.v


class LayerDepthAnomalyDiag(DiagnosticVariable):
    """Diagnostic Layer Depth Anomaly."""

    _unit = LayerDepthAnomaly.get_unit()
    _name = LayerDepthAnomaly.get_name()
    _description = LayerDepthAnomaly.get_description()
    _scope = LayerDepthAnomaly.get_scope()

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute variable value.

        Args:
            uvh (UVH): Prognostic varibales values.

        Returns:
            torch.Tensor: Value.
        """
        return uvh.h
