"""Dynamics variables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.spatial.core.grid_conversion import points_to_surfaces

if TYPE_CHECKING:
    from qgsw.masks import Masks
    from qgsw.models.variables.state import State
    from qgsw.models.variables.uvh import UVH


class PhysicalVelocity(DiagnosticVariable[tuple[torch.Tensor, torch.Tensor]]):
    """Physical zonal velocity."""

    _unit = "m.s⁻¹"
    _name = "uv_phys"
    _description = "Physical horizontal velocity."

    def __init__(self, dx: float, dy: float) -> None:
        """Instantiate the variable.

        Args:
            dx (float): Elementary distance in the x direction.
            dy (float): Elementary distance in the y direction.
        """
        self._dx = dx
        self._dy = dy

    def compute(self, uvh: UVH) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the variable value.

        Args:
            uvh (UVH): Prognostic variables.

        Returns:
            torch.Tensor: Physical zonal velocity component.
        """
        return (uvh.u / self._dx, uvh.v / self._dy)


class PhysicalLayerDepthAnomaly(DiagnosticVariable[torch.Tensor]):
    """Physical layer depth anomaly."""

    _unit = "m"
    _name = "h_phys"
    _description = "Physical layer depth anomaly."

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


class VelocityFlux(DiagnosticVariable[tuple[torch.Tensor, torch.Tensor]]):
    """Velocity flux."""

    _unit = "s⁻¹"
    _name = "UV"
    _description = "Horizontal velocity flux."

    def __init__(self, dx: float, dy: float) -> None:
        """Instantiate the variable.

        Args:
            dx (float): Elementary distance in the x direction.
            dy (float): Elementary distance in the y direction.
        """
        self._dx = dx
        self._dy = dy

    def compute(self, uvh: UVH) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            UVH: Value
        """
        return (uvh.u / self._dx**2, uvh.v / self._dy**2)


class SurfaceHeightAnomaly(DiagnosticVariable[torch.Tensor]):
    """Surface height anomaly."""

    _unit = "m"
    _name = "eta"
    _description = "Surface height anomaly."

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
        # the prognostic variable is h^* = h dx dy
        # then uvh.h must be divided by dx dy
        return reverse_cumsum(self._h_phys.compute(uvh), dim=-3)

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
        # Bind the h_phys variable
        self._h_phys = self._h_phys.bind(state)
        return super().bind(state)


class Vorticity(DiagnosticVariable[torch.Tensor]):
    """Vorticity Diagnostic Variable."""

    _unit = "m².s⁻¹"
    _name = "omega"
    _description = "Vorticity."

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
        omega: torch.Tensor = (
            self._w_valid * curl_uv
            + self._w_cornerout_bound * (1 - self._slip_coef) * curl_uv
            + self._w_vertical_bound * alpha * dx_v
            - self._w_horizontal_bound * alpha * dy_u
        )
        return omega


class PhysicalVorticity(DiagnosticVariable[torch.Tensor]):
    """Physical vorticity."""

    _unit = "s⁻¹"
    _name = "omega_phys"
    _description = "Physical vorticity."

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
    ) -> BoundDiagnosticVariable[Self, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the vorticity variable
        self._vorticity = self._vorticity.bind(state)
        return super().bind(state)


class Pressure(DiagnosticVariable[torch.Tensor]):
    """Pressure."""

    _unit = "m².s⁻²"
    _name = "p"
    _description = "Pressure per unit of mass."

    def __init__(self, g_prime: float, eta: SurfaceHeightAnomaly) -> None:
        """Instantiate the pressure variable.

        Args:
            g_prime (float): Reduced gravity
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
    ) -> BoundDiagnosticVariable[Pressure, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the eta variable
        self._eta = self._eta.bind(state)
        return super().bind(state)


class PotentialVorticity(DiagnosticVariable[torch.Tensor]):
    """Potential Vorticity."""

    _unit = "s⁻¹"
    _name = "pv"
    _description = "Potential vorticity."

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
    ) -> BoundDiagnosticVariable[Self, torch.Tensor]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the vorticity variable
        self._vorticity = self._vorticity.bind(state)
        return super().bind(state)