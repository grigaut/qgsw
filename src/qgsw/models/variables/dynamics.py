"""Dynamics variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.models.variables.core import (
    UVH,
    BoundDiagnosticVariable,
    DiagnosticVariable,
    State,
)
from qgsw.spatial.core.grid_conversion import points_to_surfaces

if TYPE_CHECKING:
    from qgsw.masks import Masks


class Vorticity(DiagnosticVariable):
    """Vorticity Diagnostic Variable."""

    def __init__(self, masks: Masks, slip_coef: float) -> None:
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
        u_ = F.pad(uvh.u, (1, 1, 0, 0))
        v_ = F.pad(uvh.v, (0, 0, 1, 1))
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


class PhysicalZonalVelocity(DiagnosticVariable):
    """Physical Zonal velocity Variable."""

    def __init__(self, dx: float) -> None:
        """Instantiate the variable.

        Args:
            dx (float): X step.
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


class PhysicalMeridionalVelocity(DiagnosticVariable):
    """Physical Zonal velocity Variable."""

    def __init__(self, dy: float) -> None:
        """Instantiate the variable.

        Args:
            dy (float): Y step.
        """
        self._dy = dy

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            torch.Tensor: Value
        """
        return uvh.v / self._dy**2


class SurfaceHeightAnomaly(DiagnosticVariable):
    """Surface heigh anomaly."""

    def __init__(self, area: float) -> None:
        """Instantiate variable.

        Args:
            area (float): Elementary area.
        """
        self._area = area

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            torch.Tensor: Surface height anomaly
        """
        # the prognostic variable is h^* = h dx dy
        # then uvh.h must be divided by dx dy
        return reverse_cumsum(uvh.h / self._area, dim=-3)


class Pressure(DiagnosticVariable):
    """Pressure."""

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

    def bind(self, state: State) -> BoundDiagnosticVariable:
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

    def __init__(
        self,
        omega: Vorticity,
        h_ref: torch.Tensor,
        area: float,
        f0: float,
    ) -> None:
        """Instantiate variable.

        Args:
            omega (Vorticity): Vorticity.
            h_ref (torch.Tensor): Reference heights.
            area (float): Elementary area.
            f0 (float): Coriolis parameter.
        """
        self._h_ref = h_ref
        self._area = area
        self._f0 = f0
        self._omega = omega
        self._interp = OptimizableFunction(points_to_surfaces)

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostioc variables

        Returns:
            torch.Tensor: Value
        """
        omega = self._interp(self._omega.compute(uvh))
        return omega / self._area - self._f0 * uvh.h / self._h_ref
