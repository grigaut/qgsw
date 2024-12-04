"""Variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, NamedTuple, Self, TypeVar

import torch
import torch.nn.functional as F  # noqa: N812

from qgsw.models.core import finite_diff
from qgsw.models.core.finite_diff import reverse_cumsum
from qgsw.models.core.utils import OptimizableFunction
from qgsw.spatial.core.grid_conversion import points_to_surfaces

if TYPE_CHECKING:
    from qgsw.masks import Masks


class UVH(NamedTuple):
    """Zonal velocity, meridional velocity and layer thickness."""

    u: torch.Tensor
    v: torch.Tensor
    h: torch.Tensor

    def __mul__(self, value: float) -> UVH:
        """Left mutlitplication."""
        return UVH(self.u * value, self.v * value, self.h * value)

    def __rmul__(self, value: float) -> UVH:
        """Right multiplication."""
        return self.__mul__(value)

    def __add__(self, value: UVH) -> UVH:
        """Addition."""
        return UVH(self.u + value.u, self.v + value.v, self.h + value.h)

    def __sub__(self, value: UVH) -> UVH:
        """Substraction."""
        return UVH(self.u - value.u, self.v - value.v, self.h - value.h)


class DiagnosticVariable(ABC):
    """Diagnostic Variable Base Class."""

    def __repr__(self) -> str:
        """Variable string representation."""
        return f"Diagnostic Variable: {self.__class__}"

    @abstractmethod
    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostic variables
        """

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a ggivent state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        bound_var = BoundDiagnosticVariable(state, self)
        state.add_bound_diagnostic_variable(bound_var)
        return bound_var


T = TypeVar("T", bound=DiagnosticVariable)


class BoundDiagnosticVariable(DiagnosticVariable, Generic[T]):
    """Bound variable."""

    _up_to_date = False

    def __init__(self, state: State, variable: T) -> None:
        """Instantiate the bound variable.

        Args:
            state (State): State to bound to.
            variable (DiagnosticVariable): Variable to bound.
        """
        self._var = variable
        self._state = state

    def __repr__(self) -> str:
        """Bound variable representation."""
        return "Bound " + self._var.__repr__()

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable value if outdated.

        Args:
            uvh (UVH): UVH.

        Returns:
            torch.Tensor: Variable value.
        """
        if self._up_to_date:
            return self._value
        self._up_to_date = True
        self._value = self._var.compute(uvh)
        return self._value

    def get(self) -> torch.Tensor:
        """Get the variable value.

        Returns:
            torch.Tensor: Variable value.
        """
        return self.compute(self._state.uvh)

    def outdated(self) -> None:
        """Set the variable as outdated.

        Next call to 'get' or 'compute' will recompute the value.
        """
        self._up_to_date = False

    def bind(self, state: State) -> BoundDiagnosticVariable:
        """Bind the variable to anotehr state if required.

        Args:
            state (State): State to bound to

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        if state is not self._state:
            return self._var.bind(state)
        return self


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
            uvh (UVH): Prognostioc variables

        Returns:
            torch.Tensor: Value
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
    """Surface heogh anomaly."""

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
            torch.Tensor: Value
        """
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
            torch.Tensor: Value
        """
        return torch.cumsum(self._g_prime * self._eta.compute(uvh), dim=-3)

    def bind(self, state: State) -> BoundDiagnosticVariable:
        """Bind the variable to a ggivent state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the eta variable
        self._eta = self._eta.bind(state)
        return super().bind(state)


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


class State:
    """State: wrapper for UVH state variables.

    This wrapper links uvh variables to diagnostic variables.
    Diagnostic variables can be bound to the state so that they are updated
    only when the state has changed.
    """

    def __init__(
        self,
        uvh: UVH,
    ) -> None:
        """Instantiate state.

        Args:
            uvh (UVH): Prognostic variables.
        """
        self.unbind()
        self.uvh = uvh

    @property
    def uvh(self) -> UVH:
        """Prognostic variables."""
        return self._uvh

    @uvh.setter
    def uvh(self, uvh: UVH) -> None:
        self._uvh = uvh
        self._updated()

    @property
    def u(self) -> torch.Tensor:
        """Prognostic zonal velocity."""
        return self._uvh.u

    @property
    def v(self) -> torch.Tensor:
        """Prognostic meriodional velocity."""
        return self._uvh.v

    @property
    def h(self) -> torch.Tensor:
        """Prognostic layer thickness anomaly."""
        return self._uvh.h

    @property
    def diag_vars(self) -> list[BoundDiagnosticVariable]:
        """List of diagnostic variables."""
        return self._diag

    def update(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> None:
        """Update prognostic variables.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meriodional velocity.
            h (torch.Tensor): Surface height anomaly.
        """
        self.uvh = UVH(u, v, h)

    def _updated(self) -> None:
        """Update diagnostic variables."""
        for var in self.diag_vars:
            var.outdated()

    def add_bound_diagnostic_variable(
        self,
        variable: BoundDiagnosticVariable,
    ) -> None:
        """Add a diagnostic variable.

        Args:
            variable (BoundDiagnosticVariable): Variable.
        """
        self._diag.add(variable)

    def unbind(self) -> None:
        """Unbind all variables from state."""
        self._diag: set[BoundDiagnosticVariable] = set()

    @classmethod
    def steady(
        cls,
        n_ens: int,
        nl: int,
        nx: int,
        ny: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Self:
        """Instaitate a steady state with zero-filled prognostic variables.

        Args:
            n_ens (int): Number of ensembles.
            nl (int): Number of layers.
            nx (int): Number of points in the x direction.
            ny (int): Number of points in the y direction.
            dtype (torch.dtype): Data type.
            device (torch.device): Device to use.

        Returns:
            Self: State.
        """
        h = torch.zeros(
            (n_ens, nl, nx, ny),
            dtype=dtype,
            device=device,
        )
        u = torch.zeros(
            (n_ens, nl, nx + 1, ny),
            dtype=dtype,
            device=device,
        )
        v = torch.zeros(
            (n_ens, nl, nx, ny + 1),
            dtype=dtype,
            device=device,
        )
        return cls.from_tensors(u, v, h)

    @classmethod
    def from_tensors(
        cls,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> Self:
        """Instantiate the state from tensors.

        Args:
            u (torch.Tensor): Zonal velocity.
            v (torch.Tensor): Meridional velocity.
            h (torch.Tensor): Surface height anomaly.

        Returns:
            Self: State.
        """
        return cls(UVH(u, v, h))
