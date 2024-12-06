"""Base classes for variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, Self, TypeVar

import torch


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


T = TypeVar("T")


class DiagnosticVariable(ABC, Generic[T]):
    """Diagnostic Variable Base Class."""

    _unit: str
    _to_bind: list | None

    @property
    def unit(self) -> str:
        """Variable unit."""
        return self._unit

    def __repr__(self) -> str:
        """Variable string representation."""
        return f"Diagnostic Variable: {self.__class__} [{self.unit}]"

    @abstractmethod
    def compute(self, uvh: UVH) -> T:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostic variables
        """

    def bind(self, state: State) -> BoundDiagnosticVariable[Self, T]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        bound_var = BoundDiagnosticVariable(state, self)
        state.add_bound_diagnostic_variable(bound_var)
        return bound_var


DiagVar = TypeVar("DiagVar", bound=DiagnosticVariable)


class BoundDiagnosticVariable(DiagnosticVariable, Generic[DiagVar, T]):
    """Bound variable."""

    _up_to_date = False

    def __init__(self, state: State, variable: DiagVar) -> None:
        """Instantiate the bound variable.

        Args:
            state (State): State to bound to.
            variable (DiagnosticVariable): Variable to bound.
        """
        self._var = variable
        self._state = state
        self._unit = self._var.unit

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

    def get(self) -> T:
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
