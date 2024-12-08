"""Base variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, Self, TypeVar

if TYPE_CHECKING:
    import torch

    from qgsw.models.variables.state import State
    from qgsw.models.variables.uvh import UVH

T = TypeVar("T")


class Variable(Generic[T]):
    """Variable."""

    _unit: str
    _name: str
    _description: str

    @property
    def unit(self) -> str:
        """Variable unit."""
        return self._unit

    def __repr__(self) -> str:
        """Variable string representation."""
        return f"Variable {self._name} [{self.unit}]: {self._description}"


class PrognosticVariable(Variable[T]):
    """Prognostic variable."""

    def __init__(self, initial: T) -> None:
        """Instantiate the variable.

        Args:
            initial (T): Initial value.
        """
        self._data = initial

    def __repr__(self) -> str:
        """Variable representation."""
        return super().__repr__() + " (Prognostic)"

    def __mul__(self, other: float) -> Self:
        """Left mutlitplication."""
        self._data.__mul__(other)
        return self

    def __rmul__(self, other: float) -> Self:
        """Right multiplication."""
        return self.__mul__(other)

    def __add__(self, other: Self) -> Self:
        """Addition."""
        self._data.__add__(other)
        return self

    def __sub__(self, other: Self) -> Self:
        """Substraction."""
        return self.__add__(-1 * other)

    def get(self) -> T:
        """Variable value.

        Returns:
            torch.Tensor: Value of the variable.
        """
        return self._data


class DiagnosticVariable(Variable[T], ABC):
    """Diagnostic Variable Base Class."""

    def __repr__(self) -> str:
        """Variable representation."""
        return super().__repr__() + " (Diagnostic)"

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

    @property
    def up_to_date(self) -> bool:
        """Whether the variable must be updated or not."""
        return self._up_to_date

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
