"""Base variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qgsw.fields.base import Field
from qgsw.fields.scope import Scope

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


if TYPE_CHECKING:
    try:
        from types import EllipsisType
    except ImportError:
        EllipsisType = type(...)

    import torch

    from qgsw.fields.variables.state import State
    from qgsw.fields.variables.uvh import UVH
    from qgsw.spatial.units._units import Unit


class Variable(Field):
    """Variable."""

    _unit: Unit

    @property
    def unit(self) -> Unit:
        """Variable unit."""
        return self._unit

    def __repr__(self) -> str:
        """Variable string representation."""
        return super().__repr__() + f" [{self.unit.value}]"

    def to_dict(self) -> dict[str, Any]:
        """Convert the variable to a dictionnary."""
        return {
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
        }

    @classmethod
    def get_unit(cls) -> Unit:
        """Retrieve the unit of the variable.

        Returns:
            Unit: Variable unit.
        """
        return cls._unit


class PrognosticVariable(ABC, Variable):
    """Prognostic variable."""

    _scope = Scope.POINT_WISE

    @Field.slices.setter
    def slices(
        self,
        slices: list[slice | EllipsisType],  # type: ignore  # noqa: ARG002, PGH003
    ) -> None:
        """Slice setter."""
        msg = "Impossible to define slice for PrognosticVariable."
        raise PermissionError(msg)

    def __init__(self, initial: torch.Tensor) -> None:
        """Instantiate the variable.

        Args:
            initial (torch.Tensor): Initial value.
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

    def update(self, data: torch.Tensor) -> None:
        """Update the variable value.

        Args:
            data (torch.Tensor): New value for the variable.

        Raises:
            ValueError: If the value shape does not match.
        """
        if self._data.shape != data.shape:
            msg = (
                f"Invalid shape, expected {self._data.shape}"
                f", received {data.shape}."
            )
            raise ValueError(msg)
        self._data = data

    def get(self) -> torch.Tensor:
        """Variable value.

        Returns:
            torch.Tensor: Value of the variable.
        """
        return self._data.__getitem__(self.slices)


class DiagnosticVariable(Variable, ABC):
    """Diagnostic Variable Base Class."""

    @abstractmethod
    def _compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostic variables
        """

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostic variables
        """
        return self._compute(uvh).__getitem__(self.slices)

    def compute_no_slice(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostic variables
        """
        return self._compute(uvh)

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
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


class BoundDiagnosticVariable(Variable, Generic[DiagVar]):
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
        self._name = self._var.name
        self._description = self._var.description
        self._scope = self._var.scope

    @property
    def up_to_date(self) -> bool:
        """Whether the variable must be updated or not."""
        return self._up_to_date

    def compute_no_slice(self, uvh: UVH) -> torch.Tensor:
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

    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the variable value if outdated.

        Args:
            uvh (UVH): UVH.

        Returns:
            torch.Tensor: Variable value.
        """
        return self.compute_no_slice(uvh).__getitem__(self.slices)

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

    def bind(self, state: State) -> BoundDiagnosticVariable[DiagVar]:
        """Bind the variable to another state if required.

        Args:
            state (State): State to bound to

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        if state is not self._state:
            return self._var.bind(state)
        return self
