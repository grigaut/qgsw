"""Base variables."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qgsw.fields.base import Field

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

    from qgsw.fields.variables.state import StateUVH
    from qgsw.fields.variables.tuples import BaseTuple
    from qgsw.utils.units._units import Unit


class Variable(Field):
    """Variable."""

    _unit: Unit

    @property
    def unit(self) -> Unit:
        """Variable unit."""
        return self._unit

    @cached_property
    def id(self) -> uuid.UUID:
        """Variable id."""
        return uuid.uuid1()

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

    _require_time = False
    _require_alpha = False

    @property
    def require_time(self) -> bool:
        """Whether the variable require time to be computed."""
        return self._require_time

    @property
    def require_alpha(self) -> bool:
        """Whether the variable require alpha to be computed."""
        return self._require_alpha

    @abstractmethod
    def _compute(self, vars_tuple: BaseTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (BaseTuple): Prognostic variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """

    def compute(self, vars_tuple: BaseTuple) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (BaseTuple): Prognostic variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        return self._compute(vars_tuple).__getitem__(self.slices)

    def compute_no_slice(
        self,
        vars_tuple: BaseTuple,
    ) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            vars_tuple (BaseTuple): Prognostic variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped
        """
        return self._compute(vars_tuple)

    def bind(self, state: StateUVH) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (StateUVH): StateUVH to bind the variable to.

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

    def __init__(self, state: StateUVH, variable: DiagVar) -> None:
        """Instantiate the bound variable.

        Args:
            state (StateUVH): StateUVH to bound to.
            variable (DiagnosticVariable): Variable to bound.
        """
        self._var = variable
        self._state = state
        self._unit = self._var.unit
        self._name = self._var.name
        self._description = self._var.description
        self._scope = self._var.scope

    @cached_property
    def id(self) -> uuid.UUID:
        """Variable id."""
        return self._var.id

    @property
    def require_time(self) -> bool:
        """Whether the variable require time to be computed."""
        return self._var.require_time

    @property
    def require_alpha(self) -> bool:
        """Whether the variable require alpha to be computed."""
        return self._var.require_alpha

    @property
    def up_to_date(self) -> bool:
        """Whether the variable must be updated or not."""
        return self._up_to_date

    def compute_no_slice(
        self,
        vars_tuple: BaseTuple,
    ) -> torch.Tensor:
        """Compute the variable value if outdated.

        Args:
            vars_tuple (BaseTuple): BaseTuple.

        Returns:
            torch.Tensor: Variable value.
        """
        if self._up_to_date:
            return self._value
        self._up_to_date = True
        self._value = self._var.compute(vars_tuple)
        return self._value

    def compute(self, vars_tuple: BaseTuple) -> torch.Tensor:
        """Compute the variable value if outdated.

        Args:
            vars_tuple (BaseTuple): BaseTuple.

        Returns:
            torch.Tensor: Variable value.
        """
        return self.compute_no_slice(vars_tuple).__getitem__(self.slices)

    def get(self) -> torch.Tensor:
        """Get the variable value.

        Returns:
            torch.Tensor: Variable value.
        """
        return self.compute(self._state.prognostic)

    def outdated(self) -> None:
        """Set the variable as outdated.

        Next call to 'get' or 'compute' will recompute the value.
        """
        self._up_to_date = False

    def bind(self, state: StateUVH) -> BoundDiagnosticVariable[DiagVar]:
        """Bind the variable to another state if required.

        Args:
            state (StateUVH): StateUVH to bound to

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        if state is not self._state:
            return self._var.bind(state)
        return self
