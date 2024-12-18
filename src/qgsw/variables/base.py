"""Base variables."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qgsw.spatial.units._units import Unit
from qgsw.variables.scope import PointWise, Scope

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


if TYPE_CHECKING:
    import datetime
    from collections.abc import Iterator

    import numpy as np
    import torch

    from qgsw.run_summary import OutputFile
    from qgsw.variables.state import State
    from qgsw.variables.uvh import UVH

T = TypeVar("T")


class Variable:
    """Variable."""

    _unit: Unit
    _name: str
    _description: str
    _scope: Scope

    @property
    def unit(self) -> Unit:
        """Variable unit."""
        return self._unit

    @property
    def name(self) -> str:
        """Variable name."""
        return self._name

    @property
    def description(self) -> str:
        """Variable description."""
        return self._description

    @property
    def scope(self) -> Scope:
        """Variable scope."""
        return self._scope

    def __repr__(self) -> str:
        """Variable string representation."""
        return f"{self._description}: {self._name} [{self.unit}]"

    def to_dict(self) -> dict[str, Any]:
        """Convert the variable to a dictionnary."""
        var_dict = {
            "name": self.name,
            "unit": self.unit,
            "description": self.description,
        }
        return var_dict | self._scope.to_dict()

    @classmethod
    def get_name(cls) -> str:
        """Retrieve the name of the variable.

        Returns:
            str: Variable name.
        """
        return cls._name


class PrognosticVariable(Variable):
    """Prognostic variable."""

    _scope = PointWise()

    def __init__(self, initial: torch.Tensor) -> None:
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
        return self._data


class DiagnosticVariable(Variable, ABC):
    """Diagnostic Variable Base Class."""

    def __repr__(self) -> str:
        """Variable representation."""
        return super().__repr__() + " (Diagnostic)"

    @abstractmethod
    def compute(self, uvh: UVH) -> torch.Tensor:
        """Compute the value of the variable.

        Args:
            uvh (UVH): Prognostic variables
        """

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

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to anotehr state if required.

        Args:
            state (State): State to bound to

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        if state is not self._state:
            return self._var.bind(state)
        return self


class ParsedVariable(Variable):
    """Variable parsed from output file."""

    def __init__(
        self,
        name: str,
        unit: str,
        description: str,
        scope: Scope,
        outputs: list[OutputFile],
    ) -> None:
        """Instantiate the variable.

        Args:
            name (str): Variable name.
            unit (str): Variable unit.
            description (str): Variable description.
            scope (Scope): Variable scope.
            outputs (list[OutputFile]): Ouputs files.
        """
        self._name = name
        self._unit = Unit(unit)
        self._description = description
        self._scope = scope
        self._outputs = outputs

    def _from_output(
        self,
        output: OutputFile,
    ) -> np.ndarray:
        return output.read()[self.name]

    def datas(self) -> Iterator[np.ndarray]:
        """Data from the outputs.

        Yields:
            Iterator[np.ndarray]: Data iterator.
        """
        return iter(self._from_output(output) for output in self._outputs)

    def steps(self) -> Iterator[int]:
        """Sorted list of steps.

        Yields:
            Iterator[float]: Steps iterator.
        """
        return (output.step for output in iter(self._outputs))

    def timesteps(self) -> Iterator[datetime.timedelta]:
        """Sorted list of timesteps.

        Yields:
            Iterator[datetime.timedelta]: Timesteps iterator.
        """
        return (output.timestep for output in iter(self._outputs))

    def seconds(self) -> Iterator[float]:
        """Sorted list of seconds.

        Yields:
            Iterator[float]: Seconds iterator.
        """
        return (output.second for output in iter(self._outputs))

    @classmethod
    def from_dict(cls, dic: dict, outputs: list[OutputFile]) -> Self:
        """Instantiate the variable from a dictionnary.

        Args:
            dic (dict): Dictionnary to use.
            outputs (list[OutputFile]): Ouputs files.

        Returns:
            Self: Parsed Variable.
        """
        return cls(
            name=dic["name"],
            unit=dic["unit"],
            description=dic["description"],
            scope=Scope.from_dict(dic),
            outputs=outputs,
        )
