"""Base class for errors."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from qgsw.fields.base import Field
from qgsw.fields.scope import Scope

if TYPE_CHECKING:
    try:
        from types import EllipsisType
    except ImportError:
        EllipsisType = type(...)
    from qgsw.fields.variables.base import DiagnosticVariable
    from qgsw.fields.variables.prognostic_tuples import BasePrognosticTuple


class Error(ABC, Field):
    """Error base class."""

    def __init__(
        self,
        variable: DiagnosticVariable,
        variable_ref: DiagnosticVariable,
    ) -> None:
        """Instantiate the error.

        Args:
            variable (DiagnosticVariable): Variable to use to compute error.
            variable_ref (DiagnosticVariable): Reference variable to use
            to compute error.

        Raises:
            ValueError: If the variable scope is invalid.
        """
        if variable.scope != self.scope or variable_ref.scope != self.scope:
            msg = "Only point wise variable are supported for such error."
            raise ValueError(msg)
        self._var = copy.deepcopy(variable)
        self._var_ref = copy.deepcopy(variable_ref)

    @Field.slices.setter
    def slices(self, slices: list[slice, EllipsisType]) -> None:  # type: ignore  # noqa: PGH003
        """Slice setter."""
        Field.slices.fset(self, slices)
        self._var.slices = slices
        self._var_ref.slices = slices

    def __repr__(self) -> str:
        """String representation of the error."""
        var_name = self._var.name
        var_unit = self._var.unit.value
        return super().__repr__() + f" on {var_name} ([{var_unit}])"

    @abstractmethod
    def _compute(
        self,
        prognostic: BasePrognosticTuple,
        prognostic_ref: BasePrognosticTuple,
    ) -> torch.Tensor:
        """Compute error.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables value.
            prognostic_ref (BasePrognosticTuple): Reference prognostic
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """


class EnsembleWiseError(Error):
    """Base class for ensemble-wise errors."""

    _scope = Scope.ENSEMBLE_WISE

    def compute_ensemble_wise(
        self,
        prognostic: BasePrognosticTuple,
        prognostic_ref: BasePrognosticTuple,
    ) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables value.
            prognostic_ref (BasePrognosticTuple): Reference prognostic
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        return self._compute(prognostic, prognostic_ref)


class LevelWiseError(Error):
    """Base class for level-wise errors."""

    _scope = Scope.LEVEL_WISE

    def compute_level_wise(
        self,
        prognostic: BasePrognosticTuple,
        prognostic_ref: BasePrognosticTuple,
    ) -> torch.Tensor:
        """Compute level-wise error.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables value.
            prognostic_ref (BasePrognosticTuple): Reference prognostic
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        return self._compute(prognostic, prognostic_ref)

    def compute_ensemble_wise(
        self,
        prognostic: BasePrognosticTuple,
        prognostic_ref: BasePrognosticTuple,
    ) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables value.
            prognostic_ref (BasePrognosticTuple): Reference prognostic
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.mean(self._compute(prognostic, prognostic_ref), dim=(-1))


class PointWiseError(Error):
    """Base class for point-wise errors."""

    _scope = Scope.POINT_WISE

    def compute_point_wise(
        self,
        prognostic: BasePrognosticTuple,
        prognostic_ref: BasePrognosticTuple,
    ) -> torch.Tensor:
        """Compute point-wise error.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables value.
            prognostic_ref (BasePrognosticTuple): Reference prognostic
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        return self._compute(prognostic, prognostic_ref)

    def compute_level_wise(
        self,
        prognostic: BasePrognosticTuple,
        prognostic_ref: BasePrognosticTuple,
    ) -> torch.Tensor:
        """Compute level-wise error.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables value.
            prognostic_ref (BasePrognosticTuple): Reference prognostic
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.mean(
            self._compute(prognostic, prognostic_ref),
            dim=(-1, -2),
        )

    def compute_ensemble_wise(
        self,
        prognostic: BasePrognosticTuple,
        prognostic_ref: BasePrognosticTuple,
    ) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            prognostic (BasePrognosticTuple): Prognostic variables value.
            prognostic_ref (BasePrognosticTuple): Reference prognostic
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.mean(
            self._compute(prognostic, prognostic_ref),
            dim=(-1, -2, -3),
        )
