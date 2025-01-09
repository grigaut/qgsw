"""Base class for errors."""

from abc import ABC, abstractmethod
from types import EllipsisType

import torch

from qgsw.fields.base import Field
from qgsw.fields.scope import Scope
from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.uvh import UVH


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
        self._var = variable
        self._var_ref = variable_ref

    @Field.slices.setter
    def slices(self, slice_list: list[slice, EllipsisType]) -> None:
        """Slice setter."""
        Field.slices.fset(self, slice_list)
        self._var.slices = slice_list
        self._var_ref.slices = slice_list

    def __repr__(self) -> str:
        """String representation of the error."""
        var_name = self._var.name
        var_unit = self._var.unit.value
        return super().__repr__() + f" on {var_name} ([{var_unit}])"

    @abstractmethod
    def _compute(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """


class EnsembleWiseError(Error):
    """Base class for ensemble-wise errors."""

    _scope = Scope.ENSEMBLE_WISE

    def compute_ensemble_wise(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return self._compute(uvh, uvh_ref)


class LevelWiseError(Error):
    """Base class for level-wise errors."""

    _scope = Scope.LEVEL_WISE

    def compute_level_wise(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute level-wise error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return self._compute(uvh, uvh_ref)

    def compute_ensemble_wise(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.mean(self._compute(uvh, uvh_ref), dim=(-1))


class PointWiseError(Error):
    """Base class for point-wise errors."""

    _scope = Scope.POINT_WISE

    def compute_point_wise(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute point-wise error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return self._compute(uvh, uvh_ref)

    def compute_level_wise(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute level-wise error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.mean(self._compute(uvh, uvh_ref), dim=(-1, -2))

    def compute_ensemble_wise(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.mean(self._compute(uvh, uvh_ref), dim=(-1, -2, -3))
