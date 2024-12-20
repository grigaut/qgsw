"""Base class for errors."""

from abc import ABC, abstractmethod

import torch

from qgsw.fields.scope import EnsembleWise, LevelWise, PointWise
from qgsw.fields.variables.base import DiagnosticVariable, Variable
from qgsw.fields.variables.uvh import UVH


class EnsembleWiseError(ABC, Variable):
    """Base class for ensemble-wise errors."""

    _scope = EnsembleWise()

    def __init__(self, variable: DiagnosticVariable) -> None:
        """Instantiate the error.

        Args:
            variable (DiagnosticVariable): Variable to use to compute error.

        Raises:
            ValueError: If the variable scope is invalid.
        """
        if not variable.scope.point_wise_at_least:
            msg = "Only point wise variable are supported for such error."
            raise ValueError(msg)
        self._var = variable

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

    def compute_ensemble_wise(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return self._compute(uvh, uvh_ref)


class LevelWiseError(ABC, Variable):
    """Base class for level-wise errors."""

    _scope = LevelWise()

    def __init__(self, variable: DiagnosticVariable) -> None:
        """Instantiate the error.

        Args:
            variable (DiagnosticVariable): Variable to use to compute error.

        Raises:
            ValueError: If the variable scope is invalid.
        """
        if not variable.scope.level_wise_at_least:
            msg = "Only point wise variable are supported for such error."
            raise ValueError(msg)
        self._var = variable

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


class PointWiseError(ABC, Variable):
    """Base class for point-wise errors."""

    _scope = PointWise()

    def __init__(self, variable: DiagnosticVariable) -> None:
        """Instantiate the error.

        Args:
            variable (DiagnosticVariable): Variable to use to compute error.

        Raises:
            ValueError: If the variable scope is invalid.
        """
        if not variable.scope.ensemble_wise_at_least:
            msg = "Only point wise variable are supported for such error."
            raise ValueError(msg)
        self._var = variable

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
