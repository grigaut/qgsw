"""Compute errors."""

import torch

from qgsw.fields.errors.base import PointWiseError
from qgsw.fields.variables.prognostic_tuples import BasePrognosticTuple


class RMSE(PointWiseError):  # noqa: N818
    """RMSE."""

    _name = "rmse"
    _description = "Root mean square error."

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
        value = self._var.compute(prognostic)
        value_ref = self._var_ref.compute(prognostic_ref)
        return torch.square(value - value_ref)

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
        return torch.sqrt(self._compute(prognostic, prognostic_ref))

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
        value = self._var.compute(prognostic)
        value_ref = self._var_ref.compute(prognostic_ref)
        point_wise = torch.square(value - value_ref)
        mean_err = torch.mean(point_wise, dim=(-1, -2))
        return torch.sqrt(mean_err)

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
        value = self._var.compute(prognostic)
        value_ref = self._var_ref.compute(prognostic_ref)
        point_wise = torch.square(value - value_ref)
        mean_err = torch.mean(point_wise, dim=(-1, -2, -3))
        return torch.sqrt(mean_err)
