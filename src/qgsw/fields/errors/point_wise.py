"""Compute errors."""

import torch

from qgsw.fields.errors.base import PointWiseError
from qgsw.fields.variables.uvh import BasePrognosticTuple


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
        norm = torch.max(
            torch.square(value_ref).flatten(-2, -1),
            dim=-1,
            keepdim=True,
        ).values.unsqueeze(-1)
        point_wise = torch.square(value - value_ref)
        sum_errs = torch.sum(point_wise / norm, dim=(-1, -2))
        _, _, nx, ny = point_wise.shape
        return torch.sqrt(sum_errs / (nx * ny))

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
        norm = torch.mean(
            torch.square(value_ref).flatten(-3, -1),
            dim=-1,
        )
        point_wise = torch.square(value - value_ref)
        sum_errs = torch.sum(point_wise / norm, dim=(-1, -2, -3))
        _, nl, nx, ny = point_wise.shape
        return torch.sqrt(sum_errs / (nl * nx * ny))
