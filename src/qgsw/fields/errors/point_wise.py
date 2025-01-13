"""Compute errors."""

import torch

from qgsw.fields.errors.base import PointWiseError
from qgsw.fields.variables.uvh import PrognosticTuple


class RMSE(PointWiseError):  # noqa: N818
    """RMSE."""

    _name = "rmse"
    _description = "Root mean square error."

    def _compute(
        self,
        prognostic: PrognosticTuple,
        prognostic_ref: PrognosticTuple,
    ) -> torch.Tensor:
        """Compute error.

        Args:
            prognostic (PrognosticTuple): Prognostic variables value.
            prognostic_ref (PrognosticTuple): Reference prognostic variables
            value.

        Returns:
            torch.Tensor: Error.
        """
        value = self._var.compute_no_slice(prognostic)
        value_ref = self._var_ref.compute_no_slice(prognostic_ref)
        return torch.square(value - value_ref).__getitem__(self.slices)

    def compute_point_wise(
        self,
        prognostic: PrognosticTuple,
        prognostic_ref: PrognosticTuple,
    ) -> torch.Tensor:
        """Compute point-wise error.

        Args:
            prognostic (PrognosticTuple): Prognostic variables value.
            prognostic_ref (PrognosticTuple): Reference prognostic variables
            value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.sqrt(self._compute(prognostic, prognostic_ref))

    def compute_level_wise(
        self,
        prognostic: PrognosticTuple,
        prognostic_ref: PrognosticTuple,
    ) -> torch.Tensor:
        """Compute level-wise error.

        Args:
            prognostic (PrognosticTuple): Prognostic variables value.
            prognostic_ref (PrognosticTuple): Reference prognostic variables
            value.

        Returns:
            torch.Tensor: Error.
        """
        point_wise = self._compute(prognostic, prognostic_ref)
        sum_errs = torch.sum(point_wise, dim=(-1, -2))
        _, _, nx, ny = point_wise.shape
        return torch.sqrt(sum_errs / (nx * ny))

    def compute_ensemble_wise(
        self,
        prognostic: PrognosticTuple,
        prognostic_ref: PrognosticTuple,
    ) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            prognostic (PrognosticTuple): Prognostic variables value.
            prognostic_ref (PrognosticTuple): Reference prognostic variables
            value.

        Returns:
            torch.Tensor: Error.
        """
        point_wise = self._compute(prognostic, prognostic_ref)
        sum_errs = torch.sum(point_wise, dim=(-1, -2, -3))
        _, nl, nx, ny = point_wise.shape
        return torch.sqrt(sum_errs / (nl * nx * ny))
