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
        return torch.abs(
            self._var.compute(prognostic) - self._var.compute(prognostic_ref),
        )
