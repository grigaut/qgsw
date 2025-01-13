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
        return torch.square(
            self._var.compute(prognostic).__getitem__(self.slices)
            - self._var_ref.compute(prognostic_ref).__getitem__(self.slices),
        )

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
        return torch.sqrt(
            torch.sum(self._compute(prognostic, prognostic_ref), dim=(-1, -2)),
        )

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
        return torch.sqrt(
            torch.sum(
                self._compute(prognostic, prognostic_ref),
                dim=(-1, -2, -3),
            ),
        )
