"""Compute errors."""

import torch

from qgsw.fields.errors.base import PointWiseError
from qgsw.fields.variables.uvh import UVH


class RMSE(PointWiseError):  # noqa: N818
    """RMSE."""

    _name = "rmse"
    _description = "Root mean square error."

    def _compute(self, uvh: UVH, uvh_ref: UVH) -> torch.Tensor:
        """Compute error.

        Args:
            uvh (UVH): Prognostic variables value.
            uvh_ref (UVH): Reference prognostic variables value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.abs(self._var.compute(uvh) - self._var.compute(uvh_ref))
