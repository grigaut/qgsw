"""Compute errors."""

import torch

from qgsw.fields.errors.base import PointWiseError
from qgsw.fields.variables.tuples import BaseTuple


class RMSE(PointWiseError):  # noqa: N818
    """RMSE."""

    _name = "rmse"
    _description = "Root mean square error."

    def _compute(
        self,
        vars_tuple: BaseTuple,
        vars_tuple_ref: BaseTuple,
    ) -> torch.Tensor:
        """Compute error.

        Args:
            vars_tuple (BaseTuple):  Variables tuple value.
            vars_tuple_ref (BaseTuple): Reference tuple
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        value = self._var.compute(vars_tuple)
        value_ref = self._var_ref.compute(vars_tuple_ref)
        return torch.square(value - value_ref)

    def compute_point_wise(
        self,
        vars_tuple: BaseTuple,
        vars_tuple_ref: BaseTuple,
    ) -> torch.Tensor:
        """Compute point-wise error.

        Args:
            vars_tuple (BaseTuple):  Variables tuple value.
            vars_tuple_ref (BaseTuple): Reference tuple
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        return torch.sqrt(self._compute(vars_tuple, vars_tuple_ref))

    def compute_level_wise(
        self,
        vars_tuple: BaseTuple,
        vars_tuple_ref: BaseTuple,
    ) -> torch.Tensor:
        """Compute level-wise error.

        Args:
            vars_tuple (BaseTuple):  Variables tuple value.
            vars_tuple_ref (BaseTuple): Reference tuple
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        value = self._var.compute(vars_tuple)
        value_ref = self._var_ref.compute(vars_tuple_ref)
        point_wise = torch.square(value - value_ref)
        mean_err = torch.mean(point_wise, dim=(-1, -2))
        return torch.sqrt(mean_err)

    def compute_ensemble_wise(
        self,
        vars_tuple: BaseTuple,
        vars_tuple_ref: BaseTuple,
    ) -> torch.Tensor:
        """Compute ensemble-wise error.

        Args:
            vars_tuple (BaseTuple):  Variables tuple value.
            vars_tuple_ref (BaseTuple): Reference tuple
            variables.
            value.

        Returns:
            torch.Tensor: Error.
        """
        value = self._var.compute(vars_tuple)
        value_ref = self._var_ref.compute(vars_tuple_ref)
        point_wise = torch.square(value - value_ref)
        mean_err = torch.mean(point_wise, dim=(-1, -2, -3))
        return torch.sqrt(mean_err)
