"""Collinearity Coefficients."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from torch._tensor import Tensor

from qgsw.fields.scope import Scope
from qgsw.fields.variables.base import (
    BoundDiagnosticVariable,
    DiagnosticVariable,
)
from qgsw.fields.variables.dynamics import StreamFunction
from qgsw.fields.variables.state import State
from qgsw.fields.variables.uvh import PrognosticTuple
from qgsw.specs import DEVICE
from qgsw.utils.least_squares_regression import (
    perform_linear_least_squares_regression,
)
from qgsw.utils.units._units import Unit


class LSRSFInferredAlpha(DiagnosticVariable):
    """Inferred collinearity from the streamfunction.

    Performs linear least squares regression to infer alpha.
    """

    _unit = Unit._
    _name = "alpha_lsr_sf"
    _description = "LSR-Stream function inferred coefficient"
    _scope = Scope.ENSEMBLE_WISE

    def __init__(self, psi_ref: StreamFunction) -> None:
        """Instantiate the variable.

        Args:
            psi_ref (StreamFunction): Reference stream function.
        """
        self._psi = psi_ref

    def _compute(self, prognostic: PrognosticTuple) -> Tensor:
        """Compute the value of alpha.

        Args:
            prognostic (PrognosticTuple): Prognostic variables.

        Returns:
            Tensor: Alpha
        """
        psi = self._psi.compute_no_slice(prognostic)
        psi_1 = psi[:, 0, ...]  # (n_ens,nx,ny) -shaped
        psi_2 = psi[:, 1, ...]  # (n_ens,nx,ny) -shaped

        x = psi_1.flatten(-2, -1).unsqueeze(1)  # (n_ens,1,nx*ny) -shaped
        x = x.transpose(-2, -1)  # (n_ens,nx*ny,1) -shaped
        y = psi_2.flatten(-2, -1)  # (n_ens,nx*ny) -shaped

        try:
            return perform_linear_least_squares_regression(x, y)[:, 0]
        except torch.linalg.LinAlgError:
            return torch.zeros(
                (y.shape[0],),
                dtype=torch.float64,
                device=DEVICE.get(),
            )

    def bind(self, state: State) -> BoundDiagnosticVariable[Self]:
        """Bind the variable to a given state.

        Args:
            state (State): State to bind the variable to.

        Returns:
            BoundDiagnosticVariable: Bound variable.
        """
        # Bind the psi variable
        self._psi = self._psi.bind(state)
        return super().bind(state)
