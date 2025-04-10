"""Variables for usual QGPSIQ."""

import torch

from qgsw.fields.scope import Scope
from qgsw.fields.variables.base import DiagnosticVariable
from qgsw.fields.variables.prognostic_tuples import BasePrognosticPSIQ
from qgsw.utils.units._units import Unit


class Psi2(DiagnosticVariable):
    """Stream function variable in second layer.

    └── (n_ens, nl, nx, ny)-shaped
    """

    _unit = Unit.M2S_1
    _name = "psi2"
    _description = "Stream function in second layer"
    _scope = Scope.POINT_WISE

    def _compute(self, prognostic: BasePrognosticPSIQ) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticUVH): Prognostic variables
            (t, α,) u,v and h.
                ├── (t: (n_ens,)-shaped)
                ├── (α: (n_ens,)-shaped)
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function in second layer.
                └── (n_ens, 1, nx, ny)-shaped
        """
        return prognostic.psi[:, 1:2, :, :]
