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
            psi and q.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├── q: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function in second layer.
                └── (n_ens, 1, nx+1, ny+1)-shaped
        """
        return prognostic.psi[:, 1:2, :, :]


class Psi21L(Psi2):
    """PSi2 for one layer models."""

    def _compute(self, prognostic: BasePrognosticPSIQ) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticUVH): Prognostic variables
            psi and q.
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                ├── q: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function in second layer.
                └── (n_ens, 1, nx+1, ny+1)-shaped
        """
        return torch.zeros_like(prognostic.psi[:, :1])
