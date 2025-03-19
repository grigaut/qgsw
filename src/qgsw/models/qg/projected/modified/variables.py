"""Variables."""

import torch
from torch._tensor import Tensor

from qgsw.fields.variables.dynamics import (
    StreamFunctionFromVorticity,
)
from qgsw.fields.variables.prognostic_tuples import (
    UVHTAlpha,
)


class RefStreamFunctionFromVorticity(StreamFunctionFromVorticity):
    """Stream function from vorticity for collinear models."""

    def _compute(self, prognostic: UVHTAlpha) -> Tensor:
        """Compute the variable value.

        Args:
            prognostic (UVHTAlpha): Prognostic variables
            t, α, u,v and h.
                ├── t: (n_ens,)-shaped
                ├── α: (n_ens, 1, nx, ny)-shaped
                ├── u: (n_ens, nl, nx+1, ny)-shaped
                ├── v: (n_ens, nl, nx, ny+1)-shaped
                └── h: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function.
                └── (n_ens, nl, nx, ny)-shaped
        """
        psi1 = super()._compute(prognostic)
        psi2 = torch.zeros_like(psi1)
        return torch.cat([psi1, psi2], dim=1)
