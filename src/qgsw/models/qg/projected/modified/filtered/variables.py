"""Variables."""

import torch
from torch._tensor import Tensor

from qgsw.fields.variables.dynamics import (
    PhysicalVorticity,
    StreamFunctionFromVorticity,
)
from qgsw.fields.variables.prognostic_tuples import (
    UVHTAlpha,
)
from qgsw.filters.base import _Filter


class ColFiltStreamFunctionFromVorticity(StreamFunctionFromVorticity):
    """Stream function from vorticity for collinear models."""

    def __init__(
        self,
        vorticity: PhysicalVorticity,
        nx: int,
        ny: int,
        dx: float,
        dy: float,
        filt: _Filter,
    ) -> None:
        """Instantiate the variable.

        Args:
            vorticity (PhysicalVorticity): Physical vorticity.
            nx (int): Number of poitn in the x direction.
            ny (int): Number of points in the y direction.
            dx (float): Infinitesimal x step.
            dy (float): Infinitesimal y step.
            filt (_Filter): Filter to apply to top stream function.
        """
        super().__init__(vorticity, nx, ny, dx, dy)
        self._filter = filt

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
        psi_filt = self._filter(psi1[0, 0]).unsqueeze(0).unsqueeze(0)
        psi2 = prognostic.alpha * psi_filt
        return torch.cat([psi1, psi2], dim=1)
