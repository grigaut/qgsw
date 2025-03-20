"""Variables."""

from torch._tensor import Tensor

from qgsw.fields.variables.dynamics import (
    Psi2,
    StreamFunctionFromVorticity,
)
from qgsw.fields.variables.prognostic_tuples import (
    UVHTAlpha,
)
from qgsw.filters.base import _Filter


class CollinearFilteredPsi2(Psi2):
    """Stream function in second layer for collinear filtered model."""

    def __init__(
        self,
        psi_vort: StreamFunctionFromVorticity,
        filt: _Filter,
    ) -> None:
        """Instantiate the variable.

        Args:
            psi_vort (StreamFunctionFromVorticity): Stream Function.
            filt (_Filter): Filter to apply to top stream function.
        """
        super().__init__(psi_vort)
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
            torch.Tensor: Stream function in second layer.
                └── (n_ens, 1, nx, ny)-shaped
        """
        psi1 = self._psi_vort.compute_no_slice(prognostic)[:, :1]
        psi_filt = self._filter(psi1[0, 0]).unsqueeze(0).unsqueeze(0)
        return prognostic.alpha * psi_filt
