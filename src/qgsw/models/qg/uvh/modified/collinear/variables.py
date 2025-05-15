"""Variables."""

from torch._tensor import Tensor

from qgsw.fields.variables.physical import (
    Psi2,
)
from qgsw.fields.variables.prognostic_tuples import (
    UVHTAlpha,
)


class CollinearPsi2(Psi2):
    """Stream function in second layer for collinear models."""

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
        return prognostic.alpha * psi1
