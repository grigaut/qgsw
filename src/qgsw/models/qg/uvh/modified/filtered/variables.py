"""Variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.fields.variables.physical import (
    Psi2,
    StreamFunctionFromVorticity,
)

if TYPE_CHECKING:
    from qgsw.fields.variables.tuples import (
        UVHTAlpha,
    )
    from qgsw.filters.base import _Filter


class CollinearFilteredPsi2(Psi2):
    """Stream function in second layer for collinear filtered model."""

    def __init__(
        self,
        psi_vort: StreamFunctionFromVorticity,
        filt: _Filter,
        offset_psi1: torch.Tensor | None = None,
        offset_psi2: torch.Tensor | None = None,
    ) -> None:
        """Instantiate the variable.

        Args:
            psi_vort (StreamFunctionFromVorticity): Stream Function.
            filt (_Filter): Filter to apply to top stream function.
            offset_psi1 (torch.Tensor | None): Offset to apply to top
                stream function.
            offset_psi2 (torch.Tensor | None): Offset to apply to
                bottom stream function.
        """
        super().__init__(psi_vort)
        self._filter = filt
        self._offset_psi1 = offset_psi1
        self._offset_psi2 = offset_psi2

    def _compute(self, prognostic: UVHTAlpha) -> torch.Tensor:
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
        if self._offset_psi1 is None:
            offset_psi1 = torch.zeros_like(psi1)
        else:
            offset_psi1 = self._offset_psi1
        if self._offset_psi2 is None:
            offset_psi2 = torch.zeros_like(psi1)
        else:
            offset_psi2 = self._offset_psi2
        psi_filt = (
            self._filter(psi1[0, 0] - offset_psi1[0, 0])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return prognostic.alpha * psi_filt + offset_psi2
