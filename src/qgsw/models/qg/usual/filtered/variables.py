"""Variables for QGPSIQCollinearALpha models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from qgsw.models.qg.usual.variables import Psi2

if TYPE_CHECKING:
    from qgsw.fields.variables.prognostic_tuples import PSIQTAlpha
    from qgsw.filters.base import _Filter


class CollinearFilteredPsi2(Psi2):
    """Stream function in second layer for collinear filtered model."""

    def __init__(
        self,
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
        self._filter = filt
        self._offset_psi1 = offset_psi1
        self._offset_psi2 = offset_psi2

    def _compute(self, prognostic: PSIQTAlpha) -> torch.Tensor:
        """Compute the variable value.

        Args:
            prognostic (BasePrognosticUVH): Prognostic variables t, α,
            psi and q.
                ├── t: (n_ens)-shaped
                ├── α: (n_ens, 1, nx+1, ny+1)-shaped
                ├── psi: (n_ens, nl, nx+1, ny+1)-shaped
                └── q: (n_ens, nl, nx, ny)-shaped

        Returns:
            torch.Tensor: Stream function in second layer.
                └── (n_ens, 1, nx+1, ny+1)-shaped
        """
        psi1 = prognostic.psi[:, :1]
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
