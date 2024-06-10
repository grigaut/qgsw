"""Modified QG Model with Colinear Sublayer Behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qgsw.models.exceptions import InvalidLayersDefinitionError
from qgsw.models.qg.core import QG

if TYPE_CHECKING:
    import torch


class QGColinearSublayer(QG):
    """Modified QG model implementing CoLinear Sublayer Behavior."""

    _supported_layers_nb: int = 2

    def _set_H(self, h: torch.Tensor) -> torch.Tensor:  # noqa: N802
        """Perform additional validation over H.

        Args:
            h (torch.Tensor): Layers thickness.

        Raises:
            ValueError: if H is not constant in space

        Returns:
            torch.Tensor: H
        """
        h = super()._set_H(h)
        if self.space.nl != self._supported_layers_nb:
            msg = (
                "QGColinearSublayer can only support"
                f"{self._supported_layers_nb} layers."
            )
            raise InvalidLayersDefinitionError(msg)
        return h

    def crop_prognostic_variables(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Crops prognostic variables.

        Args:
            u (torch.Tensor): Zonal Velocity
            v (torch.Tensor): Meridional Velocity
            h (torch.Tensor): Layer Thickness

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: u,v and h
            in top layer.
        """
        return u[..., 0, :, :], v[..., 0, :, :], h[..., 0, :, :]
