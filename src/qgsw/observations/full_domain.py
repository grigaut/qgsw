"""Full domain masks."""

import torch
from torch import Tensor

from qgsw.observations.base import BaseObservationMask


class FullDomainMask(BaseObservationMask):
    """Mask for full domain observations."""

    def at_time(self, time: Tensor) -> Tensor:  # noqa: ARG002
        """Compute observation mask at given time..

        Args:
            time (torch.Tensor): Time of the observation
                (for compatibility only).

        Returns:
            torch.Tensor: Mask.
                └── (nx, ny) shaped
        """
        return torch.ones_like(self._x, dtype=torch.bool)
