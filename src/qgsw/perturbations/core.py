"""Main perturbation class."""

import torch

from qgsw.perturbations.base import _Perturbation


class Perturbation:
    """Pertubation class."""

    def __init__(self, perturbation: _Perturbation) -> None:
        """Instantiate the perturbation."""
        self._perturbation = perturbation

    @property
    def type(self) -> str:
        """Perturbation type."""
        return self._perturbation.type

    def compute_initial_pressure(self) -> torch.Tensor:
        """Retrieve pressure values."""
        return self._perturbation.compute_initial_pressure()
