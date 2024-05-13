"""Base class for perturbations."""

from abc import ABCMeta, abstractmethod

import torch


class _Perturbation(metaclass=ABCMeta):
    """Perturbation base class."""

    @abstractmethod
    def retrieve_pressure(self) -> torch.Tensor:
        """Retrieve pressure values."""
