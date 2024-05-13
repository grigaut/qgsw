"""Base class for perturbations."""

from abc import ABCMeta, abstractmethod

import torch


class _Perturbation(metaclass=ABCMeta):
    """Perturbation base class."""

    _type: str

    @property
    def type(self) -> str:
        """Perturbation type."""
        return self._type

    @abstractmethod
    def retrieve_pressure(self) -> torch.Tensor:
        """Retrieve pressure values."""
